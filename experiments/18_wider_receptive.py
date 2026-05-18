# ============================================================
# FluidLM — Wider Receptive Field Experiment (Exp 18)
# experiments/18_wider_receptive.py
#
# Araştırma Sorusu (RQ6):
#   hop_k arttırılırsa (k=8, k=16) receptive field genişler mi
#   ve PPL düşer mi?
#
#   k=4  → scales=[1,2,4]   → 12 katman × 4  = 48  token
#   k=8  → scales=[1,2,4,8] → 12 katman × 8  = 96  token
#   k=16 → scales=[1,..16]  → 12 katman × 16 = 192 token
#
# Exp17 referans sonuçları (1B token):
#   FluidLM k=4 (FFT): 99.86 PPL
#   GPT-S:             61.95 PPL
#
# Kullanım:
#   python experiments/18_wider_receptive.py --tokens 1e9
#   python experiments/18_wider_receptive.py --tokens 1e9 --hop_ks 8
#   python experiments/18_wider_receptive.py --tokens 1e9 --hop_ks 8 16
#
# Çıktı:
#   results/18_fluid_k{K}_S_s{seed}_best.pt
#   results/18_wider_receptive.json
# ============================================================

import sys, os, math, time, json, argparse, random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─────────────────────────────────────────────────────────────────────────────
# Argümanlar
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="FluidLM Wider Receptive Field (Exp 18)")
parser.add_argument("--hop_ks",       nargs="+", type=int, default=[8, 16],
                    help="Test edilecek multihop_k değerleri (default: 8 16)")
parser.add_argument("--scale",        default="S", choices=["S", "M"])
parser.add_argument("--model",        default="fluid", choices=["fluid", "gpt", "both"])
parser.add_argument("--tokens",       type=float, default=1e9)
parser.add_argument("--seeds",        nargs="+", type=int, default=[42])
parser.add_argument("--batch_tokens", type=int,   default=None)
parser.add_argument("--seq_len",      type=int,   default=1024)
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--min_lr",       type=float, default=3e-5)
parser.add_argument("--warmup_frac",  type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--grad_clip",    type=float, default=1.0)
parser.add_argument("--dropout",      type=float, default=0.0)
parser.add_argument("--mlp_ratio",    type=int,   default=4)
parser.add_argument("--eval_interval",type=int,   default=500)
parser.add_argument("--eval_steps",   type=int,   default=100)
parser.add_argument("--wikitext_eval",action="store_true")
parser.add_argument("--out_dir",      default=None)
parser.add_argument("--compile",      action="store_true")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# DDP
# ─────────────────────────────────────────────────────────────────────────────

IS_DDP = int(os.environ.get("RANK", -1)) != -1
if IS_DDP:
    dist.init_process_group(backend="nccl")
    RANK       = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(LOCAL_RANK)
    IS_MASTER  = (RANK == 0)
else:
    RANK = 0; WORLD_SIZE = 1; LOCAL_RANK = 0; IS_MASTER = True

def master_print(*a, **kw):
    if IS_MASTER: print(*a, **kw, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cihaz
# ─────────────────────────────────────────────────────────────────────────────

device   = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
USE_BF16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE    = torch.bfloat16 if USE_BF16 else torch.float32
AUTOCAST = nullcontext() if DTYPE == torch.float32 else \
           torch.amp.autocast(device_type="cuda", dtype=DTYPE)

if device.type == "cuda":
    vram_gb  = torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1e9
    gpu_name = torch.cuda.get_device_properties(LOCAL_RANK).name
else:
    vram_gb, gpu_name = 0, "CPU"

master_print(f"[Device] {gpu_name}  VRAM={vram_gb:.0f}GB  dtype={DTYPE}")
master_print(f"[Exp18] hop_ks={args.hop_ks}  seeds={args.seeds}")

# ─────────────────────────────────────────────────────────────────────────────
# Ölçek Konfigürasyonları
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScaleConfig:
    name:    str
    d_model: int
    n_layers:int
    n_heads: int
    seq_len: int

SCALE_CONFIGS = {
    "S": ScaleConfig(name="S", d_model=768,  n_layers=12, n_heads=12, seq_len=1024),
    "M": ScaleConfig(name="M", d_model=1024, n_layers=24, n_heads=16, seq_len=1024),
}

# ─────────────────────────────────────────────────────────────────────────────
# Batch size
# ─────────────────────────────────────────────────────────────────────────────

def auto_batch(d_model: int, seq_len: int, vram_gb: float) -> tuple[int, int]:
    target = 500_000
    bytes_per_token = d_model * 28 * 2
    max_physical = max(1, int(vram_gb * 0.75 * 1e9 / (bytes_per_token * seq_len)))
    max_physical = min(max_physical, 32)
    grad_accum   = max(1, target // (max_physical * seq_len))
    return max_physical, grad_accum

BATCH_OVERRIDE = max(1, args.batch_tokens // args.seq_len) if args.batch_tokens else None

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

try:
    import tiktoken
    ENC    = tiktoken.get_encoding("gpt2")
    VOCAB  = ENC.n_vocab
    encode = lambda text: ENC.encode_ordinary(text)
    master_print(f"[Tokenizer] tiktoken gpt2  vocab={VOCAB}")
except ImportError:
    master_print("[WARNING] tiktoken bulunamadi — char-level fallback")
    ENC = None; VOCAB = None; encode = None

# ─────────────────────────────────────────────────────────────────────────────
# Veri Yükleme
# ─────────────────────────────────────────────────────────────────────────────

class TokenStream:
    def __init__(self, split: str, seq_len: int, device: torch.device):
        self.seq_len = seq_len; self.device = device
        self.buffer: List[int] = []
        self._load(split)

    def _load(self, split: str):
        global VOCAB, encode
        if ENC is not None:
            try:
                from datasets import load_dataset
                master_print(f"[Data] OpenWebText yukleniyor (split={split}, streaming=True)...")
                ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
                ds = ds.shuffle(seed=42, buffer_size=10_000)
                ds = ds.skip(990_000) if split == "val" else ds.take(990_000)
                self._stream = iter(ds)
                self._source = "openwebtext"
                master_print(f"[Data] OpenWebText stream hazir ({split})")
                return
            except Exception as e:
                master_print(f"[Data] OpenWebText yuklenemedi: {e}")

        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
        REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
        candidates = [os.path.join(REPO_ROOT, "data", "shakespeare.txt"),
                      os.path.join(SCRIPT_DIR, "shakespeare.txt")]
        data_path = next((p for p in candidates if os.path.exists(p)), None)
        if data_path is None:
            import urllib.request
            data_path = os.path.join(REPO_ROOT, "data", "shakespeare.txt")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
                "data/tinyshakespeare/input.txt", data_path)
        with open(data_path, encoding="utf-8") as f:
            text = f.read()
        if ENC is not None:
            ids = encode(text)
        else:
            chars = sorted(set(text)); VOCAB = len(chars)
            stoi  = {c: i for i, c in enumerate(chars)}
            encode = lambda s: [stoi[c] for c in s]
            ids   = encode(text)
        n = int(len(ids) * 0.9)
        ids = ids[:n] if split == "train" else ids[n:]
        self._tensor = torch.tensor(ids, dtype=torch.long)
        self._source = "shakespeare"

    def _next_from_stream(self, n_tokens: int) -> List[int]:
        buf: List[int] = []
        while len(buf) < n_tokens:
            try:
                buf.extend(encode(next(self._stream)["text"] + "\n"))
            except StopIteration:
                from datasets import load_dataset
                ds = load_dataset("Skylion007/openwebtext", split="train",
                                  streaming=True).shuffle(seed=random.randint(0, 9999))
                self._stream = iter(ds)
        return buf[:n_tokens]

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        need = (self.seq_len + 1) * batch_size
        if self._source == "openwebtext":
            if len(self.buffer) < need:
                self.buffer.extend(self._next_from_stream(need * 4))
            chunk = self.buffer[:need]; self.buffer = self.buffer[need:]
            t = torch.tensor(chunk, dtype=torch.long).view(batch_size, self.seq_len + 1)
        else:
            ix = torch.randint(len(self._tensor) - self.seq_len - 1, (batch_size,))
            t  = torch.stack([self._tensor[i:i+self.seq_len+1] for i in ix])
        return t[:, :-1].to(self.device), t[:, 1:].to(self.device)

# ─────────────────────────────────────────────────────────────────────────────
# NS Operatörleri
# ─────────────────────────────────────────────────────────────────────────────

def _hop_scales(k: int) -> List[int]:
    if k <= 1: return [1]
    return [2**i for i in range(int(math.log2(k)) + 1)]

def multihop_gradient(u: torch.Tensor, k: int) -> torch.Tensor:
    scales = _hop_scales(k)
    total  = torch.zeros_like(u)
    for s in scales:
        p     = F.pad(u, (0, 0, s, 0))[:, :-s, :]
        total = total + (u - p) / s
    return total / len(scales)

def multihop_laplacian(u: torch.Tensor, k: int) -> torch.Tensor:
    scales = _hop_scales(k)
    total  = torch.zeros_like(u)
    for s in scales:
        p     = F.pad(u, (0, 0, 2 * s, 0))
        total = total + (u - 2 * p[:, s:-s, :] + p[:, :-2*s, :]) / (s * s)
    return total / len(scales)

def multihop_divergence(u: torch.Tensor, k: int) -> torch.Tensor:
    return multihop_gradient(u, k).mean(dim=-1)

def spectral_pressure(adv: torch.Tensor, alpha: torch.Tensor, hop_k: int) -> torch.Tensor:
    """Helmholtz-Poisson: (∇²−α²)p = −div(adv)  FFT çözümü."""
    div  = multihop_divergence(adv, hop_k)
    B, L = div.shape
    f    = torch.fft.rfft(div.float(), dim=1)
    k    = torch.arange(L // 2 + 1, dtype=torch.float32, device=div.device)
    a    = alpha.float() if isinstance(alpha, torch.Tensor) else float(alpha)
    lam  = 2.0 * (torch.cos(2.0 * math.pi * k / L) - 1.0) - a ** 2
    lam  = lam.masked_fill(lam.abs() < 1e-8, -1e-8)
    p    = torch.fft.irfft(-f / lam.unsqueeze(0), n=L, dim=1).to(adv.dtype)
    return p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5) + 1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# FluidLayer v4 + multi-hop + spectral pressure
# ─────────────────────────────────────────────────────────────────────────────

class FluidLayer(nn.Module):
    def __init__(self, d_model: int, hop_k: int = 8,
                 nu: float = 0.01, dt: float = 0.05,
                 alpha: float = 1.0, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hop_k = hop_k

        self.log_nu      = nn.Parameter(torch.tensor(math.log(nu)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(dt)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_p_scale = nn.Parameter(torch.tensor(0.0))

        self._d_k = max(d_model // 8, 16)
        self.W_q  = nn.Linear(d_model, self._d_k, bias=False)
        self.W_k  = nn.Linear(d_model, self._d_k, bias=False)
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)

        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    @property
    def nu(self):      return self.log_nu.exp()
    @property
    def dt(self):      return self.log_dt.exp()
    @property
    def alpha(self):   return self.log_alpha.exp()
    @property
    def p_scale(self): return self.log_p_scale.exp()

    def _rhs(self, u: torch.Tensor) -> torch.Tensor:
        q      = self.W_q(u)
        k      = self.W_k(u)
        k_prev = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
        speed  = torch.tanh((q * k_prev).sum(-1, keepdim=True) / (self._d_k ** 0.5))

        adv    = speed * multihop_gradient(u, self.hop_k)
        p      = spectral_pressure(adv, self.alpha, self.hop_k)
        p_grad = self.p_scale * multihop_gradient(p.unsqueeze(-1), self.hop_k).expand_as(u)
        visc   = self.nu * multihop_laplacian(u, self.hop_k)
        return -adv - p_grad + visc

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = u + self.dt * self._rhs(self.norm1(u))
        u = u + self.mlp(self.norm2(u))
        return u

# ─────────────────────────────────────────────────────────────────────────────
# FluidLM
# ─────────────────────────────────────────────────────────────────────────────

class FluidLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 max_seq_len: int, hop_k: int = 8,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model   = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop_emb  = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            FluidLayer(d_model, hop_k=hop_k, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        u    = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            u = layer(u)
        return self.lm_head(self.norm(u))

    def physical_params(self) -> Dict[str, list]:
        return {
            "nu":      [l.nu.item()      for l in self.layers],
            "dt":      [l.dt.item()      for l in self.layers],
            "alpha":   [l.alpha.item()   for l in self.layers],
            "p_scale": [l.p_scale.item() for l in self.layers],
        }

# ─────────────────────────────────────────────────────────────────────────────
# GPT Baseline
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv        = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj       = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                    dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            att = att.masked_fill(
                torch.tril(torch.ones(T, T, device=x.device)).bool().logical_not(), -1e4)
            att = self.attn_drop(F.softmax(att, dim=-1))
            y   = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class GPTLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm2(x + self.attn(self.norm1(x))))


class GPTBaseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 max_seq_len: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop_emb  = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            GPTLayer(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        x    = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

# ─────────────────────────────────────────────────────────────────────────────
# FLOP Sayacı
# ─────────────────────────────────────────────────────────────────────────────

def flops_per_token(model_type: str, d: int, L: int, T: int,
                    vocab: int, mlp_ratio: int, hop_k: int = 8) -> dict:
    hidden    = d * mlp_ratio
    mlp_total = 2 * 2 * d * hidden * L

    if model_type == "fluid":
        n_scales      = len(_hop_scales(hop_k))
        d_k           = max(d // 8, 16)
        wqk_per_token = 2 * 2 * d * d_k
        ns_per_token  = wqk_per_token + d_k + n_scales * d * 9
        fft_per_token = int(math.log2(max(T, 2))) * d
        routing_total = (ns_per_token + fft_per_token) * L
        routing_label = (f"NS v4+multihop-k{hop_k}+FFT "
                         f"(scales={_hop_scales(hop_k)})")
        routing_complexity = "O(L·T·D·log(k)+L·T·log(T))"
    else:
        routing_total = (3 * 2 * d * d + 2 * T * d + 2 * d * d) * L
        routing_label = "MHA (QKV+attn+proj)"
        routing_complexity = "O(L·T²·D)"

    head_flops = 2 * vocab * d
    total = mlp_total + routing_total + head_flops
    return {
        "mlp_total_mflop":     mlp_total     / 1e6,
        "routing_total_mflop": routing_total / 1e6,
        "lm_head_mflop":       head_flops    / 1e6,
        "total_mflop":         total         / 1e6,
        "routing_label":       routing_label,
        "routing_complexity":  routing_complexity,
    }

def count_params(model: nn.Module) -> dict:
    total   = sum(p.numel() for p in model.parameters())
    emb     = sum(p.numel() for n, p in model.named_parameters()
                  if "token_emb" in n or "pos_emb" in n)
    mlp     = sum(p.numel() for n, p in model.named_parameters() if ".mlp." in n)
    routing = sum(p.numel() for n, p in model.named_parameters()
                  if any(x in n for x in
                         ["log_nu","log_dt","log_alpha","log_p_scale",
                          "W_q","W_k",".qkv.",".attn.",".proj."]))
    return {"total": total, "emb": emb, "routing": routing, "mlp": mlp}

# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int,
           lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step > total_steps:
        return min_lr
    decay = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (lr - min_lr)

# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, stream: TokenStream,
             batch_size: int, n_steps: int) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(n_steps):
        x, y = stream.get_batch(batch_size)
        with AUTOCAST:
            logits = model(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    model.train()
    return math.exp(total_loss / n_steps)

@torch.no_grad()
def eval_wikitext103(model: nn.Module, device: torch.device, seq_len: int) -> float:
    if ENC is None: return -1.0
    try:
        from datasets import load_dataset
        ds   = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        text = " ".join([r["text"] for r in ds if r["text"].strip()])
        ids  = torch.tensor(encode(text), dtype=torch.long)
        stride, nlls = seq_len // 2, []
        for i in range(0, len(ids) - seq_len, stride):
            chunk = ids[i:i+seq_len+1].to(device)
            x, y  = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
            with AUTOCAST:
                logits = model(x)
                nlls.append(F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)).item())
        ppl = math.exp(sum(nlls) / len(nlls))
        master_print(f"[WikiText-103] Zero-shot PPL = {ppl:.2f}")
        return ppl
    except Exception as e:
        master_print(f"[WikiText] Hata: {e}"); return -1.0

# ─────────────────────────────────────────────────────────────────────────────
# Tek Model Eğitimi
# ─────────────────────────────────────────────────────────────────────────────

def train_one(model_type: str, scale_cfg: ScaleConfig, seed: int,
              token_budget: int, out_dir: str, vocab_size: int,
              hop_k: int = 8) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    sc     = scale_cfg
    run_id = f"18_{model_type}_k{hop_k}_{sc.name}_s{seed}"
    rf     = hop_k * sc.n_layers   # efektif receptive field (token)

    master_print(f"\n{'='*60}")
    master_print(f"  RUN: {run_id}")
    master_print(f"  Scale: d={sc.d_model}, L={sc.n_layers}, seq={sc.seq_len}")
    master_print(f"  hop_k={hop_k}  scales={_hop_scales(hop_k)}")
    master_print(f"  Receptive field: {hop_k} × {sc.n_layers} katman = {rf} token")
    master_print(f"  (GPT-S: {sc.seq_len} token, exp16 k=4: {4*sc.n_layers} token)")
    master_print(f"  Token budget: {token_budget:,}")
    master_print(f"{'='*60}")

    if model_type == "fluid":
        model = FluidLM(
            vocab_size=vocab_size, d_model=sc.d_model,
            n_layers=sc.n_layers, max_seq_len=sc.seq_len,
            hop_k=hop_k, mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        ).to(device)
    else:
        model = GPTBaseline(
            vocab_size=vocab_size, d_model=sc.d_model,
            n_layers=sc.n_layers, n_heads=sc.n_heads,
            max_seq_len=sc.seq_len, mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
        ).to(device)

    param_info = count_params(model)
    flops_info = flops_per_token(
        model_type, sc.d_model, sc.n_layers, sc.seq_len,
        vocab_size, args.mlp_ratio, hop_k)
    master_print(f"  Parametreler: {param_info['total']/1e6:.1f}M toplam  "
                 f"| routing={param_info['routing']:,}  "
                 f"| mlp={param_info['mlp']/1e6:.1f}M")
    master_print(f"  FLOPs/token: {flops_info['total_mflop']:.1f} MFLOP  "
                 f"({flops_info['routing_label']})")

    if args.compile:
        master_print("  torch.compile() uygulanıyor...")
        model = torch.compile(model)

    raw_model = model
    if IS_DDP:
        model = DDP(model, device_ids=[LOCAL_RANK]); raw_model = model.module

    PHYSICS_NAMES = {'log_nu', 'log_dt', 'log_alpha', 'log_p_scale'}
    physics_params = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and any(k in n for k in PHYSICS_NAMES)]
    decay_params   = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.dim() >= 2
                      and not any(k in n for k in PHYSICS_NAMES)]
    nodecay_params = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.dim() < 2
                      and not any(k in n for k in PHYSICS_NAMES)]
    pg_list = [pg for pg in [
        {"params": physics_params, "lr": args.lr * 10, "weight_decay": 0.01, "lr_scale": 10.0},
        {"params": decay_params,   "weight_decay": args.weight_decay,         "lr_scale": 1.0},
        {"params": nodecay_params, "weight_decay": 0.0,                       "lr_scale": 1.0},
    ] if len(pg["params"]) > 0]
    optimizer = torch.optim.AdamW(pg_list, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    if BATCH_OVERRIDE:
        batch_size, grad_accum = BATCH_OVERRIDE, 1
    else:
        batch_size, grad_accum = auto_batch(sc.d_model, sc.seq_len, vram_gb)
    effective_batch_tokens = batch_size * grad_accum * sc.seq_len
    total_steps  = token_budget // effective_batch_tokens
    warmup_steps = max(100, int(total_steps * args.warmup_frac))
    master_print(f"  batch={batch_size}  grad_accum={grad_accum}  "
                 f"eff_batch={effective_batch_tokens/1e3:.0f}K tok/step  "
                 f"total_steps={total_steps:,}  warmup={warmup_steps}")

    train_stream = TokenStream("train", sc.seq_len, device)
    val_stream   = TokenStream("val",   sc.seq_len, device)

    best_val_ppl   = float("inf")
    best_ckpt_path = os.path.join(out_dir, f"{run_id}_best.pt")
    train_history: List[dict] = []
    scaler = torch.amp.GradScaler("cuda",
                 enabled=(device.type == "cuda" and not USE_BF16))
    tokens_seen = 0; t0_global = time.time()
    throughput_samples: List[float] = []
    resume_step = 0

    if os.path.exists(best_ckpt_path):
        master_print(f"  [Resume] {best_ckpt_path} bulundu...")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        resume_step  = ckpt.get("step", 0) + 1
        best_val_ppl = ckpt.get("val_ppl", float("inf"))
        tokens_seen  = resume_step * effective_batch_tokens
        master_print(f"  [Resume] step={resume_step}  best_ppl={best_val_ppl:.4f}")
        for _ in range(resume_step * grad_accum * batch_size):
            train_stream.get_batch(batch_size)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(resume_step, total_steps):
        lr_now = get_lr(step, warmup_steps, total_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now * pg.get("lr_scale", 1.0)

        t_step = time.time()
        for micro in range(grad_accum):
            x, y = train_stream.get_batch(batch_size)
            if IS_DDP:
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            with AUTOCAST:
                logits = model(x)
                loss   = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            scaler.scale(loss).backward()
            tokens_seen += x.numel()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        throughput_samples.append(effective_batch_tokens / (time.time() - t_step))

        if step % args.eval_interval == 0 or step == total_steps - 1:
            val_ppl  = evaluate(raw_model, val_stream, batch_size, args.eval_steps)
            elapsed  = time.time() - t0_global
            avg_tput = sum(throughput_samples[-100:]) / max(1, len(throughput_samples[-100:]))
            train_history.append({
                "step": step, "tokens": tokens_seen,
                "lr": lr_now, "val_ppl": val_ppl, "throughput": avg_tput,
            })
            master_print(
                f"  step={step:6d}/{total_steps}  "
                f"tok={tokens_seen/1e9:.2f}B  "
                f"val_ppl={val_ppl:.4f}  "
                f"lr={lr_now:.2e}  "
                f"tput={avg_tput/1e3:.0f}K tok/s  "
                f"t={elapsed/60:.0f}m"
            )
            if val_ppl < best_val_ppl and IS_MASTER:
                best_val_ppl = val_ppl
                torch.save({
                    "model_type":   model_type,
                    "scale":        sc.name,
                    "seed":         seed,
                    "hop_k":        hop_k,
                    "step":         step,
                    "val_ppl":      val_ppl,
                    "model_state":  raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": {
                        "vocab_size": vocab_size,
                        "d_model":    sc.d_model,
                        "n_layers":   sc.n_layers,
                        "n_heads":    sc.n_heads,
                        "seq_len":    sc.seq_len,
                        "hop_k":      hop_k,
                        "mlp_ratio":  args.mlp_ratio,
                    },
                }, best_ckpt_path)

    wt103_ppl = -1.0
    if args.wikitext_eval and IS_MASTER:
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        raw_model.load_state_dict(ckpt["model_state"])
        wt103_ppl = eval_wikitext103(raw_model, device, sc.seq_len)

    phys = None
    if model_type == "fluid" and IS_MASTER:
        phys    = raw_model.physical_params()
        nu_vals = phys["nu"]
        al_vals = phys["alpha"]
        master_print(f"\n  [Fiziksel Parametreler @ {run_id}]")
        master_print(f"  nu:  early={sum(nu_vals[:4])/4:.4f}  "
                     f"late={sum(nu_vals[-4:])/4:.4f}  "
                     f"range=[{min(nu_vals):.4f}, {max(nu_vals):.4f}]")
        master_print(f"  alpha: early={sum(al_vals[:4])/4:.4f}  "
                     f"late={sum(al_vals[-4:])/4:.4f}  "
                     f"range=[{min(al_vals):.4f}, {max(al_vals):.4f}]")

    avg_tput = sum(throughput_samples) / max(1, len(throughput_samples))
    return {
        "run_id":            run_id,
        "model_type":        model_type,
        "scale":             sc.name,
        "seed":              seed,
        "hop_k":             hop_k,
        "hop_scales":        _hop_scales(hop_k),
        "receptive_field":   hop_k * sc.n_layers,
        "token_budget":      token_budget,
        "tokens_seen":       tokens_seen,
        "best_val_ppl":      best_val_ppl,
        "final_val_ppl":     train_history[-1]["val_ppl"] if train_history else -1,
        "wt103_ppl":         wt103_ppl,
        "throughput_tps":    avg_tput,
        "param_info":        param_info,
        "flops_info":        flops_info,
        "train_history":     train_history,
        "physical_params":   phys,
        "ckpt_path":         best_ckpt_path,
        "gpu":               gpu_name,
        "dtype":             str(DTYPE),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Ana Akış
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
    out_dir    = args.out_dir or os.path.join(REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    sc = SCALE_CONFIGS[args.scale]
    sc = ScaleConfig(name=sc.name, d_model=sc.d_model, n_layers=sc.n_layers,
                     n_heads=sc.n_heads, seq_len=args.seq_len)

    vocab_size   = VOCAB if VOCAB is not None else 256
    token_budget = int(args.tokens)
    models_to_run = (["fluid", "gpt"] if args.model == "both" else [args.model])

    all_results: List[dict] = []
    for model_type in models_to_run:
        hop_ks_to_run = args.hop_ks if model_type == "fluid" else [1]
        for hop_k in hop_ks_to_run:
            for seed in args.seeds:
                result = train_one(
                    model_type=model_type, scale_cfg=sc, seed=seed,
                    token_budget=token_budget, out_dir=out_dir,
                    vocab_size=vocab_size, hop_k=hop_k,
                )
                all_results.append(result)

    if IS_MASTER:
        master_print(f"\n{'='*65}")
        master_print("  EXP 18 SONUCLARI — Wider Receptive Field")
        master_print(f"{'='*65}")
        master_print(f"  {'Run ID':<38} {'RF':>5} {'Best PPL':>10} {'MFLOP':>8}")
        master_print(f"  {'-'*65}")
        for r in all_results:
            rf = r.get("receptive_field", "?")
            master_print(
                f"  {r['run_id']:<38} "
                f"{str(rf):>5} "
                f"{r['best_val_ppl']:>10.4f} "
                f"{r['flops_info']['total_mflop']:>8.1f}"
            )

        # Önceki exp karşılaştırması
        prev = []
        for fname, tag in [("16_multihop_ns.json", "exp16 k=4"),
                           ("17_spectral_pressure.json", "exp17 k=4 FFT")]:
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                    for r in data:
                        if r.get("model_type") == "fluid":
                            prev.append((r["run_id"], r["best_val_ppl"],
                                         r.get("receptive_field", 4*12), tag))
                except Exception:
                    pass
        if prev:
            master_print(f"\n  [Receptive Field vs PPL — Karşılaştırma]")
            master_print(f"  {'Tag':<25} {'RF':>5} {'PPL':>10}")
            master_print(f"  {'-'*42}")
            for rid, ppl, rf, tag in prev:
                master_print(f"  {tag:<25} {rf:>5} {ppl:>10.4f}")
            for r in all_results:
                if r["model_type"] == "fluid":
                    rf = r.get("receptive_field", "?")
                    master_print(
                        f"  {'exp18 k='+str(r['hop_k']):<25} "
                        f"{str(rf):>5} "
                        f"{r['best_val_ppl']:>10.4f}"
                    )

        out_path = os.path.join(out_dir, "18_wider_receptive.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        master_print(f"\n[JSON] Kaydedildi: {out_path}")

    if IS_DDP:
        dist.destroy_process_group()
