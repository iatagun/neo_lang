# ============================================================
# FluidLM vs GPT — Industrial Scale Comparison
# experiments/14_industrial_compare.py
#
# Araştırma Soruları (RQ):
#   RQ1: NS routing, MHA'ya eşdeğer representasyon gücü sağlıyor mu?
#   RQ2: Routing parametresi → FLOP/token tradeoff nedir?
#   RQ3: Emergent ν gradyanı char-level'ın ötesinde de oluşuyor mu?
#
# Metodoloji (Chinchilla standardı):
#   • BPE tokenizer  : tiktoken gpt2 (vocab=50257)
#   • Corpus         : OpenWebText (HuggingFace datasets streaming)
#   • Eğitim birimi  : TOKEN BUDGET (epoch değil)
#   • PPL metrikleri : Token-level (char-level değil — yayın standardı)
#   • Benchmark      : WikiText-103 zero-shot token PPL
#   • Çoklu seed     : seed 42/43/44 (S scale), 42 (M scale)
#
# Ölçekler:
#   S : d=768,  L=12, equiv. GPT-2 small  (~117M)
#   M : d=1024, L=24, equiv. GPT-2 medium (~354M)
#
# Model çiftleri (izole karşılaştırma için aynı MLP):
#   FluidLM-S : d=768,  L=12, NS routing (48 param)  + MLP → ~95M
#   GPT-S     : d=768,  L=12, MHA (28.3M param)      + MLP → ~123M
#   FluidLM-M : d=1024, L=24, NS routing (96 param)  + MLP → ~285M
#   GPT-M     : d=1024, L=24, MHA (100.7M param)     + MLP → ~354M
#
# Akademik çıktı:
#   results/14_industrial_compare.json  — tüm metrikler
#   results/14_industrial_compare.png   — eğitim eğrileri + karşılaştırma
#   results/14_industrial_summary.md    — akademik özet (paper §4 taslağı)
#
# Kullanım:
#   # Colab A100 — her ikisi
#   python experiments/14_industrial_compare.py
#
#   # Sadece S scale, 2 model
#   python experiments/14_industrial_compare.py --scale S --tokens 1e9
#
#   # Sadece FluidLM (ablasyon için)
#   python experiments/14_industrial_compare.py --model fluid --scale S
#
#   # WikiText-103 zero-shot eval (eğitim olmadan, ckpt yükle)
#   python experiments/14_industrial_compare.py --eval_only \
#       --fluid_ckpt results/14_fluid_S_s42_best.pt \
#       --gpt_ckpt   results/14_gpt_S_s42_best.pt
#
# Colab kurulum:
#   !pip install tiktoken datasets -q
#   !git clone https://github.com/KULLANICI/neo_lang.git && cd neo_lang
#   !python experiments/14_industrial_compare.py --scale S --tokens 3e9
#
# Çoklu GPU (DDP):
#   torchrun --nproc_per_node=4 experiments/14_industrial_compare.py --scale M
# ============================================================

import sys, os, math, time, json, argparse, hashlib, random, subprocess
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ─── UTF-8 stdout ─────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─────────────────────────────────────────────────────────────────────────────
# Argümanlar
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="FluidLM vs GPT Industrial Comparison")
parser.add_argument("--scale",       default="both",  choices=["S","M","both"],
                    help="Model ölçeği: S (~120M), M (~350M), both")
parser.add_argument("--model",       default="both",  choices=["fluid","gpt","both"],
                    help="Eğitilecek model")
parser.add_argument("--tokens",      type=float, default=3e9,
                    help="Token budget (Chinchilla). Varsayılan: 3B token")
parser.add_argument("--seeds",       nargs="+", type=int, default=[42, 43, 44],
                    help="S ölçeği için seed listesi (M için sadece ilki kullanılır)")
parser.add_argument("--batch_tokens",type=int, default=None,
                    help="Batch başına token sayısı (VRAM'a göre otomatik)")
parser.add_argument("--seq_len",     type=int, default=1024,
                    help="Bağlam uzunluğu (GPT-2 standardı)")
parser.add_argument("--lr",          type=float, default=3e-4)
parser.add_argument("--min_lr",      type=float, default=3e-5,
                    help="Cosine decay min LR (lr/10 standart)")
parser.add_argument("--warmup_frac", type=float, default=0.01,
                    help="Token budget'ın kaçı warmup (varsayılan %1)")
parser.add_argument("--weight_decay",type=float, default=0.1)
parser.add_argument("--grad_clip",   type=float, default=1.0)
parser.add_argument("--dropout",     type=float, default=0.0,
                    help="Büyük modelde 0 önerilir (token drop zaten regularization)")
parser.add_argument("--mlp_ratio",   type=int,   default=4)
parser.add_argument("--eval_interval",type=int,  default=500,
                    help="Her N adımda bir val PPL hesapla")
parser.add_argument("--eval_steps",  type=int,   default=100,
                    help="Eval sırasında kaç batch kullanılacak")
parser.add_argument("--eval_only",   action="store_true",
                    help="Eğitim olmadan sadece eval çalıştır")
parser.add_argument("--fluid_ckpt",  default=None, help="eval_only için FluidLM ckpt")
parser.add_argument("--gpt_ckpt",    default=None, help="eval_only için GPT ckpt")
parser.add_argument("--wikitext_eval",action="store_true",
                    help="WikiText-103 zero-shot PPL hesapla (HuggingFace gerektirir)")
parser.add_argument("--out_dir",     default=None,
                    help="Çıktı dizini (varsayılan: <repo_root>/results)")
parser.add_argument("--compile",     action="store_true",
                    help="torch.compile kullan (Ampere+ GPU)")
parser.add_argument("--ddp",         action="store_true",
                    help="DDP modu (torchrun ile otomatik algılanır)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# DDP başlatma
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

device    = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
USE_BF16  = device.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE     = torch.bfloat16 if USE_BF16 else torch.float32
AUTOCAST  = nullcontext() if DTYPE == torch.float32 else \
            torch.amp.autocast(device_type="cuda", dtype=DTYPE)

if device.type == "cuda":
    vram_gb = torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1e9
    gpu_name = torch.cuda.get_device_properties(LOCAL_RANK).name
else:
    vram_gb, gpu_name = 0, "CPU"

master_print(f"[Device] {gpu_name}  VRAM={vram_gb:.0f}GB  dtype={DTYPE}  "
             f"DDP={IS_DDP}  world={WORLD_SIZE}")

# ─────────────────────────────────────────────────────────────────────────────
# Ölçek Konfigürasyonları (GPT-2 paper eşdeğeri)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScaleConfig:
    name:       str
    d_model:    int
    n_layers:   int
    n_heads:    int    # GPT için, FluidLM'de kullanılmaz
    seq_len:    int
    # Param tahmini (embedding dahil değil — vocab'a bağlı)
    gpt_params_nonemb:  int = 0   # MHA+MLP
    fluid_params_nonemb:int = 0   # NS+MLP

SCALE_CONFIGS = {
    "S": ScaleConfig(
        name="S",
        d_model=768, n_layers=12, n_heads=12, seq_len=1024,
        # GPT-S: MHA=4*768^2=2.36M, FFN=2*768*3072=4.72M → 7.08M/layer × 12 = 85M
        gpt_params_nonemb   = 85_000_000,
        # FluidLM-S: NS=4 skaler, FFN=4.72M/layer → 56.6M
        fluid_params_nonemb = 57_000_000,
    ),
    "M": ScaleConfig(
        name="M",
        d_model=1024, n_layers=24, n_heads=16, seq_len=1024,
        # GPT-M: MHA=4*1024^2=4.19M, FFN=2*1024*4096=8.39M → 12.58M/layer × 24 = 302M
        gpt_params_nonemb   = 302_000_000,
        # FluidLM-M: NS=4 skaler, FFN=8.39M/layer → 201M
        fluid_params_nonemb = 201_000_000,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Batch size otomatik belirleme
# ─────────────────────────────────────────────────────────────────────────────

def auto_batch(d_model: int, seq_len: int, vram_gb: float) -> tuple[int, int]:
    """
    (batch_size, grad_accum) döner.
    Hedef: her adımda ≥ 500K token işle (GPT-3 paper standardı).
    """
    target_batch_tokens = 500_000   # token/step

    # VRAM tabanlı fiziksel batch tahmini (BF16, yaklaşık)
    # Her token: d_model × n_layers × ~12 byte (activation memory)
    bytes_per_token = d_model * 28 * 2  # ampirik katsayı
    max_physical = max(1, int(vram_gb * 0.75 * 1e9 / (bytes_per_token * seq_len)))
    max_physical = min(max_physical, 32)  # sanity cap

    grad_accum  = max(1, target_batch_tokens // (max_physical * seq_len))
    return max_physical, grad_accum

# args'tan override
if args.batch_tokens:
    _ph = max(1, args.batch_tokens // args.seq_len)
    BATCH_OVERRIDE = _ph
else:
    BATCH_OVERRIDE = None

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

try:
    import tiktoken
    ENC    = tiktoken.get_encoding("gpt2")
    VOCAB  = ENC.n_vocab    # 50257
    encode = lambda text: ENC.encode_ordinary(text)
    master_print(f"[Tokenizer] tiktoken gpt2  vocab={VOCAB}")
except ImportError:
    master_print("[WARNING] tiktoken bulunamadı — char-level fallback kullanılıyor")
    master_print("          pip install tiktoken  ile kurun")
    ENC   = None
    VOCAB = None   # veri yüklemede belirlenir
    encode = None

# ─────────────────────────────────────────────────────────────────────────────
# Veri Yükleme
# ─────────────────────────────────────────────────────────────────────────────

class TokenStream:
    """
    Streaming token kaynağı.
    1) HuggingFace OpenWebText (BPE)
    2) Yerel shakespeare.txt (BPE veya char fallback)
    Döner: rastgele (B, T) tensörler.
    """
    def __init__(self, split: str, seq_len: int, device: torch.device):
        self.seq_len = seq_len
        self.device  = device
        self.buffer: List[int] = []
        self._load(split)

    def _load(self, split: str):
        global VOCAB, encode

        # ── A) HuggingFace OpenWebText ─────────────────────────────────────
        if ENC is not None:
            try:
                from datasets import load_dataset
                master_print(f"[Data] OpenWebText yükleniyor (split={split}, streaming=True)...")
                # OpenWebText: ~8M dokuman, ~8B token
                ds = load_dataset("Skylion007/openwebtext",
                                  split="train",
                                  streaming=True)
                # train/val ayrımı: örtüşmesiz partition
                # train: [0, 990_000), val: [990_000, ∞)
                # shuffle AYNI seed → ayrı stream nesneleri ama aynı sıra;
                # take/skip ile kesişmeyen dilimler garanti edilir.
                ds = ds.shuffle(seed=42, buffer_size=10_000)
                if split == "val":
                    ds = ds.skip(990_000)  # train'den sonraki kısım — overlap yok
                else:
                    ds = ds.take(990_000)  # [0, 990_000) — val'e girmez
                self._stream = iter(ds)
                self._source = "openwebtext"
                master_print(f"[Data] OpenWebText stream hazır ({split})")
                return
            except Exception as e:
                master_print(f"[Data] OpenWebText yüklenemedi: {e}")
                master_print("[Data] Yerel Shakespeare fallback...")

        # ── B) Yerel fallback ──────────────────────────────────────────────
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
        REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
        candidates = [
            os.path.join(REPO_ROOT, "data", "shakespeare.txt"),
            os.path.join(SCRIPT_DIR, "shakespeare.txt"),
        ]
        data_path = next((p for p in candidates if os.path.exists(p)), None)
        if data_path is None:
            import urllib.request
            data_path = os.path.join(REPO_ROOT, "data", "shakespeare.txt")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            master_print("[Data] Shakespeare indiriliyor...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
                "data/tinyshakespeare/input.txt", data_path)

        with open(data_path, encoding="utf-8") as f:
            text = f.read()

        if ENC is not None:
            ids = encode(text)
        else:
            chars  = sorted(set(text))
            VOCAB  = len(chars)
            stoi   = {c:i for i,c in enumerate(chars)}
            encode = lambda s: [stoi[c] for c in s]
            ids    = encode(text)

        n      = int(len(ids) * 0.9)
        ids    = ids[:n] if split == "train" else ids[n:]
        self._tensor = torch.tensor(ids, dtype=torch.long)
        self._source = "shakespeare"
        master_print(f"[Data] Shakespeare fallback  split={split}  "
                     f"tokens={len(ids):,}  vocab={VOCAB}")

    def _next_from_stream(self, n_tokens: int) -> List[int]:
        """OpenWebText stream'den n_tokens kadar token topla."""
        buf: List[int] = []
        while len(buf) < n_tokens:
            try:
                doc = next(self._stream)
                buf.extend(encode(doc["text"] + "\n"))
            except StopIteration:
                # Stream bitti → başa dön
                master_print("[Data] Stream bitti, yeniden başlatılıyor...")
                from datasets import load_dataset
                ds = load_dataset("Skylion007/openwebtext",
                                  split="train", streaming=True
                                  ).shuffle(seed=random.randint(0,9999))
                self._stream = iter(ds)
        return buf[:n_tokens]

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        need = (self.seq_len + 1) * batch_size

        if self._source == "openwebtext":
            if len(self.buffer) < need:
                self.buffer.extend(self._next_from_stream(need * 4))
            chunk      = self.buffer[:need]
            self.buffer = self.buffer[need:]
            t = torch.tensor(chunk, dtype=torch.long).view(batch_size, self.seq_len + 1)
        else:
            ix = torch.randint(len(self._tensor) - self.seq_len - 1, (batch_size,))
            t  = torch.stack([self._tensor[i:i+self.seq_len+1] for i in ix])

        x = t[:, :-1].to(self.device)
        y = t[:, 1:].to(self.device)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# NS Operatörleri (causal, self-contained — 10_fluidlm_faircompare.py ile aynı)
# ─────────────────────────────────────────────────────────────────────────────

def causal_gradient(u: torch.Tensor) -> torch.Tensor:
    """∂u/∂x ≈ u[i] − u[i-1]  (causal, sıfır-padding ile)"""
    return u - F.pad(u, (0, 0, 1, 0))[:, :-1, :]

def causal_laplacian(u: torch.Tensor) -> torch.Tensor:
    """∂²u/∂x² ≈ u[i] − 2u[i-1] + u[i-2]"""
    p = F.pad(u, (0, 0, 2, 0))
    return u - 2 * p[:, 1:-1, :] + p[:, :-2, :]

def causal_divergence(u: torch.Tensor) -> torch.Tensor:
    """div(u) = mean_d(∂u_d/∂x)"""
    return causal_gradient(u).mean(dim=-1)

def spectral_pressure(adv: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Helmholtz-Poisson: (∇² − α²)p = −div(adv)
    FFT çözümü: p̂_k = −f̂_k / λ_k
                λ_k = 2(cos(2πk/L)−1) − α²
    """
    div  = causal_divergence(adv)          # (B, L)
    B, L = div.shape
    f    = torch.fft.rfft(div.float(), dim=1)
    k    = torch.arange(L // 2 + 1, dtype=torch.float32, device=div.device)
    a    = alpha.float() if isinstance(alpha, torch.Tensor) else float(alpha)
    lam  = 2.0 * (torch.cos(2.0 * math.pi * k / L) - 1.0) - a ** 2
    lam  = lam.masked_fill(lam.abs() < 1e-8, -1e-8)
    p    = torch.fft.irfft(-f / lam.unsqueeze(0), n=L, dim=1).to(adv.dtype)
    return p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5) + 1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# FluidLayer (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

class FluidLayer(nn.Module):
    """
    1 katman NS timestep + MLP sublayer.
    Routing parametreleri: log_nu, log_dt, log_alpha, p_scale_raw (4 skaler).
    MLP parametreleri: d_model → hidden → d_model (asıl kapasite).
    """
    def __init__(self, d_model: int, nu: float = 0.01, dt: float = 0.05,
                 alpha: float = 1.0, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.log_nu      = nn.Parameter(torch.tensor(math.log(nu)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(dt)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_p_scale = nn.Parameter(torch.tensor(0.0))   # log(1.0)=0 → p_scale=1.0 at init, always positive

        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
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
        speed   = torch.tanh(u.norm(dim=-1, keepdim=True))
        adv     = speed * causal_gradient(u)
        # Causal cumsum: normalise BEFORE dividing by alpha so that alpha is NOT
        # cancelled by std(cumsum/alpha) = std(cumsum)/alpha.
        div_adv = causal_divergence(adv)
        cumsum  = torch.cumsum(-div_adv, dim=1)
        cumsum  = cumsum / (cumsum.std(dim=1, keepdim=True).detach() + 1e-6)  # normalise first
        p       = cumsum / (self.alpha + 1e-6)                                # alpha in gradient path
        p_grad  = self.p_scale * causal_gradient(p.unsqueeze(-1)).expand_as(u)
        visc    = self.nu * causal_laplacian(u)
        return -adv - p_grad + visc

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # NS sublayer (residual)
        u = u + self.dt * self._rhs(self.norm1(u))
        # MLP sublayer (residual)
        u = u + self.mlp(self.norm2(u))
        return u

# ─────────────────────────────────────────────────────────────────────────────
# FluidLM
# ─────────────────────────────────────────────────────────────────────────────

class FluidLM(nn.Module):
    """
    Token embedding → N × FluidLayer → LayerNorm → LM head.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 max_seq_len: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop_emb  = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            FluidLayer(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm      = nn.LayerNorm(d_model)
        self.lm_head   = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (GPT-2 standardı)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        u    = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            u = layer(u)
        logits = self.lm_head(self.norm(u))
        return logits

    def physical_params(self) -> Dict[str, list]:
        """Her katmandaki öğrenilen fiziksel parametreleri döner."""
        return {
            "nu":      [l.nu.item()      for l in self.layers],
            "dt":      [l.dt.item()      for l in self.layers],
            "alpha":   [l.alpha.item()   for l in self.layers],
            "p_scale": [l.p_scale.item() for l in self.layers],
        }

# ─────────────────────────────────────────────────────────────────────────────
# GPT Baseline (decoder-only Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Standart MHA, causal mask."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash Attention (PyTorch ≥ 2.0)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v,
                    attn_mask=None, dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=True)
        else:
            att = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5)
            att = att.masked_fill(
                torch.tril(torch.ones(T, T, device=x.device)).bool().logical_not(), -1e4)
            att = self.attn_drop(F.softmax(att, dim=-1))
            y   = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class GPTLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPTBaseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, max_seq_len: int,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop_emb  = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            GPTLayer(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x   = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

# ─────────────────────────────────────────────────────────────────────────────
# FLOP Sayacı
# ─────────────────────────────────────────────────────────────────────────────

def flops_per_token(model_type: str, d: int, L: int, T: int,
                    vocab: int, mlp_ratio: int) -> dict:
    """
    Teorik forward-pass FLOPs/token (çift sayma yok).
    Tüm değerler token başına normalize edilir (÷T).
    Referans: Kaplan et al. 2020 "Scaling Laws" metodolojisi.
    """
    # Embedding lookup: 0 FLOP (sadece indexing)
    emb_flops = 0

    # MLP sublayer: 2 linear (d→hidden, hidden→d), her biri 2×d×hidden FLOP
    # Per-token: 2 × 2 × d × hidden
    hidden    = d * mlp_ratio
    mlp_per_token = 2 * 2 * d * hidden   # forward multiply-add
    mlp_total     = mlp_per_token * L    # tüm katmanlar, per-token

    if model_type == "fluid":
        # NS routing (gerçek _rhs implementasyonu):
        #   causal_gradient:  d   multiply-add / token
        #   causal_laplacian: 2d  multiply-add / token
        #   norm (speed):     d   / token
        #   advection:        d   / token
        #   divergence:       d   mean / token
        #   cumsum:           d   / token
        #   p_grad:           d   / token
        #   viscosity:        2d  / token
        # spectral_pressure kodu _rhs'ta KULLANILMIYOR → FFT yok
        ns_per_token = d * 9     # 9 tek boyutlu op, tümü per-token
        routing_total = ns_per_token * L
        routing_label = "NS (causal grad+lap+visc)"
        routing_complexity = "O(L·T·D)"
    else:  # gpt
        n_heads  = d // 64   # standart kural: head_dim=64
        head_dim = d // n_heads
        # QKV projection: 3 × 2×d² FLOP, per-token
        qkv_per_token  = 3 * 2 * d * d
        # Attention: QK^T + attn@V = 2×T×d per token (T token seq üzerinden)
        attn_per_token = 2 * T * d
        # Out projection: 2×d² per token
        out_per_token  = 2 * d * d
        routing_total  = (qkv_per_token + attn_per_token + out_per_token) * L
        routing_label  = "MHA (QKV+attn+proj)"
        routing_complexity = "O(L·T²·D)"

    # LM head: 2×vocab×d per token
    head_flops = 2 * vocab * d

    total = emb_flops + mlp_total + routing_total + head_flops
    return {
        "mlp_total_mflop":     mlp_total     / 1e6,
        "routing_total_mflop": routing_total / 1e6,
        "lm_head_mflop":       head_flops    / 1e6,
        "total_mflop":         total         / 1e6,
        "routing_label":       routing_label,
        "routing_complexity":  routing_complexity,
    }

def count_params(model: nn.Module, group_by: str = "none") -> dict:
    """Model parametre sayımı (routing vs MLP vs embedding ayrımı)."""
    total = sum(p.numel() for p in model.parameters())

    # Grupla
    emb   = sum(p.numel() for n, p in model.named_parameters()
                if "token_emb" in n or "pos_emb" in n)
    lmh   = sum(p.numel() for n, p in model.named_parameters()
                if "lm_head" in n)
    mlp   = sum(p.numel() for n, p in model.named_parameters()
                if ".mlp." in n)
    # Routing (FluidLM: log_nu/dt/alpha/p_scale; GPT: qkv+proj)
    routing = sum(p.numel() for n, p in model.named_parameters()
                  if any(x in n for x in ["log_nu","log_dt","log_alpha","p_scale_raw",
                                           ".qkv.",".attn.",".proj."]))
    norm  = sum(p.numel() for n, p in model.named_parameters() if "norm" in n)

    return {
        "total":   total,
        "emb":     emb,
        "routing": routing,
        "mlp":     mlp,
        "norm":    norm,
        "lm_head": lmh,   # weight tied → 0 extra params
    }

# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule (cosine warmup — GPT-3 standardı)
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int,
           lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step > total_steps:
        return min_lr
    decay = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + coeff * (lr - min_lr)

# ─────────────────────────────────────────────────────────────────────────────
# Eval Fonksiyonu
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, stream: "TokenStream",
             batch_size: int, n_steps: int) -> float:
    """Val token-PPL döner."""
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

# ─────────────────────────────────────────────────────────────────────────────
# WikiText-103 Zero-Shot Eval
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_wikitext103(model: nn.Module, device: torch.device,
                     seq_len: int, batch_size: int = 4) -> float:
    """
    WikiText-103 test setinde zero-shot token PPL.
    Akademik karşılaştırma standardı (Merity et al. 2017).
    """
    if ENC is None:
        master_print("[WikiText] tiktoken yok — atlanıyor")
        return -1.0
    try:
        from datasets import load_dataset
        master_print("[WikiText] WikiText-103 yükleniyor...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="test")
        text  = " ".join([r["text"] for r in ds if r["text"].strip()])
        ids   = torch.tensor(encode(text), dtype=torch.long)
        master_print(f"[WikiText] {len(ids):,} token")

        # Stride = seq_len // 2 (sliding window — karşılaştırma standardı)
        stride = seq_len // 2
        nlls: List[float] = []
        for i in range(0, len(ids) - seq_len, stride):
            chunk = ids[i:i+seq_len+1].to(device)
            x, y  = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
            with AUTOCAST:
                logits = model(x)
                nll    = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                         y.view(-1)).item()
            nlls.append(nll)
        ppl = math.exp(sum(nlls) / len(nlls))
        master_print(f"[WikiText-103] Zero-shot PPL = {ppl:.2f}")
        return ppl
    except Exception as e:
        master_print(f"[WikiText] Hata: {e}")
        return -1.0

# ─────────────────────────────────────────────────────────────────────────────
# Tek Model Eğitimi
# ─────────────────────────────────────────────────────────────────────────────

def train_one(
    model_type: str,        # "fluid" veya "gpt"
    scale_cfg:  ScaleConfig,
    seed:       int,
    token_budget: int,
    out_dir:    str,
    vocab_size: int,
) -> dict:
    """
    Bir modeli token_budget kadar eğitir, sonuç sözlüğü döner.
    Dönen dict: run_id, model_type, scale, seed, best_val_ppl,
                final_val_ppl, train_history, flops_info, param_info,
                throughput_toks_per_sec, physical_params (fluid only).
    """
    # ── Seed ──────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    random.seed(seed)

    sc  = scale_cfg
    run_id = f"14_{model_type}_{sc.name}_s{seed}"
    master_print(f"\n{'='*60}")
    master_print(f"  RUN: {run_id}")
    master_print(f"  Scale: d={sc.d_model}, L={sc.n_layers}, seq={sc.seq_len}")
    master_print(f"  Token budget: {token_budget:,}")
    master_print(f"{'='*60}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_type == "fluid":
        model = FluidLM(
            vocab_size=vocab_size, d_model=sc.d_model,
            n_layers=sc.n_layers, max_seq_len=sc.seq_len,
            mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        ).to(device)
    else:
        model = GPTBaseline(
            vocab_size=vocab_size, d_model=sc.d_model,
            n_layers=sc.n_layers, n_heads=sc.n_heads,
            max_seq_len=sc.seq_len, mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
        ).to(device)

    # ── Parametre sayımı ──────────────────────────────────────────────────────
    param_info = count_params(model)
    master_print(f"  Parametreler: {param_info['total']/1e6:.1f}M toplam  "
                 f"| routing={param_info['routing']:,}  "
                 f"| mlp={param_info['mlp']/1e6:.1f}M  "
                 f"| emb={param_info['emb']/1e6:.1f}M")

    # ── FLOP analizi ──────────────────────────────────────────────────────────
    flops_info = flops_per_token(
        model_type, sc.d_model, sc.n_layers, sc.seq_len, vocab_size, args.mlp_ratio
    )
    master_print(f"  FLOPs/token: {flops_info['total_mflop']:.1f} MFLOP  "
                 f"({flops_info['routing_label']})")

    # ── torch.compile ─────────────────────────────────────────────────────────
    if args.compile:
        master_print("  torch.compile() uygulanıyor...")
        model = torch.compile(model)

    # ── DDP ───────────────────────────────────────────────────────────────────
    raw_model = model
    if IS_DDP:
        model = DDP(model, device_ids=[LOCAL_RANK])
        raw_model = model.module

    # ── Optimizer (AdamW, GPT-3 standardı) ───────────────────────────────────
    # Physics params (NS katman): 10× LR + hafif decay — α/p_scale serbestçe öğrensin
    # GPT modelinde bu isimler yoktur → physics_params boş olur, sorun değil
    PHYSICS_NAMES = {'log_nu', 'log_dt', 'log_alpha', 'p_scale_raw'}
    physics_params = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and any(k in n for k in PHYSICS_NAMES)]
    decay_params   = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.dim() >= 2
                      and not any(k in n for k in PHYSICS_NAMES)]
    nodecay_params = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.dim() < 2
                      and not any(k in n for k in PHYSICS_NAMES)]
    pg_list = [
        {"params": physics_params,  "lr": args.lr * 10, "weight_decay": 0.01,             "lr_scale": 10.0},
        {"params": decay_params,    "weight_decay": args.weight_decay,                     "lr_scale": 1.0},
        {"params": nodecay_params,  "weight_decay": 0.0,                                  "lr_scale": 1.0},
    ]
    # Boş param gruplarını çıkar (GPT için physics_params boş olabilir)
    pg_list = [pg for pg in pg_list if len(pg["params"]) > 0]
    optimizer = torch.optim.AdamW(pg_list, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # ── Batch size ────────────────────────────────────────────────────────────
    if BATCH_OVERRIDE:
        batch_size = BATCH_OVERRIDE
        grad_accum = 1
    else:
        batch_size, grad_accum = auto_batch(sc.d_model, sc.seq_len, vram_gb)
    effective_batch_tokens = batch_size * grad_accum * sc.seq_len
    total_steps = token_budget // effective_batch_tokens
    warmup_steps = max(100, int(total_steps * args.warmup_frac))
    master_print(f"  batch={batch_size}  grad_accum={grad_accum}  "
                 f"eff_batch={effective_batch_tokens/1e3:.0f}K tok/step  "
                 f"total_steps={total_steps:,}  warmup={warmup_steps:,}")

    # ── Veri ──────────────────────────────────────────────────────────────────
    train_stream = TokenStream("train", sc.seq_len, device)
    val_stream   = TokenStream("val",   sc.seq_len, device)

    # ── Eğitim ────────────────────────────────────────────────────────────────
    best_val_ppl    = float("inf")
    best_ckpt_path  = os.path.join(out_dir, f"{run_id}_best.pt")
    train_history: List[dict] = []
    scaler = torch.amp.GradScaler("cuda",
                 enabled=(device.type=="cuda" and not USE_BF16))

    tokens_seen  = 0
    t0_global    = time.time()
    throughput_samples: List[float] = []
    resume_step  = 0

    # ── Resume: mevcut checkpoint varsa yükle ────────────────────────────────
    if os.path.exists(best_ckpt_path):
        master_print(f"  [Resume] {best_ckpt_path} bulundu, yükleniyor...")
        resume_ckpt = torch.load(best_ckpt_path, map_location=device,
                                  weights_only=True)
        raw_model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        resume_step      = resume_ckpt.get("step", 0) + 1
        best_val_ppl     = resume_ckpt.get("val_ppl", float("inf"))
        tokens_seen      = resume_step * effective_batch_tokens
        master_print(f"  [Resume] step={resume_step}  "
                     f"tokens={tokens_seen/1e9:.2f}B  "
                     f"best_ppl={best_val_ppl:.4f}")
        # Veri akışını resume_step'e ilerlet (streaming)
        skip_batches = resume_step * grad_accum * batch_size
        master_print(f"  [Resume] {skip_batches:,} batch atlanıyor...")
        for _ in range(skip_batches):
            train_stream.get_batch(batch_size)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(resume_step, total_steps):
        # LR güncelle
        lr_now = get_lr(step, warmup_steps, total_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now * pg.get("lr_scale", 1.0)

        # Gradient accumulation
        t_step = time.time()
        for micro in range(grad_accum):
            x, y = train_stream.get_batch(batch_size)
            if IS_DDP:
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            with AUTOCAST:
                logits = model(x)
                loss   = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                ) / grad_accum
            scaler.scale(loss).backward()
            tokens_seen += x.numel()

        # Gradient clip + optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        throughput_samples.append(effective_batch_tokens / (time.time() - t_step))

        # ── Eval ──────────────────────────────────────────────────────────────
        if step % args.eval_interval == 0 or step == total_steps - 1:
            val_ppl = evaluate(raw_model, val_stream, batch_size, args.eval_steps)
            elapsed = time.time() - t0_global
            avg_tput = sum(throughput_samples[-100:]) / max(1, len(throughput_samples[-100:]))

            record = {
                "step":       step,
                "tokens":     tokens_seen,
                "lr":         lr_now,
                "val_ppl":    val_ppl,
                "throughput": avg_tput,
            }
            train_history.append(record)

            if IS_MASTER:
                master_print(
                    f"  step={step:6d}/{total_steps}  "
                    f"tok={tokens_seen/1e9:.2f}B  "
                    f"val_ppl={val_ppl:.4f}  "
                    f"lr={lr_now:.2e}  "
                    f"tput={avg_tput/1e3:.0f}K tok/s  "
                    f"t={elapsed/60:.0f}m"
                )

            # ── Checkpoint ────────────────────────────────────────────────────
            if val_ppl < best_val_ppl and IS_MASTER:
                best_val_ppl = val_ppl
                save_dict = {
                    "model_type":    model_type,
                    "scale":         sc.name,
                    "seed":          seed,
                    "step":          step,
                    "val_ppl":       val_ppl,
                    "model_state":   raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": {
                        "vocab_size": vocab_size,
                        "d_model":    sc.d_model,
                        "n_layers":   sc.n_layers,
                        "n_heads":    sc.n_heads,
                        "seq_len":    sc.seq_len,
                        "mlp_ratio":  args.mlp_ratio,
                    },
                }
                torch.save(save_dict, best_ckpt_path)

    # ── WikiText-103 zero-shot eval ───────────────────────────────────────────
    wt103_ppl = -1.0
    if args.wikitext_eval and IS_MASTER:
        # Best checkpoint'i yükle
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        raw_model.load_state_dict(ckpt["model_state"])
        wt103_ppl = eval_wikitext103(raw_model, device, sc.seq_len)

    # ── Emergent fiziksel parametreler (FluidLM only) ─────────────────────────
    phys = None
    if model_type == "fluid" and IS_MASTER:
        phys = raw_model.physical_params()
        nu_vals = phys["nu"]
        master_print(f"\n  [Fiziksel Parametreler @ {run_id}]")
        master_print(f"  ν (viscosity): early(0-3)={sum(nu_vals[:4])/4:.4f}  "
                     f"late({sc.n_layers-4}-{sc.n_layers-1})="
                     f"{sum(nu_vals[-4:])/4:.4f}")
        master_print(f"  ν range: [{min(nu_vals):.4f}, {max(nu_vals):.4f}]  "
                     f"gradient={(nu_vals[-1]-nu_vals[0]):.4f}")

    # ── Özet ──────────────────────────────────────────────────────────────────
    avg_tput = sum(throughput_samples) / max(1, len(throughput_samples))
    result = {
        "run_id":           run_id,
        "model_type":       model_type,
        "scale":            sc.name,
        "seed":             seed,
        "token_budget":     token_budget,
        "tokens_seen":      tokens_seen,
        "best_val_ppl":     best_val_ppl,
        "final_val_ppl":    train_history[-1]["val_ppl"] if train_history else -1,
        "wt103_ppl":        wt103_ppl,
        "throughput_tps":   avg_tput,
        "param_info":       param_info,
        "flops_info":       flops_info,
        "train_history":    train_history,
        "physical_params":  phys,
        "ckpt_path":        best_ckpt_path,
        "gpu":              gpu_name,
        "dtype":            str(DTYPE),
    }
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Sonuç Görselleştirme
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results: List[dict], out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        master_print("[Plot] matplotlib bulunamadı — atlanıyor")
        return

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("FluidLM vs GPT-2 — Industrial Scale Comparison", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel A: Training curves (val PPL vs tokens) ───────────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    colors = {"fluid": "#2196F3", "gpt": "#F44336"}
    styles = {"S": "-", "M": "--"}

    for r in all_results:
        if not r["train_history"]:
            continue
        xs = [h["tokens"] / 1e9 for h in r["train_history"]]
        ys = [h["val_ppl"]      for h in r["train_history"]]
        lbl = f"{r['model_type'].upper()}-{r['scale']} (s{r['seed']})"
        ax_a.plot(xs, ys,
                  color=colors[r["model_type"]],
                  linestyle=styles[r["scale"]],
                  linewidth=1.8, alpha=0.85, label=lbl)

    ax_a.set_xlabel("Tokens Seen (B)", fontsize=11)
    ax_a.set_ylabel("Val Token PPL", fontsize=11)
    ax_a.set_title("A. Training Curves — Val PPL vs Token Budget", fontweight="bold")
    ax_a.legend(fontsize=8, ncol=2)
    ax_a.grid(True, alpha=0.3)

    # ── Panel B: Bar chart — best val PPL ─────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    run_labels, ppls, bar_colors = [], [], []
    for r in all_results:
        run_labels.append(f"{r['model_type'].upper()}-{r['scale']}\n(s{r['seed']})")
        ppls.append(r["best_val_ppl"])
        bar_colors.append(colors[r["model_type"]])
    bars = ax_b.bar(range(len(ppls)), ppls, color=bar_colors, alpha=0.8, edgecolor="white")
    ax_b.set_xticks(range(len(ppls)))
    ax_b.set_xticklabels(run_labels, fontsize=7)
    ax_b.set_ylabel("Best Val PPL (↓ iyi)", fontsize=11)
    ax_b.set_title("B. Best Val PPL Karşılaştırması", fontweight="bold")
    for bar, ppl in zip(bars, ppls):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f"{ppl:.2f}", ha="center", fontsize=7, fontweight="bold")
    ax_b.grid(True, axis="y", alpha=0.3)

    # ── Panel C: FLOP karşılaştırması ─────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    seen_scales = set()
    for r in all_results:
        key = (r["model_type"], r["scale"])
        if key in seen_scales:
            continue
        seen_scales.add(key)
    flop_data = {}
    for r in all_results:
        key = f"{r['model_type'].upper()}-{r['scale']}"
        if key not in flop_data:
            fi = r["flops_info"]
            flop_data[key] = {
                "mlp":     fi["mlp_total_mflop"],
                "routing": fi["routing_total_mflop"],
                "head":    fi["lm_head_mflop"],
                "color":   colors[r["model_type"]],
            }
    keys  = list(flop_data.keys())
    mlps  = [flop_data[k]["mlp"]     for k in keys]
    routs = [flop_data[k]["routing"]  for k in keys]
    heads = [flop_data[k]["head"]     for k in keys]
    x     = np.arange(len(keys))
    ax_c.bar(x, mlps,  label="MLP",     color="#4CAF50", alpha=0.85)
    ax_c.bar(x, routs, bottom=mlps,     label="Routing", color="#FF9800", alpha=0.85)
    ax_c.bar(x, heads, bottom=np.array(mlps)+np.array(routs),
             label="LM Head", color="#9C27B0", alpha=0.85)
    ax_c.set_xticks(x); ax_c.set_xticklabels(keys, fontsize=8)
    ax_c.set_ylabel("MFLOP/token", fontsize=11)
    ax_c.set_title("C. FLOP Dağılımı (routing vs MLP)", fontweight="bold")
    ax_c.legend(fontsize=8); ax_c.grid(True, axis="y", alpha=0.3)

    # ── Panel D: Routing params karşılaştırması (log scale) ───────────────
    ax_d = fig.add_subplot(gs[1, 1])
    names  = [f"{r['model_type'].upper()}-{r['scale']}" for r in all_results]
    rparams = [r["param_info"]["routing"] for r in all_results]
    unique  = {}
    for n, rp, r in zip(names, rparams, all_results):
        if n not in unique:
            unique[n] = (rp, colors[r["model_type"]])
    u_names = list(unique.keys())
    u_vals  = [unique[n][0] for n in u_names]
    u_colors= [unique[n][1] for n in u_names]
    bars_d  = ax_d.bar(range(len(u_names)), u_vals, color=u_colors, alpha=0.8)
    ax_d.set_yscale("log")
    ax_d.set_xticks(range(len(u_names)))
    ax_d.set_xticklabels(u_names, fontsize=8)
    ax_d.set_ylabel("Routing Parametreleri (log)", fontsize=10)
    ax_d.set_title("D. Routing Param Sayısı (log scale)", fontweight="bold")
    for bar, val in zip(bars_d, u_vals):
        ax_d.text(bar.get_x() + bar.get_width()/2, val * 1.5,
                  f"{val:,}", ha="center", fontsize=7, fontweight="bold")
    ax_d.grid(True, axis="y", alpha=0.3)

    # ── Panel E: Emergent ν gradyanı (FluidLM only) ────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    fluid_runs = [r for r in all_results if r["model_type"] == "fluid" and r["physical_params"]]
    if fluid_runs:
        for r in fluid_runs:
            nu_vals = r["physical_params"]["nu"]
            lbl     = f"FluidLM-{r['scale']} (s{r['seed']})"
            ax_e.plot(nu_vals, marker="o", markersize=4,
                      label=lbl, linewidth=1.5)
        ax_e.set_xlabel("Katman", fontsize=11)
        ax_e.set_ylabel("ν (learned viscosity)", fontsize=11)
        ax_e.set_title("E. Emergent ν Gradyanı", fontweight="bold")
        ax_e.legend(fontsize=8)
        ax_e.grid(True, alpha=0.3)
    else:
        ax_e.text(0.5, 0.5, "FluidLM\nsonucu bekleniyor",
                  ha="center", va="center", transform=ax_e.transAxes, fontsize=11)
        ax_e.set_title("E. Emergent ν Gradyanı", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "14_industrial_compare.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    master_print(f"\n[Plot] Kaydedildi: {out_path}")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Akademik Markdown Özeti
# ─────────────────────────────────────────────────────────────────────────────

def write_academic_summary(all_results: List[dict], out_dir: str):
    """
    Paper §4 taslağı: metrikler, tablolar, yorumlar.
    results/14_industrial_summary.md olarak kaydedilir.
    """

    # Ortalama PPL (aynı model/scale, farklı seed)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in all_results:
        groups[f"{r['model_type']}_{r['scale']}"].append(r)

    def avg_std(vals):
        if len(vals) == 1:
            return vals[0], 0.0
        mu = sum(vals) / len(vals)
        std = math.sqrt(sum((v-mu)**2 for v in vals) / len(vals))
        return mu, std

    lines = [
        "# Experiment 14 — Industrial Scale Results",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*\n",
        "## Overview",
        "Comparative evaluation of FluidLM (NS routing) vs GPT-2 (MHA)",
        "on OpenWebText using BPE tokenization (GPT-2, vocab=50,257).\n",
        "**Research Questions:**",
        "- **RQ1**: Can NS routing match MHA representational power?",
        "- **RQ2**: What is the routing parameter → FLOP/token tradeoff?",
        "- **RQ3**: Does the emergent ν gradient generalize beyond char-level?\n",
        "## 1. Model Architecture Comparison\n",
        "| Model | Scale | d | L | Routing Params | MLP Params | Total |",
        "|-------|-------|---|---|---------------|-----------|-------|",
    ]

    for r in all_results:
        pi = r["param_info"]
        lines.append(
            f"| {r['model_type'].upper()} | {r['scale']} | "
            f"{SCALE_CONFIGS[r['scale']].d_model} | "
            f"{SCALE_CONFIGS[r['scale']].n_layers} | "
            f"**{pi['routing']:,}** | "
            f"{pi['mlp']/1e6:.1f}M | "
            f"{pi['total']/1e6:.1f}M |"
        )

    lines += ["", "## 2. Main Results\n",
              "| Model | Scale | Seeds | Val PPL (mean±σ) | WikiText-103 PPL | "
              "MFLOP/token | Throughput (tok/s) |",
              "|-------|-------|-------|-----------------|-----------------|"
              "------------|-------------------|"]

    for key, runs in groups.items():
        model_type, scale = key.split("_")
        ppls   = [r["best_val_ppl"] for r in runs]
        wt103s = [r["wt103_ppl"]    for r in runs if r["wt103_ppl"] > 0]
        tputs  = [r["throughput_tps"] for r in runs]
        mu_ppl, std_ppl = avg_std(ppls)
        mu_wt  = sum(wt103s)/max(1,len(wt103s)) if wt103s else -1
        mu_tput = sum(tputs)/max(1,len(tputs))
        fi     = runs[0]["flops_info"]
        wt_str = f"{mu_wt:.2f}" if mu_wt > 0 else "—"
        seed_str = "/".join(str(r["seed"]) for r in runs)
        lines.append(
            f"| {model_type.upper()} | {scale} | {seed_str} | "
            f"**{mu_ppl:.3f}** ± {std_ppl:.3f} | "
            f"{wt_str} | "
            f"{fi['total_mflop']:.1f} | "
            f"{mu_tput/1e3:.0f}K |"
        )

    # FLOP karşılaştırması
    lines += ["", "## 3. FLOP Analysis\n",
              "| Model | Scale | Routing MFLOP | MLP MFLOP | Total MFLOP | Routing % | Complexity |",
              "|-------|-------|--------------|----------|-------------|-----------|-----------|"]
    for key, runs in groups.items():
        model_type, scale = key.split("_")
        fi = runs[0]["flops_info"]
        pct = 100 * fi["routing_total_mflop"] / fi["total_mflop"]
        lines.append(
            f"| {model_type.upper()} | {scale} | "
            f"{fi['routing_total_mflop']:.1f} | "
            f"{fi['mlp_total_mflop']:.1f} | "
            f"{fi['total_mflop']:.1f} | "
            f"{pct:.1f}% | "
            f"{fi['routing_complexity']} |"
        )

    # Fiziksel parametreler
    fluid_results = [r for r in all_results
                     if r["model_type"] == "fluid" and r["physical_params"]]
    if fluid_results:
        lines += ["", "## 4. Emergent Physical Parameters (RQ3)\n",
                  "| Run | Scale | ν early (L0-3) | ν late (Ln-3 to Ln) | ν gradient | α range | p_scale range |",
                  "|-----|-------|---------------|-------------------|-----------|---------|--------------|"]
        for r in fluid_results:
            p  = r["physical_params"]
            nu = p["nu"]
            n  = len(nu)
            al = p["alpha"]
            ps = p["p_scale"]
            lines.append(
                f"| {r['run_id']} | {r['scale']} | "
                f"{sum(nu[:4])/4:.4f} | "
                f"{sum(nu[-4:])/4:.4f} | "
                f"**{nu[-1]-nu[0]:+.4f}** | "
                f"[{min(al):.3f}, {max(al):.3f}] | "
                f"[{min(ps):.3f}, {max(ps):.3f}] |"
            )
        lines += [
            "",
            "> **Finding (RQ3)**: If the ν gradient is positive (late layers > early layers),",
            "> this replicates the char-level finding (exp. 11) at industrial BPE scale,",
            "> suggesting the viscosity gradient is a general emergent property of NS routing,",
            "> not an artifact of character-level tokenization.",
        ]

    lines += [
        "",
        "## 5. Reproducibility",
        "",
        "```bash",
        "# Full reproduction (Colab A100, ~6h for S, ~20h for M)",
        "pip install tiktoken datasets",
        "git clone https://github.com/KULLANICI/neo_lang.git && cd neo_lang",
        "python experiments/14_industrial_compare.py \\",
        "    --scale both --tokens 3e9 --seeds 42 43 44",
        "```",
        "",
        "| Hyperparameter | Value |",
        "|---------------|-------|",
        f"| Token budget    | {int(all_results[0]['token_budget']):,} |",
        f"| Seq len         | {SCALE_CONFIGS[all_results[0]['scale']].seq_len} |",
        f"| LR              | {args.lr} |",
        f"| LR schedule     | Cosine warmup ({args.warmup_frac*100:.0f}% warmup) |",
        f"| Min LR          | {args.min_lr} |",
        f"| Weight decay    | {args.weight_decay} |",
        f"| Grad clip       | {args.grad_clip} |",
        f"| Optimizer       | AdamW (β₁=0.9, β₂=0.95) |",
        f"| Tokenizer       | tiktoken gpt2 (vocab=50,257) |",
        f"| Dataset         | OpenWebText (Skylion007/openwebtext) |",
        "",
        "---",
        "*Generated by experiments/14_industrial_compare.py*",
    ]

    out_path = os.path.join(out_dir, "14_industrial_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    master_print(f"[Summary] Kaydedildi: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Ana Akış
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
    REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
    OUT_DIR    = args.out_dir or os.path.join(REPO_ROOT, "results")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Vocab boyutunu belirle ────────────────────────────────────────────────
    # tiktoken yüklüyse VOCAB=50257 zaten set edildi.
    # Char fallback durumunda TokenStream.__init__ VOCAB'u set eder;
    # o zaman ilk train_stream açılışında doğal olarak belirlenir.
    vocab_size = VOCAB if VOCAB else 65   # tiktoken=50257, char fallback=65

    master_print(f"\n[Config] vocab={vocab_size}  scales={args.scale}  "
                 f"model={args.model}  tokens={args.tokens:.0e}")

    # ── eval_only modu ────────────────────────────────────────────────────────
    if args.eval_only:
        assert args.fluid_ckpt or args.gpt_ckpt, \
            "--fluid_ckpt ve/veya --gpt_ckpt gerekli"
        master_print("[eval_only] Checkpoint'lerden WikiText-103 değerlendirmesi...")
        for ckpt_path, model_type in [(args.fluid_ckpt, "fluid"),
                                       (args.gpt_ckpt, "gpt")]:
            if not ckpt_path:
                continue
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            cfg  = ckpt["cfg"]
            if model_type == "fluid":
                m = FluidLM(cfg["vocab_size"], cfg["d_model"], cfg["n_layers"],
                            cfg["seq_len"], cfg["mlp_ratio"]).to(device)
            else:
                m = GPTBaseline(cfg["vocab_size"], cfg["d_model"], cfg["n_layers"],
                                cfg["n_heads"], cfg["seq_len"], cfg["mlp_ratio"]).to(device)
            m.load_state_dict(ckpt["model_state"])
            ppl = eval_wikitext103(m, device, cfg["seq_len"])
            master_print(f"  {model_type.upper()} WikiText-103 PPL = {ppl:.2f}")
        return

    # ── Eğitim koşuları ───────────────────────────────────────────────────────
    scales_to_run = (["S","M"] if args.scale == "both" else [args.scale])
    models_to_run = (["fluid","gpt"] if args.model == "both" else [args.model])
    token_budget  = int(args.tokens)

    # M scale için sadece ilk seed
    def seeds_for(scale):
        return args.seeds if scale == "S" else args.seeds[:1]

    # ── JSON'dan tamamlanan run'ları yükle (crash recovery) ──────────────────
    json_path   = os.path.join(OUT_DIR, "14_industrial_compare.json")
    all_results: List[dict] = []
    completed_ids: set = set()
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                all_results = json.load(f)
            completed_ids = {r["run_id"] for r in all_results
                             if r.get("best_val_ppl", -1) > 0}
            master_print(f"[Resume] JSON yüklendi: {len(completed_ids)} run tamamlanmış: "
                         f"{sorted(completed_ids)}")
        except Exception as e:
            master_print(f"[Resume] JSON okunamadı ({e}), sıfırdan başlıyor.")
            all_results = []

    for scale in scales_to_run:
        sc = SCALE_CONFIGS[scale]
        # Interleaved sıra: fluid_s42, gpt_s42, fluid_s43, gpt_s43, ...
        # Böylece her seed çifti ardışık eğitilir; erken karşılaştırma mümkün olur.
        for seed in seeds_for(scale):
            for model_type in models_to_run:
                run_id_check = f"14_{model_type}_{sc.name}_s{seed}"
                if run_id_check in completed_ids:
                    master_print(f"  [Skip] {run_id_check} zaten tamamlandı.")
                    continue
                result = train_one(
                    model_type=model_type,
                    scale_cfg=sc,
                    seed=seed,
                    token_budget=token_budget,
                    out_dir=OUT_DIR,
                    vocab_size=vocab_size,
                )
                all_results.append(result)

                # Her run sonrası kaydet (crash recovery)
                json_path = os.path.join(OUT_DIR, "14_industrial_compare.json")
                if IS_MASTER:
                    # train_history'yi kompakt tut (ilk ve son 50 kayıt + min)
                    for r in all_results:
                        h = r.get("train_history", [])
                        if len(h) > 110:
                            r["train_history"] = h[:5] + h[-50:]
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, indent=2, default=str)
                    master_print(f"[JSON] Kaydedildi: {json_path}")

    # ── Görselleştirme + Özet ─────────────────────────────────────────────────
    if IS_MASTER and all_results:
        plot_results(all_results, OUT_DIR)
        write_academic_summary(all_results, OUT_DIR)

        # ── Konsol özet tablosu ───────────────────────────────────────────────
        master_print("\n" + "="*70)
        master_print("  SONUÇ TABLOSU")
        master_print("="*70)
        master_print(f"  {'Model':<20} {'Scale':<6} {'Seed':<5} "
                     f"{'Val PPL':>10} {'WT-103':>8} "
                     f"{'MFLOP/tok':>10} {'RoutingP':>10}")
        master_print("-"*70)
        for r in all_results:
            wt = f"{r['wt103_ppl']:.2f}" if r['wt103_ppl'] > 0 else "—"
            master_print(
                f"  {r['model_type'].upper():<20} "
                f"{r['scale']:<6} "
                f"{r['seed']:<5} "
                f"{r['best_val_ppl']:>10.4f} "
                f"{wt:>8} "
                f"{r['flops_info']['total_mflop']:>10.1f} "
                f"{r['param_info']['routing']:>10,}"
            )
        master_print("="*70)

        # RQ1 değerlendirmesi
        for scale in scales_to_run:
            fluid_ppls = [r["best_val_ppl"] for r in all_results
                          if r["model_type"]=="fluid" and r["scale"]==scale]
            gpt_ppls   = [r["best_val_ppl"] for r in all_results
                          if r["model_type"]=="gpt"   and r["scale"]==scale]
            if fluid_ppls and gpt_ppls:
                mu_f = sum(fluid_ppls)/len(fluid_ppls)
                mu_g = sum(gpt_ppls)/len(gpt_ppls)
                delta = mu_f - mu_g
                ratio = mu_f / mu_g
                fi_f  = next(r["flops_info"] for r in all_results
                             if r["model_type"]=="fluid" and r["scale"]==scale)
                fi_g  = next(r["flops_info"] for r in all_results
                             if r["model_type"]=="gpt" and r["scale"]==scale)
                flop_ratio = fi_g["total_mflop"] / fi_f["total_mflop"]
                rp_f = next(r["param_info"]["routing"] for r in all_results
                            if r["model_type"]=="fluid" and r["scale"]==scale)
                rp_g = next(r["param_info"]["routing"] for r in all_results
                            if r["model_type"]=="gpt" and r["scale"]==scale)
                rp_ratio = rp_g / max(rp_f, 1)

                master_print(f"\n  [RQ1 — Scale {scale}]")
                master_print(f"  ΔPPL (FluidLM − GPT) = {delta:+.3f}  "
                             f"({'FluidLM daha kötü' if delta > 0 else 'FluidLM daha iyi'})")
                master_print(f"  FLOP kazanımı: GPT/FluidLM = {flop_ratio:.2f}×")
                master_print(f"  Routing param azalması: {rp_ratio:,.0f}× daha az")

    if IS_DDP:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
