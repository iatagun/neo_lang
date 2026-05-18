# ============================================================
# experiments/20_fft_advection_pilot.py
#
# RQ7: FFT tabanlı global causal advection
#
# Fikir:
#   multihop_gradient(u, k) → O(L·k·D), RF = k·n_layers
#   fft_causal_gradient(u)  → O(L·logL·D), RF = L (full sekans)
#
# Causal fark operatörü frekans uzayında:
#   H(k) = 1 − e^{−2πik/L}   ← backward difference (causal)
#
# Üzerine öğrenilen spektral ağırlık eklersek:
#   G(k) = û(k) · exp(log_mag[k]) · H(k)
#   grad = IFFT(G)                ← RF = L, causal, learned
#
# Üç model karşılaştırılır (nano scale, 20M token):
#   A) FluidLM-multihop k=4   (exp16 nano baseline)
#   B) FluidLM-FFT-adv        (bu deney)
#   C) GPT-Nano               (MHA referansı)
#
# Kullanım:
#   python experiments/20_fft_advection_pilot.py
#   python experiments/20_fft_advection_pilot.py --tokens 5e6 --quick
#
# Çıktı:
#   results/20_fft_advection_pilot.json
# ============================================================

import sys, os, math, time, json, argparse, random
from contextlib import nullcontext
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

parser = argparse.ArgumentParser()
parser.add_argument("--tokens",      type=float, default=20e6)
parser.add_argument("--quick",       action="store_true",
                    help="5M token, log_every=50 (hızlı debug)")
parser.add_argument("--d_model",     type=int, default=256)
parser.add_argument("--n_layers",    type=int, default=6)
parser.add_argument("--seq_len",     type=int, default=128)
parser.add_argument("--batch_size",  type=int, default=64)
parser.add_argument("--lr",          type=float, default=3e-4)
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--log_every",   type=int,   default=100)
parser.add_argument("--eval_every",  type=int,   default=500)
parser.add_argument("--eval_steps",  type=int,   default=50)
parser.add_argument("--out_dir",     default=None)
args = parser.parse_args()

if args.quick:
    args.tokens    = 5e6
    args.log_every = 50
    args.eval_every = 200

torch.manual_seed(args.seed)
random.seed(args.seed)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE    = torch.bfloat16 if USE_BF16 else torch.float32
AUTOCAST = nullcontext() if DTYPE == torch.float32 else \
           torch.amp.autocast(device_type="cuda", dtype=DTYPE)

gpu_name = torch.cuda.get_device_properties(0).name if device.type == "cuda" else "CPU"
print(f"[Device] {gpu_name}  dtype={DTYPE}")
print(f"[Config] d={args.d_model}  L={args.n_layers}  seq={args.seq_len}  "
      f"tokens={args.tokens/1e6:.0f}M")

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer + Veri
# ─────────────────────────────────────────────────────────────────────────────

try:
    import tiktoken
    ENC   = tiktoken.get_encoding("gpt2")
    VOCAB = ENC.n_vocab
    encode = lambda t: ENC.encode_ordinary(t)
    EOS_ID = 50256
    print(f"[Tokenizer] tiktoken gpt2  vocab={VOCAB}")
except ImportError:
    ENC = None; VOCAB = None; encode = None; EOS_ID = None

class TokenStream:
    EOS_ID = 50256
    def __init__(self, split: str):
        self.buffer: List[int] = []
        self._load(split)

    def _load(self, split: str):
        global VOCAB, encode
        if ENC is not None:
            try:
                from datasets import load_dataset
                ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
                ds = ds.shuffle(seed=42, buffer_size=10_000)
                ds = ds.skip(990_000) if split == "val" else ds.take(990_000)
                self._stream = iter(ds)
                self._source = "openwebtext"
                return
            except Exception as e:
                print(f"[Data] OWT yuklenemedi: {e}")
        REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(REPO, "data", "shakespeare.txt")
        if not os.path.exists(path):
            import urllib.request
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
                "data/tinyshakespeare/input.txt", path)
        with open(path, encoding="utf-8") as f: text = f.read()
        if ENC:
            ids = encode(text)
        else:
            chars = sorted(set(text)); VOCAB = len(chars)
            stoi  = {c: i for i, c in enumerate(chars)}
            encode = lambda s: [stoi[c] for c in s]
            ids = encode(text)
        n = int(len(ids) * 0.9)
        self._tensor = torch.tensor(ids[:n] if split=="train" else ids[n:], dtype=torch.long)
        self._source = "shakespeare"

    def _fill(self, n: int):
        while len(self.buffer) < n:
            try:
                self.buffer.extend(encode(next(self._stream)["text"]))
                self.buffer.append(self.EOS_ID)
            except StopIteration:
                from datasets import load_dataset
                ds = load_dataset("Skylion007/openwebtext", split="train",
                                  streaming=True).shuffle(seed=random.randint(0,9999))
                self._stream = iter(ds)

    def get_batch(self, B: int, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        need = (T + 1) * B
        if self._source == "openwebtext":
            self._fill(need)
            chunk = self.buffer[:need]; self.buffer = self.buffer[need:]
            t = torch.tensor(chunk, dtype=torch.long).view(B, T + 1)
        else:
            ix = torch.randint(len(self._tensor) - T - 1, (B,))
            t  = torch.stack([self._tensor[i:i+T+1] for i in ix])
        return t[:, :-1].to(device), t[:, 1:].to(device)

# ─────────────────────────────────────────────────────────────────────────────
# Gradyan Operatörleri
# ─────────────────────────────────────────────────────────────────────────────

def _hop_scales(k: int) -> List[int]:
    return [2**i for i in range(int(math.log2(k)) + 1)] if k > 1 else [1]

def multihop_gradient(u: torch.Tensor, k: int) -> torch.Tensor:
    """Çok ölçekli sonlu fark gradyanı. RF = k · n_layers."""
    scales = _hop_scales(k)
    out    = torch.zeros_like(u)
    for s in scales:
        prev = F.pad(u, (0, 0, s, 0))[:, :-s, :]
        out  = out + (u - prev) / s
    return out / len(scales)

def multihop_laplacian(u: torch.Tensor, k: int) -> torch.Tensor:
    scales = _hop_scales(k)
    out    = torch.zeros_like(u)
    for s in scales:
        p   = F.pad(u, (0, 0, 2*s, 0))
        out = out + (u - 2*p[:, s:-s, :] + p[:, :-2*s, :]) / (s*s)
    return out / len(scales)

def fft_causal_gradient(u: torch.Tensor,
                        log_mag: torch.Tensor) -> torch.Tensor:
    """
    Causal gradient frekans uzayında.

    Operatör: H(k) = 1 − e^{−2πik/L}  (backward difference, causal)
    Ağırlık:  exp(log_mag[k])           (öğrenilen spektral öncelik)

    Özellikler:
    - RF = full sequence length L (tek adımda tüm geçmiş)
    - FLOP = O(L log L · D) (hop_k'dan bağımsız)
    - k=0 (DC): H(0)=0 → mean removed, fiziksel olarak doğru
    - Parametre: seq/2+1 skaler / katman (nano: 65, S: 513)
    """
    B, L, D = u.shape
    U = torch.fft.rfft(u.float(), dim=1)          # [B, L//2+1, D]

    k_idx = torch.arange(U.shape[1], device=u.device, dtype=torch.float32)
    omega  = 2.0 * math.pi * k_idx / L            # [L//2+1]

    # Causal backward difference phase
    H_r = 1.0 - torch.cos(omega)                  # Re(H)
    H_i = torch.sin(omega)                         # Im(H)

    # Learned spectral magnitude
    mag = log_mag.exp()                            # [L//2+1]

    # Combined filter coefficients
    F_r = (mag * H_r).unsqueeze(0).unsqueeze(-1)   # [1, L//2+1, 1]
    F_i = (mag * H_i).unsqueeze(0).unsqueeze(-1)   # [1, L//2+1, 1]

    # Complex multiply: (Ur + iUi)(Fr + iFi)
    Ur, Ui = U.real, U.imag
    Gr = Ur * F_r - Ui * F_i
    Gi = Ur * F_i + Ui * F_r

    G    = torch.complex(Gr, Gi)
    grad = torch.fft.irfft(G, n=L, dim=1).to(u.dtype)
    return grad

def spectral_pressure(adv: torch.Tensor, alpha: torch.Tensor,
                      use_multihop_div: bool = True, hop_k: int = 4) -> torch.Tensor:
    """Helmholtz-Poisson basınç çözümü (FFT)."""
    if use_multihop_div:
        div = (multihop_gradient(adv, hop_k) / adv.shape[-1]).sum(-1)
    else:
        # FFT-adv path: adv zaten global → divergence da FFT ile
        B, L, D = adv.shape
        A = torch.fft.rfft(adv.float(), dim=1)
        k_idx = torch.arange(A.shape[1], device=adv.device, dtype=torch.float32)
        omega  = 2.0 * math.pi * k_idx / L
        H_r = 1.0 - torch.cos(omega); H_i = torch.sin(omega)
        F_r = H_r.unsqueeze(0).unsqueeze(-1)
        F_i = H_i.unsqueeze(0).unsqueeze(-1)
        Gr = A.real * F_r - A.imag * F_i
        Gi = A.real * F_i + A.imag * F_r
        G  = torch.complex(Gr, Gi)
        div_vec = torch.fft.irfft(G, n=L, dim=1).to(adv.dtype)
        div = div_vec.mean(dim=-1)

    B, L = div.shape
    f    = torch.fft.rfft(div.float(), dim=1)
    k_p  = torch.arange(L//2+1, dtype=torch.float32, device=div.device)
    a    = alpha.float() if isinstance(alpha, torch.Tensor) else float(alpha)
    lam  = 2.0*(torch.cos(2.0*math.pi*k_p/L)-1.0) - a**2
    lam  = lam.masked_fill(lam.abs() < 1e-8, -1e-8)
    p    = torch.fft.irfft(-f / lam.unsqueeze(0), n=L, dim=1).to(adv.dtype)
    return p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5) + 1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# FluidLayer — iki varyant: multihop ve fft_adv
# ─────────────────────────────────────────────────────────────────────────────

class FluidLayerMultiHop(nn.Module):
    """Exp16/17/18 mimarisi — referans."""
    def __init__(self, d_model: int, hop_k: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.hop_k = hop_k
        self.log_nu      = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(0.05)))
        self.log_alpha   = nn.Parameter(torch.tensor(0.0))
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
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x   = self.norm1(u)
        nu  = self.log_nu.exp(); dt = self.log_dt.exp()
        alpha = self.log_alpha.exp(); p_sc = self.log_p_scale.exp()
        q   = self.W_q(x); k = self.W_k(x)
        kp  = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
        spd = torch.tanh((q * kp).sum(-1, keepdim=True) / (self._d_k ** 0.5))
        adv = spd * multihop_gradient(x, self.hop_k)
        p   = spectral_pressure(adv, alpha, use_multihop_div=True, hop_k=self.hop_k)
        pg  = p_sc * multihop_gradient(p.unsqueeze(-1), self.hop_k).expand_as(x)
        visc = nu * multihop_laplacian(x, self.hop_k)
        rhs = -adv - pg + visc
        u   = u + dt * rhs
        u   = u + self.mlp(self.norm2(u))
        return u


class FluidLayerFFTAdv(nn.Module):
    """
    FFT Advection katmanı — bu deney.

    Farklar:
    - multihop_gradient → fft_causal_gradient
    - Laplacian da FFT ile (tutarlılık için)
    - log_mag: [seq_len//2+1] öğrenilen spektral ağırlık
    - Basınç: FFT yolu (use_multihop_div=False)
    """
    def __init__(self, d_model: int, seq_len: int, mlp_ratio: int = 4):
        super().__init__()
        n_freq = seq_len // 2 + 1
        self.log_nu      = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(0.05)))
        self.log_alpha   = nn.Parameter(torch.tensor(0.0))
        self.log_p_scale = nn.Parameter(torch.tensor(0.0))
        # Spektral ağırlıklar: öğrenilen, her frekans bini için
        # init: uniform (tüm frekanslar eşit → smooth global average)
        self.log_mag     = nn.Parameter(torch.zeros(n_freq))
        # Laplacian için ayrı ağırlık
        self.log_mag_lap = nn.Parameter(torch.zeros(n_freq))
        self._d_k = max(d_model // 8, 16)
        self.W_q  = nn.Linear(d_model, self._d_k, bias=False)
        self.W_k  = nn.Linear(d_model, self._d_k, bias=False)
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_model))

    def _fft_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """FFT tabanlı ikinci türev (Laplacian) — causal olmak zorunda değil, simetrik."""
        B, L, D = u.shape
        U = torch.fft.rfft(u.float(), dim=1)
        k = torch.arange(U.shape[1], device=u.device, dtype=torch.float32)
        # Laplacian eigenvalue: 2(cos(2πk/L) - 1)
        lap_eig = 2.0 * (torch.cos(2.0 * math.pi * k / L) - 1.0)
        mag     = self.log_mag_lap.exp()
        filt    = (mag * lap_eig).unsqueeze(0).unsqueeze(-1)
        G = torch.complex(U.real * filt, U.imag * filt)
        return torch.fft.irfft(G, n=L, dim=1).to(u.dtype)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x     = self.norm1(u)
        nu    = self.log_nu.exp(); dt = self.log_dt.exp()
        alpha = self.log_alpha.exp(); p_sc = self.log_p_scale.exp()
        q     = self.W_q(x); k = self.W_k(x)
        kp    = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
        spd   = torch.tanh((q * kp).sum(-1, keepdim=True) / (self._d_k ** 0.5))
        # FFT global causal gradient
        grad  = fft_causal_gradient(x, self.log_mag)
        adv   = spd * grad
        # Basınç: FFT yolu
        p     = spectral_pressure(adv, alpha, use_multihop_div=False)
        pg    = p_sc * fft_causal_gradient(p.unsqueeze(-1).expand_as(x), self.log_mag)
        visc  = nu * self._fft_laplacian(x)
        rhs   = -adv - pg + visc
        u     = u + dt * rhs
        u     = u + self.mlp(self.norm2(u))
        return u

    def spectral_profile(self) -> dict:
        """Öğrenilen spektral ağırlık dağılımı özeti."""
        mag = self.log_mag.exp().detach().cpu()
        return {
            "mean": mag.mean().item(),
            "low_freq_mean":  mag[:len(mag)//4].mean().item(),   # düşük frekans
            "high_freq_mean": mag[3*len(mag)//4:].mean().item(),  # yüksek frekans
            "peak_freq":      int(mag.argmax().item()),
        }

# ─────────────────────────────────────────────────────────────────────────────
# LM Modelleri
# ─────────────────────────────────────────────────────────────────────────────

class NanoFluidLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 seq_len: int, layer_cls, layer_kwargs: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.layers  = nn.ModuleList([
            layer_cls(d_model=d_model, **layer_kwargs) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        u = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        for layer in self.layers: u = layer(u)
        return self.head(self.norm(u))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.h = n_heads; self.dk = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B,T,self.h,self.dk).transpose(1,2)
        k = k.view(B,T,self.h,self.dk).transpose(1,2)
        v = v.view(B,T,self.h,self.dk).transpose(1,2)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            sc = (q @ k.transpose(-2,-1)) / (self.dk**0.5)
            sc = sc.masked_fill(
                ~torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool)), -1e4)
            y = F.softmax(sc, dim=-1) @ v
        return self.proj(y.transpose(1,2).contiguous().view(B,T,C))


class NanoGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 seq_len: int, n_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.layers  = nn.ModuleList([nn.ModuleList([
            nn.LayerNorm(d_model),
            CausalSelfAttention(d_model, n_heads),
            nn.LayerNorm(d_model),
            nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                          nn.Linear(hidden, d_model)),
        ]) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        u = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        for n1, attn, n2, mlp in self.layers:
            u = u + attn(n1(u))
            u = u + mlp(n2(u))
        return self.head(self.norm(u))

# ─────────────────────────────────────────────────────────────────────────────
# Eğitim
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, stream, steps: int) -> float:
    model.eval(); total = 0.0
    for _ in range(steps):
        x, y = stream.get_batch(args.batch_size, args.seq_len)
        with AUTOCAST:
            logits = model(x)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return math.exp(total / steps)


def train(name: str, model: nn.Module, token_budget: int) -> dict:
    print(f"\n{'─'*55}")
    print(f"  MODEL: {name}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametreler: {n_params/1e6:.2f}M")
    print(f"{'─'*55}")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda" and not USE_BF16))

    total_steps = int(token_budget) // (args.batch_size * args.seq_len)
    warmup      = max(50, total_steps // 100)

    train_stream = TokenStream("train")
    val_stream   = TokenStream("val")

    best_ppl = float("inf")
    history  = []
    t0       = time.time()

    for step in range(total_steps):
        # LR schedule (cosine)
        if step < warmup:
            lr = args.lr * step / max(1, warmup)
        else:
            decay = (step - warmup) / max(1, total_steps - warmup)
            lr    = 3e-5 + 0.5 * (1 + math.cos(math.pi * decay)) * (args.lr - 3e-5)
        for pg in optimizer.param_groups: pg["lr"] = lr

        x, y = train_stream.get_batch(args.batch_size, args.seq_len)
        with AUTOCAST:
            logits = model(x)
            vocab  = logits.size(-1)
            loss   = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            tput    = (step + 1) * args.batch_size * args.seq_len / max(1, elapsed)
            print(f"  {name}  step={step:5d}/{total_steps}"
                  f"  loss={loss.item():.3f}"
                  f"  lr={lr:.2e}"
                  f"  tput={tput/1e3:.0f}K tok/s", flush=True)

        if step % args.eval_every == 0 or step == total_steps - 1:
            val_ppl = evaluate(model, val_stream, args.eval_steps)
            if val_ppl < best_ppl: best_ppl = val_ppl
            history.append({"step": step, "val_ppl": val_ppl})
            print(f"  *** {name}  EVAL step={step}  val_ppl={val_ppl:.4f}"
                  f"  best={best_ppl:.4f}  t={elapsed/60:.0f}m ***", flush=True)

    # Spektral profil (sadece FFT-adv)
    spectral_info = {}
    if hasattr(model, "layers") and isinstance(model.layers[0], FluidLayerFFTAdv):
        spectral_info = {
            f"layer_{i}": model.layers[i].spectral_profile()
            for i in range(len(model.layers))
        }
        print(f"\n  [Spektral Profil — {name}]")
        for li, prof in spectral_info.items():
            print(f"    {li}: peak_freq={prof['peak_freq']:3d}  "
                  f"low={prof['low_freq_mean']:.3f}  "
                  f"high={prof['high_freq_mean']:.3f}")

    # Fiziksel parametreler
    phys = {}
    if hasattr(model, "layers") and hasattr(model.layers[0], "log_nu"):
        phys = {
            "nu":    [l.log_nu.exp().item()    for l in model.layers],
            "alpha": [l.log_alpha.exp().item() for l in model.layers],
        }
        print(f"\n  [Fiziksel — {name}]  "
              f"nu_mean={sum(phys['nu'])/len(phys['nu']):.4f}  "
              f"alpha_mean={sum(phys['alpha'])/len(phys['alpha']):.4f}")

    return {
        "name":         name,
        "n_params":     n_params,
        "best_val_ppl": best_ppl,
        "history":      history,
        "spectral":     spectral_info,
        "physical":     phys,
        "token_budget": int(token_budget),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Ana Akış
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
    out_dir    = args.out_dir or os.path.join(REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    vocab_size   = VOCAB if VOCAB is not None else 256
    token_budget = int(args.tokens)

    D  = args.d_model
    L  = args.n_layers
    T  = args.seq_len

    print(f"\n{'='*55}")
    print(f"  EXP 20 — FFT Advection Pilot")
    print(f"  Nano: d={D}  L={L}  seq={T}  tokens={token_budget/1e6:.0f}M")
    print(f"  Karşılaştırma: multihop-k4  vs  fft-adv  vs  gpt-nano")
    print(f"{'='*55}")

    # RF bilgisi
    print(f"\n  Receptive Field:")
    print(f"    multihop k=4  : {4 * L:4d} token (k × n_layers)")
    print(f"    fft_adv       : {T:4d} token (full sequence = L)")
    print(f"    GPT-nano      : {T:4d} token (full attention)")

    # FLOP tahmini
    n_freq = T // 2 + 1
    mh_flop  = L * T * len(_hop_scales(4)) * D * 3   # yaklaşık
    fft_flop = L * T * int(math.log2(T)) * D * 2     # O(L log L D)
    mha_flop = L * T * T * D                          # O(L T^2 D) yaklaşık
    print(f"\n  FLOP/batch (yaklaşık, routing only):")
    print(f"    multihop k=4  : {mh_flop/1e6:.1f} MFLOP")
    print(f"    fft_adv       : {fft_flop/1e6:.1f} MFLOP")
    print(f"    GPT-nano MHA  : {mha_flop/1e6:.1f} MFLOP")

    results = []

    # Model A: FluidLM-multihop k=4 (referans)
    torch.manual_seed(args.seed)
    model_mh = NanoFluidLM(vocab_size, D, L, T,
                           layer_cls=FluidLayerMultiHop,
                           layer_kwargs={"hop_k": 4})
    results.append(train("fluid-multihop-k4", model_mh, token_budget))

    # Model B: FluidLM-FFT-advection (bu deney)
    torch.manual_seed(args.seed)
    model_fft = NanoFluidLM(vocab_size, D, L, T,
                            layer_cls=FluidLayerFFTAdv,
                            layer_kwargs={"seq_len": T})
    results.append(train("fluid-fft-adv", model_fft, token_budget))

    # Model C: GPT-Nano (MHA referansı)
    torch.manual_seed(args.seed)
    model_gpt = NanoGPT(vocab_size, D, L, T, n_heads=4)
    results.append(train("gpt-nano", model_gpt, token_budget))

    # Sonuç Tablosu
    print(f"\n{'='*55}")
    print(f"  EXP 20 SONUÇLARI — FFT Advection vs Multi-Hop vs MHA")
    print(f"{'='*55}")
    print(f"  {'Model':<22} {'Params':>8}  {'Best PPL':>10}  {'RF':>5}")
    print(f"  {'-'*50}")
    rf_map = {
        "fluid-multihop-k4": f"{4*L}",
        "fluid-fft-adv":     f"{T} (full)",
        "gpt-nano":          f"{T} (full)",
    }
    for r in results:
        print(f"  {r['name']:<22} "
              f"{r['n_params']/1e6:>7.2f}M  "
              f"{r['best_val_ppl']:>10.4f}  "
              f"{rf_map.get(r['name'], '?'):>10}")

    print(f"\n  Yorum:")
    ppl_mh  = results[0]["best_val_ppl"]
    ppl_fft = results[1]["best_val_ppl"]
    ppl_gpt = results[2]["best_val_ppl"]
    delta = ppl_mh - ppl_fft
    if delta > 0.5:
        print(f"  ✅ FFT-adv, multihop-k4'ten {delta:.2f} PPL daha iyi.")
        print(f"     RF = full-L, FLOP artmadan global bilgi taşıma çalıştı.")
    elif delta < -0.5:
        print(f"  ⚠️  FFT-adv, multihop-k4'ten {abs(delta):.2f} PPL daha kötü.")
        print(f"     Causal constraint veya spektral öğrenme yeterli olmadı.")
    else:
        print(f"  ≈  FFT-adv ≈ multihop-k4 (Δ={delta:+.2f}). Nötr sonuç.")
        print(f"     Fark S-ölçekte veya daha uzun eğitimde belirginleşebilir.")
    print(f"  GPT-nano: {ppl_gpt:.4f}  (full attention üst sınır)")

    # JSON kayıt
    out_path = os.path.join(out_dir, "20_fft_advection_pilot.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[JSON] Kaydedildi: {out_path}")
