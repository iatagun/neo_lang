# ============================================================
# experiments/21_fft_advection_s_scale.py
#
# RQ8: exp20'de nano-scale'de GPT'yi 12× yenen FFT-adv
#      S ölçeğine (d=768, L=12, seq=1024, 1B token) taşınıyor mu?
#
# Karşılaştırma (head-to-head):
#   A) FluidLM-FFT-adv-S  — 120M param, RF=1024 (full), FLOP O(L log L D)
#   B) GPT-S              — 124M param, RF=1024, baseline=61.95 PPL
#
# Mimari:
#   FluidLayerFFTAdv (exp20'den) → S ölçeğine genişletildi
#   log_mag: [513] per layer (1024//2+1), 6K extra parametre toplam
#
# Kullanım:
#   python experiments/21_fft_advection_s_scale.py            # 1B tok
#   python experiments/21_fft_advection_s_scale.py --quick    # 100M tok (debug)
#   python experiments/21_fft_advection_s_scale.py --model gpt  # sadece GPT-S
#
# Çıktı:
#   results/21_fft_adv_s_scale.json
#   checkpoints/21_fluid_fft_adv_S_best.pt
#   checkpoints/21_gpt_s_best.pt
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

# ─────────────────────────────────────────────────────────────────────────────
# Argümanlar
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--tokens",       type=float, default=1e9)
parser.add_argument("--quick",        action="store_true",
                    help="100M token, hızlı debug")
parser.add_argument("--model",        default="both",
                    choices=["both", "fluid", "gpt"],
                    help="Eğitilecek model")
# Model mimarisi
parser.add_argument("--d_model",      type=int,   default=768)
parser.add_argument("--n_layers",     type=int,   default=12)
parser.add_argument("--n_heads",      type=int,   default=12)
parser.add_argument("--seq_len",      type=int,   default=1024)
parser.add_argument("--mlp_ratio",    type=int,   default=4)
# Eğitim
parser.add_argument("--batch_size",   type=int,   default=32)
parser.add_argument("--grad_accum",   type=int,   default=15)
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--min_lr",       type=float, default=3e-5)
parser.add_argument("--warmup_frac",  type=float, default=0.01)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--grad_clip",    type=float, default=1.0)
parser.add_argument("--seed",         type=int,   default=42)
# Logging
parser.add_argument("--log_every",    type=int,   default=100)
parser.add_argument("--eval_every",   type=int,   default=500)
parser.add_argument("--eval_steps",   type=int,   default=100)
parser.add_argument("--save_every",   type=int,   default=500)
parser.add_argument("--out_dir",      default=None)
parser.add_argument("--ckpt_dir",     default=None)
args = parser.parse_args()

if args.quick:
    args.tokens    = 100e6
    args.log_every = 50
    args.eval_every = 200
    args.save_every = 200

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
      f"batch={args.batch_size}×{args.grad_accum}={args.batch_size*args.grad_accum*args.seq_len/1000:.0f}K eff  "
      f"tokens={args.tokens/1e9:.1f}B")

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer + Veri
# ─────────────────────────────────────────────────────────────────────────────

try:
    import tiktoken
    ENC    = tiktoken.get_encoding("gpt2")
    VOCAB  = ENC.n_vocab          # 50257
    encode = lambda t: ENC.encode_ordinary(t)
    EOS_ID = 50256
    print(f"[Tokenizer] tiktoken gpt2  vocab={VOCAB}")
except ImportError:
    raise SystemExit("tiktoken gerekli: pip install tiktoken")


class TokenStream:
    EOS_ID = 50256

    def __init__(self, split: str):
        self.buffer: List[int] = []
        self._split = split
        self._load(split)

    def _load(self, split: str):
        try:
            from datasets import load_dataset
            ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
            ds = ds.shuffle(seed=42, buffer_size=10_000)
            ds = ds.skip(990_000) if split == "val" else ds.take(990_000)
            self._stream = iter(ds)
            self._source = "openwebtext"
        except Exception as e:
            raise SystemExit(f"OpenWebText yüklenemedi: {e}")

    def _fill(self, n: int):
        while len(self.buffer) < n:
            try:
                self.buffer.extend(encode(next(self._stream)["text"]))
                self.buffer.append(self.EOS_ID)
            except StopIteration:
                # Split'e göre doğru bölümü yeniden yükle — val asla train'e düşmesin
                self._load(self._split)

    def get_batch(self, B: int, T: int):
        need = (T + 1) * B
        self._fill(need)
        chunk = self.buffer[:need]
        self.buffer = self.buffer[need:]
        t = torch.tensor(chunk, dtype=torch.long).view(B, T + 1)
        return t[:, :-1].to(device), t[:, 1:].to(device)

# ─────────────────────────────────────────────────────────────────────────────
# Operatörler
# ─────────────────────────────────────────────────────────────────────────────

def fft_causal_gradient(u: torch.Tensor,
                        log_mag: torch.Tensor) -> torch.Tensor:
    """
    Global causal gradient — frekans uzayında.
    H(k) = 1 − e^{−2πik/L}  (backward difference, causal)
    Ağırlık: exp(log_mag[k])  (öğrenilen spektral öncelik)
    RF = full L, FLOP = O(L log L D)
    log_mag, eğitim seq_len için tanımlı; farklı uzunluklarda interpolate edilir.
    """
    B, L, D = u.shape
    U = torch.fft.rfft(u.float(), dim=1)          # [B, n_freq, D]
    n_freq = U.shape[1]

    k_idx = torch.arange(n_freq, device=u.device, dtype=torch.float32)
    omega  = 2.0 * math.pi * k_idx / L

    H_r = 1.0 - torch.cos(omega)
    H_i = torch.sin(omega)

    # log_mag eğitim uzunluğu için — farklı L'de interpolate et
    if log_mag.shape[0] != n_freq:
        mag = F.interpolate(
            log_mag.view(1, 1, -1), size=n_freq, mode="linear", align_corners=False
        ).view(n_freq).exp()
    else:
        mag = log_mag.exp()

    F_r = (mag * H_r).unsqueeze(0).unsqueeze(-1)   # [1, n_freq, 1]
    F_i = (mag * H_i).unsqueeze(0).unsqueeze(-1)

    Ur, Ui = U.real, U.imag
    Gr = Ur * F_r - Ui * F_i
    Gi = Ur * F_i + Ui * F_r

    grad = torch.fft.irfft(torch.complex(Gr, Gi), n=L, dim=1).to(u.dtype)
    return grad


def spectral_pressure_fft(adv: torch.Tensor,
                          alpha: torch.Tensor) -> torch.Tensor:
    """Helmholtz-Poisson FFT basınç çözümü — FFT-adv yolu."""
    B, L, D = adv.shape
    A = torch.fft.rfft(adv.float(), dim=1)
    k_idx = torch.arange(A.shape[1], device=adv.device, dtype=torch.float32)
    omega  = 2.0 * math.pi * k_idx / L
    H_r = (1.0 - torch.cos(omega)).unsqueeze(0).unsqueeze(-1)
    H_i = torch.sin(omega).unsqueeze(0).unsqueeze(-1)
    Gr = A.real * H_r - A.imag * H_i
    Gi = A.real * H_i + A.imag * H_r
    div_vec = torch.fft.irfft(torch.complex(Gr, Gi), n=L, dim=1).to(adv.dtype)
    div = div_vec.mean(dim=-1)               # [B, L]

    f   = torch.fft.rfft(div.float(), dim=1)
    k_p = torch.arange(L // 2 + 1, dtype=torch.float32, device=div.device)
    a   = alpha.float()
    lam = 2.0 * (torch.cos(2.0 * math.pi * k_p / L) - 1.0) - a ** 2
    lam = lam.masked_fill(lam.abs() < 1e-8, -1e-8)
    p   = torch.fft.irfft(-f / lam.unsqueeze(0), n=L, dim=1).to(adv.dtype)
    return p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5) + 1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# FluidLayerFFTAdv — S ölçeği
# ─────────────────────────────────────────────────────────────────────────────

class FluidLayerFFTAdv(nn.Module):
    """
    FFT Advection katmanı — S ölçeği.
    exp20'de nano-scale'de GPT-nano'yu 12× geride bırakan mimari.

    Parametreler (S ölçeği):
      log_mag     : [513]  — advection spektral ağırlığı
      log_mag_lap : [513]  — viskozite spektral ağırlığı
      log_nu, log_dt, log_alpha, log_p_scale: fiziksel scalar'lar
      W_q, W_k: [768 × 96]  — content-dependent hız (d_k = d//8)
      LayerNorm × 2, MLP(768→3072→768)
    """
    def __init__(self, d_model: int, seq_len: int, mlp_ratio: int = 4):
        super().__init__()
        n_freq = seq_len // 2 + 1
        self.n_freq = n_freq

        self.log_nu  = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt  = nn.Parameter(torch.tensor(math.log(0.05)))

        self.log_mag = nn.Parameter(torch.zeros(n_freq))

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
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02 / math.sqrt(2))

    def _causal_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """Causal 2nd-order backward difference: u[n] - 2*u[n-1] + u[n-2].
        Strictly causal — uses only past tokens."""
        u_p1 = torch.cat([torch.zeros_like(u[:, :1]), u[:, :-1]], dim=1)
        u_p2 = torch.cat([torch.zeros_like(u[:, :2]), u[:, :-2]], dim=1)
        return u - 2 * u_p1 + u_p2

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x  = self.norm1(u)
        nu = self.log_nu.exp()
        dt = self.log_dt.exp()

        q  = self.W_q(x)
        k  = self.W_k(x)
        kp = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
        spd = torch.tanh((q * kp).sum(-1, keepdim=True) / (self._d_k ** 0.5))

        grad = fft_causal_gradient(x, self.log_mag)
        adv  = spd * grad

        visc = nu * self._causal_laplacian(x)
        rhs  = -adv + visc
        u    = u + dt * rhs
        u    = u + self.mlp(self.norm2(u))
        return u

    def spectral_profile(self) -> dict:
        mag = self.log_mag.exp().detach().cpu()
        n   = len(mag)
        return {
            "mean":           mag.mean().item(),
            "low_freq_mean":  mag[:n // 4].mean().item(),
            "high_freq_mean": mag[3 * n // 4:].mean().item(),
            "peak_freq":      int(mag.argmax().item()),
        }

# ─────────────────────────────────────────────────────────────────────────────
# FluidLM-S
# ─────────────────────────────────────────────────────────────────────────────

class FluidLM_S(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 seq_len: int, mlp_ratio: int = 4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop    = nn.Dropout(0.0)
        self.layers  = nn.ModuleList([
            FluidLayerFFTAdv(d_model, seq_len, mlp_ratio)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        u    = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for layer in self.layers:
            u = layer(u)
        return self.head(self.norm(u))

    def spectral_summary(self) -> dict:
        return {
            f"layer_{i}": self.layers[i].spectral_profile()
            for i in range(len(self.layers))
        }

# ─────────────────────────────────────────────────────────────────────────────
# GPT-S referans (exp16 eşdeğeri)
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.h  = n_heads
        self.dk = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.qkv.weight,  std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, C))


class GPT_S(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 seq_len: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop    = nn.Dropout(0.0)
        self.layers  = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(d_model),
                CausalSelfAttention(d_model, n_heads),
                nn.LayerNorm(d_model),
                nn.Sequential(
                    nn.Linear(d_model, hidden), nn.GELU(),
                    nn.Linear(hidden, d_model)),
            ])
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for ln1, attn, ln2, mlp in self.layers:
            nn.init.normal_(mlp[0].weight, std=0.02)
            nn.init.normal_(mlp[2].weight, std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        u = self.drop(self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device)))
        for ln1, attn, ln2, mlp in self.layers:
            u = u + attn(ln1(u))
            u = u + mlp(ln2(u))
        return self.head(self.norm(u))

# ─────────────────────────────────────────────────────────────────────────────
# Eğitim
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, stream, steps: int) -> float:
    model.eval()
    total = 0.0
    for _ in range(steps):
        x, y = stream.get_batch(args.batch_size, args.seq_len)
        with AUTOCAST:
            logits = model(x)
            total += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            ).item()
    model.train()
    return math.exp(total / steps)


def train(name: str, model: nn.Module, token_budget: int,
          out_dir: str, ckpt_dir: str) -> dict:
    sep = "─" * 57
    print(f"\n{sep}")
    print(f"  MODEL: {name}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametreler: {n_params / 1e6:.2f}M")
    eff_batch = args.batch_size * args.grad_accum * args.seq_len
    print(f"  Eff. batch: {eff_batch / 1e3:.0f}K tok/step")
    print(f"{sep}")

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(device.type == "cuda" and not USE_BF16))

    total_steps = int(token_budget) // eff_batch
    warmup      = max(100, int(total_steps * args.warmup_frac))
    print(f"  Toplam adım: {total_steps}  warmup: {warmup}")

    train_stream = TokenStream("train")
    val_stream   = TokenStream("val")

    best_ppl  = float("inf")
    best_ckpt = os.path.join(ckpt_dir, f"21_{name.replace('-','_')}_best.pt")
    history   = []
    t0        = time.time()
    accum_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        # Cosine LR
        if step < warmup:
            lr = args.lr * (step + 1) / max(1, warmup)
        else:
            prog = (step - warmup) / max(1, total_steps - warmup)
            lr   = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
                1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        for micro in range(args.grad_accum):
            x, y = train_stream.get_batch(args.batch_size, args.seq_len)
            with AUTOCAST:
                logits = model(x)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                ) / args.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            tput    = (step + 1) * eff_batch / max(1.0, elapsed)
            tok_B   = (step + 1) * eff_batch / 1e9
            print(
                f"  {name}  step={step:5d}/{total_steps}"
                f"  tok={tok_B:.2f}B"
                f"  loss={accum_loss:.3f}"
                f"  lr={lr:.2e}"
                f"  tput={tput/1e3:.0f}K tok/s"
                f"  t={elapsed/60:.0f}m",
                flush=True
            )
            accum_loss = 0.0

        if step % args.eval_every == 0 or step == total_steps - 1:
            val_ppl = evaluate(model, val_stream, args.eval_steps)
            elapsed = time.time() - t0
            tok_B   = (step + 1) * eff_batch / 1e9
            history.append({"step": step, "tok_B": round(tok_B, 3), "val_ppl": val_ppl})

            improved = ""
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                improved = "  ← BEST"
                torch.save({
                    "step": step, "val_ppl": val_ppl,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "model_type": "fluid_fft",
                    "cfg": {
                        "vocab_size": vocab_size,
                        "d_model": args.d_model,
                        "n_layers": args.n_layers,
                        "n_heads": args.n_heads,
                        "seq_len": args.seq_len,
                        "mlp_ratio": args.mlp_ratio,
                    },
                }, best_ckpt)

            print(
                f"  *** {name}  step={step}/{total_steps}"
                f"  tok={tok_B:.2f}B"
                f"  val_ppl={val_ppl:.4f}"
                f"  best={best_ppl:.4f}"
                f"  t={elapsed/60:.0f}m{improved} ***",
                flush=True
            )

    # Spektral profil
    spectral_info = {}
    if isinstance(model, FluidLM_S):
        spectral_info = model.spectral_summary()
        print(f"\n  [Spektral Profil — {name}]")
        for li, prof in spectral_info.items():
            print(f"    {li}: peak_freq={prof['peak_freq']:4d}  "
                  f"low={prof['low_freq_mean']:.3f}  "
                  f"high={prof['high_freq_mean']:.3f}")

    # Fiziksel parametreler
    phys = {}
    if isinstance(model, FluidLM_S):
        nu_vals    = [l.log_nu.exp().item()    for l in model.layers]
        alpha_vals = [l.log_alpha.exp().item() for l in model.layers]
        dt_vals    = [l.log_dt.exp().item()    for l in model.layers]
        phys = {"nu": nu_vals, "alpha": alpha_vals, "dt": dt_vals}
        print(f"\n  [Fiziksel — {name}]"
              f"  nu_mean={sum(nu_vals)/len(nu_vals):.4f}"
              f"  alpha_mean={sum(alpha_vals)/len(alpha_vals):.4f}"
              f"  dt_mean={sum(dt_vals)/len(dt_vals):.4f}")

    print(f"\n  [{name}] Eğitim bitti. Best PPL: {best_ppl:.4f}  ckpt: {best_ckpt}")

    return {
        "name":         name,
        "n_params":     n_params,
        "best_val_ppl": best_ppl,
        "history":      history,
        "spectral":     spectral_info,
        "physical":     phys,
        "token_budget": int(token_budget),
        "ckpt":         best_ckpt,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Ana Akış
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
    out_dir    = args.out_dir  or os.path.join(REPO_ROOT, "results")
    ckpt_dir   = args.ckpt_dir or os.path.join(REPO_ROOT, "checkpoints")
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    vocab_size   = VOCAB
    token_budget = int(args.tokens)
    D  = args.d_model
    L  = args.n_layers
    T  = args.seq_len
    H  = args.n_heads
    n_freq = T // 2 + 1

    # eff batch ve adım sayısı
    eff_batch   = args.batch_size * args.grad_accum * T
    total_steps = token_budget // eff_batch

    print(f"\n{'='*57}")
    print(f"  EXP 21 — FFT Advection S-Scale")
    print(f"  d={D}  L={L}  seq={T}  heads={H}")
    print(f"  tokens={token_budget/1e9:.1f}B  steps={total_steps}")
    print(f"  GPT-S baseline: 61.95 PPL")
    print(f"{'='*57}")

    print(f"\n  Receptive Field:")
    print(f"    fluid-fft-adv-S : {T:4d} token  (full sequence)")
    print(f"    gpt-S           : {T:4d} token  (full attention)")

    print(f"\n  Spektral log_mag boyutu: {n_freq} / layer  "
          f"(toplam {n_freq * L / 1e3:.1f}K ekstra parametre)")

    results = []
    out_path = os.path.join(out_dir, "21_fft_adv_s_scale.json")

    def save_results():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  [Kaydedildi] {out_path}")

    # Model A: FluidLM-FFT-adv-S
    if args.model in ("both", "fluid"):
        torch.manual_seed(args.seed)
        model_fluid = FluidLM_S(vocab_size, D, L, T, args.mlp_ratio)
        n_fluid = sum(p.numel() for p in model_fluid.parameters())
        print(f"\n  FluidLM-FFT-adv-S: {n_fluid/1e6:.2f}M parametre")
        res_fluid = train("fluid-fft-adv-S", model_fluid, token_budget, out_dir, ckpt_dir)
        results.append(res_fluid)
        save_results()
        del model_fluid
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Model B: GPT-S referans
    if args.model in ("both", "gpt"):
        torch.manual_seed(args.seed)
        model_gpt = GPT_S(vocab_size, D, L, T, H, args.mlp_ratio)
        n_gpt = sum(p.numel() for p in model_gpt.parameters())
        print(f"\n  GPT-S: {n_gpt/1e6:.2f}M parametre")
        res_gpt = train("gpt-S", model_gpt, token_budget, out_dir, ckpt_dir)
        results.append(res_gpt)
        save_results()
        del model_gpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Sonuç tablosu
    if results:
        print(f"\n{'='*57}")
        print(f"  EXP 21 SONUÇLARI — FFT Advection S-Scale vs GPT-S")
        print(f"{'='*57}")
        print(f"  {'Model':<22} {'Params':>8}  {'Best PPL':>10}  {'vs GPT-S':>9}")
        print(f"  {'-'*52}")
        gpt_ppl = next((r["best_val_ppl"] for r in results if r["name"] == "gpt-S"), 61.95)
        for r in results:
            delta = r["best_val_ppl"] - gpt_ppl
            sign  = "+" if delta >= 0 else ""
            print(f"  {r['name']:<22} "
                  f"{r['n_params']/1e6:>7.2f}M  "
                  f"{r['best_val_ppl']:>10.4f}  "
                  f"{sign}{delta:>8.4f}")
        print(f"  (GPT-S exp16 referansı: 61.95 PPL)")
        print(f"{'='*57}")
        save_results()
