# ============================================================
# FluidLM — Colab A100 Egitimi
# experiments/07_colab_a100.py
#
# Colab'da calistirmak icin:
#   1. Bu dosyayi Colab'a yukle (ya da Drive'a koy)
#   2. Runtime > Change runtime type > A100 GPU sec
#   3. Asagidaki hucreyi calistir:
#      !git clone https://github.com/KULLANICI/neo_lang.git
#      %cd neo_lang
#      !python experiments/07_colab_a100.py
#
# Ya da bu dosyayi tek basina calistir — asagida
# gerekli tum siniflar tamamen yeniden tanimlidir,
# src/ klasorune bagimliligi yoktur.
# ============================================================

import sys, os
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Colab Drive mount (opsiyonel — checkpoint kalici kayit icin)
SAVE_TO_DRIVE = False  # True yapinca Drive'a kaydeder
if SAVE_TO_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUTPUT_DIR = "/content/drive/MyDrive/fluidlm_a100"
    except ImportError:
        OUTPUT_DIR = "."
else:
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."

os.makedirs(OUTPUT_DIR, exist_ok=True)

import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ── Cihaz ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}  ({gpu_mem:.1f} GB VRAM)")
else:
    DEVICE = torch.device("cpu")
    print("UYARI: GPU bulunamadi, CPU kullaniliyor (cok yavas)")

USE_BF16 = (DEVICE.type == "cuda" and
            torch.cuda.is_bf16_supported())
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
print(f"Dtype: {DTYPE}  |  torch: {torch.__version__}")

# ── Causal fluid operatorleri (tamamen oz-bagimsiz) ───────────────────────────

def causal_gradient(u: torch.Tensor) -> torch.Tensor:
    """Backward difference: du/dx[i] = u[i] - u[i-1],  u[-1]=0 (causal)."""
    padded = F.pad(u, (0, 0, 1, 0))   # [B, L+1, D]
    return u - padded[:, :-1, :]

def causal_laplacian(u: torch.Tensor) -> torch.Tensor:
    """Backward 2nd diff: d2u/dx2[i] = u[i] - 2*u[i-1] + u[i-2]  (causal)."""
    padded = F.pad(u, (0, 0, 2, 0))   # [B, L+2, D]
    return u - 2 * padded[:, 1:-1, :] + padded[:, :-2, :]

def causal_divergence(u: torch.Tensor) -> torch.Tensor:
    """Mean over D of causal gradient.  -> [B, L]"""
    return causal_gradient(u).mean(dim=-1)

def causal_pressure(adv: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Causal pressure field via spectral (FFT) Poisson solver.

    Çözülen denklem (Helmholtz-Poisson):
        (∇² − α²) p = −div(adv)

    Fourier alanında:
        p̂_k = f̂_k / λ_k    λ_k = 2(cos(2πk/L)−1) − α²

    λ_k ≤ −α² < 0 olduğu için payda sıfıra gitmez → kararlı.

    [NOT — Nedensellik meselesi]
    FFT dairesel (periodic) konvolüsyon kullanır: gelecek tokenlar
    basınç üzerinden sıfır ağırlıkla da olsa etkiler. Bu, dil modeli
    için teknik olarak nedensel ihlaldir. Pratikte iki yaklaşım var:

    1. cumsum yaklaşımı (önceki versiyon):
       p[i] = −cumsum(div)[i] × α  → kesinlikle causal, ama 1. derece
       Green fonksiyonu (∂p/∂x = −div), Poisson değil.

    2. FFT Poisson (bu versiyon):
       Gerçek ∇²p = −div, global etkileşim, ama periodicity nedeniyle
       yarı-causal (sağ sınır soldan ~exp(−α·L) kadar etkileniyor).

    α büyükse (> 2) sağdan gelen etki ihmal edilebilir düzeye iner.
    Bu bir tasarım seçimi — α öğrenilen bir parametredir.
    """
    div   = causal_divergence(adv)                                # [B, L]
    B, L  = div.shape

    # FFT Poisson
    rhs_fft  = torch.fft.rfft(div.float(), dim=1)                 # [B, L//2+1]
    k        = torch.arange(L // 2 + 1, dtype=torch.float32,
                            device=div.device)
    alpha_f  = alpha.float() if isinstance(alpha, torch.Tensor) else float(alpha)
    lambda_k = 2.0 * (torch.cos(2.0 * math.pi * k / L) - 1.0) - alpha_f ** 2
    lambda_k = lambda_k.masked_fill(lambda_k.abs() < 1e-8, -1e-8) # payda güvencesi

    p_fft    = -rhs_fft / lambda_k.unsqueeze(0)                   # [B, L//2+1]
    p        = torch.fft.irfft(p_fft, n=L, dim=1)                 # [B, L]
    p        = p.to(adv.dtype)

    # Gradyan öldürmeyen normalleştirme
    p = p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5) + 1e-6)
    return p

# ── Causal FluidLayer  (NS adimi + MLP sublayer) ──────────────────────────────

class CausalFluidLayer(nn.Module):
    """
    Tek NS zaman adimi  +  MLP sublayer.

    Parametreler:
      4 adet skaler  (log_nu, log_dt, log_alpha, log_p_scale)
      MLP: d -> 4d -> d  (~8*d^2 param)
      2 adet LayerNorm

    d=1024 icin: ~8.4M param/katman
    """

    def __init__(self, d_model: int, nu=0.01, dt=0.05, alpha=1.0,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.log_nu      = nn.Parameter(torch.tensor(math.log(nu)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(dt)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_p_scale = nn.Parameter(torch.tensor(math.log(0.1)))

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
        """NS sag tarafi: -adv - p_grad + visc  (tamamen causal)."""
        # Advection
        speed  = torch.tanh(u.norm(dim=-1, keepdim=True))    # [B,L,1]
        adv    = speed * causal_gradient(u)                   # [B,L,D]
        # Causal pressure
        p      = causal_pressure(adv, self.alpha)             # [B,L]
        p_grad = self.p_scale * causal_gradient(
                     p.unsqueeze(-1)).expand_as(u)            # [B,L,D]
        # Viscosity
        visc   = self.nu * causal_laplacian(u)               # [B,L,D]
        return -adv - p_grad + visc

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # 1. NS adimi (reziduel baglanti)
        u = u + self.dt * self._rhs(self.norm1(u))
        # 2. MLP sublayer (kapasite artirici, transformer FFN analogu)
        u = u + self.mlp(self.norm2(u))
        return u


# ── CausalFluidLM ─────────────────────────────────────────────────────────────

class CausalFluidLM(nn.Module):
    """
    Tamamen causal FluidLM.
    Embedding -> N x CausalFluidLayer -> LayerNorm -> LM Head
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 1024,
        n_layers:    int   = 16,
        max_seq_len: int   = 516,
        nu:          float = 0.01,
        dt:          float = 0.05,
        alpha:       float = 1.0,
        mlp_ratio:   int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # Her katman icin hafif pertuerbasyon — simetri kirma, parametre cesitliligi
        self.layers = nn.ModuleList([
            CausalFluidLayer(d_model,
                             nu=nu    * (1.0 + 0.05 * i),
                             dt=dt    * (1.0 + 0.02 * i),
                             alpha=alpha,
                             mlp_ratio=mlp_ratio, dropout=dropout)
            for i in range(n_layers)
        ])

        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor,
                 use_checkpoint: bool = False) -> torch.Tensor:
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)
        u    = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            if use_checkpoint and self.training:
                # Gradient checkpointing: aktivasyonlari saklamaz,
                # backward sirasinda yeniden hesaplar. Bellek ~1/n_layers azalir.
                u = torch.utils.checkpoint.checkpoint(
                    layer, u, use_reentrant=False)
            else:
                u = layer(u)
        return self.lm_head(self.norm(u))   # [B, L, vocab_size]

    def physical_params(self):
        return {
            "nu":      [l.nu.item()      for l in self.layers],
            "dt":      [l.dt.item()      for l in self.layers],
            "alpha":   [l.alpha.item()   for l in self.layers],
            "p_scale": [l.p_scale.item() for l in self.layers],
        }


# ── Veri ──────────────────────────────────────────────────────────────────────

# Colab'da veri yolu: Drive'dan ya da wget ile indir
DATA_PATH = None
for candidate in [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else ".", "data", "shakespeare.txt"),
    "/content/neo_lang/data/shakespeare.txt",
    "/content/shakespeare.txt",
    "shakespeare.txt",
]:
    if os.path.exists(candidate):
        DATA_PATH = candidate
        break

if DATA_PATH is None:
    print("Shakespeare verisi indiriliyor...")
    import urllib.request
    DATA_PATH = os.path.join(OUTPUT_DIR, "shakespeare.txt")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_PATH
    )
    print(f"Indirildi: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    TEXT = f.read()

chars      = sorted(set(TEXT))
vocab      = ["<pad>", "<bos>", "<eos>"] + chars
stoi       = {c: i for i, c in enumerate(vocab)}
itos       = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(vocab)
PAD_ID, BOS_ID, EOS_ID = stoi["<pad>"], stoi["<bos>"], stoi["<eos>"]

print("=" * 70)
print("CausalFluidLM — A100 Egitimi")
print("=" * 70)
print(f"Metin   : {len(TEXT):,} karakter")
print(f"Vocab   : {VOCAB_SIZE}")

def encode(text):
    return [BOS_ID] + [stoi[c] for c in text if c in stoi] + [EOS_ID]

def decode(ids):
    return "".join(itos.get(i, "?") for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID))

# ── Hiperparametreler ─────────────────────────────────────────────────────────

# Bellek tahmini (bf16, gradient checkpointing ACIK):
#   Aktivasyon/batch = B * seq * d * 2 bytes * sabit (~4 tensor/katman)
#   d=1024, seq=512, B=32: 32*512*1024*2*4 = 134 MB  (cok uygun)
#   Parametreler: ~100M * 2 bytes = 200 MB
#   Optimizer (AdamW): 2x param = 400 MB
#   Toplam ~A100 80GB'nin ~1 GB'si
#
# A100 80GB: batch=32, GRAD_ACCUM=8  -> effective_batch = 256
# A100 40GB: batch=16, GRAD_ACCUM=8  -> effective_batch = 128
# T4   16GB: batch=8,  GRAD_ACCUM=8  -> effective_batch = 64

D_MODEL    = 1024
N_LAYERS   = 16
SEQ_LEN    = 512
MLP_RATIO  = 4
DROPOUT    = 0.2   # 0.1 → 0.2: epoch 29'da overfitting basliyor, daha fazla regularize
USE_CKPT   = True   # gradient checkpointing — bellek ~%60 azalir

if DEVICE.type == "cuda":
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 70:   BATCH_SIZE, GRAD_ACCUM = 32, 8   # A100 80GB
    elif vram_gb >= 35: BATCH_SIZE, GRAD_ACCUM = 16, 8   # A100 40GB
    elif vram_gb >= 14: BATCH_SIZE, GRAD_ACCUM = 8,  8   # T4 16GB
    else:               BATCH_SIZE, GRAD_ACCUM = 4,  4
else:
    BATCH_SIZE, GRAD_ACCUM = 4, 2

EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM

EPOCHS    = 200
LR        = 3e-4
CLIP_NORM = 1.0
PATIENCE  = 25
WARMUP_EP = 10
CKPT_PATH = os.path.join(OUTPUT_DIR, "07_best_model.pt")

# ── Veri bolme ────────────────────────────────────────────────────────────────

all_ids = encode(TEXT)
split   = int(len(all_ids) * 0.9)
train_ids, val_ids = all_ids[:split], all_ids[split:]
print(f"Train   : {len(train_ids):,} token")
print(f"Val     : {len(val_ids):,} token")

def make_batches(ids, seq_len, batch_size, shuffle=True):
    chunks = []
    for i in range(0, len(ids) - seq_len - 1, seq_len // 2):
        chunk = ids[i : i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            chunks.append(chunk)
    if shuffle:
        np.random.shuffle(chunks)
    batches = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        t = torch.tensor(chunks[i : i + batch_size], dtype=torch.long)
        batches.append((t[:, :-1].to(DEVICE), t[:, 1:].to(DEVICE)))
    return batches

# ── Model ─────────────────────────────────────────────────────────────────────

model = CausalFluidLM(
    vocab_size  = VOCAB_SIZE,
    d_model     = D_MODEL,
    n_layers    = N_LAYERS,
    max_seq_len = SEQ_LEN + 4,
    nu          = 0.01,
    dt          = 0.05,
    alpha       = 1.0,
    mlp_ratio   = MLP_RATIO,
    dropout     = DROPOUT,
).to(DEVICE)

if USE_BF16:
    model = model.to(torch.bfloat16)

n_params = model.num_parameters()
print(f"\nModel   : d={D_MODEL}, L={N_LAYERS}, seq={SEQ_LEN}, "
      f"mlp_ratio={MLP_RATIO}")
print(f"Param   : {n_params:,}  ({n_params/1e6:.1f}M)")
print(f"Dtype   : {DTYPE}")
print(f"Batch   : {BATCH_SIZE} x {GRAD_ACCUM} accum = {EFFECTIVE_BATCH} effective")
print(f"Grad ckpt: {USE_CKPT}  (bellek tasarrufu: ~%60)")

# ── Optimizer & scheduler (warmup + cosine) ───────────────────────────────────
# Fiziksel parametreler (nu, dt, alpha, p_scale) icin 10x buyuk LR:
# Bu parametreler cok az sayida skaler oldugu icin MLP agirliklarinin
# golgesinde kaliyor; daha buyuk LR ile signal alabilsinler.
phys_ids = set()
for layer in model.layers:
    for name in ['log_nu', 'log_dt', 'log_alpha', 'log_p_scale']:
        phys_ids.add(id(getattr(layer, name)))

phys_params = [p for p in model.parameters() if id(p) in phys_ids]
rest_params  = [p for p in model.parameters() if id(p) not in phys_ids]

optimizer = torch.optim.AdamW(
    [
        {"params": rest_params,  "lr": LR,      "weight_decay": 1e-2},
        {"params": phys_params,  "lr": LR * 10, "weight_decay": 0.0},
    ],
    betas=(0.9, 0.95)
)

print(f"Optimizer: {len(rest_params)} genel param grubu, "
      f"{len(phys_params)} fizik param grubu (LR x10)")

def lr_lambda(ep):
    if ep < WARMUP_EP:
        return (ep + 1) / WARMUP_EP
    t = (ep - WARMUP_EP) / max(EPOCHS - WARMUP_EP, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))  # cosine, min=10% LR

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# AMP (Automatic Mixed Precision) — A100'de bf16 kullan
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda" and not USE_BF16))

# ── Egitim dongusu ────────────────────────────────────────────────────────────

train_losses, val_losses, val_ppls = [], [], []

print(f"\nEgitim: {EPOCHS} epoch, LR={LR}, patience={PATIENCE}, "
      f"warmup={WARMUP_EP}, eff_batch={EFFECTIVE_BATCH}")
print("-" * 70)

best_val   = float("inf")
best_epoch = 0
no_improve = 0
t0_total   = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_batches = make_batches(train_ids, SEQ_LEN, BATCH_SIZE, shuffle=True)
    epoch_loss = 0.0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(train_batches):
        is_last = (step == len(train_batches) - 1)
        accum_step = (step + 1) % GRAD_ACCUM == 0 or is_last

        with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda"), dtype=DTYPE):
            logits = model(x, use_checkpoint=USE_CKPT)
            loss   = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                     y.reshape(-1), ignore_index=PAD_ID)
            loss   = loss / GRAD_ACCUM  # normalize

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * GRAD_ACCUM  # un-normalize for logging

        if accum_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    scheduler.step()
    avg_train = epoch_loss / max(len(train_batches), 1)
    train_losses.append(avg_train)

    # Val
    model.eval()
    val_batches = make_batches(val_ids, SEQ_LEN, BATCH_SIZE, shuffle=False)
    v_loss = 0.0
    with torch.no_grad(), torch.amp.autocast(
            'cuda', enabled=(DEVICE.type == "cuda"), dtype=DTYPE):
        for x, y in val_batches:
            logits = model(x, use_checkpoint=False)
            v_loss += F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                      y.reshape(-1), ignore_index=PAD_ID).item()
    avg_val = v_loss / max(len(val_batches), 1)
    val_losses.append(avg_val)
    ppl = math.exp(min(avg_val, 20))
    val_ppls.append(ppl)

    # Checkpoint
    if avg_val < best_val:
        best_val   = avg_val
        best_epoch = epoch
        no_improve = 0
        torch.save({
            "epoch":       epoch,
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "val_loss":    avg_val,
            "stoi":        stoi,
            "itos":        itos,
            "vocab_size":  VOCAB_SIZE,
            "d_model":     D_MODEL,
            "n_layers":    N_LAYERS,
            "max_seq_len": SEQ_LEN + 4,
            "mlp_ratio":   MLP_RATIO,
        }, CKPT_PATH)
        marker = " << BEST"
    else:
        no_improve += 1
        marker = ""

    if epoch % 5 == 0 or epoch == 1:
        elapsed   = time.time() - t0_total
        per_epoch = elapsed / epoch
        remaining = per_epoch * (EPOCHS - epoch)
        cur_lr    = scheduler.get_last_lr()[0]
        print(f"Ep {epoch:>3}/{EPOCHS}  train={avg_train:.4f}  "
              f"val={avg_val:.4f}  ppl={ppl:.1f}  "
              f"best={math.exp(min(best_val,20)):.1f}@{best_epoch}  "
              f"pat={no_improve}/{PATIENCE}  lr={cur_lr:.2e}  "
              f"~{remaining/60:.0f}dk{marker}")

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping — {PATIENCE} epoch iyilesme yok.")
        break

print("-" * 70)
total_min = (time.time() - t0_total) / 60
print(f"Egitim bitti. best_ppl={math.exp(min(best_val,20)):.2f} "
      f"(epoch {best_epoch})  Sure: {total_min:.1f} dk")
print(f"Checkpoint: {CKPT_PATH}")

# ── En iyi checkpoint yukle ───────────────────────────────────────────────────

print(f"\nEn iyi model yukleniyor (epoch {best_epoch})...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()
if USE_BF16:
    model = model.to(torch.bfloat16)

# ── Metin uretimi ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(prompt_text: str, max_new: int = 400,
             temperature: float = 0.8, top_k: int = 10,
             rep_penalty: float = 1.3) -> str:
    """
    rep_penalty: >1 → daha once uretilen tokenlarin logitini cez.
    1.0 = devre disi.
    """
    prompt_ids = encode(prompt_text)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    ids = ids[:, -(SEQ_LEN + 4):]
    generated: list[int] = []

    for _ in range(max_new):
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda"), dtype=DTYPE):
            logits = model(ids, use_checkpoint=False)
        next_l = logits[0, -1, :].float() / max(temperature, 1e-6)

        # Tekrar cezasi: daha once gorulen tokenlarin skorunu dusur
        if rep_penalty != 1.0 and generated:
            seen = set(generated[-64:])  # son 64 tokendan
            for tok in seen:
                if next_l[tok] > 0:
                    next_l[tok] = next_l[tok] / rep_penalty
                else:
                    next_l[tok] = next_l[tok] * rep_penalty

        if top_k > 0:
            topk_v, _ = torch.topk(next_l, min(top_k, next_l.size(-1)))
            next_l = next_l.masked_fill(next_l < topk_v[-1], -1e9)
        probs = torch.softmax(next_l, dim=-1)
        nid   = torch.multinomial(probs, 1).unsqueeze(0)
        tok   = nid.item()
        generated.append(tok)
        ids   = torch.cat([ids, nid], dim=1)[:, -(SEQ_LEN + 4):]
        if tok == EOS_ID:
            break

    # ids[0] = [prompt_tokens..., generated_tokens...]
    # prompt zaten icinde, tekrar eklemiyoruz
    return decode(ids[0].tolist())

prompts_cfg = [
    ("HAMLET greedy",    "HAMLET:",           0.3,  5),
    ("HAMLET balanced",  "HAMLET:",           0.8, 10),
    ("HAMLET creative",  "HAMLET:",           1.1, 20),
    ("Diyalog",          "OPHELIA:\nTo be,",  0.8, 10),
    ("Act 1",            "ACT I.\nSCENE I.",  0.8, 10),
]

print("\n" + "=" * 70)
print("METIN URETIMI — CausalFluidLM A100")
print("=" * 70)
for label, prompt, temp, topk in prompts_cfg:
    out = generate(prompt, max_new=400, temperature=temp, top_k=topk)
    print(f"\n[{label}]")
    print("-" * 60)
    print(out[:600])

# ── Fiziksel parametreler ─────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("OGRENILEN FIZIKSEL PARAMETRELER")
print("=" * 70)
phys = model.physical_params()
print(f"  {'Katman':<8} {'nu':<10} {'dt':<10} {'alpha':<10} {'p_scale'}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for i, (nu_v, dt_v, al_v, ps_v) in enumerate(
        zip(phys["nu"], phys["dt"], phys["alpha"], phys["p_scale"])):
    print(f"  {i:<8} {nu_v:<10.4f} {dt_v:<10.4f} {al_v:<10.4f} {ps_v:<10.4f}")

# ── Grafik ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"CausalFluidLM A100  d={D_MODEL} L={N_LAYERS} seq={SEQ_LEN}  "
             f"best epoch={best_epoch}  ppl={math.exp(min(best_val,20)):.1f}",
             fontsize=11)

ex = list(range(1, len(train_losses) + 1))
axes[0].plot(ex, train_losses, label="Train", color="steelblue", lw=1.5)
axes[0].plot(ex, val_losses,   label="Val",   color="tomato",    lw=1.5, ls="--")
axes[0].axvline(best_epoch, color="green", ls=":", lw=1.5, label=f"best={best_epoch}")
axes[0].set_title("Kayip"); axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].semilogy(ex, val_ppls, color="purple", lw=1.5)
axes[1].axvline(best_epoch, color="green", ls=":", lw=1.5)
axes[1].set_title("Val PPL (log)"); axes[1].set_xlabel("Epoch")
axes[1].grid(alpha=0.3, which="both")

ax2  = axes[2]
ax2b = ax2.twinx()
ax2.plot(phys["nu"],      marker="o", color="steelblue",  lw=2, label="nu")
ax2.plot(phys["dt"],      marker="s", color="darkorange", lw=2, label="dt")
ax2.plot(phys["p_scale"], marker="D", color="crimson",    lw=2, label="p_scale")
ax2b.plot(phys["alpha"],  marker="^", color="seagreen",   lw=2, ls="--",
          label="alpha (sag)")
ax2.set_title("Parametreler"); ax2.set_xlabel("Katman")
l1, lb1 = ax2.get_legend_handles_labels()
l2, lb2 = ax2b.get_legend_handles_labels()
ax2.legend(l1 + l2, lb1 + lb2, fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
out_img = os.path.join(OUTPUT_DIR, "07_result.png")
plt.savefig(out_img, dpi=120, bbox_inches="tight")
print(f"\nGrafik: {out_img}")
print("Deney tamamlandi.")
