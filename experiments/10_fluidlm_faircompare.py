# ============================================================
# FluidLM Adil Karşılaştırma Deneyi
# experiments/10_fluidlm_faircompare.py
#
# Motivasyon:
#   07_colab_a100.py sonuçları: FluidLM 135M → PPL 5.48
#   09_baseline Config-B:       GPT-B    85M → PPL 4.27
#
#   Parametre dökümü (experiments/_param_breakdown.py):
#     ns_physics :      64 param  (0.0%)  ← routing mekanizması
#     mlp        : 134M param (99.5%)  ← asıl kapasite
#
#   Yani 07 modeli MHA yerine NS kullanıyor, ama MLP'si GPT-B'den
#   3× büyük. PPL farkı routing'den mi, MLP kapasitesinden mi?
#
# Bu deney:
#   FluidLM-fair: d=768, L=12, mlp_ratio=4
#     NS routing:  12 katman × 4 skaler = 48 param
#     MLP:         12 × 2 × 768 × 3072 ≈ 56M param
#     Toplam:      ~57M param
#
#   GPT-B (09_baseline Config-B): d=768, L=12, h=12
#     MHA routing: 12 × 4 × 768² ≈ 28M param
#     MLP:         12 × 2 × 768 × 3072 ≈ 56M param
#     Toplam:      ~85M param
#
#   Aynı MLP, farklı routing → adil izolasyon
#
# Kullanım:
#   python experiments/10_fluidlm_faircompare.py
#   python experiments/10_fluidlm_faircompare.py --epochs 200 --patience 20
# ============================================================

import sys, os, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",   type=int,   default=200)
parser.add_argument("--patience", type=int,   default=20)
parser.add_argument("--data",     default=None)
parser.add_argument("--out_dir",  default=None)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
OUT_DIR    = args.out_dir or SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ── Cihaz ─────────────────────────────────────────────────────────────────────
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE    = torch.bfloat16 if USE_BF16 else torch.float32
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9 \
           if device.type == "cuda" else 0
print(f"Device: {device}  dtype={DTYPE}  VRAM={vram_gb:.0f}GB")

# ── Model config ───────────────────────────────────────────────────────────────
# GPT-B ile tam eşleşme: d=768, L=12, seq=512
D_MODEL   = 768
N_LAYERS  = 12
SEQ_LEN   = 512
MLP_RATIO = 4       # hidden = 768 × 4 = 3072
DROPOUT   = 0.2
LR        = 3e-4

# ── Veri ──────────────────────────────────────────────────────────────────────
DATA_PATH = args.data
if DATA_PATH is None:
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    for candidate in [
        os.path.join(REPO_ROOT, "data", "shakespeare.txt"),
        "/content/neo_lang/data/shakespeare.txt",
        os.path.join(SCRIPT_DIR, "shakespeare.txt"),
    ]:
        if os.path.exists(candidate):
            DATA_PATH = candidate
            break

if DATA_PATH is None:
    import urllib.request
    DATA_PATH = os.path.join(OUT_DIR, "shakespeare.txt")
    print("Shakespeare indiriliyor...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt", DATA_PATH)

with open(DATA_PATH, encoding="utf-8") as f:
    TEXT = f.read()

chars      = sorted(set(TEXT))
VOCAB_SIZE = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for i, c in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s]

all_ids   = torch.tensor(encode(TEXT), dtype=torch.long)
split     = int(len(all_ids) * 0.9)
TRAIN_IDS = all_ids[:split]
VAL_IDS   = all_ids[split:]
print(f"Corpus: {len(TEXT):,} char  vocab={VOCAB_SIZE}  "
      f"train={len(TRAIN_IDS):,}  val={len(VAL_IDS):,}")

def get_batch(ids, batch_size):
    ix = torch.randint(len(ids) - SEQ_LEN - 1, (batch_size,))
    x  = torch.stack([ids[i:i+SEQ_LEN]     for i in ix]).to(device)
    y  = torch.stack([ids[i+1:i+SEQ_LEN+1] for i in ix]).to(device)
    return x, y

# ── NS Operatörleri (causal, self-contained) ───────────────────────────────────

def causal_gradient(u):
    padded = F.pad(u, (0, 0, 1, 0))
    return u - padded[:, :-1, :]

def causal_laplacian(u):
    padded = F.pad(u, (0, 0, 2, 0))
    return u - 2 * padded[:, 1:-1, :] + padded[:, :-2, :]

def causal_divergence(u):
    return causal_gradient(u).mean(dim=-1)

def spectral_pressure(adv, alpha):
    """
    FFT Helmholtz-Poisson: (∇² − α²) p = −div(adv)
    p̂_k = −f̂_k / λ_k,   λ_k = 2(cos(2πk/L)−1) − α²
    """
    div  = causal_divergence(adv)
    B, L = div.shape
    f    = torch.fft.rfft(div.float(), dim=1)
    k    = torch.arange(L // 2 + 1, dtype=torch.float32, device=div.device)
    a    = alpha.float() if isinstance(alpha, torch.Tensor) else float(alpha)
    lam  = 2.0 * (torch.cos(2.0 * math.pi * k / L) - 1.0) - a ** 2
    lam  = lam.masked_fill(lam.abs() < 1e-8, -1e-8)
    p    = torch.fft.irfft(-f / lam.unsqueeze(0), n=L, dim=1).to(adv.dtype)
    return p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5) + 1e-6)

# ── FluidLayer ─────────────────────────────────────────────────────────────────

class FluidLayer(nn.Module):
    """
    NS routing (4 skaler) + MLP sublayer.
    d=768, mlp_ratio=4: MLP ≈ 2×768×3072 = 4.72M param/katman
    """
    def __init__(self, d_model, nu=0.01, dt=0.05, alpha=1.0,
                 mlp_ratio=4, dropout=0.1):
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

    def _rhs(self, u):
        speed  = torch.tanh(u.norm(dim=-1, keepdim=True))
        adv    = speed * causal_gradient(u)
        p      = spectral_pressure(adv, self.alpha)
        p_grad = self.p_scale * causal_gradient(p.unsqueeze(-1)).expand_as(u)
        visc   = self.nu * causal_laplacian(u)
        return -adv - p_grad + visc

    def forward(self, u):
        u = u + self.dt * self._rhs(self.norm1(u))
        u = u + self.mlp(self.norm2(u))
        return u

# ── FluidLM-fair ───────────────────────────────────────────────────────────────

class FluidLMFair(nn.Module):
    """
    d=768, L=12 — GPT-B ile aynı MLP boyutu, NS routing.
    Routing: 48 param (12 × 4 skaler)
    MLP:     ~56M param (12 × 2 × 768 × 3072)
    Toplam:  ~57M param
    """
    def __init__(self, vocab_size, d_model=768, n_layers=12,
                 max_seq_len=512, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            FluidLayer(d_model,
                       nu=0.01 * (1 + 0.05 * i),
                       dt=0.05 * (1 + 0.02 * i),
                       mlp_ratio=mlp_ratio,
                       dropout=dropout)
            for i in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying
        self._init()

    def _init(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        ns    = sum(p.numel() for n, p in self.named_parameters()
                    if any(x in n for x in ['log_nu','log_dt','log_alpha','log_p_scale']))
        mlp   = sum(p.numel() for n, p in self.named_parameters() if 'mlp' in n)
        emb   = sum(p.numel() for n, p in self.named_parameters()
                    if 'token_emb' in n or 'pos_emb' in n)
        return total, ns, mlp, emb

    def forward(self, ids, use_ckpt=False):
        import torch.utils.checkpoint as ck
        B, L = ids.shape
        pos  = torch.arange(L, device=ids.device).unsqueeze(0)
        u    = self.emb_drop(self.token_emb(ids) + self.pos_emb(pos))
        for layer in self.layers:
            u = ck.checkpoint(layer, u, use_reentrant=False) if use_ckpt else layer(u)
        return self.lm_head(self.norm(u))

# ── Batch boyutu ───────────────────────────────────────────────────────────────
if   vram_gb >= 70: BATCH_SIZE, GRAD_ACCUM = 32, 8
elif vram_gb >= 40: BATCH_SIZE, GRAD_ACCUM = 16, 8
elif vram_gb >= 10: BATCH_SIZE, GRAD_ACCUM = 8,  8
else:               BATCH_SIZE, GRAD_ACCUM = 4,  4

# ── Model ─────────────────────────────────────────────────────────────────────
model = FluidLMFair(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
    max_seq_len=SEQ_LEN, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
).to(device)
if USE_BF16:
    model = model.to(torch.bfloat16)

total, ns, mlp, emb = model.count_params()
print(f"\nFluidLM-fair parametre dokumu:")
print(f"  NS routing  : {ns:>10,}  ({100*ns/total:.2f}%)  ← 12 katman × 4 skaler")
print(f"  MLP         : {mlp:>10,}  ({100*mlp/total:.1f}%)")
print(f"  Embedding   : {emb:>10,}  ({100*emb/total:.1f}%)")
print(f"  TOPLAM      : {total:>10,}")
print(f"\nGPT-B icin karsilastirma:")
print(f"  MHA routing : ~28,311,552  (28M) — FluidLM'in 590,000× fazlasi")
print(f"  MLP         :  ~56,623,104 (GPT-B ile esit)")
print(f"  TOPLAM GPT-B:       ~85M")

USE_GRAD_CKPT = (vram_gb < 10 and total > 50e6)
print(f"\nBatch={BATCH_SIZE}  GradAccum={GRAD_ACCUM}  EffBatch={BATCH_SIZE*GRAD_ACCUM}")
print(f"GradCheckpoint: {'ACIK' if USE_GRAD_CKPT else 'KAPALI'}")

# ── Optimizer ─────────────────────────────────────────────────────────────────
# NS parametreleri için 10× yüksek LR (küçük skalerler yavaş öğreniyor)
ns_params  = [p for n, p in model.named_parameters()
              if any(x in n for x in ['log_nu','log_dt','log_alpha','log_p_scale'])]
ns_names   = {id(p) for p in ns_params}
base_params = [p for p in model.parameters() if id(p) not in ns_names]

optimizer = torch.optim.AdamW([
    {"params": base_params, "lr": LR},
    {"params": ns_params,   "lr": LR * 10},
], betas=(0.9, 0.95), weight_decay=0.1)

# ── LR schedule ───────────────────────────────────────────────────────────────
WARMUP = 500
total_steps = 0
def get_lr_scale(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    prog = (step - WARMUP) / max(1, 10_000 - WARMUP)
    return max(0.1, 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

# ── Eğitim ────────────────────────────────────────────────────────────────────
CKPT_PATH  = os.path.join(OUT_DIR, "10_fluidlm_fair_best.pt")
STEPS_PER  = max(1, len(TRAIN_IDS) // (SEQ_LEN * BATCH_SIZE * GRAD_ACCUM))
VAL_STEPS  = min(100, len(VAL_IDS)  // (SEQ_LEN * BATCH_SIZE))
LOG_EVERY  = max(10, STEPS_PER // 10)

print(f"\nSteps/epoch={STEPS_PER}  epochs={args.epochs}  patience={args.patience}")
print("=" * 60)

best_val   = float("inf")
no_improve = 0
t0         = time.time()

for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    for step_i in range(STEPS_PER):
        for _ in range(GRAD_ACCUM):
            x, y = get_batch(TRAIN_IDS, BATCH_SIZE)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=DTYPE):
                logits = model(x, use_ckpt=USE_GRAD_CKPT)
                loss   = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)
                ) / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

        scale = get_lr_scale(total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] if "initial_lr" in pg else LR
        optimizer.param_groups[0]["lr"] = LR * scale
        optimizer.param_groups[1]["lr"] = LR * 10 * scale

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_steps += 1

        if step_i % LOG_EVERY == 0:
            elapsed = time.time() - t0
            sps     = (step_i + 1) / max(elapsed / epoch, 1e-9)
            remain  = (STEPS_PER - step_i - 1) / max(sps, 1e-9)
            print(f"  Ep {epoch:>3}/{args.epochs}  "
                  f"step {step_i+1:>4}/{STEPS_PER}  "
                  f"loss={epoch_loss/((step_i+1)*GRAD_ACCUM):.4f}  "
                  f"ETA {remain/60:.1f}m", flush=True)

    avg_train = epoch_loss / (STEPS_PER * GRAD_ACCUM)

    # Validation
    model.eval()
    v_loss = 0.0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=DTYPE):
        for _ in range(VAL_STEPS):
            xv, yv = get_batch(VAL_IDS, BATCH_SIZE)
            v_loss += F.cross_entropy(
                model(xv, use_ckpt=False).reshape(-1, VOCAB_SIZE), yv.reshape(-1)
            ).item()
    avg_val = v_loss / VAL_STEPS
    ppl     = math.exp(min(avg_val, 20))

    marker = ""
    if avg_val < best_val:
        best_val   = avg_val
        no_improve = 0
        marker     = "  << BEST"
        torch.save({
            "epoch": epoch, "val_loss": avg_val,
            "model_state": model.state_dict(),
            "d_model": D_MODEL, "n_layers": N_LAYERS,
            "vocab_size": VOCAB_SIZE,
        }, CKPT_PATH)
    else:
        no_improve += 1

    elapsed_m = (time.time() - t0) / 60
    print(f"  Ep {epoch:>3}/{args.epochs}  "
          f"train={avg_train:.4f}  val={avg_val:.4f}  "
          f"ppl={ppl:.2f}  {elapsed_m:.1f}m{marker}", flush=True)

    if no_improve >= args.patience:
        print(f"\nEarly stopping — {args.patience} epoch iyilesme yok.")
        break

best_ppl = math.exp(min(best_val, 20))
print(f"\nFluidLM-fair tamamlandi: best_val={best_val:.4f}  PPL={best_ppl:.2f}")

# ── Karşılaştırma Tablosu ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  ROUTING IZOLASYON DENEYI — Adil Karsilastirma")
print(f"{'='*65}")
print(f"  {'Model':<32} {'Routing':>10}  {'MLP':>8}  {'Toplam':>8}  {'PPL':>6}")
print("  " + "-"*61)

# GPT-B checkpoint
gpt_b_ckpt = None
for p in [
    os.path.join(OUT_DIR, "09_gpt_B_best.pt"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "checkpoints", "09_gpt_B_best.pt"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "experiments", "09_gpt_B_best.pt"),
    "/content/neo_lang/experiments/09_gpt_B_best.pt",
]:
    if os.path.exists(p):
        gpt_b_ckpt = p
        break

if gpt_b_ckpt:
    try:
        ck = torch.load(gpt_b_ckpt, map_location="cpu", weights_only=True)
        gpt_b_ppl = math.exp(min(ck["val_loss"], 20))
        print(f"  {'GPT-B  (d=768 L=12 h=12)':<32} {'~28M':>10}  {'~56M':>8}  {'~85M':>8}  {gpt_b_ppl:>6.2f}")
    except Exception as e:
        print(f"  GPT-B checkpoint hatasi: {e}")
else:
    print(f"  {'GPT-B  (d=768 L=12 h=12)':<32} {'~28M':>10}  {'~56M':>8}  {'~85M':>8}  {'?':>6}")

print(f"  {'FluidLM-fair (d=768 L=12 NS)':<32} {'48':>10}  {'~56M':>8}  {'~57M':>8}  {best_ppl:>6.2f}")

# Orijinal FluidLM
orig_ckpt = None
for p in [
    os.path.join(os.path.dirname(SCRIPT_DIR), "checkpoints", "07_best_model.pt"),
    os.path.join(OUT_DIR, "07_best_model.pt"),
    "/content/neo_lang/checkpoints/07_best_model.pt",
]:
    if os.path.exists(p):
        orig_ckpt = p
        break
if orig_ckpt:
    try:
        ck = torch.load(orig_ckpt, map_location="cpu", weights_only=True)
        orig_ppl = math.exp(min(ck["val_loss"], 20))
        print(f"  {'FluidLM-07 (d=1024 L=16 NS)':<32} {'64':>10}  {'~134M':>8}  {'~135M':>8}  {orig_ppl:>6.2f}")
    except Exception:
        pass

print(f"\n  YORUM:")
print(f"  FluidLM-fair routing mekanizmasi: 48 parametre")
print(f"  GPT-B routing mekanizmasi:        ~28,311,552 parametre (590,000x fazla)")
if gpt_b_ckpt:
    delta = best_ppl - gpt_b_ppl
    if delta < 0:
        print(f"  Sonuc: FluidLM-fair {abs(delta):.2f} PPL DAHA IYI — NS routing kazaniyor!")
    elif delta < 0.5:
        print(f"  Sonuc: {delta:.2f} PPL fark — 590,000x daha az parameterle neredeyse esit.")
    else:
        print(f"  Sonuc: {delta:.2f} PPL fark — MHA hala avantajli, ama routing maliyeti 590,000x az.")
