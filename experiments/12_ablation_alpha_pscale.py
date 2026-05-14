# ============================================================
# 12_ablation_alpha_pscale.py
#
# Ablasyon Deneyi: α ve p_scale parametrelerini dondur
#
# Motivasyon:
#   11_flop_and_physics.py çıktısından:
#     α    (Helmholtz): tüm katmanlarda 1.0000 (değişmedi)
#     p_sc (basınç):    tüm katmanlarda 0.1006 (değişmedi)
#
#   İki hipotez:
#     H1: Bu parametreler ihtiyaç duyulmadığı için değişmedi
#         → Tamamen dondurulsa PPL değişmez (≤ 0.05 fark)
#     H2: Gradyan sinyali çok zayıf, aslında önemli ama öğrenemiyor
#         → Dondurunca PPL belirgin düşer veya yükselir
#
#   Bu deney H1 vs H2'yi test eder:
#     Koşul A: FluidLM-fair, α ve p_scale öğrenilebilir    (baseline: PPL=4.59)
#     Koşul B: FluidLM-fair, α ve p_scale sabit (frozen)
#     Koşul C: FluidLM-fair, sadece α frozen
#     Koşul D: FluidLM-fair, sadece p_scale frozen
#
#   Eğer |PPL_B - PPL_A| < 0.05 → H1 doğrulandı:
#     "Helmholtz regularizasyonu ve basınç ölçeği için tek global
#      değer yeterli, katman başına öğrenim gereksiz."
#
# Kullanım:
#   python experiments/12_ablation_alpha_pscale.py
#   python experiments/12_ablation_alpha_pscale.py --epochs 100 --conditions AB
#   python experiments/12_ablation_alpha_pscale.py --conditions ABCD --epochs 150
#
# --conditions: hangi koşullar (default: AB — hızlı)
# ============================================================

import sys, os, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",     type=int,   default=100)
parser.add_argument("--patience",   type=int,   default=15)
parser.add_argument("--conditions", default="AB",
                    help="Hangi koşullar: A=baseline, B=both frozen, C=alpha frozen, D=pscale frozen")
parser.add_argument("--data",       default=None)
parser.add_argument("--out_dir",    default=None)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
OUT_DIR    = args.out_dir or SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE    = torch.bfloat16 if USE_BF16 else torch.float32
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9 \
           if device.type == "cuda" else 0
print(f"Device: {device}  dtype={DTYPE}  VRAM={vram_gb:.0f}GB")
print(f"Kosullar: {args.conditions}  epochs={args.epochs}  patience={args.patience}")

# ── Veri ──────────────────────────────────────────────────────────────────────
D_MODEL   = 768
N_LAYERS  = 12
SEQ_LEN   = 512
MLP_RATIO = 4
DROPOUT   = 0.2
LR        = 3e-4

DATA_PATH = args.data
if DATA_PATH is None:
    for candidate in [
        os.path.join(REPO_ROOT, "data", "shakespeare.txt"),
        "/content/neo_lang/data/shakespeare.txt",
        os.path.join(SCRIPT_DIR, "shakespeare.txt"),
    ]:
        if os.path.exists(candidate):
            DATA_PATH = candidate; break

if DATA_PATH is None:
    import urllib.request
    DATA_PATH = os.path.join(OUT_DIR, "shakespeare.txt")
    print("Shakespeare indiriliyor...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt", DATA_PATH)

with open(DATA_PATH, encoding="utf-8") as f:
    TEXT = f.read()

chars = sorted(set(TEXT)); VOCAB_SIZE = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
all_ids   = torch.tensor(encode(TEXT), dtype=torch.long)
split     = int(len(all_ids) * 0.9)
TRAIN_IDS = all_ids[:split]; VAL_IDS = all_ids[split:]
print(f"Corpus: {len(TEXT):,} char  vocab={VOCAB_SIZE}")

def get_batch(ids, bs):
    ix = torch.randint(len(ids) - SEQ_LEN - 1, (bs,))
    x  = torch.stack([ids[i:i+SEQ_LEN]     for i in ix]).to(device)
    y  = torch.stack([ids[i+1:i+SEQ_LEN+1] for i in ix]).to(device)
    return x, y

if   vram_gb >= 70: BATCH_SIZE, GRAD_ACCUM = 32, 8
elif vram_gb >= 40: BATCH_SIZE, GRAD_ACCUM = 16, 8
elif vram_gb >= 10: BATCH_SIZE, GRAD_ACCUM = 8,  8
else:               BATCH_SIZE, GRAD_ACCUM = 4,  4

# ── NS Operatörleri ───────────────────────────────────────────────────────────
def causal_gradient(u):
    return u - F.pad(u, (0,0,1,0))[:,:-1,:]

def causal_laplacian(u):
    p = F.pad(u, (0,0,2,0))
    return u - 2*p[:,1:-1,:] + p[:,:-2,:]

def causal_divergence(u):
    return causal_gradient(u).mean(dim=-1)

def spectral_pressure(adv, alpha):
    div = causal_divergence(adv)
    B, L = div.shape
    f   = torch.fft.rfft(div.float(), dim=1)
    k   = torch.arange(L//2+1, dtype=torch.float32, device=div.device)
    a   = alpha.float() if isinstance(alpha, torch.Tensor) else float(alpha)
    lam = 2.0*(torch.cos(2.0*math.pi*k/L)-1.0) - a**2
    lam = lam.masked_fill(lam.abs()<1e-8, -1e-8)
    p   = torch.fft.irfft(-f/lam.unsqueeze(0), n=L, dim=1).to(adv.dtype)
    return p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5)+1e-6)

# ── FluidLayer (ablasyon destekli) ────────────────────────────────────────────

class FluidLayerAbl(nn.Module):
    """
    freeze_alpha=True → log_alpha requires_grad=False (sabit)
    freeze_pscale=True → log_p_scale requires_grad=False (sabit)
    """
    def __init__(self, d_model, i=0, freeze_alpha=False, freeze_pscale=False):
        super().__init__()
        self.log_nu      = nn.Parameter(torch.tensor(math.log(0.01*(1+0.05*i))))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(0.05*(1+0.02*i))))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(1.0)),
                                        requires_grad=not freeze_alpha)
        self.log_p_scale = nn.Parameter(torch.tensor(math.log(0.1)),
                                        requires_grad=not freeze_pscale)
        H = d_model * MLP_RATIO
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, H), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(H, d_model), nn.Dropout(DROPOUT))
        self._fa = freeze_alpha
        self._fp = freeze_pscale

    def forward(self, u):
        nu = self.log_nu.exp(); dt = self.log_dt.exp()
        al = self.log_alpha.exp(); ps = self.log_p_scale.exp()
        n  = self.norm1(u)
        sp = torch.tanh(n.norm(dim=-1, keepdim=True))
        adv = sp * causal_gradient(n)
        p   = spectral_pressure(adv, al)
        pg  = ps * causal_gradient(p.unsqueeze(-1)).expand_as(n)
        rhs = -adv - pg + nu*causal_laplacian(n)
        u = u + dt*rhs
        u = u + self.mlp(self.norm2(u))
        return u

class FluidLMAbl(nn.Module):
    def __init__(self, freeze_alpha=False, freeze_pscale=False):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb   = nn.Embedding(SEQ_LEN, D_MODEL)
        self.emb_drop  = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList([
            FluidLayerAbl(D_MODEL, i=i,
                          freeze_alpha=freeze_alpha,
                          freeze_pscale=freeze_pscale)
            for i in range(N_LAYERS)])
        self.norm    = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._init()
        self._fa = freeze_alpha; self._fp = freeze_pscale

    def _init(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, ids):
        B, L = ids.shape
        pos  = torch.arange(L, device=ids.device).unsqueeze(0)
        u    = self.emb_drop(self.token_emb(ids) + self.pos_emb(pos))
        for layer in self.layers:
            u = layer(u)
        return self.lm_head(self.norm(u))

    def frozen_info(self):
        parts = []
        if self._fa: parts.append("α sabit")
        if self._fp: parts.append("p_scale sabit")
        return ", ".join(parts) if parts else "hepsi ogreniliyor"

# ── Eğitim fonksiyonu ─────────────────────────────────────────────────────────
WARMUP = 500
def get_lr_scale(step):
    if step < WARMUP: return step / max(1, WARMUP)
    prog = (step - WARMUP) / max(1, 10_000 - WARMUP)
    return max(0.1, 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

def train_condition(label, freeze_alpha, freeze_pscale):
    print(f"\n{'='*60}")
    print(f"  KOŞUL {label}: {('α sabit' if freeze_alpha else 'α serbest')} + "
          f"{'p_scale sabit' if freeze_pscale else 'p_scale serbest'}")
    print(f"{'='*60}")

    model = FluidLMAbl(freeze_alpha=freeze_alpha, freeze_pscale=freeze_pscale)
    model = model.to(device)
    if USE_BF16: model = model.to(torch.bfloat16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Öğrenilebilir: {trainable:,}  Donuk: {frozen:,}")

    ns_params   = [p for n, p in model.named_parameters()
                   if any(x in n for x in ['log_nu','log_dt','log_alpha','log_p_scale'])
                   and p.requires_grad]
    ns_ids      = {id(p) for p in ns_params}
    base_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in ns_ids]

    param_groups = [{"params": base_params, "lr": LR}]
    if ns_params:
        param_groups.append({"params": ns_params, "lr": LR * 10})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9,0.95), weight_decay=0.1)

    STEPS_PER = max(1, len(TRAIN_IDS) // (SEQ_LEN * BATCH_SIZE * GRAD_ACCUM))
    VAL_STEPS = min(100, len(VAL_IDS) // (SEQ_LEN * BATCH_SIZE))
    LOG_EVERY = max(10, STEPS_PER // 5)

    best_val = float("inf"); no_improve = 0; total_steps = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0; optimizer.zero_grad()

        for step_i in range(STEPS_PER):
            for _ in range(GRAD_ACCUM):
                x, y = get_batch(TRAIN_IDS, BATCH_SIZE)
                with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=DTYPE):
                    loss = F.cross_entropy(
                        model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1)
                    ) / GRAD_ACCUM
                loss.backward()
                epoch_loss += loss.item() * GRAD_ACCUM

            scale = get_lr_scale(total_steps)
            for pg in optimizer.param_groups:
                base_lr = LR * 10 if len(ns_params) > 0 and pg["params"] is ns_params else LR
                pg["lr"] = base_lr * scale

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(); total_steps += 1

        avg_train = epoch_loss / (STEPS_PER * GRAD_ACCUM)

        model.eval(); v_loss = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=DTYPE):
            for _ in range(VAL_STEPS):
                xv, yv = get_batch(VAL_IDS, BATCH_SIZE)
                v_loss += F.cross_entropy(
                    model(xv).reshape(-1, VOCAB_SIZE), yv.reshape(-1)).item()
        avg_val = v_loss / VAL_STEPS
        ppl     = math.exp(min(avg_val, 20))

        marker = ""
        if avg_val < best_val:
            best_val = avg_val; no_improve = 0; marker = "  << BEST"
            torch.save({"epoch": epoch, "val_loss": avg_val, "label": label},
                       os.path.join(OUT_DIR, f"12_ablation_{label}_best.pt"))
        else:
            no_improve += 1

        if epoch % 10 == 0 or marker:
            elapsed_m = (time.time()-t0)/60
            print(f"  Ep {epoch:>3}/{args.epochs}  train={avg_train:.4f}  "
                  f"val={avg_val:.4f}  ppl={ppl:.2f}  {elapsed_m:.1f}m{marker}", flush=True)

        if no_improve >= args.patience:
            print(f"  Early stopping ep {epoch}")
            break

    best_ppl = math.exp(min(best_val, 20))
    print(f"  → Koşul {label} tamamlandı: best_val={best_val:.4f}  PPL={best_ppl:.2f}")
    return label, best_val, best_ppl

# ── Koşulları tanımla ──────────────────────────────────────────────────────────
CONDITION_MAP = {
    "A": (False, False),  # baseline — hepsi serbest
    "B": (True,  True),   # ikisi de frozen
    "C": (True,  False),  # sadece α frozen
    "D": (False, True),   # sadece p_scale frozen
}

selected = [(c, *CONDITION_MAP[c]) for c in args.conditions.upper() if c in CONDITION_MAP]
print(f"\nSeçilen koşullar: {[x[0] for x in selected]}")
print(f"Batch={BATCH_SIZE}  GradAccum={GRAD_ACCUM}  EffBatch={BATCH_SIZE*GRAD_ACCUM}")

# ── Mevcut FluidLM-fair sonucu (Koşul A baseline olarak yükle) ─────────────────
baseline_ppl = None
for p in [
    os.path.join(SCRIPT_DIR, "10_fluidlm_fair_best.pt"),
    os.path.join(REPO_ROOT, "checkpoints", "10_fluidlm_fair_best.pt"),
    "/content/neo_lang/experiments/10_fluidlm_fair_best.pt",
]:
    if os.path.exists(p):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=True)
            baseline_ppl = math.exp(min(ck["val_loss"], 20))
            print(f"\nMevcut FluidLM-fair baseline: PPL={baseline_ppl:.2f} ({p})")
        except Exception:
            pass
        break

# ── Eğit ──────────────────────────────────────────────────────────────────────
results = {}
for label, fa, fp in selected:
    lbl, best_val, best_ppl = train_condition(label, fa, fp)
    results[lbl] = (best_val, best_ppl)

# ── Ablasyon Tablosu ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  ABLASYON SONUÇLARI — α ve p_scale Etkisi")
print(f"{'='*65}")
print(f"  {'Koşul':<30} {'α':>8} {'p_scale':>10}  {'PPL':>6}  {'Δ PPL':>8}")
print("  " + "-"*60)

baseline_ref = baseline_ppl or results.get("A", (None, None))[1]

cond_labels = {
    "A": ("serbest",   "serbest",  "baseline"),
    "B": ("SABIT=1.0", "SABIT=0.1","ikisi frozen"),
    "C": ("SABIT=1.0", "serbest",  "sadece α frozen"),
    "D": ("serbest",   "SABIT=0.1","sadece p_scale frozen"),
}

all_ppls = {}
if baseline_ppl and "A" not in results:
    all_ppls["A"] = baseline_ppl
for lbl, (_, ppl) in results.items():
    all_ppls[lbl] = ppl

for lbl in ["A", "B", "C", "D"]:
    if lbl not in all_ppls and lbl not in results: continue
    ppl = all_ppls.get(lbl) or results.get(lbl, (None,None))[1]
    if ppl is None: continue
    al, ps, note = cond_labels.get(lbl, ("?","?",""))
    delta = ppl - baseline_ref if baseline_ref and lbl != "A" else 0.0
    delta_str = f"{delta:+.3f}" if lbl != "A" else "—"
    print(f"  {lbl}: {note:<26} {al:>8} {ps:>10}  {ppl:>6.2f}  {delta_str:>8}")

print()
print("  Yorum:")
ppl_b = all_ppls.get("B")
if ppl_b and baseline_ref:
    delta_b = ppl_b - baseline_ref
    if abs(delta_b) < 0.05:
        print(f"  ✓ H1 DOGRULANDI: α+p_scale dondurulunca PPL fark={delta_b:+.3f}")
        print(f"    → Bu iki parametre ogrenim icin gerekli degil.")
        print(f"    → Model basitlestirilebilir: her katmanda 4→2 ogrenilecek param.")
    elif delta_b > 0.1:
        print(f"  ✗ H2 DOGRULANDI: dondurunca PPL {delta_b:+.3f} artti")
        print(f"    → α/p_scale gradyan sinyali zayif ama etkili.")
    else:
        print(f"  ? Belirsiz: delta={delta_b:+.3f}. Daha uzun egitim gerekebilir.")

ppl_c = all_ppls.get("C"); ppl_d = all_ppls.get("D")
if ppl_c and ppl_d and baseline_ref:
    print(f"\n  Bireysel etki:")
    print(f"    α tek basina: delta={ppl_c-baseline_ref:+.3f}")
    print(f"    p_scale tek basina: delta={ppl_d-baseline_ref:+.3f}")
    dominant = "α" if abs(ppl_c-baseline_ref) > abs(ppl_d-baseline_ref) else "p_scale"
    print(f"    Daha etkili parametre: {dominant}")
