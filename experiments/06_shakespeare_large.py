"""
experiments/06_shakespeare_large.py
=====================================
FluidLM buyuk model — d_model=512, n_layers=12, seq_len=256
05 ile karsilastirilinca: 2x genis, 2x derin, 2x uzun baglam

GTX 1650 (4 GB) icin BATCH_SIZE=32 ayarlandi.

Calistir:
    cd neo_lang
    python experiments/06_shakespeare_large.py
"""

import sys, os
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fluid_lm import FluidLM

torch.manual_seed(42)
np.random.seed(42)

# ── Cihaz ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("GPU: Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("CPU modunda calisiyor (cok yavas olabilir)")

# ── Veri ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "shakespeare.txt")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    TEXT = f.read()

chars      = sorted(set(TEXT))
vocab      = ["<pad>", "<bos>", "<eos>"] + chars
stoi       = {c: i for i, c in enumerate(vocab)}
itos       = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(vocab)
PAD_ID, BOS_ID, EOS_ID = stoi["<pad>"], stoi["<bos>"], stoi["<eos>"]

def encode(text):
    return [BOS_ID] + [stoi[c] for c in text if c in stoi] + [EOS_ID]

def decode(ids):
    return "".join(itos.get(i, "?") for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID))

print("=" * 70)
print("FluidLM BUYUK MODEL — d=512, L=12, seq=256")
print("=" * 70)
print(f"Metin: {len(TEXT):,} karakter  Vocab: {VOCAB_SIZE}")

# ── Hiperparametreler ─────────────────────────────────────────────────────────
D_MODEL    = 512
N_LAYERS   = 12
SEQ_LEN    = 256
# GTX 1650 4GB: d=512 ile batch=32 guvenli; OOM olursa 16'ya duser
BATCH_SIZE = 32 if DEVICE.type in ("cuda", "mps") else 8

EPOCHS     = 400
LR         = 8e-4        # buyuk model -> daha kucuk LR
CLIP_NORM  = 1.0
PATIENCE   = 30
DROPOUT    = 0.2
CKPT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "06_best_model.pt")

# ── Veri bolme ────────────────────────────────────────────────────────────────
all_ids = encode(TEXT)
split   = int(len(all_ids) * 0.9)
train_ids, val_ids = all_ids[:split], all_ids[split:]
print(f"Train: {len(train_ids):,} token  Val: {len(val_ids):,} token")

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
model = FluidLM(
    vocab_size            = VOCAB_SIZE,
    d_model               = D_MODEL,
    n_layers              = N_LAYERS,
    max_seq_len           = SEQ_LEN + 4,
    nu                    = 0.01,
    dt                    = 0.05,
    alpha                 = 1.0,
    integrator            = "euler",
    convergence_threshold = 0.005,   # daha hassas yakinlasma
    min_steps             = 4,
    dropout               = DROPOUT,
).to(DEVICE)

n_params = model.num_parameters()
print(f"\nModel: d={D_MODEL}, L={N_LAYERS}, seq={SEQ_LEN}, "
      f"batch={BATCH_SIZE}, device={DEVICE}")
print(f"Parametre: {n_params:,}  "
      f"(05 modeline gore ~{n_params / 1 :.0f} param)")

# ── OOM koruma: kucuk batch dene ──────────────────────────────────────────────
def safe_forward(x, y):
    """OOM olursa batch'i yarisina bol ve tekrar dene."""
    try:
        logits = model(x, adaptive=False)
        return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1),
                               ignore_index=PAD_ID)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        mid = x.size(0) // 2
        l1  = F.cross_entropy(
            model(x[:mid], adaptive=False).reshape(-1, VOCAB_SIZE),
            y[:mid].reshape(-1), ignore_index=PAD_ID)
        l2  = F.cross_entropy(
            model(x[mid:], adaptive=False).reshape(-1, VOCAB_SIZE),
            y[mid:].reshape(-1), ignore_index=PAD_ID)
        return (l1 + l2) / 2

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# Warmup: ilk 10 epoch lineer yukselis, sonra cosine
def lr_lambda(ep):
    warmup = 10
    if ep < warmup:
        return ep / warmup
    progress = (ep - warmup) / max(EPOCHS - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - 5e-5 / LR) + 5e-5 / LR

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── Egitim ────────────────────────────────────────────────────────────────────
train_losses, val_losses, val_ppls = [], [], []

print(f"\nEgitim basliyor: {EPOCHS} epoch, LR={LR}, patience={PATIENCE}")
print("-" * 70)

best_val   = float("inf")
best_epoch = 0
no_improve = 0
t0_total   = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_batches = make_batches(train_ids, SEQ_LEN, BATCH_SIZE, shuffle=True)
    epoch_loss = 0.0

    for x, y in train_batches:
        optimizer.zero_grad()
        loss = safe_forward(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg_train = epoch_loss / max(len(train_batches), 1)
    train_losses.append(avg_train)

    # Val
    model.eval()
    val_batches = make_batches(val_ids, SEQ_LEN, BATCH_SIZE, shuffle=False)
    v_loss = 0.0
    with torch.no_grad():
        for x, y in val_batches:
            logits = model(x, adaptive=False)
            v_loss += F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1),
                                      ignore_index=PAD_ID).item()
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
            "model_state": model.state_dict(),
            "val_loss":    avg_val,
            "stoi":        stoi,
            "itos":        itos,
            "vocab_size":  VOCAB_SIZE,
            "d_model":     D_MODEL,
            "n_layers":    N_LAYERS,
            "max_seq_len": SEQ_LEN + 4,
        }, CKPT_PATH)
        marker = " << BEST"
    else:
        no_improve += 1
        marker = ""

    if epoch % 10 == 0 or epoch == 1:
        elapsed   = time.time() - t0_total
        per_epoch = elapsed / epoch
        remaining = per_epoch * (EPOCHS - epoch)
        cur_lr    = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:>3}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}  "
              f"ppl={ppl:.1f}  best@ep{best_epoch}({math.exp(min(best_val,20)):.1f})  "
              f"pat={no_improve}/{PATIENCE}  lr={cur_lr:.5f}  "
              f"~{remaining/60:.0f}dk{marker}")

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping: {PATIENCE} epoch iyilesme yok.")
        break

print("-" * 70)
total_min = (time.time() - t0_total) / 60
print(f"Egitim bitti. Best val_ppl={math.exp(min(best_val,20)):.2f} "
      f"(epoch {best_epoch})  Sure: {total_min:.1f} dk")
print(f"Checkpoint: {CKPT_PATH}")

# ── En iyi checkpoint yukle ───────────────────────────────────────────────────
print(f"\nEn iyi model yukleniyor (epoch {best_epoch})...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── Metin uretimi ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt_text, max_new=300, temperature=0.8, top_k=10):
    ids = torch.tensor([encode(prompt_text)], dtype=torch.long, device=DEVICE)
    ids = ids[:, -(SEQ_LEN + 4):]
    for _ in range(max_new):
        logits  = model(ids, adaptive=False)   # adaptive=False: tum katmanlar
        next_l  = logits[0, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            topk_v, _ = torch.topk(next_l, min(top_k, next_l.size(-1)))
            next_l = next_l.masked_fill(next_l < topk_v[-1], -1e9)
        probs = torch.softmax(next_l, dim=-1)
        nid   = torch.multinomial(probs, 1).unsqueeze(0)
        ids   = torch.cat([ids, nid], dim=1)[:, -(SEQ_LEN + 4):]
        if nid.item() == EOS_ID:
            break
    return prompt_text + decode(ids[0].tolist())

prompts_cfg = [
    ("HAMLET greedy",    "HAMLET:",           0.3,  5),
    ("HAMLET balanced",  "HAMLET:",           0.8, 10),
    ("HAMLET creative",  "HAMLET:",           1.1, 20),
    ("Diyalog",          "OPHELIA:\nTo be,",  0.8, 10),
    ("Act 1",            "ACT I.\nSCENE I.",  0.8, 10),
]

print("\n" + "=" * 70)
print("METIN URETIMI (06_best_model.pt)")
print("=" * 70)
for label, prompt, temp, topk in prompts_cfg:
    out = generate(prompt, max_new=300, temperature=temp, top_k=topk)
    print(f"\n[{label}]")
    print("-" * 60)
    print(out[:500])

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
fig.suptitle(f"FluidLM BUYUK — d={D_MODEL}, L={N_LAYERS}, seq={SEQ_LEN}  "
             f"best epoch={best_epoch}", fontsize=12)

epochs_x = list(range(1, len(train_losses) + 1))

axes[0].plot(epochs_x, train_losses, label="Train", color="steelblue", lw=1.5)
axes[0].plot(epochs_x, val_losses,   label="Val",   color="tomato",    lw=1.5, ls="--")
axes[0].axvline(best_epoch, color="green", ls=":", lw=1.5, label=f"best ep={best_epoch}")
axes[0].set_title("Kayip"); axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].semilogy(epochs_x, val_ppls, color="purple", lw=1.5)
axes[1].axvline(best_epoch, color="green", ls=":", lw=1.5)
axes[1].set_title("Val Perplexity (log)"); axes[1].set_xlabel("Epoch")
axes[1].grid(alpha=0.3, which="both")

ax2  = axes[2]
ax2b = ax2.twinx()
ax2.plot(phys["nu"],      marker="o", color="steelblue",  lw=2, label="nu")
ax2.plot(phys["dt"],      marker="s", color="darkorange", lw=2, label="dt")
ax2.plot(phys["p_scale"], marker="D", color="crimson",    lw=2, label="p_scale")
ax2b.plot(phys["alpha"],  marker="^", color="seagreen",   lw=2, ls="--", label="alpha (sag)")
ax2.set_title("Parametreler"); ax2.set_xlabel("Katman")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
out_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_result.png")
plt.savefig(out_img, dpi=120, bbox_inches="tight")
print(f"\nGrafik: {out_img}")
print("Deney tamamlandi.")
