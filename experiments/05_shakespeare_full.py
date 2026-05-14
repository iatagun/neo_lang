"""
experiments/05_shakespeare_full.py
====================================
Tüm Shakespeare (~1.1M karakter) ile FluidLM eğitimi
------------------------------------------------------

Model: d_model=256, n_layers=6, euler integrator
Veri : data/shakespeare.txt (tinyshakespeare)
Hedef: Anlamlı İngilizce üretimi test etmek

Çalıştır:
    cd neo_lang
    python experiments/05_shakespeare_full.py
"""

import sys, os
# Windows cp1254 terminallerde Unicode hatası yaşamamak için stdout'u UTF-8'e al
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

from fluidlm.fluid_lm import FluidLM

torch.manual_seed(42)
np.random.seed(42)

# ── Cihaz seçimi ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("GPU: Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("GPU yok, CPU kullanılıyor (yavaş olabilir)")

# ── Veri ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "shakespeare.txt")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    TEXT = f.read()

# Tüm metin (~1.1M karakter)
print(f"Toplam metin: {len(TEXT):,} karakter — tamamı kullanılıyor")

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
print("FluidLM — Shakespeare Tam Eğitim")
print("=" * 70)
print(f"Metin uzunluğu   : {len(TEXT):,} karakter")
print(f"Vocabulary boyutu: {VOCAB_SIZE}")

# ── Veri seti ─────────────────────────────────────────────────────────────────
SEQ_LEN    = 128
BATCH_SIZE = 128 if DEVICE.type in ("cuda", "mps") else 64

all_ids = encode(TEXT)
split   = int(len(all_ids) * 0.9)
train_ids, val_ids = all_ids[:split], all_ids[split:]
print(f"Train token: {len(train_ids):,}  Val token: {len(val_ids):,}")

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
    vocab_size  = VOCAB_SIZE,
    d_model     = 256,
    n_layers    = 6,
    max_seq_len = SEQ_LEN + 4,
    nu          = 0.01,
    dt          = 0.05,
    alpha       = 1.0,
    integrator  = "euler",
    convergence_threshold = 0.01,
    min_steps   = 3,
    dropout     = 0.15,
).to(DEVICE)

n_params = model.num_parameters()
print(f"\nModel: d_model=256, n_layers=6, integrator=euler, device={DEVICE}")
print(f"Parametre sayısı: {n_params:,}")

# ── Eğitim ────────────────────────────────────────────────────────────────────
EPOCHS        = 300
LR            = 3e-3
CLIP_NORM     = 1.0
PATIENCE      = 20   # early stopping: val iyileşmezse dur
CKPT_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "05_best_model.pt")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-5)

train_losses, val_losses, val_ppls = [], [], []

print(f"\nEğitim: {EPOCHS} epoch, LR={LR}, batch={BATCH_SIZE}, seq={SEQ_LEN}, patience={PATIENCE}")
print("-" * 70)

best_val      = float("inf")
best_epoch    = 0
no_improve    = 0
t0_total      = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_batches = make_batches(train_ids, SEQ_LEN, BATCH_SIZE, shuffle=True)
    epoch_loss = 0.0

    for x, y in train_batches:
        optimizer.zero_grad()
        logits = model(x, adaptive=False)
        loss   = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1),
                                 ignore_index=PAD_ID)
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

    # ── Checkpoint: en iyi modeli kaydet ──────────────────────────────
    if avg_val < best_val:
        best_val   = avg_val
        best_epoch = epoch
        no_improve = 0
        torch.save({
            "epoch":      epoch,
            "model_state": model.state_dict(),
            "val_loss":   avg_val,
            "stoi":       stoi,
            "itos":       itos,
            "vocab_size": VOCAB_SIZE,
            "d_model":    256,
            "n_layers":   6,
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
        print(f"Epoch {epoch:>3}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}  "
              f"ppl={ppl:.1f}  best_ppl={math.exp(min(best_val,20)):.1f}@ep{best_epoch}  "
              f"patience={no_improve}/{PATIENCE}  kalan~{remaining/60:.1f}dk{marker}")

    # ── Early stopping ───────────────────────────────────────────────
    if no_improve >= PATIENCE:
        print(f"\nEarly stopping: {PATIENCE} epoch iyileşme yok. Duruluyor.")
        break

print("-" * 70)
print(f"Eğitim bitti. En iyi val_ppl={math.exp(min(best_val,20)):.2f} (epoch {best_epoch})  "
      f"Toplam süre: {(time.time()-t0_total)/60:.1f} dakika")
print(f"Model kaydedildi: {CKPT_PATH}")

# ── En iyi checkpoint'i yükle ───────────────────────────────────────────────
print(f"\nEn iyi model yükleniyor: epoch {best_epoch}, val_loss={best_val:.4f}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── Üretim ────────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("ÜRETİM TESTİ")
print("═" * 70)

def generate(model, prompt, max_new=200, temperature=0.8, top_k=10):
    model.eval()
    ids = torch.tensor([encode(prompt)[:SEQ_LEN]]).to(DEVICE)
    with torch.no_grad():
        for _ in range(max_new):
            ids_in = ids[:, -model.max_seq_len:]
            logits = model(ids_in, adaptive=True)
            next_l = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                topk_v, _ = torch.topk(next_l, min(top_k, next_l.size(-1)))
                next_l = next_l.masked_fill(next_l < topk_v[:, -1:], -1e9)
            probs  = torch.softmax(next_l, dim=-1)
            nid    = torch.multinomial(probs, 1)
            ids    = torch.cat([ids, nid], dim=1)
            if nid.item() == EOS_ID:
                break
    return decode(ids[0].tolist())

prompts_cfg = [
    ("HAMLET (greedy)",   "HAMLET:",            0.3,  5),
    ("HAMLET (balanced)", "HAMLET:",            0.8, 10),
    ("HAMLET (creative)", "HAMLET:",            1.0, 20),
    ("Act 1 balanced",    "ACT I.\nSCENE I.",   0.8, 10),
    ("Diyalog",           "OPHELIA:\nTo be,",   0.8, 10),
]

for label, prompt, temp, topk in prompts_cfg:
    out = generate(model, prompt, max_new=200, temperature=temp, top_k=topk)
    print(f"\n{'─'*65}")
    print(f"[{label}]")
    print(f"{'─'*65}")
    print(out[:400])

# ── Fiziksel parametreler ────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("ÖĞRENILEN FİZİKSEL PARAMETRELER")
print("═" * 70)
phys = model.physical_params()
print(f"  {'Katman':<8} {'ν':<10} {'Δt':<10} {'α':<10} {'p_scale'}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for i, (nu_v, dt_v, al_v, ps_v) in enumerate(
        zip(phys["nu"], phys["dt"], phys["alpha"], phys["p_scale"])):
    print(f"  {i:<8} {nu_v:<10.4f} {dt_v:<10.4f} {al_v:<10.4f} {ps_v:<10.4f}")

# ── Görselleştirme ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"FluidLM - Shakespeare (1.1M chars, d=256, L=6)  best epoch={best_epoch}",
             fontsize=13)

epochs_x = list(range(1, len(train_losses) + 1))

axes[0].plot(epochs_x, train_losses, label="Train", color="steelblue", lw=1.5)
axes[0].plot(epochs_x, val_losses,   label="Val",   color="tomato",    lw=1.5, ls="--")
axes[0].axvline(best_epoch, color="green", ls=":", lw=1.5, label=f"best ep={best_epoch}")
axes[0].set_title("Kayip"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].semilogy(epochs_x, val_ppls, color="purple", lw=1.5)
axes[1].axvline(best_epoch, color="green", ls=":", lw=1.5)
axes[1].set_title("Val Perplexity (log)"); axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.3, which="both")

# alpha ayri sag eksende (diger parametrelerle olcek uyumsuz olabilir)
ax2  = axes[2]
ax2b = ax2.twinx()
ax2.plot(phys["nu"],      marker="o", color="steelblue",  lw=2, label="nu")
ax2.plot(phys["dt"],      marker="s", color="darkorange", lw=2, label="dt")
ax2.plot(phys["p_scale"], marker="D", color="crimson",    lw=2, label="p_scale")
ax2b.plot(phys["alpha"],  marker="^", color="seagreen",   lw=2, ls="--", label="alpha (sag)")
ax2.set_title("Ogrenilen Parametreler"); ax2.set_xlabel("Katman")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_shakespeare_result.png")
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nGrafik kaydedildi: {out_path}")
print("\nDeney tamamlandı.")
