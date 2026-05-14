"""
experiments/04_full_train_and_generate.py
==========================================
Tam Eğitim + Kapsamlı Üretim Testi
------------------------------------

Bu deney:
  1. Daha büyük model (d=128, 12 katman) uzun süre eğitir
  2. Eğitim + validasyon kaybını izler
  3. Perplexity hesaplar
  4. Çoklu prompt ile üretim yapar (greedy, temperature, top-k, top-p)
  5. Adaptif derinliği istatistiksel olarak analiz eder
  6. Öğrenilen fiziksel parametrelerin evrimini görselleştirir

Çalıştır:
    cd neo_lang
    python experiments/04_full_train_and_generate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from src.fluid_lm import FluidLM

torch.manual_seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  VERİ SETİ
#     Shakespeare'in küçük bir parçası + tekrarlayan İngilizce düzyazı.
#     Karakter düzeyi modeli için yeterince çeşitli.
# ═══════════════════════════════════════════════════════════════════════════════

TEXT = """
first citizen:
before we proceed any further, hear me speak.

all:
speak, speak.

first citizen:
you are all resolved rather to die than to famish?

all:
resolved. resolved.

first citizen:
first, you know caius marcius is chief enemy to the people.

all:
we know't, we know't.

first citizen:
let us kill him, and we'll have corn at our own price.
is't a verdict?

all:
no more talking on't; let it be done: away, away!

second citizen:
one word, good citizens.

first citizen:
we are accounted poor citizens, the patricians good.
what authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. let us revenge this with
our pikes, ere we become rakes: for the gods know i
speak this in hunger for bread, not in thirst for revenge.

second citizen:
would you proceed especially against caius marcius?

all:
against him first: he's a very dog to the commonalty.

second citizen:
consider you what services he has done for his country?

first citizen:
very well; and could be content to give him good
report for't, but that he pays himself with being proud.

second citizen:
nay, but speak not maliciously.

first citizen:
i say unto you, what he hath done famously, he did it to
that end: though soft-conscienced men can be content to
say it was for his country he did it to please his mother
and to be partly proud; which he is, even till the
altitude of his virtue.

second citizen:
what he cannot help in his nature, you account a vice in him.
you must in no way say he is covetous.

first citizen:
if i must not, i need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
what shouts are these? the other side o' the city is risen:
why stay we prating here? to the capitol!

all:
come, come.
""".strip()

# ─── Vocabulary ───────────────────────────────────────────────────────────────

chars      = sorted(set(TEXT))
vocab      = ["<pad>", "<bos>", "<eos>"] + chars
stoi       = {c: i for i, c in enumerate(vocab)}
itos       = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(vocab)
PAD_ID     = stoi["<pad>"]
BOS_ID     = stoi["<bos>"]
EOS_ID     = stoi["<eos>"]

def encode(text: str) -> List[int]:
    return [BOS_ID] + [stoi[c] for c in text if c in stoi] + [EOS_ID]

def decode(ids: List[int]) -> str:
    return "".join(itos.get(i, "?") for i in ids
                   if i not in (PAD_ID, BOS_ID, EOS_ID))

print("=" * 70)
print("FluidLM — Tam Eğitim ve Üretim Testi")
print("=" * 70)
print(f"Metin uzunluğu   : {len(TEXT):,} karakter")
print(f"Benzersiz karakter: {len(chars)}")
print(f"Vocabulary boyutu: {VOCAB_SIZE}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  VERİ HAZIRLAMA — train / val split
# ═══════════════════════════════════════════════════════════════════════════════

SEQ_LEN    = 64
BATCH_SIZE = 32

all_ids = encode(TEXT)
split   = int(len(all_ids) * 0.9)
train_ids, val_ids = all_ids[:split], all_ids[split:]

def make_batches(ids: List[int], seq_len: int, batch_size: int,
                 shuffle: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor]]:
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
        batches.append((t[:, :-1], t[:, 1:]))
    return batches

print(f"Train token sayısı: {len(train_ids):,}")
print(f"Val   token sayısı: {len(val_ids):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

model = FluidLM(
    vocab_size  = VOCAB_SIZE,
    d_model     = 128,
    n_layers    = 12,
    max_seq_len = SEQ_LEN + 4,
    nu          = 0.02,
    dt          = 0.08,
    alpha       = 1.0,
    integrator  = "euler",       # euler: 4× daha hızlı, bu ölçekte yeterli
    convergence_threshold = 0.01,  # normalised ΔKE eşiği (||Δu||²/||u||²)
    min_steps   = 4,
    dropout     = 0.1,
)

n_params = model.num_parameters()
print(f"\nModel: d_model={model.d_model}, n_layers={model.n_layers}, "
      f"integrator=euler")
print(f"Parametre sayısı : {n_params:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EĞİTİM
# ═══════════════════════════════════════════════════════════════════════════════

EPOCHS    = 200
LR        = 5e-3
CLIP_NORM = 1.0

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

train_losses: List[float] = []
val_losses:   List[float] = []
val_ppls:     List[float] = []

print(f"\nEğitim: {EPOCHS} epoch, LR={LR}, batch={BATCH_SIZE}, seq={SEQ_LEN}")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
    # ── Eğitim ──────────────────────────────────────────────────────────────
    model.train()
    train_batches = make_batches(train_ids, SEQ_LEN, BATCH_SIZE, shuffle=True)
    epoch_loss = 0.0

    for x, y in train_batches:
        optimizer.zero_grad()
        logits = model(x, adaptive=False)
        loss   = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=PAD_ID)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg_train = epoch_loss / max(len(train_batches), 1)
    train_losses.append(avg_train)

    # ── Validasyon ──────────────────────────────────────────────────────────
    model.eval()
    val_batches = make_batches(val_ids, SEQ_LEN, BATCH_SIZE, shuffle=False)
    if not val_batches:
        # Val seti çok küçükse train'in son 10%'inden batch al
        val_batches = make_batches(train_ids[-len(train_ids)//10:],
                                   SEQ_LEN, BATCH_SIZE, shuffle=False)

    v_loss = 0.0
    with torch.no_grad():
        for x, y in (val_batches or []):
            logits = model(x, adaptive=False)
            v_loss += F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), y.reshape(-1),
                ignore_index=PAD_ID).item()

    avg_val = v_loss / max(len(val_batches), 1)
    val_losses.append(avg_val)
    val_ppls.append(math.exp(min(avg_val, 20)))   # clamp for log display

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train_loss={avg_train:.4f}  "
              f"val_loss={avg_val:.4f}  "
              f"val_ppl={val_ppls[-1]:.1f}  "
              f"lr={scheduler.get_last_lr()[0]:.5f}")

print("-" * 70)
print(f"Eğitim tamamlandı. Son val_ppl={val_ppls[-1]:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  ADAPTİF DERİNLİK ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("ADAPTİF DERİNLİK ANALİZİ")
print("═" * 70)

test_texts = [
    ("Kısa / basit",     "all:"),
    ("Orta",             "first citizen:"),
    ("Uzun / karmaşık",  "we are accounted poor citizens, the patricians"),
    ("Diyalog açılışı",  "second citizen:"),
    ("Cümle ortası",     "what he cannot help in his nature,"),
]

model.eval()
step_counts = []
for label, prompt in test_texts:
    ids = torch.tensor([encode(prompt)[:SEQ_LEN]])
    with torch.no_grad():
        _, stats = model(ids, adaptive=True, return_stats=True)
    n  = stats["n_steps"]
    dk = stats["delta_kes"][-1]
    step_counts.append(n)
    print(f"  [{label:<25}]  '{prompt[:35]}'")
    print(f"    → {n} katman kullanıldı  |  son ΔKE={dk:.2e}")

print(f"\n  Ortalama katman: {np.mean(step_counts):.1f} / {model.n_layers}")
print(f"  Min: {min(step_counts)}  Max: {max(step_counts)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ÜRETİM TESTLERİ
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("ÜRETİM TESTLERİ")
print("═" * 70)

generation_configs = [
    {"label": "Greedy (temp=0.1, top_k=1)",    "temperature": 0.1,  "top_k": 1,  "top_p": 1.0},
    {"label": "Conservative (temp=0.5, top_k=5)", "temperature": 0.5, "top_k": 5,  "top_p": 1.0},
    {"label": "Balanced (temp=0.8, top_k=10)", "temperature": 0.8,  "top_k": 10, "top_p": 1.0},
    {"label": "Creative (temp=1.0, top_k=20)", "temperature": 1.0,  "top_k": 20, "top_p": 1.0},
    {"label": "Top-p (temp=0.9, p=0.9)",       "temperature": 0.9,  "top_k": 0,  "top_p": 0.9},
]

# Top-p nucleus sampling desteği için generate'i genişlet
def generate_extended(
    model: FluidLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 120,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    adaptive: bool = True,
) -> str:
    model.eval()
    ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ids_in = ids[:, -model.max_seq_len:]
            logits  = model(ids_in, adaptive=adaptive)
            next_l  = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                topk_vals, _ = torch.topk(next_l, min(top_k, next_l.size(-1)))
                next_l = next_l.masked_fill(next_l < topk_vals[:, -1:], -1e9)

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_l, sorted_idx = torch.sort(next_l, descending=True)
                cum_p = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
                # Remove tokens after cumulative prob exceeds top_p
                sorted_l[cum_p - torch.softmax(sorted_l, dim=-1) > top_p] = -1e9
                next_l = torch.zeros_like(next_l).scatter_(1, sorted_idx, sorted_l)

            probs   = torch.softmax(next_l, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids     = torch.cat([ids, next_id], dim=1)

            if next_id.item() == EOS_ID:
                break

    generated = ids[0].tolist()
    return decode(generated)

prompts = [
    "first citizen:",
    "all:\nspeak",
    "second citizen:\nconsider you",
    "what he cannot",
]

for prompt in prompts:
    prompt_ids = torch.tensor([encode(prompt)[:SEQ_LEN]])
    print(f"\n{'─'*65}")
    print(f"PROMPT: {repr(prompt)}")
    print(f"{'─'*65}")
    for cfg in generation_configs:
        out = generate_extended(
            model, prompt_ids,
            max_new_tokens=100,
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
            adaptive=True,
        )
        # Çıktıyı 80 karla sınırla (konsol için)
        display = out[:120].replace("\n", "↵")
        print(f"  [{cfg['label']:<35}] {display}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ÖĞRENILEN FİZİKSEL PARAMETRELER
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("ÖĞRENILEN FİZİKSEL PARAMETRELER")
print("═" * 70)
phys = model.physical_params()
print(f"  {'Katman':<8} {'ν':<10} {'Δt':<10} {'α':<10} {'p_scale'}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for i, (nu_v, dt_v, al_v, ps_v) in enumerate(zip(phys["nu"], phys["dt"], phys["alpha"], phys["p_scale"])):
    print(f"  {i:<8} {nu_v:<10.4f} {dt_v:<10.4f} {al_v:<10.4f} {ps_v:<10.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("FluidLM — Tam Eğitim ve Analiz Sonuçları", fontsize=14, fontweight="bold")

epochs_x = range(1, EPOCHS + 1)

# 1. Kayıp eğrisi
ax = axes[0, 0]
ax.plot(epochs_x, train_losses, label="Train loss", color="steelblue",   lw=1.5)
ax.plot(epochs_x, val_losses,   label="Val loss",   color="tomato",      lw=1.5, ls="--")
ax.set_title("Eğitim / Validasyon Kaybı")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.legend()
ax.grid(alpha=0.3)

# 2. Perplexity
ax = axes[0, 1]
ax.semilogy(epochs_x, val_ppls, color="purple", lw=1.5)
ax.set_title("Validasyon Perplexity (log ölçek)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Perplexity")
ax.grid(alpha=0.3, which="both")

# 3. Öğrenilen ν
ax = axes[0, 2]
ax.plot(phys["nu"],    marker="o", color="steelblue", lw=2, label="ν (viskozite)")
ax.set_title("Katman başına öğrenilen ν\n(regularizasyon gücü)")
ax.set_xlabel("Katman")
ax.set_ylabel("ν")
ax.grid(alpha=0.3)

# 4. Öğrenilen Δt
ax = axes[1, 0]
ax.plot(phys["dt"], marker="s", color="seagreen", lw=2, label="Δt")
ax.set_title("Katman başına öğrenilen Δt\n(evrim hızı)")
ax.set_xlabel("Katman")
ax.set_ylabel("Δt")
ax.grid(alpha=0.3)

# 5. Öğrenilen α
ax = axes[1, 1]
ax.plot(phys["alpha"], marker="^", color="darkorange", lw=2)
ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="Başlangıç α=1")
ax.set_title("Katman başına öğrenilen α\n(basınç etkileşim mesafesi)")
ax.set_xlabel("Katman")
ax.set_ylabel("α")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 6. Adaptif derinlik: son val batch üzerinde dağılım
ax = axes[1, 2]
model.eval()
all_steps = []
val_b2 = make_batches(train_ids, SEQ_LEN, BATCH_SIZE, shuffle=False)[:20]
with torch.no_grad():
    for x, _ in val_b2:
        _, stats = model(x, adaptive=True, return_stats=True)
        all_steps.append(stats["n_steps"])

ax.hist(all_steps, bins=range(1, model.n_layers + 2), color="mediumorchid",
        edgecolor="white", rwidth=0.85, align="left")
ax.set_title("Adaptif Derinlik Dağılımı\n(kaç batch kaç katmanda bitti)")
ax.set_xlabel("Kullanılan katman sayısı")
ax.set_ylabel("Frekans")
ax.set_xticks(range(1, model.n_layers + 1))
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "04_full_train_result.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nGrafik kaydedildi: {out_path}")
print("\nDeney tamamlandı.")
