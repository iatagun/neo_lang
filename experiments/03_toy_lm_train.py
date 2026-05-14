"""
experiments/03_toy_lm_train.py
================================
Deney 3: FluidLM'i Karakter Düzeyinde Eğitme
----------------------------------------------

Küçük bir karakter-düzeyi dil modeli eğitiyoruz.
Veri seti: basit tekrarlayan desenler ve İngilizce kelimeler.

Bu deneyin amacı:
  1. FluidLM'in gradyan akışını (eğitilebilirliğini) doğrulamak
  2. Kayıp eğrisini izlemek
  3. Adaptif derinlik istatistiklerini görmek
  4. Basit bir üretim örneği çalıştırmak

Çalıştırmak için:
    cd neo_lang
    python experiments/03_toy_lm_train.py

Not: Bu bir "proof of concept" eğitimidir.  Gerçek bir dil modeli için
     çok daha büyük veri, daha uzun eğitim ve hiper-parametre araması gerekir.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple

from src.fluid_lm import FluidLM

torch.manual_seed(0)
np.random.seed(0)

# ─── Veri seti ─────────────────────────────────────────────────────────────
# Karakter düzeyinde küçük bir metin.
# Transformer ve FluidLM arasındaki farkı görmek için yeterince büyük.

TEXT = """
the quick brown fox jumps over the lazy dog
a stitch in time saves nine
all that glitters is not gold
to be or not to be that is the question
the cat sat on the mat
peter piper picked a peck of pickled peppers
she sells seashells by the seashore
how much wood would a woodchuck chuck
the rain in spain stays mainly in the plain
jack and jill went up the hill to fetch a pail of water
humpty dumpty sat on a wall humpty dumpty had a great fall
twinkle twinkle little star how i wonder what you are
row row row your boat gently down the stream
mary had a little lamb its fleece was white as snow
baa baa black sheep have you any wool yes sir yes sir three bags full
""".strip()

# ─── Vocabulary ─────────────────────────────────────────────────────────────

chars   = sorted(set(TEXT))
vocab   = ["<pad>", "<bos>", "<eos>"] + chars
stoi    = {c: i for i, c in enumerate(vocab)}
itos    = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(vocab)
PAD_ID = stoi["<pad>"]
BOS_ID = stoi["<bos>"]
EOS_ID = stoi["<eos>"]

print("=" * 65)
print("FluidLM — Karakter Düzeyi Dil Modeli Eğitimi")
print("=" * 65)
print(f"Metin uzunluğu   : {len(TEXT)} karakter")
print(f"Vocabulary boyutu: {VOCAB_SIZE} token")
print(f"Vocab: {''.join(chars[:30])}{'...' if len(chars) > 30 else ''}")

# ─── Veri hazırlama ─────────────────────────────────────────────────────────

def encode(text: str) -> List[int]:
    return [BOS_ID] + [stoi[c] for c in text if c in stoi] + [EOS_ID]

def make_batches(
    text: str,
    seq_len: int = 32,
    batch_size: int = 16,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Sliding-window batches for next-character prediction."""
    ids = encode(text)
    chunks = []
    for i in range(0, len(ids) - seq_len - 1, seq_len // 2):   # 50% overlap
        chunk = ids[i : i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            chunks.append(chunk)

    # Shuffle and batch
    np.random.shuffle(chunks)
    batches = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        batch = chunks[i : i + batch_size]
        t     = torch.tensor(batch, dtype=torch.long)   # [B, seq_len+1]
        x     = t[:, :-1]                               # [B, seq_len]   input
        y     = t[:, 1:]                                # [B, seq_len]   target
        batches.append((x, y))
    return batches

SEQ_LEN    = 32
BATCH_SIZE = 16
batches    = make_batches(TEXT, SEQ_LEN, BATCH_SIZE)
print(f"Toplam batch     : {len(batches)}  (her biri {BATCH_SIZE}×{SEQ_LEN})")

# ─── Model ─────────────────────────────────────────────────────────────────

model = FluidLM(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    n_layers=8,
    max_seq_len=SEQ_LEN + 4,
    nu=0.05,
    dt=0.10,
    integrator="euler",          # Euler — hızlı prototipler için
    convergence_threshold=1e-4,
    min_steps=3,
    dropout=0.1,
)

n_params = model.num_parameters()
print(f"\nModel parametre sayısı : {n_params:,}")
print(f"  (d_model={model.d_model}, n_layers={model.n_layers})")

# Karşılaştırma için basit Transformer boyutunu hesapla
d = model.d_model
# 1 Transformer katmanı ≈ 4*d² (MHA) + 8*d² (FFN) = 12*d²
n_transformer_equiv = 12 * d * d * model.n_layers + 2 * VOCAB_SIZE * d
print(f"Eşdeğer Transformer    : ~{n_transformer_equiv:,} parametre")
print(f"Oran FluidLM/Transformer: {n_params / n_transformer_equiv:.3f}")

# ─── Eğitim ────────────────────────────────────────────────────────────────

EPOCHS     = 40
LR         = 3e-3
CLIP_NORM  = 1.0

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses: List[float] = []
layer_usage:  List[float] = []   # ortalama kullanılan katman sayısı

print(f"\nEğitim başlıyor: {EPOCHS} epoch, lr={LR}")
print("-" * 55)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss    = 0.0
    epoch_steps   = 0.0
    n_batches     = 0

    for x, y in batches:
        optimizer.zero_grad()

        # Forward
        logits, stats = model(x, adaptive=False, return_stats=True)

        # Cross-entropy loss (next token prediction)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            ignore_index=PAD_ID,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        epoch_loss  += loss.item()
        epoch_steps += stats["n_steps"]
        n_batches   += 1

    scheduler.step()

    avg_loss  = epoch_loss  / n_batches
    avg_steps = epoch_steps / n_batches
    train_losses.append(avg_loss)
    layer_usage.append(avg_steps)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3} | kayıp={avg_loss:.4f} | "
              f"ortalama katman={avg_steps:.1f} | "
              f"lr={scheduler.get_last_lr()[0]:.5f}")

print("-" * 55)
print("Eğitim tamamlandı.")

# ─── Adaptif derinlik testi ────────────────────────────────────────────────

print("\nAdaptif derinlik testi (evaluate mode):")
model.eval()

test_phrases = [
    "the cat",          # basit cümle
    "to be or not to",  # karmaşık cümle
    "a",                # tek karakter
]

for phrase in test_phrases:
    ids = torch.tensor([encode(phrase)[:-1]])   # BOS + chars (no EOS)
    _, stats = model(ids, adaptive=True, return_stats=True)
    print(f"  '{phrase}' → {stats['n_steps']} katman kullanıldı  "
          f"(son ΔKE={stats['delta_kes'][-1]:.6f})")

# ─── Örnek üretim ─────────────────────────────────────────────────────────

print("\nÖrnek üretim (temperature=0.8, top_k=5):")
model.eval()
prompt     = "the "
prompt_ids = torch.tensor([encode(prompt)[:-1]])   # BOS + chars

with torch.no_grad():
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=5,
        adaptive=True,
    )

generated_chars = [
    itos.get(i.item(), "?")
    for i in generated_ids[0]
    if i.item() not in (PAD_ID, BOS_ID, EOS_ID)
]
print(f"  Prompt: '{prompt}'")
print(f"  Üretim: {''.join(generated_chars)}")

# ─── Öğrenilen fiziksel parametreler ──────────────────────────────────────

print(f"\nÖğrenilen fiziksel parametreler (nu, dt, alpha) per katman:")
phys = model.physical_params()
print(f"  {'Katman':<8} {'ν (viskozite)':<18} {'Δt (zaman adımı)':<18} {'α (etkileşim mesafesi)':<22}")
print(f"  {'-'*8} {'-'*18} {'-'*18} {'-'*22}")
for i, (nu_val, dt_val, al_val) in enumerate(zip(phys["nu"], phys["dt"], phys["alpha"])):
    print(f"  {i:<8} {nu_val:<18.6f} {dt_val:<18.6f} {al_val:<22.6f}")

print("\nGözlem: Erken katmanlar daha büyük α (kısa mesafeli) kullanabilir,")
print("        geç katmanlar daha küçük α (uzun mesafeli global etki).")

# ─── Görselleştirme ────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("FluidLM Eğitim Sonuçları", fontsize=12, fontweight="bold")

# Kayıp eğrisi
ax = axes[0]
ax.plot(range(1, EPOCHS + 1), train_losses, color="steelblue", lw=2)
ax.set_title("Eğitim Kaybı")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.grid(alpha=0.3)

# Öğrenilen ν ve Δt
ax = axes[1]
x_layers = range(len(phys["nu"]))
ax.plot(x_layers, phys["nu"], label="ν (viskozite)", color="tomato",   lw=2, marker="o")
ax.plot(x_layers, phys["dt"], label="Δt (zaman adımı)", color="green", lw=2, marker="s")
ax.set_title("Öğrenilen Fiziksel Parametreler")
ax.set_xlabel("Katman index")
ax.legend()
ax.grid(alpha=0.3)

# Vocabulary üzerindeki son tahmin dağılımı
ax = axes[2]
model.eval()
sample_ids = torch.tensor([encode("the quick brown fox")[:SEQ_LEN]])
with torch.no_grad():
    logits = model(sample_ids)
probs = torch.softmax(logits[0, -1, :], dim=-1).numpy()
top10_idx = np.argsort(probs)[-10:][::-1]
top10_probs = probs[top10_idx]
top10_labels = [repr(itos.get(i, "?")) for i in top10_idx]
ax.barh(top10_labels[::-1], top10_probs[::-1], color="mediumpurple")
ax.set_title("Son token olasılıkları\n('the quick brown fox' → sonraki karakter)")
ax.set_xlabel("Olasılık")

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "03_training_result.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nGrafik kaydedildi: {out_path}")
print("\nDeney tamamlandı.")
