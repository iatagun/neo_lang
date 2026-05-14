"""
experiments/05_test_checkpoint.py
===================================
05_best_model.pt checkpoint'ini yükleyip interaktif metin üretimi yapar.
Eğitim devam ederken ayrı bir terminalde çalıştırılabilir.

Kullanım:
    cd neo_lang
    python experiments/05_test_checkpoint.py
    python experiments/05_test_checkpoint.py --prompt "HAMLET:" --temp 0.8 --topk 10 --n 300
"""

import sys, os, argparse
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math

from fluidlm.fluid_lm import FluidLM

# ── Argümanlar ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",   default=None,        help="Checkpoint yolu (.pt)")
parser.add_argument("--prompt", default=None,        help="Başlangıç metni (None=interaktif)")
parser.add_argument("--temp",   type=float, default=0.8,  help="Sıcaklık (varsayılan 0.8)")
parser.add_argument("--topk",   type=int,   default=10,   help="Top-k örnekleme (0=kapalı)")
parser.add_argument("--n",      type=int,   default=300,  help="Üretilecek token sayısı")
args = parser.parse_args()

# ── Checkpoint yolu ───────────────────────────────────────────────────────────
CKPT_PATH = args.ckpt or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "05_best_model.pt"
)

if not os.path.exists(CKPT_PATH):
    print(f"HATA: Checkpoint bulunamadı: {CKPT_PATH}")
    print("Eğitim henüz tamamlanmadı ya da ilk val iyileşmesi olmadı.")
    sys.exit(1)

# ── Cihaz ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── Checkpoint yükle ──────────────────────────────────────────────────────────
print(f"Checkpoint yukleniyor: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)

saved_epoch  = ckpt.get("epoch", "?")
saved_val    = ckpt.get("val_loss", float("nan"))
saved_ppl    = math.exp(min(saved_val, 20))
vocab_size   = ckpt["vocab_size"]
d_model      = ckpt.get("d_model", 256)
n_layers     = ckpt.get("n_layers", 6)
max_seq_len  = ckpt.get("max_seq_len", 132)
stoi         = ckpt["stoi"]
itos         = ckpt["itos"]

PAD_ID = stoi.get("<pad>", 0)
BOS_ID = stoi.get("<bos>", 1)
EOS_ID = stoi.get("<eos>", 2)

print(f"  Epoch       : {saved_epoch}")
print(f"  Val loss    : {saved_val:.4f}")
print(f"  Val PPL     : {saved_ppl:.2f}")
print(f"  Vocab size  : {vocab_size}")
print(f"  d_model     : {d_model}  n_layers: {n_layers}")
print(f"  Cihaz       : {DEVICE}")

# ── Modeli kur ve yükle ───────────────────────────────────────────────────────
model = FluidLM(
    vocab_size            = vocab_size,
    d_model               = d_model,
    n_layers              = n_layers,
    max_seq_len           = max_seq_len,
    nu                    = 0.01,
    dt                    = 0.05,
    alpha                 = 1.0,
    integrator            = "euler",
    convergence_threshold = 0.01,
    min_steps             = 3,
    dropout               = 0.0,   # inference'da dropout kapalı
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"  Parametreler: {model.num_parameters():,}\n")

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────
def encode(text):
    return [BOS_ID] + [stoi[c] for c in text if c in stoi]

def decode(ids):
    return "".join(itos.get(i, "?") for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID))

@torch.no_grad()
def generate(prompt_text, max_new=300, temperature=0.8, top_k=10):
    ids = torch.tensor([encode(prompt_text)], dtype=torch.long, device=DEVICE)
    ids = ids[:, -max_seq_len:]
    for _ in range(max_new):
        logits  = model(ids, adaptive=False)         # inference: tum katmanlar
        next_l  = logits[0, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            topk_v, _ = torch.topk(next_l, min(top_k, next_l.size(-1)))
            next_l = next_l.masked_fill(next_l < topk_v[-1], -1e9)
        probs = torch.softmax(next_l, dim=-1)
        nid   = torch.multinomial(probs, 1).unsqueeze(0)
        ids   = torch.cat([ids, nid], dim=1)[:, -max_seq_len:]
        if nid.item() == EOS_ID:
            break
    return prompt_text + decode(ids[0].tolist())

# ── Fiziksel parametreler özeti ───────────────────────────────────────────────
phys = model.physical_params()
print("=" * 60)
print("OGRENILEN FIZIKSEL PARAMETRELER")
print("=" * 60)
print(f"  {'Katman':<8} {'nu':<10} {'dt':<10} {'alpha':<10} {'p_scale'}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for i, (nu_v, dt_v, al_v, ps_v) in enumerate(
        zip(phys["nu"], phys["dt"], phys["alpha"], phys["p_scale"])):
    print(f"  {i:<8} {nu_v:<10.4f} {dt_v:<10.4f} {al_v:<10.4f} {ps_v:<10.4f}")
print()

# ── Val loss hesapla (opsiyonel, veri varsa) ──────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "shakespeare.txt"
)
if os.path.exists(DATA_PATH):
    import numpy as np
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        TEXT = f.read()
    all_ids = [BOS_ID] + [stoi[c] for c in TEXT if c in stoi] + [EOS_ID]
    val_ids = all_ids[int(len(all_ids) * 0.9):]
    SEQ_LEN = max_seq_len - 4

    chunks = []
    for i in range(0, len(val_ids) - SEQ_LEN - 1, SEQ_LEN // 2):
        chunk = val_ids[i : i + SEQ_LEN + 1]
        if len(chunk) == SEQ_LEN + 1:
            chunks.append(chunk)

    BATCH = 64
    v_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(chunks) - BATCH + 1, BATCH):
            t = torch.tensor(chunks[i : i + BATCH], dtype=torch.long, device=DEVICE)
            x, y = t[:, :-1], t[:, 1:]
            logits = model(x, adaptive=False)
            v_loss += F.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=PAD_ID
            ).item()
            n_batches += 1
    if n_batches > 0:
        cur_val  = v_loss / n_batches
        cur_ppl  = math.exp(min(cur_val, 20))
        print(f"Anlık val_loss={cur_val:.4f}  val_ppl={cur_ppl:.2f}  "
              f"(checkpoint: ppl={saved_ppl:.2f})")
        print()

# ── Metin üretimi ─────────────────────────────────────────────────────────────
PROMPTS = [
    ("HAMLET greedy",    "HAMLET:",           0.3,  5),
    ("HAMLET balanced",  "HAMLET:",           0.8, 10),
    ("HAMLET creative",  "HAMLET:",           1.2, 20),
    ("Diyalog",          "OPHELIA:\nTo be,",  0.8, 10),
    ("Act 1",            "ACT I.\nSCENE I.",  0.8, 10),
]

if args.prompt:
    # Komut satırından gelen tek prompt
    print("=" * 60)
    print(f"[PROMPT] {args.prompt!r}  temp={args.temp}  top_k={args.topk}")
    print("=" * 60)
    out = generate(args.prompt, max_new=args.n, temperature=args.temp, top_k=args.topk)
    print(out)
else:
    # Tüm hazır promptları çalıştır
    for label, prompt, temp, topk in PROMPTS:
        print("=" * 60)
        print(f"[{label}]  temp={temp}  top_k={topk}")
        print("=" * 60)
        out = generate(prompt, max_new=args.n, temperature=temp, top_k=topk)
        print(out[:500])
        print()

print("\nTest tamamlandi.")
