# ============================================================
# experiments/19_doc_length_analysis.py
#
# OpenWebText belge uzunluklarını analiz eder.
# Amaç: receptive field seçimini veri dağılımına göre yapmak.
#
# Çıktı:
#   results/19_doc_length_stats.json
#
# Kullanım:
#   python experiments/19_doc_length_analysis.py
#   python experiments/19_doc_length_analysis.py --n_docs 50000
# ============================================================

import sys, os, json, math, argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--n_docs",   type=int, default=20_000,
                    help="Kaç belge analiz edilsin (default: 20000)")
parser.add_argument("--split",    default="train")
parser.add_argument("--out_dir",  default=None)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
out_dir    = args.out_dir or os.path.join(REPO_ROOT, "results")
os.makedirs(out_dir, exist_ok=True)

import tiktoken
ENC = tiktoken.get_encoding("gpt2")
EOS_ID = ENC.encode_single_token("<|endoftext|>")   # 50256

print(f"[Setup] tiktoken gpt2  vocab={ENC.n_vocab}  EOS={EOS_ID}")

from datasets import load_dataset
print(f"[Data] OpenWebText yukleniyor (streaming, ilk {args.n_docs} belge)...")
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=10_000)

lengths = []
for i, row in enumerate(ds):
    if i >= args.n_docs:
        break
    toks = ENC.encode_ordinary(row["text"])
    lengths.append(len(toks))
    if i % 2000 == 0:
        print(f"  {i:6d}/{args.n_docs}  son belge={lengths[-1]} token", flush=True)

lengths.sort()
N = len(lengths)

def percentile(p):
    idx = int(p / 100 * N)
    return lengths[min(idx, N-1)]

mean   = sum(lengths) / N
stdev  = math.sqrt(sum((x - mean)**2 for x in lengths) / N)

print(f"\n{'='*55}")
print(f"  OWT Belge Uzunluğu Dağılımı  (n={N:,} belge)")
print(f"{'='*55}")
print(f"  min        : {lengths[0]:6d} token")
print(f"  p5         : {percentile(5):6d} token")
print(f"  p10        : {percentile(10):6d} token")
print(f"  p25        : {percentile(25):6d} token")
print(f"  medyan     : {percentile(50):6d} token")
print(f"  ortalama   : {mean:6.0f} token  (±{stdev:.0f})")
print(f"  p75        : {percentile(75):6d} token")
print(f"  p90        : {percentile(90):6d} token")
print(f"  p95        : {percentile(95):6d} token")
print(f"  p99        : {percentile(99):6d} token")
print(f"  max        : {lengths[-1]:6d} token")
print(f"{'='*55}")

# Receptive field eşleşmesi
print(f"\n  [RF — Kapsama Analizi]")
print(f"  RF ile kıyasla: bir RF penceresi ortalama belgenin ne kadarını görür?")
n_layers = 12
for k in [1, 4, 8, 16, 32, 64]:
    rf = k * n_layers
    # Kaç belge RF'den kısadır (RF > doc → konu sınırı aşılır)?
    n_longer = sum(1 for l in lengths if l > rf)
    pct_within = 100.0 * (N - n_longer) / N   # belgelerin yüzde kaçı RF içine sığar
    print(f"  k={k:3d}  RF={rf:4d} tok  → "
          f"belgelerin %{pct_within:5.1f}'i bu pencereye sığar  "
          f"({N - n_longer:,}/{N:,})")

print()
print(f"  Yorumlar:")
print(f"  - RF < medyan → model çoğu belgeyi 'tam' göremez (konu sınırı aşılır)")
print(f"  - Ideal RF ≈ p75 belge uzunluğu: belgelerin %75'i tek pencerede kalır")
ideal_rf = percentile(75)
ideal_k  = math.ceil(ideal_rf / n_layers)
print(f"  - p75={ideal_rf} tok → ideal k ≈ {ideal_k} (RF={ideal_k*n_layers})")
print(f"  - p90={percentile(90)} tok → agresif k ≈ {math.ceil(percentile(90)/n_layers)}")

# Histogram (bucket: 0-64, 64-128, ..., 1024+)
buckets = [0, 64, 128, 256, 384, 512, 768, 1024, 2048, 4096, 999999]
counts  = Counter()
for l in lengths:
    for i in range(len(buckets) - 1):
        if buckets[i] <= l < buckets[i+1]:
            counts[i] += 1
            break
print(f"\n  [Histogram]")
print(f"  {'Aralık':<18} {'Belge':>8} {'%':>6}  bar")
for i in range(len(buckets) - 1):
    lo   = buckets[i]; hi = buckets[i+1]
    label = f"{lo}-{hi}" if hi < 999999 else f"{lo}+"
    c    = counts[i]
    pct  = 100.0 * c / N
    bar  = "█" * int(pct / 2)
    print(f"  {label:<18} {c:>8,} {pct:>5.1f}%  {bar}")

stats = {
    "n_docs": N, "mean": round(mean, 1), "stdev": round(stdev, 1),
    "min": lengths[0], "max": lengths[-1],
    "p5": percentile(5),   "p10": percentile(10),
    "p25": percentile(25), "p50": percentile(50),
    "p75": percentile(75), "p90": percentile(90),
    "p95": percentile(95), "p99": percentile(99),
    "rf_coverage": {
        f"k{k}_rf{k*n_layers}": round(100.0 * sum(1 for l in lengths if l <= k*n_layers) / N, 2)
        for k in [1, 4, 8, 16, 32, 64]
    },
    "ideal_k_p75": ideal_k,
    "ideal_rf_p75": ideal_k * n_layers,
}
out_path = os.path.join(out_dir, "19_doc_length_stats.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print(f"\n[JSON] Kaydedildi: {out_path}")
