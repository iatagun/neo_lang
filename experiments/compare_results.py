"""
Checkpoint'lerden val_loss okuyup karşılaştırma tablosu yazdırır.
Eğitim yapmaz.

Kullanım:
    python experiments/compare_results.py
    python experiments/compare_results.py --dir experiments   # farklı klasör
"""
import os, math, argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default=None, help="checkpoint klasörü (varsayılan: otomatik arar)")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

# Aranacak klasörler
SEARCH_DIRS = [
    args.dir,
    os.path.join(REPO_ROOT, "checkpoints"),   # önce checkpoints/
    SCRIPT_DIR,
    os.path.join(REPO_ROOT, "experiments"),
    "/content/neo_lang/checkpoints",
    "/content/neo_lang/experiments",
]

def find_ckpt(name: str) -> str | None:
    for d in SEARCH_DIRS:
        if d and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None

def read_ppl(path: str) -> float:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    val_loss = ck.get("val_loss") or ck.get("best_val_loss")
    if val_loss is None:
        raise KeyError(f"val_loss anahtarı bulunamadı: {path}")
    return math.exp(min(float(val_loss), 20))

MODELS = [
    ("FluidLM  (NS, d=1024 L=16)",   "~135M", "07_best_model.pt"),
    ("GPT-A    (d=1024 L=16 h=16)",  "~202M", "09_gpt_A_best.pt"),
    ("GPT-B    (d=768  L=12 h=12)",  " ~85M", "09_gpt_B_best.pt"),
]

print(f"\n{'='*60}")
print("  KARŞILAŞTIRMA TABLOSU  (Tiny Shakespeare, char-level)")
print(f"{'='*60}")
print(f"  {'Model':<35} {'Param':>6}  {'Val PPL':>8}")
print("  " + "-"*54)

found_any = False
for label, params, fname in MODELS:
    path = find_ckpt(fname)
    if path:
        try:
            ppl = read_ppl(path)
            print(f"  {label:<35} {params:>6}  {ppl:>8.2f}")
            found_any = True
        except Exception as e:
            print(f"  {label:<35}  HATA: {e}")
    else:
        print(f"  {label:<35} {params:>6}  {'(henüz yok)':>8}")

print(f"{'='*60}\n")

if not found_any:
    print("Hiç checkpoint bulunamadı.")
    print("Beklenen dosyalar:", [m[2] for m in MODELS])
