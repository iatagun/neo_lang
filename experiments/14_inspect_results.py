# ============================================================
# Deney 14 — Sonuç İnceleyici
# experiments/14_inspect_results.py
#
# Eğitim bitmeden veya bittikten sonra çalıştırılır.
# results/14_industrial_compare.json okunur, her şey yazdırılır.
#
# Kullanım:
#   python experiments/14_inspect_results.py
#   python experiments/14_inspect_results.py --json results/14_industrial_compare.json
#   python experiments/14_inspect_results.py --plot   (PNG tekrar üretir)
# ============================================================

import sys, os, json, math, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json", default=None, help="JSON dosyası yolu")
parser.add_argument("--plot", action="store_true", help="PNG üret")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
JSON_PATH  = args.json or os.path.join(REPO_ROOT, "results", "14_industrial_compare.json")

# ─── Renk kodları (terminal) ──────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def bold(s):   return f"{BOLD}{s}{RESET}"
def green(s):  return f"{GREEN}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"

# ─── JSON yükle ───────────────────────────────────────────────────────────────
if not os.path.exists(JSON_PATH):
    print(red(f"JSON bulunamadı: {JSON_PATH}"))
    print("Eğitim henüz başlamadı veya farklı dizinde kaydedildi.")
    sys.exit(1)

with open(JSON_PATH, encoding="utf-8") as f:
    all_results = json.load(f)

print(bold(f"\n{'='*68}"))
print(bold(f"  FluidLM vs GPT — Industrial Results ({len(all_results)} run yüklendi)"))
print(bold(f"{'='*68}"))
print(f"  Kaynak: {JSON_PATH}\n")

# ─── Her run için özet ────────────────────────────────────────────────────────
for r in all_results:
    mt    = r["model_type"].upper()
    scale = r["scale"]
    seed  = r["seed"]
    ppl   = r.get("best_val_ppl", -1)
    wt    = r.get("wt103_ppl", -1)
    tput  = r.get("throughput_tps", 0)
    tok   = r.get("tokens_seen", 0)
    pi    = r.get("param_info", {})
    fi    = r.get("flops_info", {})
    phys  = r.get("physical_params")

    color = CYAN if r["model_type"] == "fluid" else YELLOW
    tag   = f"{color}{bold(mt+'-'+scale)}{RESET}"

    print(f"  {tag}  seed={seed}  tokens={tok/1e9:.2f}B")
    print(f"    Val PPL  : {bold(f'{ppl:.4f}') if ppl > 0 else red('henüz yok')}")
    if wt > 0:
        print(f"    WT-103   : {wt:.4f}")
    print(f"    Params   : {pi.get('total',0)/1e6:.1f}M toplam  "
          f"routing={pi.get('routing',0):,}  mlp={pi.get('mlp',0)/1e6:.1f}M")
    print(f"    MFLOP/tok: {fi.get('total_mflop',0):.1f}  "
          f"({fi.get('routing_complexity','')})")
    print(f"    Throughput: {tput/1e3:.0f}K tok/s")

    # Fiziksel parametreler (FluidLM only)
    if phys:
        nu      = phys["nu"]
        dt      = phys["dt"]
        alpha   = phys["alpha"]
        p_scale = phys["p_scale"]
        n       = len(nu)
        nu_early = sum(nu[:4]) / 4
        nu_late  = sum(nu[-4:]) / 4
        grad     = nu[-1] - nu[0]
        grad_str = green(f"+{grad:.4f}") if grad > 0 else red(f"{grad:.4f}")
        print(f"    ν early(0-3)  : {nu_early:.4f}")
        print(f"    ν late ({n-4}-{n-1}) : {nu_late:.4f}")
        print(f"    ν gradient    : {grad_str}  "
              f"range=[{min(nu):.4f}, {max(nu):.4f}]")
        print(f"    α range       : [{min(alpha):.3f}, {max(alpha):.3f}]")
        print(f"    p_scale range : [{min(p_scale):.3f}, {max(p_scale):.3f}]")

    # Eğitim geçmişi özeti
    hist = r.get("train_history", [])
    if hist:
        first = hist[0]
        last  = hist[-1]
        best_step = min(hist, key=lambda h: h.get("val_ppl", 9999))
        print(f"    Eğitim adımı : {last['step']:,}  "
              f"(best @ step {best_step['step']:,})")
        if len(hist) >= 2:
            # PPL düşüş trendi
            ppls = [h["val_ppl"] for h in hist if "val_ppl" in h]
            drop = ppls[0] - ppls[-1]
            print(f"    PPL yolculuğu: {ppls[0]:.2f} → {ppls[-1]:.2f}  "
                  f"(Δ={drop:+.2f})")

    print()

# ─── Karşılaştırma tablosu ────────────────────────────────────────────────────
print(bold(f"{'─'*68}"))
print(bold("  KARŞILAŞTIRMA TABLOSU"))
print(bold(f"{'─'*68}"))
print(f"  {'Model':<14} {'Scale':<6} {'Seed':<5} "
      f"{'Val PPL':>9} {'WT-103':>8} "
      f"{'MFLOP/tok':>10} {'RoutingP':>11}")
print(f"  {'─'*14} {'─'*6} {'─'*5} "
      f"{'─'*9} {'─'*8} "
      f"{'─'*10} {'─'*11}")

for r in all_results:
    ppl = r.get("best_val_ppl", -1)
    wt  = r.get("wt103_ppl", -1)
    fi  = r.get("flops_info", {})
    pi  = r.get("param_info", {})
    ppl_str = f"{ppl:.4f}" if ppl > 0 else "—"
    wt_str  = f"{wt:.2f}"  if wt  > 0 else "—"
    color   = CYAN if r["model_type"] == "fluid" else YELLOW
    model_str = f"{color}{r['model_type'].upper():<14}{RESET}"
    print(f"  {model_str} {r['scale']:<6} {r['seed']:<5} "
          f"{ppl_str:>9} {wt_str:>8} "
          f"{fi.get('total_mflop',0):>10.1f} "
          f"{pi.get('routing',0):>11,}")

# ─── RQ analizi ───────────────────────────────────────────────────────────────
from collections import defaultdict
groups = defaultdict(list)
for r in all_results:
    groups[f"{r['model_type']}_{r['scale']}"].append(r)

print(bold(f"\n{'─'*68}"))
print(bold("  ARAŞTIRMA SORULARI (mevcut veriyle)"))
print(bold(f"{'─'*68}"))

for scale in ["S", "M"]:
    fluid_runs = groups.get(f"fluid_{scale}", [])
    gpt_runs   = groups.get(f"gpt_{scale}",   [])

    if not fluid_runs:
        print(f"\n  [{scale} scale] — henüz sonuç yok")
        continue

    fluid_ppls = [r["best_val_ppl"] for r in fluid_runs if r["best_val_ppl"] > 0]
    gpt_ppls   = [r["best_val_ppl"] for r in gpt_runs   if r["best_val_ppl"] > 0]

    def mean_std(vals):
        if not vals: return None, None
        mu  = sum(vals) / len(vals)
        std = math.sqrt(sum((v-mu)**2 for v in vals)/len(vals)) if len(vals)>1 else 0
        return mu, std

    mu_f, std_f = mean_std(fluid_ppls)
    mu_g, std_g = mean_std(gpt_ppls)

    print(f"\n  [{bold(scale+' scale')}]  "
          f"FluidLM: {len(fluid_runs)} run  GPT: {len(gpt_runs)} run")

    # RQ3 — emergent ν  (eğitim loguyla tutarlı: late_avg(8-11) - early_avg(0-3))
    nu_grads = []
    for r in fluid_runs:
        p = r.get("physical_params")
        if p:
            nu   = p["nu"]
            n    = len(nu)
            early = sum(nu[:4]) / 4
            late  = sum(nu[max(0, n-4):]) / 4
            g     = late - early
            nu_grads.append(g)
    if nu_grads:
        avg_grad = sum(nu_grads) / len(nu_grads)
        # Tutarlılık: tüm seedler aynı yönde (hepsi pozitif VEYA hepsi negatif)
        consistent = all(g > 0 for g in nu_grads) or all(g < 0 for g in nu_grads)
        direction  = "erken>geç (yüksek→düşük ν)" if avg_grad < 0 else "erken<geç (düşük→yüksek ν)"
        verdict = green(f"✓ DOĞRULANDI ({direction})") if consistent else yellow("~ TUTARSIZ (seedler ters yönde)")
        print(f"  RQ3 ν gradyanı : {verdict}  "
              f"ort. Δν={avg_grad:+.4f}  "
              f"seeds={[f'{g:+.4f}' for g in nu_grads]}")

    if mu_f is None:
        print(f"  RQ1/RQ2        : FluidLM PPL henüz yok")
        continue

    print(f"  FluidLM PPL    : {mu_f:.4f}" +
          (f" ± {std_f:.4f}" if std_f else "") +
          (f"  ({len(fluid_ppls)} seed)" if len(fluid_ppls)>1 else ""))

    if mu_g is None:
        print(f"  GPT PPL        : henüz yok")
        continue

    print(f"  GPT PPL        : {mu_g:.4f}" +
          (f" ± {std_g:.4f}" if std_g else "") +
          (f"  ({len(gpt_ppls)} seed)" if len(gpt_ppls)>1 else ""))

    delta = mu_f - mu_g
    delta_str = (red if delta > 1.0 else yellow if delta > 0.5
                 else green)(f"{delta:+.4f}")
    print(f"  ΔPPL (F−G)     : {delta_str}")

    # RQ1 verdict
    if delta < 0.3:
        rq1 = green("✓ GÜÇLÜ — NS routing MHA ile kıyaslanabilir")
    elif delta < 0.7:
        rq1 = yellow("~ ORTA — küçük fark, parametre verimliliği savunulabilir")
    else:
        rq1 = red("✗ ZAYIF — anlamlı PPL farkı var")
    print(f"  RQ1 verdict    : {rq1}")

    # RQ2 FLOP karşılaştırması
    if fluid_runs and gpt_runs:
        fi_f = fluid_runs[0].get("flops_info", {})
        fi_g = gpt_runs[0].get("flops_info", {})
        pi_f = fluid_runs[0].get("param_info", {})
        pi_g = gpt_runs[0].get("param_info", {})
        if fi_f and fi_g:
            flop_ratio = fi_g.get("total_mflop",1) / max(fi_f.get("total_mflop",1), 1)
            rp_ratio   = pi_g.get("routing",1)     / max(pi_f.get("routing",1),     1)
            tput_f = sum(r.get("throughput_tps",0) for r in fluid_runs) / len(fluid_runs)
            tput_g = sum(r.get("throughput_tps",0) for r in gpt_runs)   / len(gpt_runs)
            tput_ratio = tput_f / max(tput_g, 1)
            print(f"  RQ2 FLOP ratio : GPT/FluidLM = {green(f'{flop_ratio:.2f}×')}")
            print(f"  RQ2 RoutingP   : {green(f'{rp_ratio:,.0f}×')} daha az  "
                  f"({pi_f.get('routing',0):,} vs {pi_g.get('routing',0):,})")
            if tput_f > 0 and tput_g > 0:
                tc = green if tput_ratio > 1 else red
                print(f"  RQ2 Throughput : FluidLM {tc(f'{tput_ratio:.2f}×')} "
                      f"({'daha hızlı' if tput_ratio>1 else 'daha yavaş'})")

# ─── Son durum ────────────────────────────────────────────────────────────────
done  = [r for r in all_results if r.get("best_val_ppl", -1) > 0]
total_expected = 8  # S×3seed×2model + M×1seed×2model

print(bold(f"\n{'─'*68}"))
print(f"  Tamamlanan : {len(done)}/{total_expected} run")
remaining = total_expected - len(done)
if remaining > 0:
    print(f"  Kalan      : {remaining} run — eğitim devam ediyor")
else:
    print(green("  Tüm runlar tamamlandı!"))
print(bold(f"{'='*68}\n"))

# ─── Plot (isteğe bağlı) ──────────────────────────────────────────────────────
if args.plot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
        import importlib.util

        OUT_DIR = os.path.join(REPO_ROOT, "results")
        spec = importlib.util.spec_from_file_location(
            "industrial_compare",
            os.path.join(SCRIPT_DIR, "14_industrial_compare.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.plot_results(all_results, OUT_DIR)
        print(f"PNG oluşturuldu: {OUT_DIR}/14_industrial_compare.png")
    except Exception as e:
        print(f"Plot hatası: {e}")
