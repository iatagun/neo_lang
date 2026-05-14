"""
13_visualize_3d.py — FluidLM Architecture 3D Visualization
===========================================================

Her NS katmanındaki hız alanı u'yu [B,L,D] → [katman × token] uzayında görselleştirir.

4 panel:
  A (3D surface) : u-norm yüzeyi (X=token, Y=katman, Z=||u||), basınç renkli
  B (heatmap)    : ||u|| ısı haritası [katman × token]
  C (heatmap)    : Basınç alanı p [katman × token]
  D (çizgi)      : ΔKE + ν/Δt per katman (adaptif derinlik sinyali)

Kullanım:
  python experiments/13_visualize_3d.py
  python experiments/13_visualize_3d.py --ckpt experiments/10_fluidlm_fair_best.pt
  python experiments/13_visualize_3d.py --ckpt experiments/10_fluidlm_fair_best.pt --text "To be or not to be"
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401 (registers 3d projection)
from matplotlib import cm
from matplotlib.colors import Normalize

from fluidlm.fluid_ops import gradient, divergence
from fluidlm.ns_layer import FluidLayer
from fluidlm.fluid_lm import FluidLM

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",    type=str,  default=None,
                    help="Checkpoint dosyası (isteğe bağlı)")
parser.add_argument("--text",    type=str,  default=None,
                    help="Görselleştirilecek metin (char-level checkpoint için)")
parser.add_argument("--seq_len", type=int,  default=48,
                    help="Kullanılacak dizi uzunluğu")
parser.add_argument("--out",     type=str,  default="results/13_architecture_3d.png")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[13] device={device}")

# ─────────────────────────────────────────────────────────────────────────────
# Model yükle veya sıfırdan oluştur
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create(ckpt_path, seq_len):
    """Checkpoint varsa yükle, yoksa varsayılan config ile oluştur."""

    # Varsayılan config (10_fluidlm_faircompare ile aynı)
    cfg = dict(
        vocab_size   = 70,     # TinyShakespeare karakter sayısı ~65
        d_model      = 768,
        n_layers     = 12,
        max_seq_len  = 512,
        nu           = 0.01,
        dt           = 0.05,
        alpha        = 1.0,
        integrator   = "rk4",
        dropout      = 0.0,    # görselleştirmede dropout kapalı
        causal       = True,
    )

    stoi = None

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[13] Checkpoint yükleniyor: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # state_dict'ten vocab_size ve d_model çıkar
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        emb_w = sd.get("token_emb.weight")
        if emb_w is not None:
            cfg["vocab_size"] = emb_w.shape[0]
            cfg["d_model"]    = emb_w.shape[1]

        # Katman sayısını çıkar
        n_layers = sum(1 for k in sd if k.startswith("layers.") and k.endswith(".log_nu"))
        if n_layers > 0:
            cfg["n_layers"] = n_layers

        # stoi (karakter → ID) varsa al
        stoi = ckpt.get("stoi", None)

        model = FluidLM(**cfg)
        model.load_state_dict(sd, strict=False)
        print(f"[13] Model: vocab={cfg['vocab_size']}, d={cfg['d_model']}, L={cfg['n_layers']}")
    else:
        if ckpt_path:
            print(f"[13] UYARI: checkpoint bulunamadı ({ckpt_path}), rastgele ağırlıklar kullanılıyor.")
        else:
            print("[13] Checkpoint belirtilmedi, rastgele ağırlıklı model oluşturuluyor.")
        model = FluidLM(**cfg)

    model = model.eval().to(device)
    return model, cfg, stoi


model, cfg, stoi = load_or_create(args.ckpt, args.seq_len)
L      = args.seq_len
D      = cfg["d_model"]
N      = cfg["n_layers"]
V      = cfg["vocab_size"]

# ─────────────────────────────────────────────────────────────────────────────
# Girdi hazırla
# ─────────────────────────────────────────────────────────────────────────────

if args.text and stoi:
    # Gerçek metin
    ids = [stoi.get(c, 0) for c in args.text[:L]]
    ids += [0] * max(0, L - len(ids))
    input_ids = torch.tensor([ids[:L]], dtype=torch.long, device=device)
    print(f"[13] Metin girdi: '{args.text[:L]}'")
else:
    # Rastgele token IDs (sabit seed → tekrarlanabilir)
    torch.manual_seed(42)
    input_ids = torch.randint(0, V, (1, L), device=device)
    print(f"[13] Rastgele girdi (seed=42), L={L}, V={V}")

# ─────────────────────────────────────────────────────────────────────────────
# İleri geçiş + u durumlarını yakala
# ─────────────────────────────────────────────────────────────────────────────

u_states  = []   # Liste[tensor [L, D]] — her katman çıktısı + başlangıç
p_states  = []   # Liste[tensor [L]]    — her katmandaki basınç alanı
ke_states = []   # Liste[float]         — ΔKE per katman

# Hook: her FluidLayer forward() çağrısında u_in ve u_out yakala
def make_hook(layer_idx):
    def hook(module, inp, out):
        u_in  = inp[0]          # [1, L, D]
        u_out = out[0]          # [1, L, D]  (out = (u_new, delta_ke))

        # Başlangıç durumunu yalnızca ilk katmanda ekle
        if layer_idx == 0:
            u_states.append(u_in[0].detach().cpu())

        u_states.append(u_out[0].detach().cpu())

        # Basınç alanını yeniden hesapla (causal cumsum yöntemi)
        with torch.no_grad():
            adv = torch.tanh(u_in[0].norm(dim=-1, keepdim=True)) * \
                  gradient(u_in[0].unsqueeze(0), causal=True)[0]
            div_adv = divergence(adv.unsqueeze(0), causal=True)[0]
            p = torch.cumsum(-div_adv, dim=0) / (module.alpha.item() + 1e-6)
            p = p / (p.std() + 1e-6)
        p_states.append(p.detach().cpu())

    return hook

hooks = []
for i, layer in enumerate(model.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

with torch.no_grad():
    _, stats = model(input_ids, return_stats=True)

ke_states = stats["delta_kes"]

# Hook'ları temizle
for h in hooks:
    h.remove()

print(f"[13] Yakalanan katman sayısı: {len(u_states)-1} (+ başlangıç)")
print(f"[13] ΔKE: min={min(ke_states):.2e}, max={max(ke_states):.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# Veri matrislerini oluştur
# ─────────────────────────────────────────────────────────────────────────────

# ||u|| norm matrisi: shape [N+1, L]
norm_matrix = np.stack([u.norm(dim=-1).numpy() for u in u_states], axis=0)  # [N+1, L]

# Basınç matrisi: shape [N, L]  (katman sayısı kadar)
p_matrix = np.stack([p.numpy() for p in p_states], axis=0)    # [N, L]

# Normalize
norm_min, norm_max = norm_matrix.min(), norm_matrix.max()
norm_n = (norm_matrix - norm_min) / (norm_max - norm_min + 1e-8)

p_abs = np.abs(p_matrix)
p_min, p_max = p_abs.min(), p_abs.max()
p_n = (p_abs - p_min) / (p_max - p_min + 1e-8)

# Meshgrid için
tokens = np.arange(L)
layers = np.arange(N + 1)
T_grid, Lay_grid = np.meshgrid(tokens, layers)   # her ikisi [N+1, L]

# ─────────────────────────────────────────────────────────────────────────────
# Görselleştirme
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use("dark_background")
fig = plt.figure(figsize=(20, 16), facecolor="#0d0d0d")
fig.suptitle("FluidLM — Navier-Stokes Hız Alanının Katmanlar Boyunca Evrimi",
             fontsize=16, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.38, wspace=0.32,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

CMAP_MAIN = cm.plasma
CMAP_HEAT = cm.inferno
CMAP_PRES = cm.coolwarm

# ── Panel A: 3D yüzey ────────────────────────────────────────────────────────
ax3d = fig.add_subplot(gs[0, 0], projection="3d")
ax3d.set_facecolor("#0d0d0d")

# Basınç renklemesi için facecolors hazırla
# p_n sadece N katman için var, N+1 satır için son satırı tekrarla
p_n_ext = np.vstack([p_n, p_n[-1:]])   # [N+1, L]

facecolors = CMAP_MAIN(p_n_ext)

surf = ax3d.plot_surface(
    T_grid, Lay_grid, norm_n,
    facecolors=facecolors,
    alpha=0.88,
    linewidth=0,
    antialiased=True,
    shade=True,
)

# Katman sınırlarına ince tel örgü çiz
for layer_i in range(0, N + 1, 3):
    ax3d.plot(tokens, [layer_i] * L, norm_n[layer_i],
              color="white", alpha=0.15, lw=0.6)

ax3d.set_xlabel("Token pozisyonu", color="lightgray", labelpad=8, fontsize=9)
ax3d.set_ylabel("Katman (NS zamanı)", color="lightgray", labelpad=8, fontsize=9)
ax3d.set_zlabel("‖u‖ (norm.", color="lightgray", labelpad=6, fontsize=9)
ax3d.set_title("A — Hız Alanı Evrimi\n(renk = basınç büyüklüğü)",
               color="white", fontsize=11, pad=10)
ax3d.tick_params(colors="gray", labelsize=7)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor("#333333")
ax3d.yaxis.pane.set_edgecolor("#333333")
ax3d.zaxis.pane.set_edgecolor("#333333")
ax3d.view_init(elev=28, azim=-55)

# Renk çubuğu (basınç)
sm_a = plt.cm.ScalarMappable(cmap=CMAP_MAIN, norm=Normalize(vmin=p_min, vmax=p_max))
sm_a.set_array([])
cb_a = fig.colorbar(sm_a, ax=ax3d, shrink=0.45, pad=0.04)
cb_a.set_label("|p| (basınç büyüklüğü)", color="lightgray", fontsize=8)
cb_a.ax.yaxis.set_tick_params(color="gray", labelsize=7)

# ── Panel B: ‖u‖ ısı haritası ────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor("#0d0d0d")

im_b = ax_b.imshow(norm_matrix, aspect="auto", origin="upper",
                   cmap=CMAP_HEAT, interpolation="bilinear",
                   extent=[0, L-1, N, 0])
ax_b.set_xlabel("Token pozisyonu", color="lightgray", fontsize=10)
ax_b.set_ylabel("Katman", color="lightgray", fontsize=10)
ax_b.set_title("B — ‖u‖ Norm Isı Haritası\n(koyu = düşük enerji, açık = yüksek)", 
               color="white", fontsize=11)
ax_b.tick_params(colors="gray", labelsize=8)
for spine in ax_b.spines.values():
    spine.set_edgecolor("#444")
cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.035, pad=0.02)
cb_b.set_label("‖u‖ (norm.)", color="lightgray", fontsize=8)
cb_b.ax.yaxis.set_tick_params(color="gray", labelsize=7)

# ── Panel C: Basınç alanı ısı haritası ──────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_facecolor("#0d0d0d")

im_c = ax_c.imshow(p_matrix, aspect="auto", origin="upper",
                   cmap=CMAP_PRES, interpolation="bilinear",
                   extent=[0, L-1, N-1, 0])
ax_c.set_xlabel("Token pozisyonu", color="lightgray", fontsize=10)
ax_c.set_ylabel("Katman", color="lightgray", fontsize=10)
ax_c.set_title("C — Basınç Alanı p [katman × token]\n(kırmızı = pozitif basınç, mavi = negatif)",
               color="white", fontsize=11)
ax_c.tick_params(colors="gray", labelsize=8)
for spine in ax_c.spines.values():
    spine.set_edgecolor("#444")
cb_c = fig.colorbar(im_c, ax=ax_c, fraction=0.035, pad=0.02)
cb_c.set_label("p", color="lightgray", fontsize=8)
cb_c.ax.yaxis.set_tick_params(color="gray", labelsize=7)

# ── Panel D: ΔKE + ν + Δt per katman ────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor("#111111")

layer_ids = list(range(len(ke_states)))

# ΔKE (sol eksen)
color_ke = "#f97316"
ax_d.bar(layer_ids, ke_states, color=color_ke, alpha=0.65, width=0.6, label="ΔKE")
ax_d.set_xlabel("Katman", color="lightgray", fontsize=10)
ax_d.set_ylabel("ΔKE (kinetik enerji değişimi)", color=color_ke, fontsize=9)
ax_d.tick_params(axis="y", colors=color_ke, labelsize=8)
ax_d.tick_params(axis="x", colors="gray", labelsize=8)
ax_d.set_xlim(-0.5, len(ke_states) - 0.5)
for spine in ax_d.spines.values():
    spine.set_edgecolor("#444")

# ν ve Δt (sağ eksen)
ax_d2 = ax_d.twinx()
nus  = [layer.nu.item()  for layer in model.layers[:len(ke_states)]]
dts  = [layer.dt.item()  for layer in model.layers[:len(ke_states)]]

ax_d2.plot(layer_ids, nus, "o-", color="#38bdf8", lw=2, ms=5, label="ν (viskozite)")
ax_d2.plot(layer_ids, dts, "s--", color="#a78bfa", lw=1.5, ms=4, label="Δt (adım)")
ax_d2.set_ylabel("ν / Δt değeri", color="lightgray", fontsize=9)
ax_d2.tick_params(axis="y", colors="lightgray", labelsize=8)
ax_d2.spines["right"].set_edgecolor("#444")

# Efsaneler
lines1, labels1 = ax_d.get_legend_handles_labels()
lines2, labels2 = ax_d2.get_legend_handles_labels()
ax_d.legend(lines1 + lines2, labels1 + labels2,
            loc="upper right", fontsize=8,
            facecolor="#1a1a1a", edgecolor="#555", labelcolor="lightgray")

ax_d.set_title("D — ΔKE + Öğrenilen Fiziksel Parametreler\n(ν gradyanı: erken=keskin, geç=pürüzsüz)",
               color="white", fontsize=11)

# ── Alt açıklama ─────────────────────────────────────────────────────────────
source = f"checkpoint: {os.path.basename(args.ckpt)}" if args.ckpt else "rastgele ağırlıklar"
fig.text(0.5, 0.005,
         f"FluidLM  |  d={D}, N={N}, L={L}  |  {source}  |  "
         f"∂u/∂t = −(u·∇)u − ∇p + ν∇²u",
         ha="center", color="#666666", fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Kaydet
# ─────────────────────────────────────────────────────────────────────────────

plt.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[13] Görsel kaydedildi: {args.out}")

# Özet istatistikler
print(f"\n── Özet ────────────────────────────────────────")
print(f"‖u‖ ortalaması  : {norm_matrix.mean():.4f}  (ort. katman = {norm_matrix.mean(axis=1)})")
print(f"Basınç std       : {p_matrix.std(axis=1).tolist()}")
print(f"ΔKE seyrinde     : {[f'{k:.2e}' for k in ke_states]}")
print(f"ν değerleri      : {[f'{v:.5f}' for v in nus]}")
print(f"Δt değerleri     : {[f'{v:.5f}' for v in dts]}")
