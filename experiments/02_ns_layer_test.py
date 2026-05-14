"""
experiments/02_ns_layer_test.py
================================
Deney 2: Navier-Stokes Katmanı — Basınç ve Adveksiyon Görselleştirmesi
------------------------------------------------------------------------

Bu deneyde tam NS katmanını (adveksiyon + basınç + viskozite) çalıştırıyor
ve her terimin token embedding'lerine nasıl farklı etkiler yaptığını
görselleştiriyoruz.

Anahtar gözlemler:
  • Adveksiyon: "güçlü" tokenlar (yüksek norm) komşularını iter
  • Basınç: global constraint, uzaktaki tokenlara etki eder (attention analog!)
  • Viskozite: düzleştirici, her pozisyonda yerel etki
  • Basıncın etki alanı Σᵢ cos(2πki/L)/k² — uzun mesafeli, attention gibi

Çalıştırmak için:
    cd neo_lang
    python experiments/02_ns_layer_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fluid_ops import gradient, laplacian, divergence, solve_poisson
from src.ns_layer import FluidLayer

torch.manual_seed(42)

# ─── Ayarlar ───────────────────────────────────────────────────────────────

B, L, D = 1, 32, 64      # batch=1, 32 token, 64-boyutlu embedding
nu  = 0.05
dt  = 0.10
n_steps = 20              # kaç NS adımı izleyeceğiz

# ─── Başlangıç hız alanı ──────────────────────────────────────────────────
# "Cümle": bazı tokenlar yüksek normlu (önemli kelimeler), bazıları düşük
# Token 10'a "vurgulu" bir embedding ver — basıncın nasıl yayıldığını gör

u = torch.randn(B, L, D) * 0.5
u[0, 10, :] += 5.0     # "güçlü" token (vurgulu kelime)
u[0, 20, :] += 3.0     # "orta güçlü" token
u_init = u.clone()

# ─── Her terimi ayrı ayrı hesapla (tek adım) ──────────────────────────────

layer = FluidLayer(D, nu=nu, dt=dt, integrator="euler")

with torch.no_grad():
    # Adveksiyon
    speed = u.norm(dim=-1, keepdim=True)      # [1, L, 1]
    adv   = speed * gradient(u)               # [1, L, D]

    # Basınç
    rhs_p = -divergence(adv)                  # [1, L]
    p     = solve_poisson(rhs_p)              # [1, L]
    p_grad = gradient(p.unsqueeze(-1)).expand_as(u)  # [1, L, D]

    # Viskozite
    visc  = nu * laplacian(u)                 # [1, L, D]

    # Birleşik RHS
    rhs   = -adv - p_grad + visc             # [1, L, D]

# ─── Token normları — hangi token ne kadar "aktif"? ───────────────────────

u_norm      = u.norm(dim=-1).squeeze()        # [L]
adv_norm    = adv.norm(dim=-1).squeeze()
p_squeezed  = p.squeeze()                     # [L]  — skaler basınç alanı
visc_norm   = visc.norm(dim=-1).squeeze()
rhs_norm    = rhs.norm(dim=-1).squeeze()

print("=" * 65)
print("Navier-Stokes Katmanı — Term Analizi")
print("=" * 65)
print(f"\nTop-5 en aktif token (‖u‖ sıralaması):")
top5 = torch.topk(u_norm, 5)
for idx, val in zip(top5.indices, top5.values):
    print(f"  Token {idx.item():>3}: ‖u‖ = {val.item():.3f}")

print(f"\nBasınç alanı min/max: [{p_squeezed.min().item():.4f}, "
      f"{p_squeezed.max().item():.4f}]")
print(f"Basınç p[10] = {p_squeezed[10].item():.4f}  (güçlü token etrafı)")
print(f"Basınç p[0]  = {p_squeezed[0].item():.4f}   (uzaktaki token)")
print("\nGözlem: Basınç, token 10'dan uzaklara yayılıyor — global etki!")
print("Bu, Transformer attention'ının yaptığına benzer ancak fizikten türüyor.")

# ─── Çok adımlı simülasyon ─────────────────────────────────────────────────

print(f"\n{n_steps} NS adımı simülasyonu...")
u = u_init.clone()
norms_over_time = []
delta_kes = []

for step in range(n_steps):
    with torch.no_grad():
        u_new, dke = layer(u)
        norms_over_time.append(u.norm(dim=-1).squeeze().numpy().copy())
        delta_kes.append(dke.item())
        u = u_new

print(f"\nAdaptif durdurma simülasyonu:")
threshold = 1e-3
for i, dke in enumerate(delta_kes):
    marker = " ← DUR (ΔKE < eşik)" if dke < threshold else ""
    if i < 8 or dke < threshold:
        print(f"  Adım {i+1:>2}: ΔKE = {dke:.6f}{marker}")
    if dke < threshold:
        print(f"  ... Bu cümle için {i+1} katman yeterli!")
        break

# ─── Görselleştirme ────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Navier-Stokes Katmanı: Token Etkileşimlerinin Fiziksel Analizi",
             fontsize=12, fontweight="bold")

tokens = np.arange(L)

# 1. Başlangıç token normları
ax = axes[0, 0]
ax.bar(tokens, u_norm.numpy(), color="steelblue", alpha=0.8)
ax.axvline(10, color="red",    linestyle="--", label="Güçlü token (pos=10)")
ax.axvline(20, color="orange", linestyle="--", label="Orta token (pos=20)")
ax.set_title("Başlangıç ‖u[i]‖ — Token 'Güç' Dağılımı")
ax.set_xlabel("Token pozisyonu")
ax.set_ylabel("‖u‖ (embedding norm)")
ax.legend(fontsize=7)

# 2. Basınç alanı — global etki
ax = axes[0, 1]
ax.plot(tokens, p_squeezed.numpy(), color="darkviolet", linewidth=2)
ax.axvline(10, color="red",    linestyle="--", alpha=0.7)
ax.axvline(20, color="orange", linestyle="--", alpha=0.7)
ax.set_title("Basınç Alanı p(x) — Attention'ın Fiziksel Karşılığı\n"
             "(Token 10'dan tüm konumlara yayılıyor)")
ax.set_xlabel("Token pozisyonu")
ax.set_ylabel("p(x)")
ax.grid(alpha=0.3)

# 3. Her terimin katkısı
ax = axes[0, 2]
ax.plot(tokens, adv_norm.numpy(),  label="Adveksiyon ‖adv‖",  color="tomato",      lw=2)
ax.plot(tokens, visc_norm.numpy(), label="Viskozite ‖visc‖",  color="steelblue",   lw=2)
ax.plot(tokens, rhs_norm.numpy(),  label="Toplam ‖F(u)‖",     color="black",       lw=2, ls="--")
ax.set_title("Her Terimin Token Başına Büyüklüğü")
ax.set_xlabel("Token pozisyonu")
ax.set_ylabel("‖term‖")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 4. Token normlarının zaman içinde evrimi (ısı haritası)
ax = axes[1, 0]
heatmap = np.array(norms_over_time)   # [n_steps, L]
im = ax.imshow(heatmap.T, aspect="auto", cmap="hot",
               extent=[1, n_steps, 0, L], origin="lower")
ax.set_title("Token Normları Zaman İçinde (‖u[i,t]‖)")
ax.set_xlabel("Adım (katman index)")
ax.set_ylabel("Token pozisyonu")
plt.colorbar(im, ax=ax)

# 5. Delta KE — adaptif durdurma kriteri
ax = axes[1, 1]
ax.semilogy(range(1, n_steps + 1), delta_kes, color="seagreen", lw=2, marker="o", ms=4)
ax.axhline(threshold, color="red", linestyle="--", label=f"Eşik = {threshold}")
ax.set_title("ΔKE (Kinetik Enerji Değişimi) — Adaptif Durdurma Kriteri\n"
             "Kırmızı çizginin altında → katman gerekmiyor")
ax.set_xlabel("Adım (katman index)")
ax.set_ylabel("ΔKE  [log ölçek]")
ax.legend()
ax.grid(alpha=0.3)

# 6. Karşılaştırma: başlangıç vs son durum
ax = axes[1, 2]
ax.plot(tokens, u_norm.numpy(),
        label="Başlangıç ‖u₀‖", color="steelblue", lw=2, ls="--")
ax.plot(tokens, u.norm(dim=-1).squeeze().detach().numpy(),
        label=f"Son ‖u_{n_steps}‖",  color="tomato",    lw=2)
ax.set_title(f"Token Normları: Başlangıç → {n_steps}. Adım")
ax.set_xlabel("Token pozisyonu")
ax.set_ylabel("‖u‖")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "02_ns_layer_result.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nGrafik kaydedildi: {out_path}")

# ─── Parametre sayısı ──────────────────────────────────────────────────────

n_params = sum(p.numel() for p in layer.parameters())
print(f"\nFluidLayer parametre sayısı: {n_params}")
print(f"  log_nu: 1 parametre")
print(f"  log_dt: 1 parametre")
print(f"  Karşılaştırma — Transformer MHA (D={D}): {4 * D * D} parametre")
print(f"  FluidLayer / MHA oranı: {n_params / (4 * D * D):.4f}  "
      f"({100 * n_params / (4 * D * D):.2f}%)")
print("\nFluidLM katmanı, eşdeğer Transformer MHA'ya göre çok daha az parametre kullanıyor.")
