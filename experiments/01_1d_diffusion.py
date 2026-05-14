"""
experiments/01_1d_diffusion.py
===============================
Deney 1: 1-D Isı / Difüzyon Denklemi
--------------------------------------

∂u/∂t = ν ∂²u/∂x²

Bu en basit akışkan denklemidir.  Token dizisini bir "sıcaklık alanı" gibi
düşün: her token bir enerji değeri taşıyor ve zaman içinde komşularıyla
etkileşerek düzleşiyor.

Bu deneyin amacı:
  1. Difüzyon operatörünün nasıl çalıştığını görselleştirmek
  2. Laplacian (∇²u) ile viskozite terimini elle anlamak
  3. Euler ve RK4 integratörlerini karşılaştırmak
  4. "Katman = zaman adımı" metaforunu somutlaştırmak

Çalıştırmak için:
    cd neo_lang
    python experiments/01_1d_diffusion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — saves PNG instead of showing window
import matplotlib.pyplot as plt
from src.fluid_ops import laplacian
import torch

# ─── Parametreler ──────────────────────────────────────────────────────────

N   = 64       # token sayısı (sequence length — "uzaysal" grid)
nu  = 0.10     # viskozite katsayısı  (regularizasyon gücü)
dt  = 0.05     # zaman adımı         (katman derinliği)
T   = 100      # toplam adım sayısı  (toplam katman sayısı)

# ─── Başlangıç koşulu ─────────────────────────────────────────────────────
# Token embedding'i rastgele gürültü + iki tepe — karmaşık başlangıç durumu

x = np.linspace(0, 2 * np.pi, N, endpoint=False)
u_numpy = (
    np.random.randn(N) * 0.3            # gürültü
    + np.sin(x)                         # 1. harmonik tepe
    + 0.5 * np.sin(3 * x)              # 3. harmonik tepe
)

# ─── Simülasyon — Euler yöntemi (elle, numpy ile) ─────────────────────────

print("=" * 60)
print("1-D Difüzyon Denklemi — Token Embedding Evrimi")
print("=" * 60)
print(f"  N (token sayısı)     = {N}")
print(f"  ν (viskozite)        = {nu}")
print(f"  Δt (zaman adımı)     = {dt}")
print(f"  T (toplam adım)      = {T}")
print()

snapshots_numpy = [u_numpy.copy()]
u = u_numpy.copy()

for step in range(T):
    # Laplacian  ∂²u/∂x² ≈ u[i+1] − 2u[i] + u[i−1]   (periodic BCs)
    lap = np.roll(u, -1) - 2 * u + np.roll(u, 1)
    u = u + dt * nu * lap
    if step in (4, 9, 24, 49, 99):
        snapshots_numpy.append(u.copy())
        print(f"  Adım {step+1:>3}:  std(u) = {u.std():.4f},  "
              f"max|u| = {np.abs(u).max():.4f}")

print()
print("Gözlem: std(u) giderek küçülüyor → viskozite embedding'i düzleştiriyor.")
print("Bu Transformer'daki LayerNorm / dropout'a benzer bir etki.")

# ─── Aynı simülasyonu PyTorch operatörleriyle doğrula ──────────────────────

print()
print("PyTorch fluid_ops.laplacian ile doğrulama...")
u_torch = torch.tensor(u_numpy[None, :, None])   # [1, N, 1]  (B=1, D=1)
lap_torch = laplacian(u_torch).squeeze().numpy()
lap_numpy = np.roll(u_numpy, -1) - 2 * u_numpy + np.roll(u_numpy, 1)
max_err = np.abs(lap_torch - lap_numpy).max()
print(f"  Maksimum fark (numpy vs torch): {max_err:.2e}")
assert max_err < 1e-5, "Laplacian uyuşmazlığı!"
print("  ✓  Operatörler tutarlı.")

# ─── Görselleştirme ────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("1-D Difüzyon: Token Embedding'inin Akışkan Gibi Evrimi",
             fontsize=13, fontweight="bold")

# Sol: zaman adımlarında snapshot'lar
ax = axes[0]
labels = ["t=0 (başlangıç)", "t=5", "t=10", "t=25", "t=50", "t=100"]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(snapshots_numpy)))
for snap, label, color in zip(snapshots_numpy, labels, colors):
    ax.plot(x, snap, label=label, color=color, linewidth=1.8)
ax.set_title("Embedding değerleri zaman içinde difüze oluyor")
ax.set_xlabel("Token pozisyonu (x)")
ax.set_ylabel("Embedding değeri u(x)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Sağ: enerjinin azalması (std'nin düşüşü)
ax = axes[1]
u_track = u_numpy.copy()
stds = [u_track.std()]
for _ in range(T):
    lap = np.roll(u_track, -1) - 2 * u_track + np.roll(u_track, 1)
    u_track = u_track + dt * nu * lap
    stds.append(u_track.std())
ax.semilogy(stds, color="steelblue", linewidth=2)
ax.set_title("Kinetik enerji (std) → sıfıra yaklaşıyor\n(Adaptif durdurma kriteri)")
ax.set_xlabel("Adım (katman index)")
ax.set_ylabel("std(u)  [log ölçek]")
ax.axhline(1e-2, color="red", linestyle="--", alpha=0.6, label="Örnek eşik 1e-2")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "01_diffusion_result.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print()
print(f"Grafik kaydedildi: {out_path}")

# ─── Euler vs RK4 karşılaştırması ─────────────────────────────────────────

print()
print("Euler vs RK4 doğruluk karşılaştırması (dt=0.1):")
dt_test = 0.1
u0 = u_numpy.copy()

# "Referans": küçük adımlarla Euler (dt=0.001, 1000 adım → t=0.1)
u_ref = u0.copy()
for _ in range(1000):
    lap = np.roll(u_ref, -1) - 2 * u_ref + np.roll(u_ref, 1)
    u_ref += 0.001 * nu * lap

# Euler, 1 adım dt=0.1
u_euler = u0.copy()
lap = np.roll(u_euler, -1) - 2 * u_euler + np.roll(u_euler, 1)
u_euler = u_euler + dt_test * nu * lap

# RK4, 1 adım dt=0.1
def rhs(u_):
    return nu * (np.roll(u_, -1) - 2 * u_ + np.roll(u_, 1))

k1 = rhs(u0)
k2 = rhs(u0 + dt_test * 0.5 * k1)
k3 = rhs(u0 + dt_test * 0.5 * k2)
k4 = rhs(u0 + dt_test * k3)
u_rk4 = u0 + (dt_test / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

err_euler = np.abs(u_euler - u_ref).mean()
err_rk4   = np.abs(u_rk4   - u_ref).mean()
print(f"  Euler hata  (MAE): {err_euler:.6f}")
print(f"  RK4   hata  (MAE): {err_rk4:.6f}")
print(f"  RK4 / Euler oranı: {err_rk4 / err_euler:.4f}  (küçük → RK4 daha iyi)")
print()
print("Sonuç: Her 'katman' fiziksel bir zaman adımı.")
print("       Karmaşık cümleler daha fazla adım gerektirir.")
print("       Enerji yakınsayınca erken durabilirsin → adaptif derinlik.")
