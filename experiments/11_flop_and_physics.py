# ============================================================
# 11_flop_and_physics.py
#
# İki ölçüm:
#   A) FLOP/token analizi:  FluidLM-fair vs GPT-B
#      - Her modelin bir forward pass'ında kaç FLOP harcandığını hesapla
#      - Throughput (token/saniye) karşılaştır
#
#   B) Fiziksel parametre analizi:  10_fluidlm_fair_best.pt
#      - Katman katman ν, Δt, α, p_scale değerlerini çıkart
#      - Farklı katmanlar farklı dinamikler mi öğrendi?
#      - Terminal tablosu + matplotlib figürü (opsiyonel)
#
# Kullanım:
#   python experiments/11_flop_and_physics.py
#   python experiments/11_flop_and_physics.py --no_plot   # matplotlib yoksa
# ============================================================

import sys, os, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

parser = argparse.ArgumentParser()
parser.add_argument("--no_plot",    action="store_true", help="matplotlib gerekmez")
parser.add_argument("--ckpt_fluid", default=None,  help="FluidLM-fair checkpoint")
parser.add_argument("--ckpt_gpt",   default=None,  help="GPT-B checkpoint")
parser.add_argument("--out_dir",    default=None)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
OUT_DIR    = args.out_dir or SCRIPT_DIR

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 \
          if device.type == "cuda" else 0

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM A — FLOP Analizi
# ─────────────────────────────────────────────────────────────────────────────
#
# Teorik FLOP sayımı (multiply-add = 2 FLOP):
#
# FluidLM-fair forward (B=1, seq=L, d=D, layers=N, hidden=H=D*4):
#   Embedding lookup: 0 FLOP (sadece indeksleme)
#   NS operators per layer:
#     causal_gradient:   2*L*D  (fark + pad)
#     causal_laplacian:  3*L*D
#     spectral_pressure: FFT(L) ~ L*log(L) + rfft divide + irfft
#     Total NS routing:  ~10*L*D per layer   (çok küçük)
#   MLP per layer (2 matmuls):
#     Linear(D→H): 2*L*D*H
#     Linear(H→D): 2*L*H*D
#     = 4*L*D*H  = 4*L*D*(4D)  = 16*L*D²
#   LayerNorm: ~5*L*D per layer (küçük)
#   LM head:   2*L*D*V
#
# GPT-B forward (B=1, seq=L, d=D, heads=h, layers=N, hidden=H=D*4):
#   MHA per layer:
#     QKV proj:    2*L*D*(3D)  = 6*L*D²
#     QK^T:        2*L²*D      (per head, h heads = 2*L²*D)
#     Softmax:     ~5*L²*h
#     AV:          2*L²*D
#     Out proj:    2*L*D²
#     Total MHA:   ~10*L*D² + 4*L²*D
#   MLP per layer:  16*L*D²  (aynı)
#   Total GPT-B per layer: ~26*L*D² + 4*L²*D
#
# FluidLM-fair per layer: ~16*L*D² + 10*L*D  ≈ 16*L*D²  (NS ihmal edilebilir)
# → FluidLM-fair / GPT-B ≈ 16 / (26 + 4L/D)
#   L=512, D=768: 4L/D = 2.67  → ratio ≈ 16/28.67 ≈ 0.558
#   Yani FluidLM-fair teorik olarak ~1.8× daha az FLOP

def flops_fluidlm(L, D, N, H_mul=4, V=65):
    """FluidLM-fair teorik FLOP (multiply-add her biri 1 sayılır)"""
    per_layer_mlp = 2 * L * D * (D * H_mul) + 2 * L * (D * H_mul) * D
    per_layer_ns  = 10 * L * D + L * math.log2(max(L, 2))  # FFT terimi
    lm_head       = 2 * L * D * V
    return N * (per_layer_mlp + per_layer_ns) + lm_head

def flops_gpt(L, D, N, h=12, H_mul=4, V=65):
    """GPT-B teorik FLOP"""
    per_layer_mha = (6 * L * D * D         # QKV
                   + 2 * L * L * D          # QK^T + AV
                   + 2 * L * D * D)         # Out proj
    per_layer_mlp = 2 * L * D * (D*H_mul) + 2 * L * (D*H_mul) * D
    lm_head       = 2 * L * D * V
    return N * (per_layer_mha + per_layer_mlp) + lm_head

L, D, N = 512, 768, 12
V = 65

f_fluid = flops_fluidlm(L, D, N, V=V)
f_gpt   = flops_gpt(L, D, N, V=V)

print("=" * 65)
print("  BÖLÜM A — Teorik FLOP/token Analizi")
print("=" * 65)
print(f"  L={L}  D={D}  N={N}  V={V}")
print()
print(f"  FluidLM-fair:   {f_fluid/1e9:.3f} GFLOP/sample  "
      f"({f_fluid/L/1e6:.2f} MFLOP/token)")
print(f"  GPT-B:          {f_gpt/1e9:.3f} GFLOP/sample  "
      f"({f_gpt/L/1e6:.2f} MFLOP/token)")
print(f"  Oran (Fluid/GPT): {f_fluid/f_gpt:.3f}  "
      f"→ FluidLM-fair {1/(f_fluid/f_gpt):.2f}× daha az FLOP")
print()

# ── Gerçek throughput (ölçüm) ─────────────────────────────────────────────────
print("  Throughput ölçümü (warm-up + 50 iter)...")

try:
    import time

    # FluidLM-fair tanımları (10_fluidlm_faircompare.py'den kopyalandı)
    def causal_gradient(u):
        return u - F.pad(u, (0,0,1,0))[:,:-1,:]

    def causal_laplacian(u):
        p = F.pad(u, (0,0,2,0))
        return u - 2*p[:,1:-1,:] + p[:,:-2,:]

    def causal_divergence(u):
        return causal_gradient(u).mean(dim=-1)

    def spectral_pressure(adv, alpha):
        div = causal_divergence(adv)
        B, Lx = div.shape
        f   = torch.fft.rfft(div.float(), dim=1)
        k   = torch.arange(Lx//2+1, dtype=torch.float32, device=div.device)
        a   = alpha.float() if isinstance(alpha, torch.Tensor) else float(alpha)
        lam = 2.0*(torch.cos(2.0*math.pi*k/Lx)-1.0) - a**2
        lam = lam.masked_fill(lam.abs()<1e-8, -1e-8)
        p   = torch.fft.irfft(-f/lam.unsqueeze(0), n=Lx, dim=1).to(adv.dtype)
        return p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5)+1e-6)

    class FluidLayerBench(nn.Module):
        def __init__(self, d_model, i=0):
            super().__init__()
            self.log_nu      = nn.Parameter(torch.tensor(math.log(0.01*(1+0.05*i))))
            self.log_dt      = nn.Parameter(torch.tensor(math.log(0.05*(1+0.02*i))))
            self.log_alpha   = nn.Parameter(torch.tensor(math.log(1.0)))
            self.log_p_scale = nn.Parameter(torch.tensor(math.log(0.1)))
            H = d_model * 4
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, H), nn.GELU(), nn.Linear(H, d_model))
        def forward(self, u):
            nu = self.log_nu.exp(); dt = self.log_dt.exp()
            al = self.log_alpha.exp(); ps = self.log_p_scale.exp()
            n  = self.norm1(u)
            sp = torch.tanh(n.norm(dim=-1, keepdim=True))
            adv = sp * causal_gradient(n)
            p   = spectral_pressure(adv, al)
            pg  = ps * causal_gradient(p.unsqueeze(-1)).expand_as(n)
            rhs = -adv - pg + nu*causal_laplacian(n)
            u = u + dt*rhs
            u = u + self.mlp(self.norm2(u))
            return u

    class FluidBench(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(V, D)
            self.pos  = nn.Embedding(L, D)
            self.lyrs = nn.ModuleList([FluidLayerBench(D, i) for i in range(N)])
            self.norm = nn.LayerNorm(D)
            self.head = nn.Linear(D, V, bias=False)
            self.head.weight = self.emb.weight
        def forward(self, x):
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            u   = self.emb(x) + self.pos(pos)
            for lyr in self.lyrs: u = lyr(u)
            return self.head(self.norm(u))

    # GPT-B
    class GPTBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(D)
            self.ln2 = nn.LayerNorm(D)
            self.qkv = nn.Linear(D, 3*D)
            self.out = nn.Linear(D, D)
            self.mlp = nn.Sequential(
                nn.Linear(D, D*4), nn.GELU(), nn.Linear(D*4, D))
            h = 12
            mask = torch.tril(torch.ones(L, L)).unsqueeze(0).unsqueeze(0)
            self.register_buffer("mask", mask)
            self.h = h
        def forward(self, x):
            B, Lx, _ = x.shape
            q,k,v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
            head_d = D // self.h
            def split(t): return t.view(B, Lx, self.h, head_d).transpose(1,2)
            q,k,v = split(q), split(k), split(v)
            scores = (q @ k.transpose(-2,-1)) / math.sqrt(head_d)
            scores = scores.masked_fill(self.mask[:,:,:Lx,:Lx]==0, float('-inf'))
            a = F.softmax(scores, dim=-1) @ v
            a = a.transpose(1,2).contiguous().view(B, Lx, D)
            x = x + self.out(a)
            x = x + self.mlp(self.ln2(x))
            return x

    class GPTBench(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(V, D)
            self.pos  = nn.Embedding(L, D)
            self.lyrs = nn.ModuleList([GPTBlock() for _ in range(N)])
            self.norm = nn.LayerNorm(D)
            self.head = nn.Linear(D, V, bias=False)
            self.head.weight = self.emb.weight
        def forward(self, x):
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            u   = self.emb(x) + self.pos(pos)
            for lyr in self.lyrs: u = lyr(u)
            return self.head(self.norm(u))

    fluid_m = FluidBench().to(device).eval()
    gpt_m   = GPTBench().to(device).eval()
    dummy   = torch.randint(0, V, (4, L), device=device)

    def bench(model, x, n=50):
        with torch.no_grad():
            for _ in range(5): model(x)            # warm-up
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n): model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n      # saniye/batch

    fluid_t = bench(fluid_m, dummy)
    gpt_t   = bench(gpt_m,   dummy)

    fluid_tok = (4 * L) / fluid_t
    gpt_tok   = (4 * L) / gpt_t

    print(f"\n  Throughput (batch=4, seq={L}, device={device}):")
    print(f"  FluidLM-fair:   {fluid_tok:>10,.0f} token/sn  ({fluid_t*1000:.1f} ms/batch)")
    print(f"  GPT-B:          {gpt_tok:>10,.0f} token/sn  ({gpt_t*1000:.1f} ms/batch)")
    print(f"  Hiz orani:      {fluid_tok/gpt_tok:.2f}×  (FluidLM / GPT-B)")
    print()
    print(f"  → FluidLM-fair {fluid_tok/gpt_tok:.2f}× {'DAHA HIZLI' if fluid_tok > gpt_tok else 'daha yavas'} "
          f"@ {device}  (batchsize=4)")
    print()

    # Token-normalize
    print(f"  Efektif maliyet (PPL-normalize):")
    fluid_ppl = 4.59   # 10_fluidlm_faircompare sonucu
    gpt_ppl   = 4.27   # 09_baseline Config-B sonucu
    print(f"  Ayni PPL icin FluidLM-fair {gpt_tok/fluid_tok * (math.log(gpt_ppl)/math.log(fluid_ppl)):.2f}× "
          f"daha az hesap kaynak harcaiyor (tahmin)")

    del fluid_m, gpt_m
    if device.type == "cuda": torch.cuda.empty_cache()

except Exception as e:
    print(f"  Throughput olcumu basarisiz: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM B — Öğrenilen Fiziksel Parametreler
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BÖLÜM B — Öğrenilen Fiziksel Parametreler")
print("=" * 65)

# Checkpoint bul
def find_ckpt(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

fluid_ckpt = find_ckpt([
    args.ckpt_fluid,
    os.path.join(SCRIPT_DIR, "10_fluidlm_fair_best.pt"),
    os.path.join(REPO_ROOT, "checkpoints", "10_fluidlm_fair_best.pt"),
    "/content/neo_lang/experiments/10_fluidlm_fair_best.pt",
])

if fluid_ckpt is None:
    print("  UYARI: 10_fluidlm_fair_best.pt bulunamadi.")
    print("  Once python experiments/10_fluidlm_faircompare.py calistirin.")
    sys.exit(0)

print(f"  Checkpoint: {fluid_ckpt}")
ck = torch.load(fluid_ckpt, map_location="cpu", weights_only=True)
state = ck["model_state"]

# Katman bazlı fiziksel parametreler
print(f"\n  Epoch: {ck.get('epoch','?')}  val_loss: {ck.get('val_loss',0):.4f}  "
      f"PPL: {math.exp(min(ck.get('val_loss',99), 20)):.2f}")
print()

n_layers_found = sum(1 for k in state if "log_nu" in k and "layers." in k)
print(f"  {'Katman':<8} {'ν (viskozite)':>16} {'Δt (adım)':>14} "
      f"{'α (Helmholtz)':>16} {'p_scale':>12}  Yorum")
print("  " + "-"*80)

layer_data = []
for i in range(24):   # maks 24 katman ara
    prefix = f"layers.{i}."
    keys   = [prefix+"log_nu", prefix+"log_dt", prefix+"log_alpha", prefix+"log_p_scale"]
    if not all(k in state for k in keys):
        break
    nu      = math.exp(state[keys[0]].item())
    dt      = math.exp(state[keys[1]].item())
    alpha   = math.exp(state[keys[2]].item())
    p_scale = math.exp(state[keys[3]].item())
    layer_data.append((i, nu, dt, alpha, p_scale))

    # Yorumlama
    regime = []
    if nu < 0.005:   regime.append("ideal akis")
    elif nu > 0.1:   regime.append("yuksek viskozite")
    else:             regime.append("orta viskozite")
    if dt > 0.1:     regime.append("cesur adim")
    elif dt < 0.01:  regime.append("kucuk adim")
    if alpha > 2.0:  regime.append("kuvvetli Helm.")
    elif alpha < 0.5: regime.append("zayif Helm.")

    print(f"  {i:<8} {nu:>16.6f} {dt:>14.6f} {alpha:>16.6f} {p_scale:>12.6f}"
          f"  {', '.join(regime)}")

if not layer_data:
    print("  UYARI: Katman parametreleri state_dict'te bulunamadi.")
    print("  Mevcut anahtarlar:", [k for k in state if "log_" in k][:10])
    sys.exit(0)

# İstatistikler
nus     = [x[1] for x in layer_data]
dts     = [x[2] for x in layer_data]
alphas  = [x[3] for x in layer_data]
pscales = [x[4] for x in layer_data]

print()
print(f"  Ozet istatistikler:")
print(f"  ν    (viskozite): min={min(nus):.6f}  max={max(nus):.6f}  ort={sum(nus)/len(nus):.6f}")
print(f"  Δt   (adim):      min={min(dts):.6f}  max={max(dts):.6f}  ort={sum(dts)/len(dts):.6f}")
print(f"  α    (Helmholtz): min={min(alphas):.4f}  max={max(alphas):.4f}  ort={sum(alphas)/len(alphas):.4f}")
print(f"  p_sc (basinc):    min={min(pscales):.6f}  max={max(pscales):.6f}  ort={sum(pscales)/len(pscales):.6f}")

# Katmandan katmana değişim var mı?
nu_range = max(nus) - min(nus)
dt_range = max(dts) - min(dts)
print()
print(f"  Katman bazli cesitlilik analizi:")
print(f"  ν range / ort:  {nu_range/max(sum(nus)/len(nus),1e-9):.3f}  "
      f"({'YÜKSEK — farkli katmanlar farkli dinamik' if nu_range/max(sum(nus)/len(nus),1e-9)>0.5 else 'dusuk — homojen'})")
print(f"  Δt range / ort: {dt_range/max(sum(dts)/len(dts),1e-9):.3f}  "
      f"({'YÜKSEK — adim boyutu ogreniLDI' if dt_range/max(sum(dts)/len(dts),1e-9)>0.5 else 'dusuk — homojen'})")

# Erken/orta/geç katman analizi
n = len(layer_data)
early  = layer_data[:n//3]
middle = layer_data[n//3:2*n//3]
late   = layer_data[2*n//3:]

def avg(lst, idx): return sum(x[idx] for x in lst) / max(len(lst), 1)

print()
print(f"  Derin katman pattern analizi (n={n} katman):")
print(f"  {'':15} {'ν':>10} {'Δt':>10} {'α':>10} {'p_scale':>10}")
print(f"  {f'Erken (0..{n//3-1})':15} {avg(early,1):>10.5f} {avg(early,2):>10.5f} "
      f"{avg(early,3):>10.4f} {avg(early,4):>10.5f}")
print(f"  {f'Orta ({n//3}..{2*n//3-1})':15} {avg(middle,1):>10.5f} {avg(middle,2):>10.5f} "
      f"{avg(middle,3):>10.4f} {avg(middle,4):>10.5f}")
print(f"  {f'Gec ({2*n//3}..{n-1})':15} {avg(late,1):>10.5f} {avg(late,2):>10.5f} "
      f"{avg(late,3):>10.4f} {avg(late,4):>10.5f}")

# ── Matplotlib figürü ─────────────────────────────────────────────────────────
if not args.no_plot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle("FluidLM-fair: Learned Physical Parameters per Layer\n"
                     "(d=768, L=12 NS, PPL=4.59  vs  GPT-B 85M PPL=4.27)", fontsize=11)

        layers_idx = [x[0] for x in layer_data]
        labels     = [r"$\nu$ (viscosity)", r"$\Delta t$ (step)", r"$\alpha$ (Helmholtz)", r"$p_{scale}$ (pressure)"]
        vals       = [nus, dts, alphas, pscales]
        colors     = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

        for ax, lbl, v, c in zip(axes.flat, labels, vals, colors):
            ax.plot(layers_idx, v, "o-", color=c, linewidth=2, markersize=7)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("Layer index")
            ax.set_xticks(layers_idx)
            ax.grid(alpha=0.3)
            # Init değeri referans çizgisi
            ax.axhline(v[0], color="gray", linestyle="--", alpha=0.4, label="init")

        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, "11_physics_params.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n  Figur kaydedildi: {fig_path}")
    except ImportError:
        print("\n  matplotlib yok, gorsel atilandi. --no_plot ile sessiz calis.")
    except Exception as e:
        print(f"\n  Figur hatasi: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  ÖZET")
print("=" * 65)
print(f"  PPL karsilastirmasi:")
print(f"    GPT-B  (d=768 L=12) 85M param   : 4.27  [28M routing]")
print(f"    FluidLM-fair        57M param   : 4.59  [48 routing]")
print(f"    Delta PPL: +0.32   Routing azaltma: 590,000×")
print()
print(f"  Teorik FLOP:")
print(f"    FluidLM-fair: {f_fluid/L/1e6:.2f} MFLOP/token")
print(f"    GPT-B:        {f_gpt/L/1e6:.2f} MFLOP/token")
print(f"    FluidLM daha az FLOP: {(1-f_fluid/f_gpt)*100:.1f}% tasarruf")
print()
print(f"  Fiziksel parametreler: {len(layer_data)} katmanda ogrenildi")
print(f"    ν cesitlilik: {nu_range/max(sum(nus)/len(nus),1e-9):.3f}")
print(f"    Δt cesitlilik: {dt_range/max(sum(dts)/len(dts),1e-9):.3f}")
