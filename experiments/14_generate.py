# ============================================================
# FluidLM / GPT — Metin Üretici
# experiments/14_generate.py
#
# Exp 14 checkpoint'inden metin üretir.
# Hem interaktif (prompt gir → cevap al) hem tek seferlik çalışır.
#
# Kullanım:
#   # En iyi FluidLM-S checkpoint
#   python experiments/14_generate.py --ckpt results/14_fluid_S_s42_best.pt
#
#   # Karşılaştırmalı: aynı prompt'u iki modele ver
#   python experiments/14_generate.py \
#       --ckpt   results/14_fluid_S_s42_best.pt \
#       --ckpt2  results/14_gpt_S_s42_best.pt \
#       --prompt "The history of artificial intelligence"
#
#   # Tek seferlik (interaktif değil)
#   python experiments/14_generate.py \
#       --ckpt results/14_fluid_S_s42_best.pt \
#       --prompt "In the beginning" \
#       --tokens 200 --temp 0.8
# ============================================================

import sys, os, math, time, argparse, threading
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8","utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Windows terminali için UTF-8 zorla
if sys.platform == "win32":
    import subprocess
    subprocess.run(["chcp", "65001"], shell=True, capture_output=True)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",    required=True,   help="FluidLM checkpoint")
parser.add_argument("--ckpt2",   default=None,    help="GPT checkpoint (karşılaştırma için)")
parser.add_argument("--prompt",  default=None,    help="Başlangıç metni (None → interaktif)")
parser.add_argument("--tokens",  type=int, default=300,  help="Üretilecek token sayısı")
parser.add_argument("--temp",    type=float, default=0.8, help="Sıcaklık (0.1=deterministik, 1.5=yaratıcı)")
parser.add_argument("--top_k",   type=int, default=50,   help="Top-k örnekleme (0=kapalı)")
parser.add_argument("--top_p",   type=float, default=0.95, help="Top-p (nucleus) örnekleme")
parser.add_argument("--rep_penalty", type=float, default=1.1,
                    help="Tekrar cezası (1.0=kapalı, >1 tekrarları azaltır)")
parser.add_argument("--seed",    type=int, default=None, help="Rastgelelik seed")
parser.add_argument("--show_probs", action="store_true",
                    help="Her token için olasılık dağılımını göster")
args = parser.parse_args()

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE    = torch.bfloat16 if USE_BF16 else torch.float32

if args.seed is not None:
    torch.manual_seed(args.seed)

# ─── Tokenizer ────────────────────────────────────────────────────────────────
try:
    import tiktoken
    ENC    = tiktoken.get_encoding("gpt2")
    encode = lambda t: ENC.encode_ordinary(t)
    decode = lambda ids: ENC.decode(ids)
    VOCAB  = ENC.n_vocab
    print(f"[Tokenizer] tiktoken gpt2  vocab={VOCAB}")
except ImportError:
    print("[HATA] tiktoken kurulu değil:  pip install tiktoken")
    sys.exit(1)

# ─── NS Operatörleri (14_industrial_compare ile aynı) ────────────────────────

def causal_gradient(u):
    return u - F.pad(u, (0, 0, 1, 0))[:, :-1, :]

def causal_laplacian(u):
    p = F.pad(u, (0, 0, 2, 0))
    return u - 2 * p[:, 1:-1, :] + p[:, :-2, :]

def causal_divergence(u):
    return causal_gradient(u).mean(dim=-1)

def _hop_scales(k):
    if k <= 1: return [1]
    return [2**i for i in range(int(math.log2(k)) + 1)]

def multihop_gradient(u, k=4):
    scales = _hop_scales(k)
    total  = torch.zeros_like(u)
    for s in scales:
        p     = F.pad(u, (0, 0, s, 0))[:, :-s, :]
        total = total + (u - p) / s
    return total / len(scales)

def multihop_laplacian(u, k=4):
    scales = _hop_scales(k)
    total  = torch.zeros_like(u)
    for s in scales:
        p     = F.pad(u, (0, 0, 2 * s, 0))
        total = total + (u - 2 * p[:, s:-s, :] + p[:, :-2*s, :]) / (s * s)
    return total / len(scales)

def multihop_divergence(u, k=4):
    return multihop_gradient(u, k).mean(dim=-1)

def spectral_pressure(adv, alpha, hop_k=1):
    div  = multihop_divergence(adv, hop_k) if hop_k > 1 else causal_divergence(adv)
    B, L = div.shape
    f    = torch.fft.rfft(div.float(), dim=1)
    k    = torch.arange(L // 2 + 1, dtype=torch.float32, device=div.device)
    a    = alpha.float() if isinstance(alpha, torch.Tensor) else float(alpha)
    lam  = 2.0 * (torch.cos(2.0 * math.pi * k / L) - 1.0) - a ** 2
    lam  = lam.masked_fill(lam.abs() < 1e-8, -1e-8)
    p    = torch.fft.irfft(-f / lam.unsqueeze(0), n=L, dim=1).to(adv.dtype)
    return p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5) + 1e-6)

def cumsum_pressure(adv, alpha, hop_k=1):
    div_adv = multihop_divergence(adv, hop_k) if hop_k > 1 else causal_divergence(adv)
    cumsum  = torch.cumsum(-div_adv, dim=1)
    cumsum  = cumsum / (cumsum.std(dim=1, keepdim=True).detach() + 1e-6)
    return cumsum / (alpha + 1e-6)

class FluidLayer(nn.Module):
    def __init__(self, d_model, nu=0.01, dt=0.05, alpha=1.0,
                 mlp_ratio=4, dropout=0.0,
                 hop_k=1, use_spectral=False):
        super().__init__()
        self.hop_k        = hop_k
        self.use_spectral = use_spectral
        self.log_nu      = nn.Parameter(torch.tensor(math.log(nu)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(dt)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_p_scale = nn.Parameter(torch.tensor(0.0))
        # v4 content-dependent speed (W_q/W_k)
        self._d_k = max(d_model // 8, 16)
        self.W_q  = nn.Linear(d_model, self._d_k, bias=False)
        self.W_k  = nn.Linear(d_model, self._d_k, bias=False)
        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )
    @property
    def nu(self):      return self.log_nu.exp()
    @property
    def dt(self):      return self.log_dt.exp()
    @property
    def alpha(self):   return self.log_alpha.exp()
    @property
    def p_scale(self): return self.log_p_scale.exp()
    def _rhs(self, u):
        # v4 speed
        q      = self.W_q(u)
        k      = self.W_k(u)
        k_prev = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)
        speed  = torch.tanh((q * k_prev).sum(-1, keepdim=True) / (self._d_k ** 0.5))
        # multi-hop advection
        grad_fn = multihop_gradient if self.hop_k > 1 else causal_gradient
        adv     = speed * (grad_fn(u, self.hop_k) if self.hop_k > 1 else grad_fn(u))
        # pressure
        if self.use_spectral:
            p = spectral_pressure(adv, self.alpha, self.hop_k)
        else:
            div_adv = multihop_divergence(adv, self.hop_k) if self.hop_k > 1 else causal_divergence(adv)
            cumsum  = torch.cumsum(-div_adv, dim=1)
            cumsum  = cumsum / (cumsum.std(dim=1, keepdim=True).detach() + 1e-6)
            p       = cumsum / (self.alpha + 1e-6)
        p_grad  = self.p_scale * (multihop_gradient(p.unsqueeze(-1), self.hop_k)
                                   if self.hop_k > 1 else
                                   causal_gradient(p.unsqueeze(-1))).expand_as(u)
        visc_fn = multihop_laplacian if self.hop_k > 1 else causal_laplacian
        visc    = self.nu * (visc_fn(u, self.hop_k) if self.hop_k > 1 else visc_fn(u))
        return -adv - p_grad + visc
    def forward(self, u):
        u = u + self.dt * self._rhs(self.norm1(u))
        u = u + self.mlp(self.norm2(u))
        return u

class FluidLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, max_seq_len,
                 mlp_ratio=4, dropout=0.0, hop_k=1, use_spectral=False):
        super().__init__()
        self.d_model   = d_model
        self.n_layers  = n_layers
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop_emb  = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            FluidLayer(d_model, mlp_ratio=mlp_ratio,
                       hop_k=hop_k, use_spectral=use_spectral)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
    def forward(self, input_ids):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        u    = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            u = layer(u)
        return self.lm_head(self.norm(u))
    def physical_params(self):
        return {
            "nu":      [l.nu.item()      for l in self.layers],
            "dt":      [l.dt.item()      for l in self.layers],
            "alpha":   [l.alpha.item()   for l in self.layers],
            "p_scale": [l.p_scale.item() for l in self.layers],
        }

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.drop(self.proj(y.transpose(1,2).contiguous().view(B, T, C)))

class GPTLayer(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPTBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads,
                 max_seq_len, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop_emb  = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            GPTLayer(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x   = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

# ─── Enerji Ölçer ────────────────────────────────────────────────────────────

class EnergyMeter:
    """
    GPU güç tüketimini arka planda örnekler (pynvml).
    pynvml yoksa FLOP tabanlı teorik tahmin kullanır.
    """
    def __init__(self, device_idx: int = 0, interval: float = 0.1):
        self.device_idx = device_idx
        self.interval   = interval
        self._samples:  list[float] = []
        self._running   = False
        self._thread    = None
        self.has_nvml   = False
        self._handle    = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle  = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self._pynvml  = pynvml
            self.has_nvml = True
        except Exception:
            pass

    def _poll(self):
        while self._running:
            try:
                w = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self._samples.append(w)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._samples.clear()
        if self.has_nvml:
            self._running = True
            self._thread  = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._samples:
            avg_w = sum(self._samples) / len(self._samples)
            max_w = max(self._samples)
            return {"avg_w": avg_w, "max_w": max_w, "samples": len(self._samples)}
        return {"avg_w": None, "max_w": None, "samples": 0}

    @staticmethod
    def theoretical_mflop(d_model: int, n_layers: int, seq_len: int,
                          model_type: str, n_heads: int = 12) -> float:
        """
        Token başına teorik MFLOP tahmini.
        FluidLM: FFT + diff ops (O(L log L))
        GPT    : QKV matmul + attention (O(L²))
        """
        L = seq_len
        if model_type == "fluid":
            # NS routing: causal grad/laplacian + FFT pressure
            ns_flop = (
                3 * d_model          # causal_gradient × 3 op
                + d_model            # causal_laplacian
                + L * math.log2(L)   # FFT (rfft+irfft)
                + 3 * d_model        # pressure grad + visc + advection
            ) * n_layers
        else:
            # MHA: QKV proj + attn scores + value agg
            ns_flop = (
                3 * d_model * d_model   # QKV
                + L * d_model           # attn scores (causal, amortized)
                + d_model * d_model     # output proj
            ) * n_layers * 2           # ×2 for matmul FMA

        # MLP (her iki modelde aynı): 2 × d × 4d × 2 (in+out)
        mlp_flop = 2 * 2 * d_model * (4 * d_model) * n_layers

        total = ns_flop + mlp_flop
        return total / 1e6  # MFLOP/token

    @staticmethod
    def energy_per_token_uj(avg_w: float, tps: float) -> float:
        """Watt ve tok/s'den token başına joule."""
        return avg_w / max(tps, 1)   # J/tok


# ─── Enerji Raporu yazdır ────────────────────────────────────────────────────

def print_energy_report(meter_result: dict, model_type: str, cfg: dict,
                        n_tokens: int, elapsed: float, tps: float):
    d      = cfg["d_model"]
    L      = cfg["n_layers"]
    seq    = cfg["seq_len"]
    nheads = cfg.get("n_heads", 12)

    mflop_tok = EnergyMeter.theoretical_mflop(d, L, seq, model_type, nheads)

    # Karşıt model için teorik MFLOP
    alt_type  = "gpt" if model_type == "fluid" else "fluid"
    mflop_alt = EnergyMeter.theoretical_mflop(d, L, seq, alt_type, nheads)

    print(f"\n  {'─'*56}")
    print(f"  ENERJI ANALİZİ  ({model_type.upper()}-{cfg.get('scale','S')})")
    print(f"  {'─'*56}")

    # ── Gerçek GPU ölçümü ─────────────────────────────────────────────────
    avg_w = meter_result.get("avg_w")
    ratio = mflop_alt / max(mflop_tok, 1)
    if avg_w:
        j_per_tok   = EnergyMeter.energy_per_token_uj(avg_w, tps)   # J/tok
        total_j     = avg_w * elapsed
        total_wh    = total_j / 3600

        # Dinamik birim seçimi: < 0.01 J → mJ, < 1000 J → J, ≥ 1000 J → kJ
        def fmt_j(val):
            if val < 0.001:  return f"{val*1e6:.1f} µJ"
            if val < 1.0:    return f"{val*1000:.2f} mJ"
            if val < 1000:   return f"{val:.3f} J"
            return f"{val/1000:.3f} kJ"

        print(f"  GPU ortalama güç   : {avg_w:.1f} W  "
              f"(maks {meter_result['max_w']:.1f} W)")
        print(f"  Token hızı (batch=1): {tps:.1f} tok/s")
        print(f"  Token başına enerji: {fmt_j(j_per_tok)}/tok")
        print(f"  Toplam enerji      : {fmt_j(total_j)}  "
              f"({total_wh*1000:.3f} mWh)  [{n_tokens} token]")

        # Karşıt modeli aynı GPU'da çalıştırsaydık (FLOP oranına göre ölçekle)
        alt_j_tok   = j_per_tok * ratio
        alt_total_j = total_j * ratio
        saved_pct   = abs(1 - 1/ratio) * 100
        direction   = "tasarruf" if ratio > 1 else "fazla harcama"
        print(f"\n  [{alt_type.upper()} olsaydı — teorik ölçek]")
        print(f"  Token başına enerji: {fmt_j(alt_j_tok)}/tok  (×{ratio:.2f})")
        print(f"  Toplam enerji      : {fmt_j(alt_total_j)}  "
              f"({abs(saved_pct):.1f}% {direction})")
    else:
        print(f"  GPU güç ölçümü     : pynvml yok — teorik mod")
        print(f"  (pip install pynvml ile gerçek ölçüm aktif olur)")

    # ── Teorik FLOP karşılaştırması ───────────────────────────────────────
    ratio = mflop_alt / max(mflop_tok, 1)
    print(f"\n  Teorik MFLOP/token:")
    cur_bar  = "█" * min(int(mflop_tok / max(mflop_alt, mflop_tok) * 30), 30)
    alt_bar  = "█" * min(int(mflop_alt / max(mflop_alt, mflop_tok) * 30), 30)
    print(f"  {model_type.upper():<8} {mflop_tok:8.1f}  {cur_bar}")
    print(f"  {alt_type.upper():<8} {mflop_alt:8.1f}  {alt_bar}")

    if model_type == "fluid":
        print(f"\n  FluidLM, GPT'ye göre ~{ratio:.2f}× daha az routing FLOP kullanır.")
        print(f"  Routing farkı: 48 param (NS) vs {d*d*3*4//1000}K param (MHA)")
    else:
        print(f"\n  GPT, FluidLM'e göre ~{ratio:.2f}× daha fazla routing FLOP kullanır.")
        print(f"  FluidLM, aynı PPL için daha verimli olabilir (bkz. RQ2).")

    # ── Ölçek projeksiyonu ────────────────────────────────────────────────
    print(f"\n  [1T token eğitim projeksiyonu]")
    tok_1t = 1e12
    if avg_w and tps > 0:
        # Eğitim hızı tahmini: batch=32 × grad_accum=15 → ~480× daha hızlı
        # Gerçek Chinchilla eğitimi için enerji tahmini kullan, süre değil
        train_tps_est = tps * 32 * 15   # yaklaşık

        # kWh = Watt × saat → avg_w aynı kalacak varsayımı
        # (eğitimde backward pass ~2× daha fazla hesap → ×3 güç tahmini)
        train_w_est   = avg_w * 3          # forward + backward + optimizer
        hours_1t      = tok_1t / train_tps_est / 3600
        kwh_1t        = train_w_est * hours_1t / 1000
        co2_1t        = kwh_1t * 0.233

        print(f"  [Not: eğitim tahmini batch=32×grad_acc=15, backward×3 güç]")
        print(f"  Tahmini eğitim süresi : {hours_1t:.0f} saat  "
              f"({hours_1t/24:.0f} gün)")
        print(f"  Tahmini enerji        : {kwh_1t:.0f} kWh  "
              f"≈ {co2_1t:.0f} kg CO₂  (Türkiye grid: ~0.233 kg/kWh)")
        alt_kwh = kwh_1t * ratio
        if model_type == "fluid":
            print(f"  GPT ile olsaydı       : {alt_kwh:.0f} kWh  "
                  f"({(alt_kwh - kwh_1t):.0f} kWh fazla, ×{ratio:.2f})")
        else:
            print(f"  FluidLM ile olsaydı   : {alt_kwh:.0f} kWh  "
                  f"({(kwh_1t - alt_kwh):.0f} kWh tasarruf, ×{ratio:.2f})")
    else:
        mflop_1t = mflop_tok * tok_1t
        print(f"  Toplam MFLOP          : {mflop_1t:.2e}")
        print(f"  {alt_type.upper()} ile olsaydı   : {mflop_alt * tok_1t:.2e}  "
              f"(×{ratio:.2f})")
    print(f"  {'─'*56}")



def load_model(ckpt_path: str):
    """Checkpoint'ten model tipini otomatik algılayarak yükler."""
    print(f"\n[Yükleniyor] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    model_type = ckpt.get("model_type", "fluid")
    cfg        = ckpt["cfg"]
    val_ppl    = ckpt.get("val_ppl", -1)
    step       = ckpt.get("step", -1)

    print(f"  Tip      : {model_type.upper()}")
    print(f"  Config   : d={cfg['d_model']}  L={cfg['n_layers']}  "
          f"seq={cfg['seq_len']}")
    print(f"  Val PPL  : {val_ppl:.4f}  (step {step:,})")

    if model_type == "fluid":
        model = FluidLM(
            vocab_size   = cfg["vocab_size"],
            d_model      = cfg["d_model"],
            n_layers     = cfg["n_layers"],
            max_seq_len  = cfg["seq_len"],
            mlp_ratio    = cfg.get("mlp_ratio", 4),
            hop_k        = cfg.get("hop_k", 1),
            use_spectral = cfg.get("use_spectral", False),
        )
    else:
        model = GPTBaseline(
            vocab_size   = cfg["vocab_size"],
            d_model      = cfg["d_model"],
            n_layers     = cfg["n_layers"],
            n_heads      = cfg["n_heads"],
            max_seq_len  = cfg["seq_len"],
            mlp_ratio    = cfg.get("mlp_ratio", 4),
        )

    # Eski checkpoint uyumu: p_scale_raw → log_p_scale
    state = ckpt["model_state"]
    state = {k.replace(".p_scale_raw", ".log_p_scale"): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval().to(device)
    if USE_BF16:
        model = model.to(DTYPE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params   : {n_params/1e6:.1f}M")

    # Fiziksel parametreler (FluidLM)
    if model_type == "fluid" and hasattr(model, "physical_params"):
        phys = model.physical_params()
        nu   = phys["nu"]
        grad = nu[-1] - nu[0]
        print(f"  ν early  : {sum(nu[:4])/4:.4f}  "
              f"ν late: {sum(nu[-4:])/4:.4f}  "
              f"gradient: {grad:+.4f}")

    return model, model_type, cfg

# ─── Örnekleme ────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_token(logits_1d, temp, top_k, top_p, rep_penalty,
                 generated_ids, show_probs=False):
    """
    Tek token örnekle.
    logits_1d: (vocab,) tensor
    Döner: token_id (int), seçilen token olasılığı (float)
    """
    # Tekrar cezası
    if rep_penalty != 1.0 and generated_ids:
        for tid in set(generated_ids[-64:]):   # son 64 tokenı cezalandır
            if logits_1d[tid] > 0:
                logits_1d[tid] /= rep_penalty
            else:
                logits_1d[tid] *= rep_penalty

    # Sıcaklık
    logits_1d = logits_1d / max(temp, 1e-8)

    # Top-k
    if top_k > 0:
        topk_vals, _ = torch.topk(logits_1d, min(top_k, logits_1d.size(-1)))
        logits_1d[logits_1d < topk_vals[-1]] = -float("inf")

    probs = F.softmax(logits_1d, dim=-1)

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        sorted_probs /= sorted_probs.sum()
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)

    if show_probs:
        top5_p, top5_i = torch.topk(probs, 5)
        candidates = [(ENC.decode([i.item()]), p.item())
                      for i, p in zip(top5_i, top5_p)]
        print(f"\n  [Top-5] ", end="")
        for word, p in candidates:
            print(f"'{word}':{p:.3f}  ", end="")
        print()

    token_id = torch.multinomial(probs, 1).item()
    return token_id, probs[token_id].item()

@torch.no_grad()
def generate(model, prompt_ids, n_tokens, temp, top_k, top_p,
             rep_penalty, seq_len, show_probs=False, stream=True):
    """
    Otoregressif metin üretimi.
    stream=True: token token yazdırır (canlı görünüm).
    Döner: üretilen token listesi, süre, EnergyMeter sonucu
    """
    ctx = list(prompt_ids)
    generated = []
    t0 = time.time()

    meter = EnergyMeter(device_idx=0)
    meter.start()

    for i in range(n_tokens):
        # Bağlam penceresini kırp
        window = ctx[-seq_len:]
        x = torch.tensor([window], dtype=torch.long, device=device)

        with torch.amp.autocast(device_type=device.type, dtype=DTYPE,
                                 enabled=(DTYPE != torch.float32)):
            logits = model(x)

        next_logits = logits[0, -1, :].float().clone()
        token_id, prob = sample_token(
            next_logits, temp, top_k, top_p, rep_penalty,
            generated, show_probs
        )

        ctx.append(token_id)
        generated.append(token_id)

        if stream:
            word = ENC.decode([token_id])
            print(word, end="", flush=True)

    elapsed = time.time() - t0
    energy  = meter.stop()
    tps = n_tokens / elapsed
    return generated, tps, elapsed, energy

# ─── Model(ler) yükle ─────────────────────────────────────────────────────────

models = []
model1, type1, cfg1 = load_model(args.ckpt)
models.append((model1, type1, cfg1, args.ckpt))

if args.ckpt2:
    model2, type2, cfg2 = load_model(args.ckpt2)
    models.append((model2, type2, cfg2, args.ckpt2))

SEQ_LEN = cfg1["seq_len"]

# ─── Üretim parametreleri özeti ───────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  Üretim ayarları:")
print(f"  tokens={args.tokens}  temp={args.temp}  "
      f"top_k={args.top_k}  top_p={args.top_p}  "
      f"rep_penalty={args.rep_penalty}")
print(f"{'─'*60}")

# ─── Üretim döngüsü ───────────────────────────────────────────────────────────

def run_generation(prompt_text: str):
    prompt_ids = encode(prompt_text)
    print(f"\n  Prompt ({len(prompt_ids)} token): '{prompt_text}'\n")

    for model, model_type, cfg, ckpt_path in models:
        label = f"{model_type.upper()}-{cfg.get('scale','S')}  ({os.path.basename(ckpt_path)})"
        print(f"\n{'━'*60}")
        print(f"  {label}")
        print(f"{'━'*60}")

        # Prompt'u yaz
        print(f"\033[90m{prompt_text}\033[0m", end="", flush=True)

        generated, tps, elapsed, energy = generate(
            model, prompt_ids,
            n_tokens    = args.tokens,
            temp        = args.temp,
            top_k       = args.top_k,
            top_p       = args.top_p,
            rep_penalty = args.rep_penalty,
            seq_len     = SEQ_LEN,
            show_probs  = args.show_probs,
        )

        full_text = decode(generated)
        print(f"\n\n  [{tps:.0f} tok/s  |  {args.tokens} token üretildi]")

        # Kalite metrikleri
        unique_tokens   = len(set(generated))
        unique_ratio    = unique_tokens / max(len(generated), 1)
        # Tekrar oranı: 5-gram tekrarı
        ngrams = [tuple(generated[i:i+5]) for i in range(len(generated)-4)]
        rep_rate = 1 - len(set(ngrams)) / max(len(ngrams), 1) if ngrams else 0
        print(f"  Benzersiz token oranı : {unique_ratio:.2%}")
        print(f"  5-gram tekrar oranı   : {rep_rate:.2%}  "
              f"({'iyi' if rep_rate < 0.15 else 'yüksek — temp artır'})")

        # Enerji raporu
        print_energy_report(energy, model_type, cfg, args.tokens, elapsed, tps)

    # Karşılaştırmalı modda fark analizi
    if len(models) == 2:
        print(f"\n{'─'*60}")
        print("  [Karşılaştırma notu]")
        print("  Aynı prompt, aynı random seed, farklı routing mekanizması.")
        print("  PPL farkı küçükse → NS routing = MHA eşdeğeri")
        print(f"{'─'*60}")


# ─── Ana akış ────────────────────────────────────────────────────────────────

if args.prompt:
    # Tek seferlik
    run_generation(args.prompt)
else:
    # İnteraktif mod
    PROMPTS = [
        "The study of artificial intelligence has",
        "In 2025, researchers discovered that",
        "The relationship between language and thought",
        "Once upon a time in a world where",
        "The most important lesson in machine learning is",
    ]

    print("\n  İnteraktif mod — prompt gir veya hazır seçeneklerden seç:")
    print("  (boş bırak → rastgele hazır prompt  |  'q' → çıkış)\n")
    for i, p in enumerate(PROMPTS):
        print(f"  [{i+1}] {p}")
    print()

    while True:
        try:
            raw = input("  Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Çıkılıyor...")
            break

        if raw.lower() in ("q", "quit", "exit", "çıkış"):
            break

        if raw == "":
            import random
            prompt_text = random.choice(PROMPTS)
            print(f"  → Seçilen: '{prompt_text}'")
        elif raw.isdigit() and 1 <= int(raw) <= len(PROMPTS):
            prompt_text = PROMPTS[int(raw) - 1]
        else:
            prompt_text = raw

        run_generation(prompt_text)
        print()
