# ============================================================
# FluidLM vs GPT — Nano Pilot (GTX 1650 / 4GB VRAM)
# experiments/15_nano_pilot.py
#
# Amaç: "NS öğreniyor mu?" sorusunu gözlemlemek.
#   • ν gradyanı oluşuyor mu? (erken katman vs geç katman ν farkı)
#   • α (screening range) değişiyor mu?
#   • p_scale sıfıra mı düşüyor, yoksa anlam kazanıyor mu?
#
# Config (4GB VRAM içine sıgacak şekilde ayarlandı):
#   d=256, L=6, seq=128, batch=8, grad_accum=8, dtype=float16
#   Vocab: tiktoken gpt2 (50257)
#   Corpus: OpenWebText streaming (sonsuz)
#   Token budget: 50M (yaklaşık 1.5-2 saat GTX 1650'de)
#
# Kullanım:
#   pip install tiktoken datasets
#   python experiments/15_nano_pilot.py
#   python experiments/15_nano_pilot.py --tokens 20e6   # hızlı test ~30dk
#   python experiments/15_nano_pilot.py --tokens 100e6  # uzun run ~3-4 saat
#   python experiments/15_nano_pilot.py --cpu           # GPU yoksa CPU ile
# ============================================================

import sys, os, math, time, json, argparse, random
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

# UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─────────────────────────────────────────────────────────────────────────────
# Argümanlar
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--tokens",    type=float, default=50e6,  help="Token budget")
parser.add_argument("--d_model",   type=int,   default=256)
parser.add_argument("--n_layers",  type=int,   default=6)
parser.add_argument("--seq_len",   type=int,   default=128)
parser.add_argument("--batch",     type=int,   default=8)
parser.add_argument("--grad_accum",type=int,   default=8)
parser.add_argument("--lr",        type=float, default=3e-4)
parser.add_argument("--seed",      type=int,   default=42)
parser.add_argument("--model",     default="both",
                    choices=["fluid","gpt","both","fluid_v2","v1v2","all"],
                    help="Eğitilecek model. v1v2=fluid+fluid_v2 karşılaştırması. all=fluid+fluid_v2+gpt")
parser.add_argument("--log_every", type=int,   default=100,  help="Her N adımda log")
parser.add_argument("--eval_every",type=int,   default=500,  help="Her N adımda val PPL")
parser.add_argument("--cpu",       action="store_true", help="GPU yoksa CPU kullan")
parser.add_argument("--out",       default="results/15_nano_pilot.json")
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

# ─────────────────────────────────────────────────────────────────────────────
# Cihaz
# ─────────────────────────────────────────────────────────────────────────────

if args.cpu or not torch.cuda.is_available():
    device = torch.device("cpu")
    USE_FP16 = False
    print("[Device] CPU modu")
else:
    device = torch.device("cuda")
    props   = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    # bfloat16: sadece Ampere+ (SM8.0+). GTX 1650 = SM7.5 → float16 kullan
    USE_BF16 = props.major >= 8
    USE_FP16 = not USE_BF16
    print(f"[Device] {props.name}  VRAM={vram_gb:.1f}GB  "
          f"{'bfloat16' if USE_BF16 else 'float16'}")

DTYPE = (torch.bfloat16 if (not args.cpu and USE_BF16)
         else torch.float16 if (not args.cpu and USE_FP16)
         else torch.float32)

AUTOCAST = (torch.amp.autocast(device_type="cuda", dtype=DTYPE)
            if device.type == "cuda" else nullcontext())

# float16 için GradScaler gerekli (bfloat16 veya fp32 için None)
scaler = (torch.cuda.amp.GradScaler() if (device.type == "cuda" and USE_FP16)
          else None)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    VOCAB = enc.n_vocab   # 50257
    print(f"[Tokenizer] tiktoken gpt2  vocab={VOCAB}")
except ImportError:
    print("HATA: tiktoken bulunamadı. Kur: pip install tiktoken")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Veri — OpenWebText streaming
# ─────────────────────────────────────────────────────────────────────────────

try:
    from datasets import load_dataset
except ImportError:
    print("HATA: datasets bulunamadı. Kur: pip install datasets")
    sys.exit(1)

print("[Data] OpenWebText yükleniyor (streaming)...")
train_ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                        trust_remote_code=True)
train_ds = train_ds.shuffle(seed=args.seed, buffer_size=5000)

class TokenStream:
    """BPE token stream — seq_len+1 uzunluğunda bloklar üretir."""
    def __init__(self, dataset, seq_len: int):
        self.seq_len = seq_len
        self.iter    = iter(dataset)
        self.buf: list[int] = []

    def _fill(self):
        while len(self.buf) < (self.seq_len + 1) * 64:
            try:
                text = next(self.iter)["text"]
            except StopIteration:
                # stream bitti → yeniden başlat
                self.iter = iter(train_ds)
                text = next(self.iter)["text"]
            self.buf.extend(enc.encode_ordinary(text))
            self.buf.append(enc.eot_token)

    def get_batch(self, batch_size: int):
        need = (self.seq_len + 1) * batch_size
        while len(self.buf) < need:
            self._fill()
        x_list, y_list = [], []
        for _ in range(batch_size):
            chunk = self.buf[:self.seq_len + 1]
            self.buf = self.buf[self.seq_len + 1:]
            x_list.append(torch.tensor(chunk[:-1], dtype=torch.long))
            y_list.append(torch.tensor(chunk[1:],  dtype=torch.long))
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)
        return x, y

stream = TokenStream(train_ds, args.seq_len)

# Val için ayrı stream (farklı shuffle seed)
val_ds  = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                       trust_remote_code=True)
val_ds  = val_ds.shuffle(seed=args.seed + 999, buffer_size=1000)
val_stream = TokenStream(val_ds, args.seq_len)
print("[Data] Stream hazır")

# ─────────────────────────────────────────────────────────────────────────────
# NS Operatörler (self-contained — fluidlm/ import gerekmez)
# ─────────────────────────────────────────────────────────────────────────────

def causal_gradient(u: torch.Tensor) -> torch.Tensor:
    """∂u/∂x causal backward difference: du[i] = u[i] - u[i-1]"""
    return u - torch.cat([u[:, :1], u[:, :-1]], dim=1)

def causal_laplacian(u: torch.Tensor) -> torch.Tensor:
    """∇²u causal: u[i] - 2u[i-1] + u[i-2]"""
    u_m1 = torch.cat([u[:, :1], u[:, :-1]], dim=1)
    u_m2 = torch.cat([u[:, :2], u[:, :-2]], dim=1)
    return u - 2 * u_m1 + u_m2

def causal_divergence(u: torch.Tensor) -> torch.Tensor:
    """div(u): mean over D dimension of gradient"""
    return causal_gradient(u).mean(dim=-1)  # [B, L]


# ── v2 operators (O(h²) causal gradient) ─────────────────────────────────────

def causal_gradient_v2(u: torch.Tensor) -> torch.Tensor:
    """O(h²) three-point backward difference: (3u[i] - 4u[i-1] + u[i-2]) / 2"""
    u_m1 = torch.cat([u[:, :1, :], u[:, :-1, :]], dim=1)
    u_m2 = torch.cat([u[:, :2, :], u[:, :-2, :]], dim=1)
    return (3.0 * u - 4.0 * u_m1 + u_m2) * 0.5


def causal_divergence_v2(u: torch.Tensor) -> torch.Tensor:
    """mean divergence using O(h²) gradient"""
    return causal_gradient_v2(u).mean(dim=-1)  # [B, L]

# ─────────────────────────────────────────────────────────────────────────────
# NS Layer (bağımsız, fluidlm/ package gerektirmez)
# ─────────────────────────────────────────────────────────────────────────────

class NanoFluidLayer(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        d_ff = d_model * mlp_ratio
        # Fiziksel parametreler (4 skaler)
        self.log_nu      = nn.Parameter(torch.tensor(math.log(0.01)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(0.1)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(1.0)))
        self.p_scale_raw = nn.Parameter(torch.tensor(1.0))   # 1.0: α gradyanı için yeterli sinyal
        # MLP (transformer ile aynı)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout),
        )

    @property
    def nu(self):    return self.log_nu.exp()
    @property
    def dt(self):    return self.log_dt.exp()
    @property
    def alpha(self): return self.log_alpha.exp()

    def _rhs(self, u: torch.Tensor) -> torch.Tensor:
        # Advection
        speed = torch.tanh(u.norm(dim=-1, keepdim=True))
        adv   = speed * causal_gradient(u)
        # Pressure (causal cumsum)
        # Normalise cumsum BEFORE dividing by alpha — otherwise std(cumsum/alpha)
        # = std(cumsum)/alpha cancels alpha out of the gradient path entirely.
        div_adv = causal_divergence(adv)                           # [B, L]
        cumsum  = torch.cumsum(-div_adv, dim=1)                    # [B, L]
        cumsum  = cumsum / (cumsum.std(dim=1, keepdim=True).detach() + 1e-6)  # normalise first
        p = cumsum / (self.alpha + 1e-6)                           # alpha in gradient path
        p_grad = causal_gradient(p.unsqueeze(-1)).expand_as(u)
        p_grad = self.p_scale_raw * p_grad
        # Viscosity
        visc = self.nu * causal_laplacian(u)
        return -adv - p_grad + visc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NS integrate (Euler) + residual
        u   = self.ln1(x)
        du  = self.dt * self._rhs(u)
        x   = x + du
        # MLP block
        x   = x + self.mlp(self.ln2(x))
        return x


class NanoFluidLayerV2(NanoFluidLayer):
    """
    FluidLM-Nano v2 — iki iyileştirme:
      1. O(h²) kausal gradyan  (v1: O(h¹) backward diff)
      2. Channel mixing        (zero-init residual Linear)
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__(d_model, dropout=dropout)
        # Zero-init: başlangıçta pasif, gradient akışıyla devreye girer
        self.channel_mix = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.channel_mix.weight)
        self.channel_mix._zero_init = True   # NanoLM._init_weights tarafından ezilmez

    def _rhs(self, u: torch.Tensor) -> torch.Tensor:
        # Advection — O(h²) gradient
        speed   = torch.tanh(u.norm(dim=-1, keepdim=True))
        adv     = speed * causal_gradient_v2(u)
        # Pressure — O(h²) divergence
        div_adv = causal_divergence_v2(adv)
        cumsum  = torch.cumsum(-div_adv, dim=1)
        cumsum  = cumsum / (cumsum.std(dim=1, keepdim=True).detach() + 1e-6)
        p       = cumsum / (self.alpha + 1e-6)
        p_grad  = causal_gradient_v2(p.unsqueeze(-1)).expand_as(u)
        p_grad  = self.p_scale_raw * p_grad
        # Viscosity (laplacian unchanged)
        visc    = self.nu * causal_laplacian(u)
        return -adv - p_grad + visc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NS step (pre-norm, same as v1)
        u    = self.ln1(x)
        du   = self.dt * self._rhs(u)
        x_ns = x + du
        # Channel mixing — cross-embedding coupling
        x_ns = x_ns + self.channel_mix(x_ns)
        # MLP block
        x    = x_ns + self.mlp(self.ln2(x_ns))
        return x


class NanoGPTLayer(nn.Module):
    """Standart causal MHA + MLP katmanı (FluidLM ile karşılaştırma için)"""
    def __init__(self, d_model: int, n_heads: int, seq_len: int,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        d_ff = d_model * mlp_ratio
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout),
        )
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=self.causal_mask[:L, :L],
                                need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

# ─────────────────────────────────────────────────────────────────────────────
# Model: FluidLM-Nano / GPT-Nano
# ─────────────────────────────────────────────────────────────────────────────

class NanoLM(nn.Module):
    def __init__(self, kind: str, vocab: int, d_model: int, n_layers: int,
                 seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert kind in ("fluid", "fluid_v2", "gpt")
        self.kind = kind
        n_heads   = max(1, d_model // 64)  # head_dim = 64

        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop    = nn.Dropout(dropout)

        if kind == "fluid":
            self.layers = nn.ModuleList([
                NanoFluidLayer(d_model, dropout=dropout) for _ in range(n_layers)
            ])
        elif kind == "fluid_v2":
            self.layers = nn.ModuleList([
                NanoFluidLayerV2(d_model, dropout=dropout) for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                NanoGPTLayer(d_model, n_heads, seq_len, dropout=dropout)
                for _ in range(n_layers)
            ])

        self.ln_f  = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        # init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if getattr(m, '_zero_init', False):
                return   # channel_mix: zero-init korunur, üzerine yazılmaz
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.shape
        pos  = torch.arange(L, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def physical_params(self) -> dict:
        """FluidLM için per-layer fiziksel parametre değerleri."""
        if self.kind not in ("fluid", "fluid_v2"):
            return {}
        nu_vals      = [l.nu.item()            for l in self.layers]
        alpha_vals   = [l.alpha.item()         for l in self.layers]
        p_scale_vals = [l.p_scale_raw.item()   for l in self.layers]
        dt_vals      = [l.dt.item()            for l in self.layers]
        n = len(nu_vals)
        early = nu_vals[:max(1, n//3)]
        late  = nu_vals[max(1, 2*n//3):]
        return {
            "nu": nu_vals,
            "alpha": alpha_vals,
            "p_scale": p_scale_vals,
            "dt": dt_vals,
            "nu_early_mean": sum(early)/len(early),
            "nu_late_mean":  sum(late)/len(late),
            "nu_gradient":   sum(late)/len(late) - sum(early)/len(early),
        }

# ─────────────────────────────────────────────────────────────────────────────
# LR schedule (cosine + warmup)
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= total:
        return lr_min
    t = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))

# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: NanoLM, n_batches: int = 50) -> float:
    model.eval()
    total_loss, total_tok = 0.0, 0
    for _ in range(n_batches):
        x, y = val_stream.get_batch(args.batch)
        with AUTOCAST:
            logits = model(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tok  += y.numel()
    model.train()
    return math.exp(total_loss / total_tok)

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer — 3-grup (fiziksel parametreler 10× LR)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: NanoLM, lr: float):
    PHYSICS = {'log_nu', 'log_dt', 'log_alpha', 'p_scale_raw'}
    physics  = [p for n,p in model.named_parameters()
                if p.requires_grad and any(k in n for k in PHYSICS)]
    decay    = [p for n,p in model.named_parameters()
                if p.requires_grad and p.dim() >= 2
                and not any(k in n for k in PHYSICS)]
    nodecay  = [p for n,p in model.named_parameters()
                if p.requires_grad and p.dim() < 2
                and not any(k in n for k in PHYSICS)]
    groups = [
        {"params": physics, "lr": lr * 10, "weight_decay": 0.01, "lr_scale": 10.0},
        {"params": decay,   "lr": lr,      "weight_decay": 0.1,  "lr_scale": 1.0},
        {"params": nodecay, "lr": lr,      "weight_decay": 0.0,  "lr_scale": 1.0},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)

# ─────────────────────────────────────────────────────────────────────────────
# Model oluştur
# ─────────────────────────────────────────────────────────────────────────────

RUN_FLUID    = args.model in ("fluid", "both", "v1v2", "all")
RUN_FLUID_V2 = args.model in ("fluid_v2", "v1v2", "all")
RUN_GPT      = args.model in ("gpt",   "both", "all")

fluid_model    = NanoLM("fluid",    VOCAB, args.d_model, args.n_layers, args.seq_len).to(device) if RUN_FLUID    else None
fluid_v2_model = NanoLM("fluid_v2", VOCAB, args.d_model, args.n_layers, args.seq_len).to(device) if RUN_FLUID_V2 else None
gpt_model      = NanoLM("gpt",      VOCAB, args.d_model, args.n_layers, args.seq_len).to(device) if RUN_GPT      else None

if fluid_model:    print(f"\n[Model] FluidLM-Nano v1   params={fluid_model.num_params()/1e6:.2f}M")
if fluid_v2_model: print(f"[Model] FluidLM-Nano v2   params={fluid_v2_model.num_params()/1e6:.2f}M")
if gpt_model:      print(f"[Model] GPT-Nano          params={gpt_model.num_params()/1e6:.2f}M")

# ─────────────────────────────────────────────────────────────────────────────
# Training setup
# ─────────────────────────────────────────────────────────────────────────────

token_budget  = int(args.tokens)
tokens_per_step = args.batch * args.grad_accum * args.seq_len
total_steps   = token_budget // tokens_per_step
warmup_steps  = max(50, total_steps // 100)

print(f"\n[Train] token_budget={token_budget/1e6:.0f}M  "
      f"batch={args.batch}  grad_accum={args.grad_accum}  "
      f"seq={args.seq_len}")
print(f"[Train] tokens_per_step={tokens_per_step/1e3:.1f}K  "
      f"total_steps={total_steps:,}  warmup={warmup_steps}")

fluid_opt    = build_optimizer(fluid_model,    args.lr) if RUN_FLUID    else None
fluid_v2_opt = build_optimizer(fluid_v2_model, args.lr) if RUN_FLUID_V2 else None
gpt_opt      = build_optimizer(gpt_model,      args.lr) if RUN_GPT      else None

# ─────────────────────────────────────────────────────────────────────────────
# Eğitim döngüsü
# ─────────────────────────────────────────────────────────────────────────────

results = {
    "config": {
        "d_model": args.d_model, "n_layers": args.n_layers,
        "seq_len": args.seq_len, "batch": args.batch,
        "grad_accum": args.grad_accum, "lr": args.lr,
        "token_budget": token_budget, "total_steps": total_steps,
        "dtype": str(DTYPE), "seed": args.seed,
    },
    "fluid":    {"train_loss": [], "val_ppl": [], "physics": []},
    "fluid_v2": {"train_loss": [], "val_ppl": [], "physics": []},
    "gpt":      {"train_loss": [], "val_ppl": []},
}

print(f"\n{'='*60}")
print(f"  Eğitim başlıyor — her iki model paralel")
print(f"{'='*60}\n")

if fluid_model:    fluid_model.train()
if fluid_v2_model: fluid_v2_model.train()
if gpt_model:      gpt_model.train()

if fluid_opt:    fluid_opt.zero_grad(set_to_none=True)
if fluid_v2_opt: fluid_v2_opt.zero_grad(set_to_none=True)
if gpt_opt:      gpt_opt.zero_grad(set_to_none=True)

t_start = time.time()

fluid_acc_loss    = 0.0
fluid_v2_acc_loss = 0.0
gpt_acc_loss      = 0.0

for step in range(total_steps):
    lr_now = get_lr(step, warmup_steps, total_steps, args.lr, args.lr * 0.1)
    for _opt in (fluid_opt, fluid_v2_opt, gpt_opt):
        if _opt is not None:
            for pg in _opt.param_groups:
                pg["lr"] = lr_now * pg.get("lr_scale", 1.0)

    # Gradient accumulation
    for micro in range(args.grad_accum):
        x, y = stream.get_batch(args.batch)

        # FluidLM forward
        if RUN_FLUID:
            with AUTOCAST:
                f_logits = fluid_model(x)
                f_loss   = F.cross_entropy(f_logits.view(-1, f_logits.size(-1)),
                                           y.view(-1)) / args.grad_accum
            if scaler:
                scaler.scale(f_loss).backward()
            else:
                f_loss.backward()
            fluid_acc_loss += f_loss.item()

        # FluidLM v2 forward
        if RUN_FLUID_V2:
            with AUTOCAST:
                fv2_logits = fluid_v2_model(x)
                fv2_loss   = F.cross_entropy(fv2_logits.view(-1, fv2_logits.size(-1)),
                                             y.view(-1)) / args.grad_accum
            if scaler:
                scaler.scale(fv2_loss).backward()
            else:
                fv2_loss.backward()
            fluid_v2_acc_loss += fv2_loss.item()

        # GPT forward
        if RUN_GPT:
            with AUTOCAST:
                g_logits = gpt_model(x)
                g_loss   = F.cross_entropy(g_logits.view(-1, g_logits.size(-1)),
                                           y.view(-1)) / args.grad_accum
            if scaler:
                scaler.scale(g_loss).backward()
            else:
                g_loss.backward()
            gpt_acc_loss += g_loss.item()

    # Optimizer step
    if scaler:
        if RUN_FLUID:
            scaler.unscale_(fluid_opt)
            torch.nn.utils.clip_grad_norm_(fluid_model.parameters(), 1.0)
            scaler.step(fluid_opt)
        if RUN_FLUID_V2:
            scaler.unscale_(fluid_v2_opt)
            torch.nn.utils.clip_grad_norm_(fluid_v2_model.parameters(), 1.0)
            scaler.step(fluid_v2_opt)
        if RUN_GPT:
            scaler.unscale_(gpt_opt)
            torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), 1.0)
            scaler.step(gpt_opt)
        scaler.update()
    else:
        if RUN_FLUID:
            torch.nn.utils.clip_grad_norm_(fluid_model.parameters(), 1.0)
            fluid_opt.step()
        if RUN_FLUID_V2:
            torch.nn.utils.clip_grad_norm_(fluid_v2_model.parameters(), 1.0)
            fluid_v2_opt.step()
        if RUN_GPT:
            torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), 1.0)
            gpt_opt.step()

    if fluid_opt:    fluid_opt.zero_grad(set_to_none=True)
    if fluid_v2_opt: fluid_v2_opt.zero_grad(set_to_none=True)
    if gpt_opt:      gpt_opt.zero_grad(set_to_none=True)

    # ── Loglama ──────────────────────────────────────────────────────────────
    if step % args.log_every == 0:
        elapsed = time.time() - t_start
        tok_done = (step + 1) * tokens_per_step
        tps = tok_done / elapsed if elapsed > 0 else 0
        eta_min = (token_budget - tok_done) / tps / 60 if tps > 0 else 0

        parts = [f"  step={step:5d}/{total_steps}"]
        if RUN_FLUID:
            f_ppl = math.exp(min(fluid_acc_loss / args.log_every, 20))
            parts.append(f"v1={f_ppl:.2f}")
            results["fluid"]["train_loss"].append({"step": step, "ppl": f_ppl})
            fluid_acc_loss = 0.0
        if RUN_FLUID_V2:
            fv2_ppl = math.exp(min(fluid_v2_acc_loss / args.log_every, 20))
            parts.append(f"v2={fv2_ppl:.2f}")
            results["fluid_v2"]["train_loss"].append({"step": step, "ppl": fv2_ppl})
            fluid_v2_acc_loss = 0.0
        if RUN_GPT:
            g_ppl = math.exp(min(gpt_acc_loss / args.log_every, 20))
            parts.append(f"gpt={g_ppl:.2f}")
            results["gpt"]["train_loss"].append({"step": step, "ppl": g_ppl})
            gpt_acc_loss = 0.0
        parts += [f"lr={lr_now:.2e}", f"tok={tok_done/1e6:.1f}M",
                  f"tps={tps:.0f}", f"ETA={eta_min:.0f}dk"]
        print("  ".join(parts))

    # ── Eval + fiziksel parametreler ─────────────────────────────────────────
    if step % args.eval_every == 0 and step > 0:
        print(f"\n  ── Eval @ step {step} ──────────────────────────────────")
        if RUN_FLUID:
            f_val = evaluate(fluid_model)
            phys  = fluid_model.physical_params()
            print(f"  FluidLM v1  val_ppl = {f_val:.4f}")
            results["fluid"]["val_ppl"].append({"step": step, "ppl": f_val})
            results["fluid"]["physics"].append({"step": step, **phys})
            print(f"    ν grad={phys['nu_gradient']:+.4f}  "
                  f"({'✓' if abs(phys['nu_gradient']) > 0.002 else '—'})")
        if RUN_FLUID_V2:
            fv2_val  = evaluate(fluid_v2_model)
            phys_v2  = fluid_v2_model.physical_params()
            print(f"  FluidLM v2  val_ppl = {fv2_val:.4f}")
            results["fluid_v2"]["val_ppl"].append({"step": step, "ppl": fv2_val})
            results["fluid_v2"]["physics"].append({"step": step, **phys_v2})
            print(f"    ν grad={phys_v2['nu_gradient']:+.4f}  "
                  f"({'✓' if abs(phys_v2['nu_gradient']) > 0.002 else '—'})")
        if RUN_FLUID and RUN_FLUID_V2:
            print(f"  v1 vs v2 ΔPPL = {f_val - fv2_val:+.4f}  "
                  f"({'v2 daha iyi' if fv2_val < f_val else 'v1 daha iyi'})")
        if RUN_GPT:
            g_val = evaluate(gpt_model)
            print(f"  GPT         val_ppl = {g_val:.4f}")
            results["gpt"]["val_ppl"].append({"step": step, "ppl": g_val})
        if RUN_FLUID and RUN_GPT:
            print(f"  v1 vs GPT ΔPPL = {f_val - g_val:+.4f}  "
                  f"({'FluidLM kazanıyor' if f_val < g_val else 'GPT kazanıyor'})")
        if RUN_FLUID_V2 and RUN_GPT:
            print(f"  v2 vs GPT ΔPPL = {fv2_val - g_val:+.4f}  "
                  f"({'FluidLM v2 kazanıyor' if fv2_val < g_val else 'GPT kazanıyor'})")
        print()

        # JSON kaydet
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# Final rapor
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  SONUÇ")
print(f"{'='*60}")

f_final   = evaluate(fluid_model,    n_batches=100) if RUN_FLUID    else None
fv2_final = evaluate(fluid_v2_model, n_batches=100) if RUN_FLUID_V2 else None
g_final   = evaluate(gpt_model,      n_batches=100) if RUN_GPT      else None
phys      = fluid_model.physical_params()           if RUN_FLUID    else {}
phys_v2   = fluid_v2_model.physical_params()        if RUN_FLUID_V2 else {}

if f_final:   print(f"\n  FluidLM-Nano v1  final val_ppl = {f_final:.4f}")
if fv2_final: print(f"  FluidLM-Nano v2  final val_ppl = {fv2_final:.4f}")
if g_final:   print(f"  GPT-Nano         final val_ppl = {g_final:.4f}")
if f_final and fv2_final:
    print(f"  v1 vs v2 ΔPPL = {f_final - fv2_final:+.4f}  "
          f"({'v2 daha iyi ✓' if fv2_final < f_final else 'v1 daha iyi'})") 
if f_final and g_final:
    print(f"  v1 vs GPT ΔPPL = {f_final - g_final:+.4f}")
if fv2_final and g_final:
    print(f"  v2 vs GPT ΔPPL = {fv2_final - g_final:+.4f}")

def _print_physics(label, phys, n_layers):
    if not phys:
        return
    print(f"\n  [Fiziksel Parametreler — {label}]")
    for i in range(n_layers):
        print(f"    Layer {i:2d}:  ν={phys['nu'][i]:.4f}  "
              f"α={phys['alpha'][i]:.4f}  "
              f"p_scale={phys['p_scale'][i]:.4f}  "
              f"dt={phys['dt'][i]:.4f}")
    print(f"    ν gradient: {phys['nu_gradient']:+.4f}  "
          f"({'OLUŞTU ✓' if abs(phys['nu_gradient']) > 0.002 else 'OLUŞMADI ✗'})")

_print_physics("v1",  phys,    args.n_layers)
_print_physics("v2",  phys_v2, args.n_layers)

def _verdict(delta):
    if delta is None:   return "tek model"
    if delta < 0.3:     return "GÜÇLÜ — FluidLM GPT'ye eşdeğer veya daha iyi"
    if delta < 0.7:     return "ORTA — FluidLM yakın ama biraz geride"
    if delta < 2.0:     return "ZAYIF — fark var ama izolasyon çalışıyor"
    return "NEGATİF — FluidLM bu scalede yetersiz"

delta_ppl    = (f_final   - g_final) if (f_final   and g_final) else None
delta_v2_gpt = (fv2_final - g_final) if (fv2_final and g_final) else None
verdict    = _verdict(delta_ppl)
verdict_v2 = _verdict(delta_v2_gpt)

if delta_ppl    is not None: print(f"\n  v1  Yorum: {verdict}")
if delta_v2_gpt is not None: print(f"  v2  Yorum: {verdict_v2}\n")

results["final"] = {
    "fluid_val_ppl":    f_final,   "fluid_v2_val_ppl": fv2_final,
    "gpt_val_ppl":      g_final,
    "delta_ppl":        delta_ppl, "delta_v2_gpt": delta_v2_gpt,
    "verdict":          verdict,   "verdict_v2": verdict_v2,
    "physics_final":    phys,      "physics_v2_final": phys_v2,
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"  Sonuçlar kaydedildi: {args.out}")
