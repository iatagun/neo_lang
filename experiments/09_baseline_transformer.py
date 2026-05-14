# ============================================================
# FluidLM vs nanoGPT Baseline Karşılaştırması
# experiments/09_baseline_transformer.py
#
# Amaç: FluidLM ile aynı veri seti + benzer parametre sayısında
#        standart GPT (decoder-only Transformer) eğiterek yan yana
#        karşılaştırma tablosu üretmek.
#
# İki config çalıştırılır:
#   A) "matched"   — FluidLM ile aynı d=1024, L=16  (~202M param, MHA ekstra)
#   B) "iso-param" — FluidLM ~135M ile eşit parametre  (d=768, L=12, h=12)
#
# Kullanım:
#   python experiments/09_baseline_transformer.py              # her iki config
#   python experiments/09_baseline_transformer.py --config A   # sadece A
#   python experiments/09_baseline_transformer.py --config B   # sadece B
#   python experiments/09_baseline_transformer.py --epochs 50  # daha uzun
# ============================================================

import sys, os, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config",  default="both", choices=["A","B","both"])
parser.add_argument("--epochs",  type=int, default=200)
parser.add_argument("--patience",type=int, default=20)
parser.add_argument("--data",    default=None, help="shakespeare.txt yolu")
parser.add_argument("--out_dir", default=None)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
OUT_DIR    = args.out_dir or SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config tanımları
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    # A: FluidLM ile aynı d/L — GPT daha fazla parametre
    "A": dict(
        name       = "GPT-matched (d=1024 L=16 h=16)",
        d_model    = 1024,
        n_layers   = 16,
        n_heads    = 16,
        seq_len    = 512,
        mlp_ratio  = 4,
        dropout    = 0.2,
        lr         = 3e-4,
        batch_size = None,   # otomatik
        grad_accum = None,
        ckpt_name  = "09_gpt_A_best.pt",
    ),
    # B: FluidLM ile aynı parametre sayısı (~135M)
    # GPT layer: d=768, L=12, h=12
    #   MHA: 4×768²=2.36M  FFN: 2×768×3072=4.72M  → 7.1M/layer × 12 = 85M
    #   + emb + head ≈ 90M  (FluidLM'den biraz az — en yakın config)
    "B": dict(
        name       = "GPT-iso-param (d=768 L=12 h=12) ~90M",
        d_model    = 768,
        n_layers   = 12,
        n_heads    = 12,
        seq_len    = 512,
        mlp_ratio  = 4,
        dropout    = 0.2,
        lr         = 3e-4,
        batch_size = None,
        grad_accum = None,
        ckpt_name  = "09_gpt_B_best.pt",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Cihaz
# ─────────────────────────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE    = torch.bfloat16 if USE_BF16 else torch.float32
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9 \
           if device.type == "cuda" else 0

print(f"Device: {device}  dtype={DTYPE}  VRAM={vram_gb:.0f}GB")

# ─────────────────────────────────────────────────────────────────────────────
# Veri
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = args.data
if DATA_PATH is None:
    for candidate in [
        os.path.join(os.path.dirname(SCRIPT_DIR), "data", "shakespeare.txt"),
        "/content/neo_lang/data/shakespeare.txt",
        os.path.join(SCRIPT_DIR, "shakespeare.txt"),
    ]:
        if os.path.exists(candidate):
            DATA_PATH = candidate
            break

if DATA_PATH is None:
    import urllib.request
    DATA_PATH = os.path.join(OUT_DIR, "shakespeare.txt")
    print("Shakespeare indiriliyor...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt", DATA_PATH)

with open(DATA_PATH, encoding="utf-8") as f:
    TEXT = f.read()

chars      = sorted(set(TEXT))
VOCAB_SIZE = len(chars)
stoi       = {c: i for i, c in enumerate(chars)}
itos       = {i: c for i, c in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda ids: "".join(itos[i] for i in ids)

all_ids   = torch.tensor(encode(TEXT), dtype=torch.long)
split     = int(len(all_ids) * 0.9)
TRAIN_IDS = all_ids[:split]
VAL_IDS   = all_ids[split:]

print(f"Corpus: {len(TEXT):,} char  vocab={VOCAB_SIZE}  "
      f"train={len(TRAIN_IDS):,}  val={len(VAL_IDS):,}")

def get_batch(ids: torch.Tensor, seq_len: int, batch_size: int):
    ix = torch.randint(len(ids) - seq_len - 1, (batch_size,))
    x  = torch.stack([ids[i:i+seq_len]   for i in ix]).to(device)
    y  = torch.stack([ids[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

# ─────────────────────────────────────────────────────────────────────────────
# Model — Decoder-only GPT (standart Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, B, h, L, hd]
        q, k, v = qkv.unbind(0)             # each [B, h, L, hd]

        # Flash attention varsa kullan
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = self.head_dim ** -0.5
            att   = (q @ k.transpose(-2, -1)) * scale
            att   = att.masked_fill(
                torch.ones(L, L, device=x.device).tril() == 0, float("-inf"))
            att   = F.softmax(att, dim=-1)
            att   = self.attn_drop(att)
            y     = att @ v

        y = y.transpose(1, 2).reshape(B, L, D)
        return self.proj_drop(self.proj(y))


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        hidden     = d_model * mlp_ratio
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, seq_len: int, mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            GPTBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                 use_ckpt: bool = False) -> torch.Tensor:
        import torch.utils.checkpoint as ck
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x    = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            if use_ckpt:
                x = ck.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.lm_head(self.norm(x))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

# ─────────────────────────────────────────────────────────────────────────────
# Eğitim döngüsü
# ─────────────────────────────────────────────────────────────────────────────

def auto_batch(vram_gb: float, d_model: int, seq_len: int):
    """VRAM'a göre batch ve grad_accum ayarla."""
    # Transformer activation (bf16): B × seq × d × layers × ~12 bytes
    if vram_gb >= 70:  return 32, 8    # eff=256
    if vram_gb >= 40:  return 16, 8    # eff=128
    if vram_gb >= 20:  return 8,  8    # eff=64
    return 4, 4                         # eff=16


def train_gpt(cfg: dict, run_id: str):
    print(f"\n{'='*70}")
    print(f"  {run_id}: {cfg['name']}")
    print(f"{'='*70}")

    SEQ_LEN    = cfg["seq_len"]
    BATCH_SIZE = cfg["batch_size"] or auto_batch(vram_gb, cfg["d_model"], SEQ_LEN)[0]
    GRAD_ACCUM = cfg["grad_accum"] or auto_batch(vram_gb, cfg["d_model"], SEQ_LEN)[1]

    model = NanoGPT(
        vocab_size = VOCAB_SIZE,
        d_model    = cfg["d_model"],
        n_layers   = cfg["n_layers"],
        n_heads    = cfg["n_heads"],
        seq_len    = SEQ_LEN,
        mlp_ratio  = cfg["mlp_ratio"],
        dropout    = cfg["dropout"],
    ).to(device)

    if USE_BF16:
        model = model.to(torch.bfloat16)

    n_params = model.count_params()
    print(f"  Parametreler: {n_params/1e6:.1f}M")
    print(f"  Batch={BATCH_SIZE}  GradAccum={GRAD_ACCUM}  EffBatch={BATCH_SIZE*GRAD_ACCUM}")

    # fused AdamW bazı GPU/driver kombinasyonlarında sessiz OOM yapıyor — devre dışı
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    # 4GB VRAM'da büyük modeller için grad checkpointing
    USE_GRAD_CKPT = (vram_gb < 10 and n_params > 100e6)
    if USE_GRAD_CKPT:
        print(f"  Gradient checkpointing: AÇIK (VRAM={vram_gb:.0f}GB, {n_params/1e6:.0f}M param)")
    # bf16 loss scaling gerektirmez; GradScaler sadece fp16 için
    USE_SCALER = (device.type == "cuda" and DTYPE == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)

    # Warmup + cosine
    WARMUP_STEPS  = 500
    total_steps   = 0
    def get_lr(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, 10_000 - WARMUP_STEPS)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    CKPT_PATH   = os.path.join(OUT_DIR, cfg["ckpt_name"])
    best_val    = float("inf")
    no_improve  = 0
    STEPS_PER   = max(1, len(TRAIN_IDS) // (SEQ_LEN * BATCH_SIZE * GRAD_ACCUM))
    VAL_STEPS   = min(100, len(VAL_IDS) // (SEQ_LEN * BATCH_SIZE))
    LOG_EVERY   = max(10, STEPS_PER // 10)

    print(f"  Steps/epoch={STEPS_PER}  epochs={args.epochs}  patience={args.patience}")
    t0 = time.time()

    train_history = []
    val_history   = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step_i in range(STEPS_PER):
            for accum_i in range(GRAD_ACCUM):
                x, y = get_batch(TRAIN_IDS, SEQ_LEN, BATCH_SIZE)
                with torch.amp.autocast("cuda",
                        enabled=(device.type == "cuda"), dtype=DTYPE):
                    logits = model(x, use_ckpt=USE_GRAD_CKPT)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)
                    ) / GRAD_ACCUM
                if USE_SCALER:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                epoch_loss += loss.item() * GRAD_ACCUM

            lr_scale = get_lr(total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["lr"] * lr_scale

            if USE_SCALER:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            total_steps += 1

            if step_i % LOG_EVERY == 0:
                elapsed = time.time() - t0
                sps = (step_i + 1) / max(elapsed / epoch, 1e-9)
                remain = (STEPS_PER - step_i - 1) / max(sps, 1e-9)
                print(f"  Ep {epoch:>3}/{args.epochs}  "
                      f"step {step_i+1:>5}/{STEPS_PER}  "
                      f"loss={epoch_loss/((step_i+1)*GRAD_ACCUM):.4f}  "
                      f"ETA {remain/60:.1f}m", flush=True)

        avg_train = epoch_loss / (STEPS_PER * GRAD_ACCUM)
        train_history.append(avg_train)

        # Validation
        model.eval()
        v_loss = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda",
                enabled=(device.type == "cuda"), dtype=DTYPE):
            for _ in range(VAL_STEPS):
                xv, yv = get_batch(VAL_IDS, SEQ_LEN, BATCH_SIZE)
                v_loss += F.cross_entropy(
                    model(xv, use_ckpt=False).reshape(-1, VOCAB_SIZE), yv.reshape(-1)
                ).item()
        avg_val = v_loss / VAL_STEPS
        ppl     = math.exp(min(avg_val, 20))
        val_history.append(avg_val)

        marker = ""
        if avg_val < best_val:
            best_val   = avg_val
            no_improve = 0
            marker     = "  << BEST"
            torch.save({
                "epoch": epoch, "val_loss": avg_val,
                "model_state": model.state_dict(),
                "cfg": cfg, "vocab_size": VOCAB_SIZE,
                "stoi": stoi, "itos": itos,
            }, CKPT_PATH)
        else:
            no_improve += 1

        elapsed_total = (time.time() - t0) / 60
        print(f"  Ep {epoch:>3}/{args.epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"ppl={ppl:.2f}  {elapsed_total:.1f}m{marker}", flush=True)

        if no_improve >= args.patience:
            print(f"\n  Early stopping — {args.patience} epoch iyileşme yok.")
            break

    print(f"\n  {run_id} tamamlandı.")
    print(f"  En iyi val_loss={best_val:.4f}  PPL={math.exp(min(best_val,20)):.2f}")
    return {
        "config":       cfg["name"],
        "n_params_M":   n_params / 1e6,
        "best_val":     best_val,
        "best_ppl":     math.exp(min(best_val, 20)),
        "train_history": train_history,
        "val_history":  val_history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Koştur
# ─────────────────────────────────────────────────────────────────────────────
results = {}

configs_to_run = (
    ["A", "B"] if args.config == "both"
    else [args.config]
)

for cfg_id in configs_to_run:
    results[cfg_id] = train_gpt(CONFIGS[cfg_id], f"Config-{cfg_id}")

# ─────────────────────────────────────────────────────────────────────────────
# Karşılaştırma Tablosu
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  KARŞILAŞTIRMA TABLOSU")
print(f"{'='*70}")
print(f"  {'Model':45s}  {'Param':>7}  {'ValPPL':>8}")
print("  " + "-"*63)

# FluidLM sonuçlarını oku (varsa)
fluidlm_ckpt = None
for p in [
    os.path.join(OUT_DIR, "07_best_model.pt"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "checkpoints", "07_best_model.pt"),
]:
    if os.path.exists(p):
        fluidlm_ckpt = p
        break

if fluidlm_ckpt:
    try:
        import math as _math
        ck = torch.load(fluidlm_ckpt, map_location="cpu", weights_only=True)
        fluidlm_ppl = _math.exp(min(ck.get("val_loss", 20), 20))
        print(f"  {'FluidLM (char, d=1024 L=16)':45s}  {'~135M':>7}  {fluidlm_ppl:>8.2f}  ← senin model")
    except Exception:
        print(f"  FluidLM checkpoint okunamadı: {fluidlm_ckpt}")

for cfg_id, r in results.items():
    print(f"  {r['config'][:45]:45s}  {r['n_params_M']:>6.0f}M  {r['best_ppl']:>8.2f}")

print(f"\n  NOT:")
print(f"  - FluidLM'de NS katmanı sadece 4 skaler (ν,dt,α,p_scale) + MLP taşır.")
print(f"  - Config-A GPT fazladan ~67M param içeriyor (MHA = 4×d²×L).")
print(f"  - Config-B GPT parametre eşit ama seq domain farklı.")
print(f"  - Karşılaştırma geçerliliği için: aynı veri, aynı seq_len, aynı donanım.")
