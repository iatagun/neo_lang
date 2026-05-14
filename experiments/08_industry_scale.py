# ============================================================
# FluidLM — Industry Scale (~1B param)
# experiments/08_industry_scale.py
#
# Char-level Shakespeare'den ne farklı:
#   ✦ BPE tokenizer     : tiktoken (GPT-2, vocab=50257)
#   ✦ Gerçek corpus     : HuggingFace FineWeb-Edu (600B token)
#   ✦ Model             : d=2048, L=24, seq=1024  → ~950M param
#   ✦ Multi-GPU         : DDP (tek GPU'da da çalışır)
#   ✦ WandB logging     : opsiyonel
#   ✦ Checkpoint resume : eğitimi kaldığı yerden sürdür
#
# Colab kurulum:
#   !pip install tiktoken datasets wandb -q
#   !git clone https://github.com/KULLANICI/neo_lang.git
#   %cd neo_lang
#   !python experiments/08_industry_scale.py
#
# Tek GPU (manuel):
#   python experiments/08_industry_scale.py
#
# Çok GPU (DDP, 2 GPU):
#   torchrun --nproc_per_node=2 experiments/08_industry_scale.py
# ============================================================

import sys, os, math, time, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

# UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8","utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─────────────────────────────────────────────────────────────────────────────
# Konfigürasyon
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # Model
    "d_model":     2048,
    "n_layers":    24,
    "seq_len":     1024,
    "mlp_ratio":   4,
    "dropout":     0.1,
    "nu":          0.01,
    "dt":          0.05,
    "alpha":       1.0,

    # Veri
    "dataset":     "HuggingFaceFW/fineweb-edu",   # ya da "openwebtext"
    "dataset_cfg": "sample-10BT",                  # fineweb-edu konfigürasyonu
    "fallback_url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
                    "data/tinyshakespeare/input.txt",  # internet yoksa fallback

    # Eğitim
    "epochs":      20,           # büyük corpus — 20 epoch yeterli
    "batch_size":  None,         # otomatik VRAM'a göre
    "grad_accum":  None,         # otomatik
    "lr":          2e-4,
    "warmup_steps": 2000,        # step bazlı warmup (epoch değil)
    "clip_norm":   1.0,
    "patience":    5,            # epoch-bazlı early stopping
    "weight_decay": 1e-1,

    # Altyapı
    "use_grad_ckpt": True,
    "compile":       False,      # torch.compile — ilk çalıştırmayı yavaşlatır
    "use_wandb":     False,      # True yapınca wandb ile izle
    "wandb_project": "fluidlm-industry",
    "resume_ckpt":   None,       # eğitimi sürdür: "path/to/ckpt.pt"

    # Kayıt
    "save_to_drive": False,
    "output_dir":    None,       # None → script dizini
    "ckpt_every":    1,          # her N epoch'ta kaydet
}

# ─────────────────────────────────────────────────────────────────────────────
# DDP Kurulum
# ─────────────────────────────────────────────────────────────────────────────

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")
    ddp_rank       = dist.get_rank()
    ddp_world_size = dist.get_world_size()
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    device         = torch.device(f"cuda:{ddp_local_rank}")
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank       = 0
    ddp_world_size = 1
    master_process = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg):
    if master_process:
        print(msg, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cihaz & dtype
# ─────────────────────────────────────────────────────────────────────────────

if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(device)
    vram_gb  = torch.cuda.get_device_properties(device).total_memory / 1e9
    log(f"GPU: {gpu_name}  ({vram_gb:.0f} GB VRAM)  |  torch {torch.__version__}")
    USE_BF16 = torch.cuda.is_bf16_supported()
else:
    vram_gb  = 0
    USE_BF16 = False
    log(f"CPU modu  |  torch {torch.__version__}")

DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
log(f"Dtype: {DTYPE}")

# Output dizini
if CFG["save_to_drive"]:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        CFG["output_dir"] = "/content/drive/MyDrive/fluidlm_industry"
    except ImportError:
        pass

if CFG["output_dir"] is None:
    CFG["output_dir"] = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
os.makedirs(CFG["output_dir"], exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Batch / Grad Accum — VRAM'a göre otomatik
# ─────────────────────────────────────────────────────────────────────────────
# d=2048, seq=1024, bf16, grad ckpt:
#   Parametre: ~950M × 2 bytes = 1.9 GB
#   Optimizer:                   3.8 GB
#   Aktivasyon/batch (ckpt ile): B × 1024 × 2048 × 2 × ~4 = B × 16 MB
#   B=8 → 128 MB aktivasyon  ✓ toplam ~6 GB
#
if CFG["batch_size"] is None:
    if   vram_gb >= 80: CFG["batch_size"], CFG["grad_accum"] = 16, 16  # 256 effective
    elif vram_gb >= 40: CFG["batch_size"], CFG["grad_accum"] = 8,  16  # 128 effective
    elif vram_gb >= 20: CFG["batch_size"], CFG["grad_accum"] = 4,  16  # 64  effective
    else:               CFG["batch_size"], CFG["grad_accum"] = 2,  8

effective_batch = CFG["batch_size"] * CFG["grad_accum"] * ddp_world_size

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer (BPE — GPT-2, vocab=50257)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = enc.n_vocab          # 50257
    PAD_ID     = 0
    EOS_ID     = enc.eot_token        # 50256

    def tokenize(text: str) -> list[int]:
        return enc.encode_ordinary(text)

    def detokenize(ids: list[int]) -> str:
        return enc.decode([i for i in ids if i != PAD_ID])

    log(f"Tokenizer: tiktoken GPT-2  vocab={VOCAB_SIZE}")
except ImportError:
    # Fallback: char-level (tiktoken kurulu değilse)
    log("UYARI: tiktoken bulunamadı — char-level tokenizer kullanılıyor")
    log("       Kurmak için: pip install tiktoken")
    _chars = None
    _stoi  = None
    _itos  = None

    def _build_char_tokenizer(text: str):
        global _chars, _stoi, _itos, VOCAB_SIZE, EOS_ID, PAD_ID
        _chars = sorted(set(text))
        _stoi  = {c: i+1 for i, c in enumerate(_chars)}   # 0=PAD
        _itos  = {i+1: c for i, c in enumerate(_chars)}
        VOCAB_SIZE = len(_chars) + 1
        PAD_ID     = 0
        EOS_ID     = VOCAB_SIZE - 1

    def tokenize(text: str) -> list[int]:
        if _stoi is None:
            _build_char_tokenizer(text)
        return [_stoi.get(c, 0) for c in text]

    def detokenize(ids: list[int]) -> str:
        return "".join(_itos.get(i, "") for i in ids if i != 0)

    VOCAB_SIZE = None   # build zamanı belirlenir
    PAD_ID     = 0
    EOS_ID     = None

# ─────────────────────────────────────────────────────────────────────────────
# Veri yükleme
# ─────────────────────────────────────────────────────────────────────────────

def load_data_tokens(cfg: dict) -> tuple[list[int], list[int]]:
    """
    HuggingFace datasets varsa FineWeb-Edu kullan,
    yoksa Shakespeare fallback.
    Döndürür: (train_ids, val_ids)
    """
    try:
        from datasets import load_dataset
        log(f"Dataset yükleniyor: {cfg['dataset']} / {cfg['dataset_cfg']} ...")
        log("(İlk çalıştırmada ~10 dk sürebilir, sonrasında cache'den hızlı)")

        ds = load_dataset(
            cfg["dataset"],
            name        = cfg["dataset_cfg"],
            split       = "train",
            streaming   = True,
            trust_remote_code = True,
        )

        # Streaming'den örnek çek — ~500M token hedef (Blackwell için)
        MAX_CHARS = 2_500_000_000   # ~500M BPE token ≈ 2.5B karakter

        texts = []
        total_chars = 0
        for row in ds:
            t = row.get("text", "")
            texts.append(t)
            total_chars += len(t)
            if total_chars >= MAX_CHARS:
                break

        log(f"  {len(texts):,} döküman, {total_chars/1e9:.1f}B karakter toplandı")
        log("  Tokenize ediliyor...")
        all_ids: list[int] = []
        for i, t in enumerate(texts):
            all_ids.extend(tokenize(t))
            all_ids.append(EOS_ID)   # döküman sınırı
            if i % 10000 == 0 and i > 0:
                log(f"    {i:,}/{len(texts):,} döküman  ({len(all_ids)/1e6:.0f}M token)")

        log(f"  Toplam: {len(all_ids)/1e6:.0f}M token")

    except Exception as e:
        log(f"HuggingFace dataset yüklenemedi ({e}), Shakespeare fallback...")
        import urllib.request
        fallback_path = os.path.join(cfg["output_dir"], "shakespeare.txt")
        if not os.path.exists(fallback_path):
            urllib.request.urlretrieve(cfg["fallback_url"], fallback_path)
        with open(fallback_path, encoding="utf-8") as f:
            text = f.read()
        # char-level tokenizer build
        if VOCAB_SIZE is None:
            tokenize(text)   # side effect: builds char tokenizer
        all_ids = tokenize(text)
        log(f"  Shakespeare: {len(all_ids):,} token  vocab={VOCAB_SIZE}")

    split   = int(len(all_ids) * 0.9)
    return all_ids[:split], all_ids[split:]


# ─────────────────────────────────────────────────────────────────────────────
# Causal Fluid Operatörleri
# ─────────────────────────────────────────────────────────────────────────────

def causal_gradient(u: torch.Tensor) -> torch.Tensor:
    """Backward difference: du/dx[i] = u[i] - u[i-1]  (causal)."""
    padded = F.pad(u, (0, 0, 1, 0))
    return u - padded[:, :-1, :]

def causal_laplacian(u: torch.Tensor) -> torch.Tensor:
    """Causal 2nd order: d²u/dx²[i] = u[i] - 2u[i-1] + u[i-2]  (causal)."""
    padded = F.pad(u, (0, 0, 2, 0))
    return u - 2 * padded[:, 1:-1, :] + padded[:, :-2, :]

def causal_divergence(u: torch.Tensor) -> torch.Tensor:
    return causal_gradient(u).mean(dim=-1)   # [B, L]

def causal_pressure(adv: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    p[i] = alpha * cumsum(-div(adv))[i]
    alpha gradyanını korumak için normalizasyonda .detach() kullan.
    """
    div = causal_divergence(adv)
    p   = torch.cumsum(-div, dim=1) * alpha
    p   = p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5) + 1e-6)
    return p   # [B, L]

# ─────────────────────────────────────────────────────────────────────────────
# Model — CausalFluidLayer
# ─────────────────────────────────────────────────────────────────────────────

class CausalFluidLayer(nn.Module):
    """
    Navier-Stokes adımı + MLP sublayer.

    d=2048, mlp_ratio=4:
      MLP params: 2 × 2048 × 8192 ≈ 33.6M
      NS params:  4 skaler + 2 LayerNorm
      Toplam/katman: ~33.6M
    """

    def __init__(self, d_model: int, nu=0.01, dt=0.05, alpha=1.0,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.log_nu      = nn.Parameter(torch.tensor(math.log(nu)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(dt)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_p_scale = nn.Parameter(torch.tensor(math.log(0.1)))

        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model, bias=False),
            nn.Dropout(dropout),
        )

    @property
    def nu(self):      return self.log_nu.exp()
    @property
    def dt(self):      return self.log_dt.exp()
    @property
    def alpha(self):   return self.log_alpha.exp()
    @property
    def p_scale(self): return self.log_p_scale.exp()

    def _rhs(self, u: torch.Tensor) -> torch.Tensor:
        speed  = torch.tanh(u.norm(dim=-1, keepdim=True))
        adv    = speed * causal_gradient(u)
        p      = causal_pressure(adv, self.alpha)
        p_grad = self.p_scale * causal_gradient(
                     p.unsqueeze(-1)).expand_as(u)
        visc   = self.nu * causal_laplacian(u)
        return -adv - p_grad + visc

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = u + self.dt * self._rhs(self.norm1(u))
        u = u + self.mlp(self.norm2(u))
        return u


class CausalFluidLM(nn.Module):
    """
    ~950M parametre konfigürasyonu:
      d=2048, L=24, seq=1024, vocab=50257
      token_emb:  50257×2048 = 103M  (lm_head ile weight-tied)
      pos_emb:    4096×2048  =   8M
      24 katman:  24×33.6M   = 806M
      LayerNorm:  ~1M
      Toplam:     ~918M param
    """

    def __init__(self, vocab_size, d_model=2048, n_layers=24,
                 max_seq_len=1028, nu=0.01, dt=0.05, alpha=1.0,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # Simetri kırma: her katman farklı nu/dt init
        self.layers = nn.ModuleList([
            CausalFluidLayer(
                d_model,
                nu      = nu * (1.0 + 0.05 * i),
                dt      = dt * (1.0 + 0.02 * i),
                alpha   = alpha,
                mlp_ratio = mlp_ratio,
                dropout   = dropout,
            )
            for i in range(n_layers)
        ])

        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self._init_weights()

    def _init_weights(self):
        std = 0.02 / math.sqrt(2 * self.n_layers)   # GPT-2 style scaling
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor,
                use_checkpoint: bool = False) -> torch.Tensor:
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)
        u    = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            if use_checkpoint and self.training:
                u = torch.utils.checkpoint.checkpoint(
                    layer, u, use_reentrant=False)
            else:
                u = layer(u)
        return self.lm_head(self.norm(u))

    def physical_params(self):
        return {
            "nu":      [l.nu.item()      for l in self.layers],
            "dt":      [l.dt.item()      for l in self.layers],
            "alpha":   [l.alpha.item()   for l in self.layers],
            "p_scale": [l.p_scale.item() for l in self.layers],
        }

# ─────────────────────────────────────────────────────────────────────────────
# Veri pipeline — step bazlı (corpus hafızaya sığmayabilir)
# ─────────────────────────────────────────────────────────────────────────────

class TokenDataset:
    """
    Büyük token listesinden rastgele mini-batch üretir.
    Hafızada tutulur (numpy array olarak).
    """
    def __init__(self, ids: list[int], seq_len: int):
        import numpy as np
        self.data    = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.n       = len(ids) - seq_len - 1

    def __len__(self):
        return self.n

    def get_batch(self, batch_size: int, device):
        idx = torch.randint(self.n, (batch_size,))
        x   = torch.stack([self.data[i:i+self.seq_len]   for i in idx])
        y   = torch.stack([self.data[i+1:i+self.seq_len+1] for i in idx])
        return x.to(device), y.to(device)

# ─────────────────────────────────────────────────────────────────────────────
# Metin Üretimi
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt_text: str, max_new: int = 300,
             temperature: float = 0.8, top_k: int = 50,
             top_p: float = 0.95, rep_penalty: float = 1.3,
             seq_len: int = 1024) -> str:
    """
    Nucleus (top-p) + top-k + repetition penalty sampling.
    """
    model.eval()
    prompt_ids = tokenize(prompt_text)
    ids        = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    ids        = ids[:, -seq_len:]
    generated  = []

    for _ in range(max_new):
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=DTYPE):
            logits = model(ids, use_checkpoint=False)

        next_l = logits[0, -1, :].float() / max(temperature, 1e-6)

        # Repetition penalty
        if rep_penalty != 1.0 and generated:
            for tok in set(generated[-128:]):
                next_l[tok] = next_l[tok] / rep_penalty if next_l[tok] > 0 \
                              else next_l[tok] * rep_penalty

        # Top-k
        if top_k > 0:
            topk_v, _ = torch.topk(next_l, min(top_k, next_l.size(-1)))
            next_l    = next_l.masked_fill(next_l < topk_v[-1], -1e9)

        # Top-p (nucleus)
        if 0.0 < top_p < 1.0:
            sorted_l, sorted_idx = torch.sort(next_l, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
            remove    = cum_probs - torch.softmax(sorted_l, dim=-1) > top_p
            sorted_l[remove] = -1e9
            next_l = torch.scatter(next_l, 0, sorted_idx, sorted_l)

        probs = torch.softmax(next_l, dim=-1)
        nid   = torch.multinomial(probs, 1).unsqueeze(0)
        tok   = nid.item()
        generated.append(tok)
        ids   = torch.cat([ids, nid], dim=1)[:, -seq_len:]
        if tok == EOS_ID:
            break

    return detokenize(ids[0].tolist())

# ─────────────────────────────────────────────────────────────────────────────
# Ana program
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Veri ─────────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("FluidLM — Industry Scale (~1B param)")
    log("=" * 70)

    train_ids, val_ids = load_data_tokens(CFG)
    log(f"Train: {len(train_ids)/1e6:.1f}M token  |  Val: {len(val_ids)/1e6:.1f}M token")

    train_ds = TokenDataset(train_ids, CFG["seq_len"])
    val_ds   = TokenDataset(val_ids,   CFG["seq_len"])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CausalFluidLM(
        vocab_size  = VOCAB_SIZE,
        d_model     = CFG["d_model"],
        n_layers    = CFG["n_layers"],
        max_seq_len = CFG["seq_len"] + 4,
        nu          = CFG["nu"],
        dt          = CFG["dt"],
        alpha       = CFG["alpha"],
        mlp_ratio   = CFG["mlp_ratio"],
        dropout     = CFG["dropout"],
    ).to(device)

    if USE_BF16:
        model = model.to(torch.bfloat16)

    # torch.compile (PyTorch 2.x) — ilk adım yavaş, sonrası ~%20 hız
    if CFG["compile"]:
        log("torch.compile() uygulanıyor...")
        model = torch.compile(model)

    # DDP sarma
    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module

    n_params = raw_model.num_parameters()
    log(f"\nModel   : d={CFG['d_model']}, L={CFG['n_layers']}, "
        f"seq={CFG['seq_len']}, vocab={VOCAB_SIZE}")
    log(f"Param   : {n_params:,}  ({n_params/1e6:.0f}M)")
    log(f"Dtype   : {DTYPE}")
    log(f"Batch   : {CFG['batch_size']} × {CFG['grad_accum']} accum "
        f"× {ddp_world_size} GPU = {effective_batch} effective")
    log(f"Grad ckpt: {CFG['use_grad_ckpt']}")

    # ── Optimizer (fizik param'lar için 10x LR) ───────────────────────────────
    phys_ids = set()
    for layer in raw_model.layers:
        for name in ["log_nu", "log_dt", "log_alpha", "log_p_scale"]:
            phys_ids.add(id(getattr(layer, name)))

    phys_params = [p for p in raw_model.parameters() if id(p) in phys_ids]
    rest_params  = [p for p in raw_model.parameters() if id(p) not in phys_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": rest_params, "lr": CFG["lr"],      "weight_decay": CFG["weight_decay"]},
            {"params": phys_params, "lr": CFG["lr"] * 10, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),  # fused AdamW: ~%15 hız kazancı
    )

    # Checkpoint resume
    start_epoch = 1
    best_val    = float("inf")
    best_epoch  = 0
    no_improve  = 0
    if CFG["resume_ckpt"] and os.path.exists(CFG["resume_ckpt"]):
        log(f"\nCheckpoint resume: {CFG['resume_ckpt']}")
        ckpt = torch.load(CFG["resume_ckpt"], map_location=device, weights_only=True)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["val_loss"]
        best_epoch  = ckpt["epoch"]
        log(f"  Epoch {ckpt['epoch']}'dan devam, best_val={best_val:.4f}")

    # Step bazlı warmup + cosine decay LR
    # Warmup_steps adım sonrası cosine; epoch değil step sayar
    total_steps = 0

    def get_lr(step: int) -> float:
        warmup = CFG["warmup_steps"]
        max_steps = CFG["epochs"] * (len(train_ds) // (CFG["batch_size"] * CFG["grad_accum"]) + 1)
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(max_steps - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not USE_BF16))

    # WandB
    if CFG["use_wandb"] and master_process:
        try:
            import wandb
            wandb.init(project=CFG["wandb_project"], config={**CFG,
                "n_params": n_params, "vocab_size": VOCAB_SIZE,
                "effective_batch": effective_batch})
        except ImportError:
            log("wandb kurulu değil, atlıyoruz")
            CFG["use_wandb"] = False

    # ── Eğitim ────────────────────────────────────────────────────────────────
    CKPT_PATH      = os.path.join(CFG["output_dir"], "08_best_model.pt")
    CKPT_LAST_PATH = os.path.join(CFG["output_dir"], "08_last_model.pt")
    train_losses, val_losses = [], []

    log(f"\nEğitim başlıyor: {CFG['epochs']} epoch, "
        f"LR={CFG['lr']}, patience={CFG['patience']}")
    log("-" * 70)

    # Kaç step/epoch?
    steps_per_epoch = max(1, len(train_ds) // (CFG["batch_size"] * CFG["grad_accum"]))
    val_steps       = min(200, len(val_ds) // CFG["batch_size"])

    t0_total = time.time()

    for epoch in range(start_epoch, CFG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for accum_i in range(steps_per_epoch * CFG["grad_accum"]):
            step_i  = accum_i // CFG["grad_accum"]
            is_last_accum = (accum_i + 1) % CFG["grad_accum"] == 0 or \
                            accum_i == steps_per_epoch * CFG["grad_accum"] - 1

            # LR güncelle
            lr_scale = get_lr(total_steps)
            for pg in optimizer.param_groups:
                base = CFG["lr"] if pg is optimizer.param_groups[0] else CFG["lr"] * 10
                pg["lr"] = base * lr_scale

            x, y = train_ds.get_batch(CFG["batch_size"], device)

            # DDP gradient sync sadece son accum adımında
            ctx = model.no_sync() if (ddp and not is_last_accum) else nullcontext()
            with ctx:
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"),
                                        dtype=DTYPE):
                    logits = model(x, use_checkpoint=CFG["use_grad_ckpt"])
                    loss   = F.cross_entropy(
                        logits.reshape(-1, VOCAB_SIZE),
                        y.reshape(-1),
                        ignore_index=PAD_ID,
                    ) / CFG["grad_accum"]

                scaler.scale(loss).backward()
            epoch_loss += loss.item() * CFG["grad_accum"]

            if is_last_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                total_steps += 1

        avg_train = epoch_loss / (steps_per_epoch * CFG["grad_accum"])
        train_losses.append(avg_train)

        # Val
        model.eval()
        v_loss = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda",
                enabled=(device.type == "cuda"), dtype=DTYPE):
            for _ in range(val_steps):
                xv, yv = val_ds.get_batch(CFG["batch_size"], device)
                v_loss += F.cross_entropy(
                    model(xv, use_checkpoint=False).reshape(-1, VOCAB_SIZE),
                    yv.reshape(-1), ignore_index=PAD_ID).item()
        avg_val = v_loss / val_steps
        val_losses.append(avg_val)
        ppl = math.exp(min(avg_val, 20))

        marker = ""
        if avg_val < best_val:
            best_val   = avg_val
            best_epoch = epoch
            no_improve = 0
            marker = "  << BEST"
            if master_process:
                torch.save({
                    "epoch":          epoch,
                    "model_state":    {k: v.cpu() for k, v in raw_model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss":       avg_val,
                    "cfg":            CFG,
                    "vocab_size":     VOCAB_SIZE,
                    "tokenizer":      "tiktoken_gpt2",
                }, CKPT_PATH)
        else:
            no_improve += 1

        # Periyodik checkpoint (her N epoch)
        if master_process and epoch % CFG["ckpt_every"] == 0:
            torch.save({
                "epoch":          epoch,
                "model_state":    {k: v.cpu() for k, v in raw_model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "val_loss":       avg_val,
                "cfg":            CFG,
                "vocab_size":     VOCAB_SIZE,
            }, CKPT_LAST_PATH)

        if master_process:
            elapsed = time.time() - t0_total
            remaining = (elapsed / epoch) * (CFG["epochs"] - epoch)
            cur_lr = optimizer.param_groups[0]["lr"]
            log(f"Ep {epoch:>3}/{CFG['epochs']}  "
                f"train={avg_train:.4f}  val={avg_val:.4f}  ppl={ppl:.1f}  "
                f"best={math.exp(min(best_val,20)):.1f}@{best_epoch}  "
                f"pat={no_improve}/{CFG['patience']}  lr={cur_lr:.2e}  "
                f"~{remaining/60:.0f}dk{marker}")

        if CFG["use_wandb"] and master_process:
            wandb.log({"train_loss": avg_train, "val_loss": avg_val,
                       "ppl": ppl, "lr": cur_lr, "epoch": epoch})

        if no_improve >= CFG["patience"]:
            log(f"\nEarly stopping — {CFG['patience']} epoch iyileşme yok.")
            break

    log("-" * 70)
    total_min = (time.time() - t0_total) / 60
    log(f"Eğitim bitti. best_ppl={math.exp(min(best_val,20)):.2f} "
        f"(epoch {best_epoch})  Süre: {total_min:.1f} dk")

    if ddp:
        dist.destroy_process_group()

    if not master_process:
        sys.exit(0)

    # ── En iyi model yükle ────────────────────────────────────────────────────
    log(f"\nEn iyi model yükleniyor (epoch {best_epoch})...")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    raw_model.load_state_dict(ckpt["model_state"])
    raw_model.eval()
    if USE_BF16:
        raw_model = raw_model.to(torch.bfloat16)

    # ── Metin Üretimi ─────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("METIN URETIMI — FluidLM Industry Scale")
    log("=" * 70)

    prompts_cfg = [
        ("Greedy",    "The history of",    0.3, 10,  0.95),
        ("Balanced",  "Once upon a time",  0.8, 50,  0.95),
        ("Creative",  "In the year 2150,", 1.0, 100, 0.9),
        ("Technical", "The neural network", 0.7, 50, 0.95),
    ]

    for label, prompt, temp, topk, topp in prompts_cfg:
        out = generate(raw_model, prompt, max_new=300, temperature=temp,
                       top_k=topk, top_p=topp, rep_penalty=1.3,
                       seq_len=CFG["seq_len"])
        log(f"\n[{label}]")
        log("-" * 60)
        log(out[:500])

    # ── Fiziksel parametreler ─────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("ÖĞRENILEN FIZIKSEL PARAMETRELER")
    log("=" * 70)
    pp = raw_model.physical_params()
    log(f"  {'Katman':8s}  {'nu':10s}  {'dt':10s}  {'alpha':10s}  {'p_scale':10s}")
    log("  " + "-" * 52)
    for i in range(len(pp["nu"])):
        log(f"  {i:<8d}  {pp['nu'][i]:<10.4f}  {pp['dt'][i]:<10.4f}  "
            f"{pp['alpha'][i]:<10.4f}  {pp['p_scale'][i]:<10.4f}")

    # ── Grafik ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses, label="train")
    axes[0].plot(val_losses,   label="val")
    if best_epoch <= len(val_losses):
        axes[0].axvline(best_epoch - 1, color="red", linestyle="--",
                        label=f"best@{best_epoch}")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    ppls = [math.exp(min(v, 20)) for v in val_losses]
    axes[1].plot(ppls, color="green")
    axes[1].set_title("Validation Perplexity"); axes[1].set_xlabel("Epoch")

    axes[2].plot(pp["nu"],    label="nu")
    axes[2].plot(pp["dt"],    label="dt")
    axes[2].plot(pp["alpha"], label="alpha")
    axes[2].set_title("Physical Params (per layer)")
    axes[2].legend(); axes[2].set_xlabel("Layer")

    plt.tight_layout()
    chart_path = os.path.join(CFG["output_dir"], "08_result.png")
    plt.savefig(chart_path, dpi=150)
    log(f"\nGrafik: {chart_path}")
    log("Deney tamamlandı.")
