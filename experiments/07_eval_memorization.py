# ============================================================
# FluidLM — Ezberleme vs Genelleme Analizi
# experiments/07_eval_memorization.py
#
# 5 Test Kategorisi:
#   1. PPL Gap          — train/val perplexity farkı (overfitting göstergesi)
#   2. N-gram Overlap   — üretilen metin training corpus'unda var mı?
#   3. OOD Generalization — Shakespeare dışı prompt'lara cevap
#   4. Diversity        — aynı prompt, 5 farklı çıktı → Self-BLEU
#   5. Distribüsyon     — karakter frekansı: üretim ≈ corpus mu?
#
# Kullanım (Colab):
#   !python experiments/07_eval_memorization.py
#   !python experiments/07_eval_memorization.py --ckpt /path/to/07_best_model.pt
# ============================================================

import sys, os, math, re, argparse, collections
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None,
                    help="Checkpoint yolu (varsayılan: script dizininde 07_best_model.pt)")
parser.add_argument("--n_samples", type=int, default=20,
                    help="Diversity testi için üretilecek örnek sayısı")
parser.add_argument("--ngram",    type=int, default=5,
                    help="N-gram overlap için N değeri")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
CKPT_PATH  = args.ckpt or os.path.join(SCRIPT_DIR, "07_best_model.pt")

if not os.path.exists(CKPT_PATH):
    # Colab konumunu dene
    for candidate in [
        "/content/neo_lang/experiments/07_best_model.pt",
        "/content/07_best_model.pt",
    ]:
        if os.path.exists(candidate):
            CKPT_PATH = candidate
            break

print(f"Checkpoint: {CKPT_PATH}")
assert os.path.exists(CKPT_PATH), f"Checkpoint bulunamadı: {CKPT_PATH}"

# ─────────────────────────────────────────────────────────────────────────────
# Model sınıfları (07 ile özdeş — checkpoint'i yükleyebilmek için)
# ─────────────────────────────────────────────────────────────────────────────

def causal_gradient(u):
    padded = F.pad(u, (0, 0, 1, 0))
    return u - padded[:, :-1, :]

def causal_laplacian(u):
    padded = F.pad(u, (0, 0, 2, 0))
    return u - 2 * padded[:, 1:-1, :] + padded[:, :-2, :]

def causal_divergence(u):
    return causal_gradient(u).mean(dim=-1)

def causal_pressure(adv, alpha):
    div = causal_divergence(adv)
    p   = torch.cumsum(-div, dim=1) * alpha
    p   = p / (p.detach().std(dim=1, keepdim=True).clamp(min=0.5) + 1e-6)
    return p

class CausalFluidLayer(nn.Module):
    def __init__(self, d_model, nu=0.01, dt=0.05, alpha=1.0,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.log_nu      = nn.Parameter(torch.tensor(math.log(nu)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(dt)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_p_scale = nn.Parameter(torch.tensor(math.log(0.1)))
        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
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

    def _rhs(self, u):
        speed  = torch.tanh(u.norm(dim=-1, keepdim=True))
        adv    = speed * causal_gradient(u)
        p      = causal_pressure(adv, self.alpha)
        p_grad = self.p_scale * causal_gradient(p.unsqueeze(-1)).expand_as(u)
        visc   = self.nu * causal_laplacian(u)
        return -adv - p_grad + visc

    def forward(self, u):
        u = u + self.dt * self._rhs(self.norm1(u))
        u = u + self.mlp(self.norm2(u))
        return u

class CausalFluidLM(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_layers=16,
                 max_seq_len=516, nu=0.01, dt=0.05, alpha=1.0,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            CausalFluidLayer(d_model,
                             nu=nu*(1+0.05*i), dt=dt*(1+0.02*i),
                             alpha=alpha, mlp_ratio=mlp_ratio, dropout=dropout)
            for i in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, use_checkpoint=False):
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)
        u    = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            u = layer(u)
        return self.lm_head(self.norm(u))

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint yükle
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
DTYPE    = torch.bfloat16 if USE_BF16 else torch.float32

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)

stoi       = ckpt["stoi"]
itos       = ckpt["itos"]
VOCAB_SIZE = ckpt["vocab_size"]
D_MODEL    = ckpt.get("d_model",     1024)
N_LAYERS   = ckpt.get("n_layers",    16)
SEQ_LEN    = ckpt.get("max_seq_len", 516) - 4
MLP_RATIO  = ckpt.get("mlp_ratio",   4)

def encode(s): return [stoi.get(c, 0) for c in s]
def decode(ids): return "".join(itos.get(i, "?") for i in ids)

model = CausalFluidLM(
    vocab_size  = VOCAB_SIZE,
    d_model     = D_MODEL,
    n_layers    = N_LAYERS,
    max_seq_len = SEQ_LEN + 4,
    mlp_ratio   = MLP_RATIO,
    dropout     = 0.0,   # eval: dropout kapalı
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
if USE_BF16:
    model = model.to(torch.bfloat16)

print(f"Model: d={D_MODEL}, L={N_LAYERS}, seq={SEQ_LEN}, vocab={VOCAB_SIZE}")
print(f"Checkpoint epoch={ckpt.get('epoch','?')}  "
      f"val_loss={ckpt.get('val_loss',0):.4f}  "
      f"val_ppl={math.exp(min(ckpt.get('val_loss',0),20)):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Corpus yükle
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = None
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
    DATA_PATH = os.path.join(SCRIPT_DIR, "shakespeare.txt")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_PATH)

with open(DATA_PATH, encoding="utf-8") as f:
    TEXT = f.read()

all_ids   = encode(TEXT)
split     = int(len(all_ids) * 0.9)
TRAIN_IDS = all_ids[:split]
VAL_IDS   = all_ids[split:]
TRAIN_TEXT = TEXT[:int(len(TEXT)*0.9)]
VAL_TEXT   = TEXT[int(len(TEXT)*0.9):]

print(f"Corpus: {len(TEXT):,} karakter  "
      f"train={len(TRAIN_TEXT):,}  val={len(VAL_TEXT):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(ids: list, batch_size=32) -> float:
    """Token listesi üzerinde perplexity hesapla."""
    total_loss = 0.0
    total_tok  = 0
    for i in range(0, len(ids) - SEQ_LEN - 1, SEQ_LEN):
        chunk = ids[i:i+SEQ_LEN+1]
        if len(chunk) < SEQ_LEN + 1:
            break
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]],  dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=DTYPE):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        total_loss += loss.item() * (SEQ_LEN)
        total_tok  += SEQ_LEN
    return math.exp(min(total_loss / max(total_tok, 1), 20))

@torch.no_grad()
def generate(prompt: str, max_new=200, temperature=0.8,
             top_k=10, rep_penalty=1.3) -> str:
    ids       = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    ids       = ids[:, -SEQ_LEN:]
    generated = []
    for _ in range(max_new):
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=DTYPE):
            logits = model(ids)
        nxt = logits[0, -1, :].float() / max(temperature, 1e-6)
        if rep_penalty != 1.0 and generated:
            for tok in set(generated[-64:]):
                nxt[tok] = nxt[tok]/rep_penalty if nxt[tok]>0 else nxt[tok]*rep_penalty
        if top_k > 0:
            tv, _ = torch.topk(nxt, min(top_k, nxt.size(-1)))
            nxt   = nxt.masked_fill(nxt < tv[-1], -1e9)
        probs = torch.softmax(nxt, dim=-1)
        nid   = torch.multinomial(probs, 1).unsqueeze(0)
        generated.append(nid.item())
        ids   = torch.cat([ids, nid], dim=1)[:, -SEQ_LEN:]
    return decode(ids[0].tolist())

def ngram_set(text: str, n: int) -> set:
    return {text[i:i+n] for i in range(len(text)-n+1)}

def ngram_overlap(generated: str, reference: str, n: int) -> float:
    """Üretilen metindeki n-gram'ların kaçı referansta var? [0,1]"""
    gen_ngrams = ngram_set(generated, n)
    ref_ngrams = ngram_set(reference, n)
    if not gen_ngrams:
        return 0.0
    overlap = gen_ngrams & ref_ngrams
    return len(overlap) / len(gen_ngrams)

def longest_common_substring(s: str, ref: str, min_len=20) -> tuple[int, str]:
    """Üretilen metinde training corpus ile eşleşen en uzun alt dizi."""
    best_len = 0
    best_str = ""
    for length in range(min(len(s), 200), min_len-1, -1):
        found = False
        for i in range(len(s) - length + 1):
            sub = s[i:i+length]
            if sub in ref:
                best_len = length
                best_str = sub
                found = True
                break
        if found:
            break
    return best_len, best_str

def type_token_ratio(text: str) -> float:
    """Kelime çeşitliliği: unique_kelime / toplam_kelime. Yüksek = daha az tekrar."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def self_bleu(texts: list[str], n: int = 3) -> float:
    """
    Self-BLEU: her metin için diğerlerindeki n-gram örtüşmesi.
    Düşük = çeşitli = daha az ezber.
    """
    scores = []
    for i, t in enumerate(texts):
        others = texts[:i] + texts[i+1:]
        ref_ngrams = set()
        for o in others:
            ref_ngrams |= ngram_set(o, n)
        gen_ngrams = ngram_set(t, n)
        if gen_ngrams:
            scores.append(len(gen_ngrams & ref_ngrams) / len(gen_ngrams))
    return sum(scores) / max(len(scores), 1)

def char_freq_kl(text: str, ref: str) -> float:
    """
    Üretilen metnin karakter frekansı ile referans arasındaki KL divergence.
    Düşük = benzer dağılım = corpus'u taklit ediyor (iyi ya da kötü olabilir).
    """
    def freq(t):
        c = collections.Counter(t)
        total = sum(c.values())
        return {k: v/total for k, v in c.items()}
    eps = 1e-9
    p = freq(text)
    q = freq(ref)
    vocab = set(p) | set(q)
    kl = sum(p.get(c, eps) * math.log((p.get(c, eps)) / (q.get(c, eps) + eps))
             for c in vocab)
    return kl

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — PPL Gap (Overfitting Ölçüsü)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("TEST 1: PPL GAP (Overfitting / Generalization)")
print("="*70)
print("Hesaplanıyor... (birkaç dakika sürebilir)")

# Hız için sadece ilk 50K token al
N_EVAL = 50_000
train_ppl = compute_ppl(TRAIN_IDS[:N_EVAL])
val_ppl   = compute_ppl(VAL_IDS[:N_EVAL])
gap_ratio = val_ppl / train_ppl

print(f"\n  Train PPL : {train_ppl:.2f}")
print(f"  Val PPL   : {val_ppl:.2f}")
print(f"  Gap Ratio : {gap_ratio:.2f}x  (1.0=mükemmel, >2.0=overfitting)")

if gap_ratio < 1.3:
    verdict1 = "✓ GENELLEME — Train/Val farkı minimal"
elif gap_ratio < 2.0:
    verdict1 = "~ ORTA — Hafif overfitting var"
else:
    verdict1 = "✗ EZBERLEMİŞ — Train/Val farkı büyük"
print(f"  Karar    : {verdict1}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — N-gram Overlap (Kopya Testi)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("TEST 2: N-GRAM OVERLAP (Corpus Kopyalama Testi)")
print("="*70)

EVAL_PROMPTS = [
    "HAMLET:", "ROMEO:", "KING RICHARD:", "OPHELIA:", "ACT I.\nSCENE I.",
    "To be, or not", "Shall I compare", "Friends, Romans,",
]

all_generated = []
overlap_train = []
overlap_val   = []
lcs_results   = []

for prompt in EVAL_PROMPTS:
    out = generate(prompt, max_new=200, temperature=0.8, top_k=10)
    # Sadece yeni üretilen kısım (prompt sonrası)
    gen_only = out[len(prompt):]
    all_generated.append(gen_only)

    # N-gram overlap
    n = args.ngram
    ov_tr = ngram_overlap(gen_only, TRAIN_TEXT, n)
    ov_va = ngram_overlap(gen_only, VAL_TEXT,   n)
    overlap_train.append(ov_tr)
    overlap_val.append(ov_va)

    # En uzun ortak alt dizi
    lcs_len, lcs_str = longest_common_substring(gen_only, TRAIN_TEXT, min_len=15)
    lcs_results.append((lcs_len, lcs_str, prompt))

print(f"\n  {args.ngram}-gram overlap (üretilen → eğitim corpus'u):")
print(f"  {'Prompt':30s}  {'Train':8s}  {'Val':8s}")
print("  " + "-"*50)
for i, p in enumerate(EVAL_PROMPTS):
    print(f"  {p[:30]:30s}  {overlap_train[i]:.3f}    {overlap_val[i]:.3f}")

avg_tr_ov = sum(overlap_train)/len(overlap_train)
avg_va_ov = sum(overlap_val)/len(overlap_val)
print(f"\n  Ortalama train overlap: {avg_tr_ov:.3f}")
print(f"  Ortalama val overlap  : {avg_va_ov:.3f}")
print(f"\n  NOT: {args.ngram}-gram overlap ~0.3-0.5 beklenir (dil istatistikleri doğal örtüşür).")
print(f"  0.8+ ise büyük olasılıkla doğrudan kopyalama var.")

# En uzun LCS
print(f"\n  En Uzun Ortak Alt Dizi (eğitim corpus'u ile):")
for lcs_len, lcs_str, prompt in sorted(lcs_results, key=lambda x: -x[0]):
    if lcs_len >= 15:
        print(f"  [{lcs_len:3d} karakter] prompt='{prompt[:20]}...'")
        print(f"    \"{lcs_str[:80]}\"")

if max(l for l,_,_ in lcs_results) > 50:
    verdict2 = "✗ EZBERLEMİŞ — 50+ karakterlik kopyalar var"
elif max(l for l,_,_ in lcs_results) > 30:
    verdict2 = "~ ORTA — Kısa öbekler kopyalanıyor (<30 karakter)"
else:
    verdict2 = "✓ GENELLEME — Uzun verbatim kopya yok"
print(f"\n  Karar: {verdict2}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — OOD (Out-of-Distribution) Generalization
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("TEST 3: OOD GENERALIZATION (Shakespeare Dışı Prompt'lar)")
print("="*70)
print("  Not: Char-level model, eğitim alfabe'sini kullanır.")
print("  OOD karakterleri (örn. rakam, emoji) <pad> olur.")
print("  Önemli olan: model tutarlı, gramer olarak makul cümleler üretiyor mu?\n")

ood_prompts = [
    ("Modern İngilizce",   "The president said that"),
    ("Soru formu",         "Why dost thou not"),
    ("Sahne yönergesi",    "Enter a soldier, armed with"),
    ("Hitap",              "My dearest friend, I must"),
    ("Ağıt başlangıcı",    "Alas, poor"),
    ("Şiir başlangıcı",    "Shall I compare thee to"),
]

for label, prompt in ood_prompts:
    out = generate(prompt, max_new=120, temperature=0.85, top_k=15)
    gen_only = out[len(prompt):]
    ttr  = type_token_ratio(gen_only)
    kl   = char_freq_kl(gen_only, TRAIN_TEXT[:50000])
    ov   = ngram_overlap(gen_only, TRAIN_TEXT, 5)
    print(f"  [{label}]")
    print(f"  Prompt : '{prompt}'")
    print(f"  Üretim : '{gen_only[:120].strip()}'")
    print(f"  TTR={ttr:.3f}  KL={kl:.4f}  5-gram_ov={ov:.3f}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Diversity (Self-BLEU)
# ─────────────────────────────────────────────────────────────────────────────

print("="*70)
print("TEST 4: ÇEŞİTLİLİK — Self-BLEU (Aynı Prompt, Farklı Sıcaklıklar)")
print("="*70)

DIV_PROMPT = "HAMLET:"
N = args.n_samples

for temp, topk in [(0.5, 5), (0.8, 10), (1.0, 20), (1.2, 40)]:
    samples = []
    for _ in range(N):
        out = generate(DIV_PROMPT, max_new=100, temperature=temp,
                       top_k=topk, rep_penalty=1.2)
        samples.append(out[len(DIV_PROMPT):])

    sb3 = self_bleu(samples, n=3)
    sb5 = self_bleu(samples, n=5)
    ttrs = [type_token_ratio(s) for s in samples]
    avg_ttr = sum(ttrs)/len(ttrs)

    print(f"  temp={temp}  top_k={topk}:  "
          f"Self-BLEU-3={sb3:.3f}  Self-BLEU-5={sb5:.3f}  "
          f"TTR={avg_ttr:.3f}")
    # Örnek çıktı
    print(f"    örnek: '{samples[0][:80].strip()}'")

print(f"""
  NOT: Self-BLEU yorumu (3-gram):
    0.0-0.3  → Çok çeşitli — model farklı yollar keşfediyor
    0.3-0.6  → Normal — bazı ortak kelime öbekleri var
    0.6-0.9  → Az çeşitli — model belirli kalıpları tekrarlıyor
    0.9-1.0  → Neredeyse aynı çıktılar — muhtemel ezber
""")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Karakter Dağılımı (İstatistiksel Benzerlik)
# ─────────────────────────────────────────────────────────────────────────────

print("="*70)
print("TEST 5: KARAKter DAĞILIMI (KL Divergence)")
print("="*70)

# Büyük miktarda metin üret
BIG_GEN = ""
for p in ["HAMLET:", "ROMEO:", "KING:", "DUKE:", "Enter"]:
    BIG_GEN += generate(p, max_new=300, temperature=0.85, top_k=10)

# Karşılaştırmalar
kl_gen_train = char_freq_kl(BIG_GEN, TRAIN_TEXT[:100000])
kl_gen_val   = char_freq_kl(BIG_GEN, VAL_TEXT)
kl_val_train = char_freq_kl(VAL_TEXT, TRAIN_TEXT[:100000])  # baz çizgisi

# Kelime frekansları
def top_words(text, n=10):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return collections.Counter(words).most_common(n)

print(f"\n  KL Divergence (karakter frekansı):")
print(f"  Üretim   ↔ Train  : {kl_gen_train:.5f}")
print(f"  Üretim   ↔ Val    : {kl_gen_val:.5f}")
print(f"  Val      ↔ Train  : {kl_val_train:.5f}  ← baz çizgisi")
print(f"\n  NOT: KL(üretim,train) ≈ KL(val,train) ise model corpus dağılımını")
print(f"  öğrenmiş ama tam kopya çekmemiş demektir.")

print(f"\n  En sık kelimeler:")
gen_words   = top_words(BIG_GEN, 15)
train_words = top_words(TRAIN_TEXT[:100000], 15)
print(f"  Üretim : {[w for w,_ in gen_words]}")
print(f"  Train  : {[w for w,_ in train_words]}")

# Örtüşen top kelimeler
gen_set   = set(w for w,_ in gen_words)
train_set = set(w for w,_ in train_words)
overlap_w = gen_set & train_set
print(f"  Top-15 kelime örtüşmesi: {len(overlap_w)}/15")
if len(overlap_w) >= 12:
    verdict5 = "✓ GENELLEME — Top kelime dağılımı corpus'a uyuyor"
else:
    verdict5 = "~ ORTA — Top kelime dağılımı kısmen farklı"
print(f"  Karar: {verdict5}")

# ─────────────────────────────────────────────────────────────────────────────
# ÖZET RAPOR
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("ÖZET RAPOR")
print("="*70)

print(f"""
  TEST 1 — PPL Gap       : {verdict1}
             train_ppl={train_ppl:.2f}  val_ppl={val_ppl:.2f}  gap={gap_ratio:.2f}x

  TEST 2 — N-gram Kopya  : {verdict2}
             {args.ngram}-gram train overlap={avg_tr_ov:.3f}

  TEST 3 — OOD           : Model Shakespeare dışı prompt'lara
             gramer yapısını koruyarak yanıt veriyor mu? (yukarıya bak)

  TEST 4 — Çeşitlilik    : Sıcaklık arttıkça Self-BLEU düştü mü?
             (aynı prompt → farklı çıktılar = genelleme)

  TEST 5 — Dağılım       : {verdict5}
             KL(üretim,train)={kl_gen_train:.5f}  baz={kl_val_train:.5f}
""")

# Genel karar
scores = []
if gap_ratio < 1.5:  scores.append(2)
elif gap_ratio < 2.5: scores.append(1)
else: scores.append(0)

max_lcs = max(l for l,_,_ in lcs_results)
if max_lcs < 30:   scores.append(2)
elif max_lcs < 60: scores.append(1)
else: scores.append(0)

if kl_gen_train <= kl_val_train * 2: scores.append(2)
else: scores.append(1)

total = sum(scores)
if total >= 5:
    final = "✓✓ GÜÇLÜ GENELLEME — Model corpus'u ezberlememiş, dil yapısını öğrenmiş"
elif total >= 3:
    final = "✓~ ORTA — Kısmi genelleme, hafif ezber izleri var"
else:
    final = "✗ EZBERLEMİŞ — Model corpus'u ezberlemiş, genelleme zayıf"

print(f"  ╔══════════════════════════════════════════════════════════════╗")
print(f"  ║  GENEL KARAR: {final[:55]:<55s}║")
print(f"  ╚══════════════════════════════════════════════════════════════╝")
print()
