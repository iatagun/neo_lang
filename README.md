# FluidLM — A Language Model Built on Navier-Stokes Dynamics

> **Transformer attention'ı elle tasarladı. FluidLM'de bu etkileşim fizikten türüyor.**

Standart Transformer'da attention şöyle çalışır:
```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V    ← elle tasarlanmış
```

FluidLM'de token etkileşimleri 1-D sıkıştırılamaz Navier-Stokes denkleminden türüyor:
```
∂u/∂t = −(u·∇)u  −  ∇p  +  ν ∇²u
           ↓           ↓       ↓
       adveksiyon   basınç  viskozite
     (anlam taşıma) (attention) (dropout)
```

Basınç alanı `p`, **tüm token dizisinden** aynı anda etkileniyor (Poisson denklemi `∇²p = −∇·adv` ile global). Bu, attention'ın "her token her tokena bakabilir" özelliğini elle tasarlamadan fizikten elde ediyor.

---

## Mimari Karşılaştırması

| Özellik | Transformer | FluidLM |
|---|---|---|
| Token etkileşimi | Q·Kᵀ attention matrisi | ∇p basınç alanı (Poisson) |
| Nonlinearity | FFN (2 linear + ReLU) | Adveksiyon (u·∇)u |
| Regularizasyon | Dropout + LayerNorm | Viskozite ν∇²u |
| Katman başı parametre | 4D² (MHA) + 8D² (FFN) | **2 skaler** (ν, Δt) |
| Derinlik | Sabit N katman | **Adaptif** (ΔKE < eşik → dur) |
| Teori temeli | İnductive bias | Akışkan dinamiği |

---

## Proje Yapısı

```
neo_lang/
├── fluidlm/                        # Çekirdek kütüphane (pip install -e . ile kullanılabilir)
│   ├── __init__.py
│   ├── fluid_ops.py                # ∇, ∇², div, Poisson çözücü (causal, FFT tabanlı)
│   ├── ns_layer.py                 # Navier-Stokes katmanı (causal=True default)
│   └── fluid_lm.py                 # Tam dil modeli + adaptif derinlik + üretim
│
├── experiments/                    # Çalıştırılabilir scriptler (koda dokunmaz)
│   ├── 01_1d_diffusion.py          # 1D difüzyon — temel kavramlar
│   ├── 02_ns_layer_test.py         # NS katmanı analizi
│   ├── 03_toy_lm_train.py          # İlk karakter düzeyi eğitim
│   ├── 04_full_train_and_generate.py
│   ├── 05_shakespeare_full.py      # d=256, L=6 — erken durdurma
│   ├── 05_test_checkpoint.py
│   ├── 06_shakespeare_large.py
│   ├── 07_colab_a100.py            # ★  135M param, char-level — Blackwell
│   ├── 07_eval_memorization.py     # Ezberleme vs genelleme (5 test)
│   └── 08_industry_scale.py        # ★★ ~950M param, BPE, FineWeb-Edu
│
├── checkpoints/                    # Kaydedilen ağırlıklar — git'e dahil değil
│   └── 07_best_model.pt
│
├── results/                        # Grafikler, loglar — git'e dahil değil
│   └── *.png
│
├── configs/
│   └── base_config.yaml
├── data/                           # Ham veri — git'e dahil değil
├── .gitignore
├── requirements.txt
└── LICENSE
```

### SOLID Tasarım İlkeleri

| İlke | Uygulama |
|------|----------|
| **S** — Tek Sorumluluk | `fluidlm/` yalnızca model kodu; `experiments/` yalnızca çalıştırılabilir scriptler; `checkpoints/` yalnızca ağırlıklar |
| **O** — Açık/Kapalı | Yeni deney eklemek `fluidlm/`'e dokunmayı gerektirmiyor |
| **I** — Arayüz Ayrımı | `fluid_ops`, `ns_layer`, `fluid_lm` bağımsız modüller — sadece ihtiyaç duyulan import edilir |
| **D** — Bağımlılık Tersine Çevirme | Experimentler `fluidlm` soyutlamalarına bağlı, somut dosya yollarına değil |

---

## Sonuçlar

### 07 — Karakter Düzeyi FluidLM (135M param)

**Donanım:** NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM)  
**Veri:** TinyShakespeare (~1M karakter)  
**Config:** d=1024, L=16, seq=512, dropout=0.2

| Metrik | Değer |
|--------|-------|
| Val PPL | **5.14** |
| Train PPL | 4.64 |
| Gap Ratio | **1.11×** (overfitting yok) |
| Eğitim süresi | ~21 dakika |
| Erken dur | Epoch 29 / 200 |

**Ezberleme Analizi (`07_eval_memorization.py`):**

| Test | Sonuç |
|------|-------|
| PPL Gap (1.11×) | ✓ Genelleme |
| LCS < 30 karakter | ✓ Verbatim kopya yok |
| Self-BLEU-3 @ temp=0.8 | 0.635 — Normal |
| Uydurma kelimeler (`glimagining`, `birtue`) | ✓ Ezber değil |
| KL(üretim, train) | 0.049 (baz: 0.027) |
| **Genel Karar** | **✓✓ GÜÇLÜ GENELLEME** |

### 08 — Endüstri Ölçeği FluidLM (~950M param)

**Donanım:** NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM)  
**Veri:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — 500M token (streaming)  
**Config:** d=2048, L=24, seq=1024, vocab=50257 (GPT-2 BPE / tiktoken)

| Bileşen | Boyut |
|---------|-------|
| Token embedding | 103M param |
| 24 × NS katmanı | 24 × 33.6M = 807M param |
| **Toplam** | **~950M param** |

**Bellek bütçesi (B=16, grad checkpointing ON):**

```
Parametreler (bf16, ~950M)          : ~1.9 GB
Optimizer state (fp32, param+m+v)   : ~11.4 GB  ← 3 × 950M × 4 bytes
Gradyanlar (bf16)                   : ~1.9 GB
Aktivasyonlar (grad ckpt ile, B=16) : ~0.1 GB
─────────────────────────────────────────────
Toplam                              : ~15–17 GB / 102 GB
```

> Not: AdamW üç fp32 buffer tutar (parametre kopyası + 1. moment + 2. moment). Optimizer state dominant maliyettir.

**Eğitim şu an devam ediyor.**

---

## Kurulum

```bash
pip install -r requirements.txt

# 08 için ek paketler
pip install tiktoken datasets
```

## Deneyleri Çalıştır

```bash
# 07 — Karakter düzeyi eğitim (Colab / Blackwell)
python experiments/07_colab_a100.py

# 07 — Ezberleme analizi (eğitim bitti sonrası)
python experiments/07_eval_memorization.py --ckpt checkpoints/07_best_model.pt

# 08 — Endüstri ölçeği (tek GPU)
python experiments/08_industry_scale.py

# 08 — Çoklu GPU (DDP)
torchrun --nproc_per_node=4 experiments/08_industry_scale.py
```

---

## Temel Kavramlar

### 1 — Token = Hız Alanı

```
"kedi"  → u(x, t=0) ∈ ℝᵈ    (d = embedding boyutu)
katman 1 → u(x, t=0.1)        (NS ile güncelle)
katman 2 → u(x, t=0.2)
   ⋮
çıkış    → u(x, t=1.0) → LM head → kelime olasılıkları
```

Her **katman** bir öncekinin **fiziksel sonucu**.  
Transformer'da katmanlar bağımsız; FluidLM'de katmanlar bir akışkanın zaman içindeki evrimi.

### 2 — Basınç = Attention (fizikten türeyen)

Poisson denklemi `(∇² − α²)p = −div(adv)` tüm diziyi aynı anda görür:
- Spektral çözüm (FFT): her frekans bileşeni bağımsız → `P̂_k = f̂_k / (λ_k − α²)`
- α > 0: Helmholtz regularizasyonu → payda hiçbir zaman sıfır değil, blowup yok
- Uzak tokenlar bile basınç üzerinden birbirini etkiler

Bu, **elle yazılmış** attention yerine **denklemden türeyen** global etkileşimdir.

### 3 — Adaptif Derinlik

```python
for layer in self.layers:
    u, delta_ke = layer(u)
    if adaptive and delta_ke < threshold:
        break   # akış stabilleşti, daha fazla katman gereksiz
```

- Basit cümleler: 3–4 katman yeterli  
- Karmaşık cümleler: 10–12 katman gerekebilir  
- Her cümle için farklı hesap maliyeti → enerji verimli

### 4 — RK4 İntegratör

```
k₁ = F(u)
k₂ = F(u + Δt/2 · k₁)
k₃ = F(u + Δt/2 · k₂)
k₄ = F(u + Δt   · k₃)
u_new = u + Δt/6 · (k₁ + 2k₂ + 2k₃ + k₄)
```

Euler yerine RK4 kullanmak: daha büyük Δt → daha az katman → daha verimli.

---

## Tasarım Kararları

### x nedir — token pozisyonu mu, embedding boyutu mu?

`x = token pozisyonu` (sequence index, L boyutu).

Tüm diferansiyel operatörler (`∂/∂x`, `∂²/∂x²`) L boyutu üzerinde çalışır.
Her token `u[b, i, :] ∈ ℝᴰ`, D-boyutlu bir durum vektörü taşır.
Bu, 1 uzamsal boyutlu, D-bileşenli bir vektör alanıdır.

```
u[b, i, d] = batch b'nin i. tokeninin d. embedding bileşeni
              ^            ^              ^
              batch        x = pozisyon   alan bileşeni
```

**Alternatif** (`x = embedding boyutu`) düşünülüp reddedildi: o durumda her token kendi içinde bağımsız bir alan olur, tokenlar arası bilgi akışı olmaz — global context kaybolur.

**Kısıtlama:** `x = L` seçimi, sequence length'in sabit olmasını gerektirmez (periodic BC padding ile farklı uzunluklar desteklenebilir), ama mevcut implementasyonda batch içinde sabit L varsayılmaktadır.

---

### Backpropagation — adjoint method gerekiyor mu?

**Hayır, gerekmiyor** — mevcut tasarımda.

`integrator='euler'` ile ileri adım şöyle:

```
u_new = u + dt · F(u)
```

Bu bir **residual bağlantıdır** (ResNet katmanı ile özdeştir). PyTorch autograd bunu tam olarak, O(n\_layers · B · L · D) bellekle differentiate eder. Adjoint method gerekmez.

`integrator='rk4'` ile 4 ara tensor vardır, bellek 4× artar ama yine de O(L·D) per sample düzeyinde kalır — pratik sequence uzunlukları için sorun değil.

**Adjoint method ne zaman gerekir?** Çok sayıda integrasyon adımı (örn. ODE solver ile adaptive step-size kontrolü, binlerce adım) uygulandığında bellek O(adım\_sayısı · B · L · D)'ye çıkar. O durumda `torchdiffeq.odeint_adjoint` kullanılarak bellek O(B·L·D)'ye düşürülebilir. Bu, **Neural ODE**'nin yaptığı tam olarak budur. Mevcut sabit-adım Euler/RK4'te bu trade-off gereksizdir.

---

### ∇·u = 0 (sıkıştırılamazlık) varsayımı neden yok?

**Fiziksel akışkanlıkta** `∇·u = 0` kütlenin korunmasından (süreklilik denklemi) gelir. Dil modelinde korunan bir "kütle" yoktur — bu fiziksel kısıtın dil alanında geçerli olmasını gerektiren bir neden yoktur.

**Mevcut implementasyon compressible NS kullanır:**

- `∇·u ≠ 0` olmasına izin verilir
- `p` basınç alanı, `u`'yu divergence-free alt uzayına **proje etmez**
- `p` yalnızca global bir **coupling sinyali** olarak kullanılır: advection'ın lokal ıraksama bilgisi spektral çözümle tüm tokenlar arasında yayılır

Bu tercih iki somut avantaj sağlar:

1. **Helmholtz projeksiyonu atlanır** → her katmanda ekstra bir FFT + assign gerekmez
2. **Model daha esnek** → sıkıştırılamazlık kısıtı öğrenmeyi gereğinden fazla sınırlamaz

Gelecekte `∇·u = 0` zorlamak istenirse Helmholtz ayrışımı eklenerek `u ← u − ∇φ` projeksiyonu yapılabilir (bir FFT çözüm ek maliyet).

---

## Baseline Karşılaştırması

FluidLM sonuçlarının anlamlı olabilmesi için **aynı veri, aynı donanım, benzer parametre sayısında** standart Transformer (nanoGPT) ile karşılaştırılması gerekir.

```bash
# Her iki config'i çalıştır (Config-A: aynı d/L, Config-B: param-eşit)
python experiments/09_baseline_transformer.py --epochs 200
```

| Model | Parametre | Val PPL | Notlar |
|-------|-----------|---------|--------|
| FluidLM (d=1024, L=16) | ~135M | **5.14** | NS katmanı: 4 skaler + MLP |
| GPT Config-A (d=1024, L=16, h=16) | ~202M | _çalıştırılıyor_ | Aynı d/L, fazladan MHA |
| GPT Config-B (d=768, L=12, h=12) | ~90M | _çalıştırılıyor_ | Param-eşit |

> FluidLM'in avantajı parametre verimliliğidir: MHA (4D²/katman) yerine 4 skaler fiziksel parametre kullanıyor.

---



### Causal Navier-Stokes Operatörler

Standart NS'te basınç tüm diziyi aynı anda görür (global). Dil modeli için **nedensellik** (causality) zorunlu — model gelecek tokenlara bakamaz. Bunu sağlamak için tüm operatörler **geriye fark** (backward difference) ile yeniden tanımlandı:

```python
# Gradient: sadece geçmiş tokenları kullan
∇u[i] = u[i] - u[i-1]   (causal backward difference)

# Laplacian:
∇²u[i] = u[i] - 2u[i-1] + u[i-2]

# Basınç: cumsum ile kausal integrasyon
p = cumsum(-div) × α  →  normalize(detach std)
```

Bu tasarım, `α` ve `p_scale` parametrelerinin gradyan almasını korurken std normalizasyonunun `α`'nın etkisini silmesini engeller.

### Fiziksel Parametre Öğrenimi

Her NS katmanının 4 öğrenilebilir fiziksel parametresi vardır:

| Parametre | Fiziksel Anlam | Başlangıç |
|-----------|----------------|-----------|
| `ν` (nu) | Viskozite — pürüzleştirme | `0.01 × (1 + 0.05i)` |
| `Δt` (dt) | Zaman adımı — katman derinliği | `0.05 × (1 + 0.02i)` |
| `α` (alpha) | Basınç ölçeği — global etki | 1.0 |
| `p_scale` | Basınç gradyanı ağırlığı | 0.1 |

Her katmana farklı başlangıç değeri verilir (simetri kırma) — böylece katmanlar farklı fiziksel dinamikler öğrenir.

---

## Öğrenme Yol Haritası

Bu mimariyi anlamak için önerilen sıra:

1. **3Blue1Brown** — "Differential Equations" YouTube serisi *(3 gün)*
2. **MIT OCW 18.336** — Numerical Methods for PDEs *(1 hafta)*
3. **Spectral Methods in Fluid Dynamics** — Canuto et al., Bölüm 1–3
4. **MIT 6.003** — LTI sistemler ve Laplace dönüşümü

Pratik başlangıç: `experiments/01_1d_diffusion.py` — sadece numpy ile 20 satır.

---

## Referanslar

- Navier, C-L. (1827). *Mémoire sur les lois du mouvement des fluides.*
- Stokes, G. G. (1845). *On the theories of the internal friction of fluids in motion.*
- Canuto, C. et al. (1988). *Spectral Methods in Fluid Dynamics.* Springer.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Chen, R. T. Q. et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS.
- Press, O. & Wolf, L. (2017). *Using the Output Embedding to Improve Language Models.* EACL.
