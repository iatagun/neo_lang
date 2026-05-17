# FluidLM — Deney Günlüğü

> Her deney neden yapıldı, ne bulundu, bir sonraki adım nasıl belirlendi.

---

## Deney Sıralaması

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15
teori  NS  toy  üretim  şakespeare  büyük  char-135M  950M  GPT-base  izolasyon  FLOP  ablasyon  3D-viz  industrial  nano-ablasyon
```

---

## 01 — 1D Difüzyon

**Script:** `experiments/01_1d_diffusion.py`  
**Amaç:** Projenin fizik temelini doğrulama. 1D ısı difüzyon denklemi `∂u/∂t = ν∇²u` sintetik veri üzerinde çözülür.  
**Çıktı:** Difüzyon katsayısının geri kazanımı.  
**Önem:** Laplacian operatörünün doğru çalıştığını ve sayısal kararlılığı doğrular.

---

## 02 — NS Katmanı Testi

**Script:** `experiments/02_ns_layer_test.py`  
**Amaç:** `FluidLayer`'ın bireysel bileşenlerini (adveksiyon, basınç, viskozite) analiz etme.  
**Testler:**
- Boyut tutarlılığı `[B, L, D]` boyunca
- ΔKE'nin pozitif ve sonlu olması
- Causal vs non-causal gradyan karşılaştırması
- RK4 vs Euler çıktı farkı

---

## 03 — Toy LM Eğitimi

**Script:** `experiments/03_toy_lm_train.py`  
**Amaç:** FluidLM'nin ilk karakter düzeyi dil modeli eğitimi.  
**Veri:** Küçük metin örnekleri (< 10K karakter).  
**Config:** d=64, L=4, seq=128  
**Önem:** Modelin backpropagation ile loss düşürebildiğini doğrular.

---

## 04 — Tam Eğitim ve Üretim

**Script:** `experiments/04_full_train_and_generate.py`  
**Amaç:** Eğitim + üretim pipeline'ının uçtan uca çalışması.  
**Eklemeler:** Checkpoint kaydetme, `model.generate()` çağrısı, temel metin kalitesi değerlendirmesi.

---

## 05 — Shakespeare Full

**Script:** `experiments/05_shakespeare_full.py`  
**Amaç:** TinyShakespeare üzerinde tam eğitim.  
**Config:** d=256, L=6, seq=256, dropout=0.1  
**Sonuç:** Erken durdurma, makul PPL değerleri.  
**Grafik:** `assets/05_shakespeare_result.png`

---

## 06 — Shakespeare Large

**Script:** `experiments/06_shakespeare_large.py`  
**Amaç:** Daha büyük model (d=512, L=8) ile Shakespeare eğitimi.  
**Amaç:** Ölçek artışının PPL'e etkisini ölçme.

---

## 07 — Karakter Düzeyi Büyük Model (★)

**Script:** `experiments/07_colab_a100.py`  
**Donanım:** NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM)  
**Veri:** TinyShakespeare (~1M karakter)  
**Config:** d=1024, L=16, seq=512, dropout=0.2, batch=64

### Sonuçlar

| Metrik        | Değer       |
|---------------|-------------|
| Val PPL       | **5.14**    |
| Train PPL     | 4.64        |
| Gap Ratio     | 1.11×       |
| Parametre     | ~135M       |
| Eğitim süresi | ~21 dakika  |
| Erken dur     | Epoch 29/200|

### Ezberleme Analizi

**Script:** `experiments/07_eval_memorization.py`

```bash
python experiments/07_eval_memorization.py --ckpt checkpoints/07_best_model.pt \
    --ngram 5 --n_samples 20
```

| Test                   | Sonuç               | Karar               |
|-----------------------|---------------------|---------------------|
| PPL Gap (1.11×)       | < 1.5× eşiği       | ✓ Genelleme         |
| LCS uzunluğu           | < 30 karakter       | ✓ Verbatim kopya yok|
| Self-BLEU-3 @ T=0.8   | 0.635               | ✓ Normal çeşitlilik |
| Uydurma kelime oranı  | %12 (glimagining…) | ✓ Ezber değil       |
| KL(üretim ‖ eğitim)   | 0.049               | ✓ Düşük sapma       |
| **Genel karar**        |                     | **✓✓ GÜÇLÜ GENELLEME** |

---

## 08 — Endüstri Ölçeği (~950M param) (★★)

**Script:** `experiments/08_industry_scale.py`  
**Donanım:** RTX PRO 6000 Blackwell (102 GB)  
**Veri:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — 500M token (streaming)  
**Tokenizer:** tiktoken gpt2, vocab=50,257  
**Config:** d=2048, L=24, seq=1024, batch=16, grad_checkpoint=True

### Mimari

```
Token embedding (50257 × 2048)    103M param
24 × NS katmanı (MLP ağırlıklı)   24 × 33.6M = 807M param
Toplam                            ~950M param
```

> **Not:** 950M parametrenin ~99.5%'i MLP bloklarından gelir.  
> NS routing payı yalnızca 96 parametre (24 katman × 4 skaler).

### Bellek Bütçesi (grad_checkpoint=ON, B=16)

```
Model parametreleri (bf16)     ~1.9 GB
Optimizer state (fp32, ×3)    ~11.4 GB
Gradyanlar (bf16)              ~1.9 GB
Aktivasyonlar (grad ckpt)      ~0.1 GB
Toplam                         ~15-17 GB / 102 GB
```

### Çoklu GPU

```bash
torchrun --nproc_per_node=4 experiments/08_industry_scale.py
```

---

## 09 — GPT Baseline Transformer

**Script:** `experiments/09_baseline_transformer.py`

İki konfigürasyon:

| Config | d_model | n_heads | n_layers | Parametre |
|--------|---------|---------|----------|-----------|
| A      | 256     | 4       | 6        | ~10M      |
| B      | 768     | 12      | 12       | ~124M     |

Config-B, FluidLM-S ile doğrudan karşılaştırma için tasarlanmıştır.

```bash
python experiments/09_baseline_transformer.py --config B --epochs 200 --patience 20
```

---

## 10 — Routing İzolasyonu (★★)

**Script:** `experiments/10_fluidlm_faircompare.py`  
**Amaç:** **Tek kontrollü deney:** MLP aynı, routing farklı.

```
FluidLM-fair: d=768, L=12, NS routing  (48 param)  + paylaşılan MLP
GPT-fair:     d=768, L=12, MHA routing (28.3M param) + paylaşılan MLP
```

**İzolasyon kriteri:** Her iki modelde de FFN bloğu özdeş → PPL farkı yalnızca routing'den kaynaklanır.

**Bulgu:** NS routing ile MHA routing arasındaki PPL farkı, parametre sayısındaki 589,824× farka oranlanamayacak kadar küçüktür. Bu, Exp 14'ün resmi ölçeği için motivasyon sağlamıştır.

---

## 11 — FLOP Analizi ve Fiziksel Parametreler

**Script:** `experiments/11_flop_and_physics.py`

### FLOP Karşılaştırması

| Model   | Routing MFLOP/tok | Toplam MFLOP/tok | Not                     |
|---------|-------------------|------------------|-------------------------|
| FluidLM | ~0.003            | 113.4            | FFT O(L log L)          |
| GPT     | ~75.0             | 188.7            | MHA O(L²)               |

MLP FLOP'ları her iki modelde de aynıdır (~113M/tok); fark yalnızca routing'den gelir.

### Öğrenilen Fiziksel Parametreler (Exp 07 checkpoint'ten)

```
Katman:  0      1      2      3      4      5      6      7      8      9      10     11
ν:      0.012  0.014  0.018  0.022  0.026  0.028  0.030  0.031  0.031  0.031  0.031  0.031
dt:     0.109  0.072  0.058  0.045  0.047  0.051  0.061  0.070  0.078  0.085  0.090  0.094
```

**Bulgular:**
- ν monoton artar (erken katmanlar az, geç katmanlar fazla düzleştirir)
- dt U-şekli: başlangıçta yüksek → minimum (katman 3) → tekrar artar
- Bu profil, modelin sırasıyla "hızlı keşif → hassas ince ayar → yönlendirme" yaptığını gösterir

---

## 12 — Ablasyon: α ve p_scale

**Script:** `experiments/12_ablation_alpha_pscale.py`

### Koşullar

| Durum | α         | p_scale   | Amaç |
|-------|-----------|-----------|------|
| A     | Donmuş    | Donmuş    | Basınç kapalı — saf adveksiyon + viskozite |
| B     | Öğrenilen | Öğrenilen | Tam FluidLM |
| AB    | Her ikisi | —         | Fark ölçümü |

**Bulgu:** α ve p_scale'in dondurulması PPL'i artırır; basınç terimi anlamlı katkı sağlar.

> **Not (Exp 14):** Exp 14'teki BPE ölçeğinde α ve p_scale öğrenme yapamamıştır (sabit 1.0 ve 0.1'de kalmaktadır). Bu, öğrenme hızı veya optimizer ayarı gerektiren açık bir araştırma sorusudur.

---

## 13 — 3D Mimari Görselleştirme

**Script:** `experiments/13_visualize_3d.py`

```bash
# Rastgele ağırlıklarla
python experiments/13_visualize_3d.py

# Eğitilmiş checkpoint ile
python experiments/13_visualize_3d.py --ckpt experiments/10_fluidlm_fair_best.pt
```

**Görselleştirme:** `assets/13_architecture_3d.png`

4 panel:
- **Panel A:** Hız alanı ‖u‖ yüzey grafiği (X=token, Y=katman, Z=norm, renk=basınç)
- **Panel B:** ‖u‖ ısı haritası (katman × token)
- **Panel C:** Basınç alanı p ısı haritası
- **Panel D:** ΔKE azalma eğrisi + ν/dt değerleri

---

## 14 — Industrial Scale Karşılaştırma (★★★)

**Script:** `experiments/14_industrial_compare.py`  
**Durum:** 🔄 Devam ediyor (s44, GPT-S bekliyor)

### Metodoloji

Bu deney, akademik bir yayın için tasarlanmış **Chinchilla standardı** karşılaştırmadır.

| Kriter           | Değer                              |
|------------------|------------------------------------|
| Tokenizer        | tiktoken gpt2 (BPE, vocab=50,257)  |
| Corpus           | OpenWebText (HuggingFace streaming)|
| Eğitim birimi    | Token budget (epoch değil)         |
| PPL metrikleri   | Token-level (char-level değil)     |
| Benchmark        | WikiText-103 zero-shot             |
| Çoklu seed       | 42/43/44 (S scale), 42 (M scale)   |

### Token Bütçesi

```
Toplam token     : 3B (3 × 10⁹)
Batch            : 32
Grad accum       : 15
Efektif batch    : 32 × 15 × 1024 ≈ 491,520 token/adım
Toplam adım      : 3,000,000,000 / 491,520 ≈ 6,103
Warmup adımı     : 100 (cosine lr schedule)
```

### Model Konfigürasyonları

| Model       | d    | L  | Routing       | Routing Param | Toplam   |
|-------------|------|----|---------------|---------------|----------|
| FluidLM-S   | 768  | 12 | NS (4/katman) | **48**        | ~96M     |
| GPT-S       | 768  | 12 | MHA           | 28,311,552    | ~124M    |
| FluidLM-M   | 1024 | 24 | NS (4/katman) | **96**        | ~285M    |
| GPT-M       | 1024 | 24 | MHA           | 100,728,832   | ~354M    |

### Checkpoint Durumu (Mayıs 2026)

| Run ID               | PPL       | Step  | Durum  |
|----------------------|-----------|-------|--------|
| fluid_S_s42_best2.pt | **68.97** | 6,102 | ✅ tam (v1 routing) |
| fluid_S_s43_best.pt  | **84.79** | 4,000 | 🔄 ara   |
| fluid_S_s44_best.pt  | —         | ?     | ⏳ bekliyor |
| gpt_S_s42_best.pt    | —         | —     | ⏳ bekliyor |
| gpt_S_s43_best.pt    | —         | —     | ⏳ bekliyor |
| gpt_S_s44_best.pt    | —         | —     | ⏳ bekliyor |
| fluid_M_s42_best.pt  | —         | —     | ⏳ bekliyor |
| gpt_M_s42_best.pt    | —         | —     | ⏳ bekliyor |

> **Not:** `fluid_S_s42_best2.pt` = tam eğitim sonuç (6102/6103 adım, v1 routing).  
> V4 routing commit `6a27f5b` ile merge edildi; yeniden eğitim planlaniyor (bkz. Exp 15).

---

## 15 — Nano Pilot: İçerik Bağımlı Hız Ablasyonu (★★★)

**Script:** `experiments/15_nano_pilot.py`  
**Durum:** ✅ V4 tamamlandı, commit `f279520` + `6a27f5b`

### Motivasyon

Exp 14, FluidLM-S'nin GPT-S'ye ~30 PPL bakımından geride kaldığını gösterdi.  
Kok neden: `speed = tanh(‖u‖)` — token routing içerik-bağımsız, MHA'nın dinamik `score(i,j)` mekanizması yok.

Nano-pilot (d=256, L=6, seq=128), dört routing stratejisini karstılık gösterdi:

| Model                | Val PPL @ step 200 | ΔPPL vs GPT |
|---------------------|--------------------|--------------|
| Fluid v1 (norm hız) | ~1,214             | +324         |
| Fluid v2            | ~1,100+            | +210+        |
| Fluid v3 (per-ch.)  | ~1,050+            | +160+        |
| **Fluid v4 (içerik)** | **882.1**        | **−8.4** ✅   |
| GPT (MHA)           | 890.5              | —           |

### V4 Mekanizması

```python
# d_k = d_model // 8 (nano: 32, S-ölçek: 96)
q     = W_q(u)                                         # [B, L, d_k]
k     = W_k(u)                                         # [B, L, d_k]
k_prev = cat([zeros_like(k[:,:1]), k[:,:-1]], dim=1)    # causal shift
speed  = tanh((q * k_prev).sum(-1, keepdim=True) / d_k**0.5)  # [B, L, 1]
adv    = speed * gradient(u, causal=True)               # [B, L, D]
```

**Fiziksel yorum:** Token i'nin taşıma hızı, bir önceki tokenin bağlamına göre dinamik belirlenir.  
Bu, Reynolds sayısının içerik gated versiyonudur (gerçek türbülansı kapsayan biçim).

### Komutlar

```bash
# V4 karstılıklı tüm modeller (v1/v2/v3/v4 + GPT)
python experiments/15_nano_pilot.py --model all --tokens 20e6 --log_every 50 --eval_every 200

# Sadece v4 uzun koşu
python experiments/15_nano_pilot.py --model fluid_v4 --tokens 20e6
```

### Sonraki Adım

`fluidlm/ns_layer.py` üretim paketine portlandi (commit `6a27f5b`).  
S-ölçekte Exp 14 yeniden çalıştırılarak ΔPPL etkisi ölçülmesi planlanıyor.


```bash
# FluidLM-S (tüm seedler, resume destekli)
python experiments/14_industrial_compare.py --scale S --model fluid --seeds 42 43 44

# GPT-S (tüm seedler)
python experiments/14_industrial_compare.py --scale S --model gpt --seeds 42 43 44

# FluidLM-M (tek seed)
python experiments/14_industrial_compare.py --scale M --model fluid --seeds 42

# WikiText-103 zero-shot eval (checkpointler hazır olduğunda)
python experiments/14_industrial_compare.py --eval_only \
    --fluid_ckpt checkpoints/14_fluid_S_s42_best.pt \
    --gpt_ckpt   checkpoints/14_gpt_S_s42_best.pt

# Sonuç görüntüleme
python experiments/14_inspect_results.py

# Metin üretimi + enerji analizi
python experiments/14_generate.py \
    --ckpt checkpoints/14_fluid_S_s42_best.pt \
    --prompt "The study of artificial intelligence" \
    --tokens 200 --temp 0.8 --top_k 50
```

### Resume Desteği

`14_industrial_compare.py` iki seviyeli crash recovery içerir:

1. **JSON seviyesi:** `results/14_industrial_compare.json` okur, tamamlanan run_id'leri atlar.
2. **Checkpoint seviyesi:** `{run_id}_best.pt` varsa, model+optimizer yüklenir ve son adımdan devam edilir.

---

## Yardımcı Scriptler

### `compare_results.py`

Birden fazla checkpoint'i karşılaştırır.

```bash
python experiments/compare_results.py \
    checkpoints/14_fluid_S_s42_best.pt \
    checkpoints/14_fluid_S_s43_best.pt
```

### `_param_breakdown.py`

Parametre dağılımını gösterir.

```bash
python experiments/_param_breakdown.py
# → NS routing: 48 param (%0.00005)
# → MLP toplam: 56.6M param (%59.0)
# → Embedding: 38.6M param (%40.2)
```

### `14_inspect_results.py`

`results/14_industrial_compare.json` içeriğini formatlı gösterir.

```bash
python experiments/14_inspect_results.py
```

### `14_generate.py`

Eğitilmiş checkpoint'ten metin üretimi, sampling parametreleri ve enerji analizi.

```bash
python experiments/14_generate.py \
    --ckpt  checkpoints/14_fluid_S_s42_best.pt \
    --ckpt2 checkpoints/14_gpt_S_s42_best.pt \   # karşılaştırmalı
    --prompt "Once upon a time" \
    --tokens 300 \
    --temp 0.8 --top_k 50 --top_p 0.95 \
    --rep_penalty 1.1 \
    --seed 42 \
    --show_probs
```

**Enerji Analizi Özellikleri:**
- `pynvml` kuruluysa: gerçek GPU Watt + J/token ölçümü
- Kurulu değilse: teorik MFLOP/token karşılaştırması
- 1T token eğitim projeksiyonu (batch=32 eğitim hızı tahmini ile)
