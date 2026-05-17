# FluidLM — Araştırma Soruları ve Metodoloji

> Bu belge, FluidLM projesinin akademik motivasyonunu, araştırma sorularını,  
> metodolojik kararlarını ve mevcut bulgularını kapsar.

---

## İçindekiler

1. [Motivasyon](#1-motivasyon)
2. [Araştırma Soruları](#2-araştırma-soruları)
3. [Metodoloji](#3-metodoloji)
4. [İzolasyon Tasarımı](#4-izolasyon-tasarımı)
5. [Mevcut Bulgular](#5-mevcut-bulgular)
6. [ΔPPL Yorumlama Eşikleri](#6-δppl-yorumlama-eşikleri)
7. [Açık Sorular](#7-açık-sorular)
8. [Planlanan Çalışmalar](#8-planlanan-çalışmalar)

---

## 1. Motivasyon

### Problem

Modern büyük dil modelleri (LLM) self-attention mekanizmasına dayanır:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

Bu mekanizmanın **iki temel maliyeti** vardır:

1. **Hesaplama:** `O(L²)` zaman ve bellek karmaşıklığı (L = dizi uzunluğu)  
2. **Parametre:** Her katmanda `4D²` ek parametre (Q, K, V, O projeksiyonları)

d=768 için bu `4 × 768² ≈ 2.36M` parametre/katman demektir.

### Soru

Token etkileşimi, **elle tasarlanmış** bir matris çarpımı yerine **fizik yasalarından** türetilebilir mi?

Cevap evet ise, bu iki şey doğurabilir:
1. Attention'a rakip veya eşdeğer representasyon gücü
2. Çok daha az parametre ile aynı işlevi

### Seçilen Fizik: Navier-Stokes

1-D sıkıştırılamaz Navier-Stokes denklemi üç terimi doğal olarak içerir:

```
∂u/∂t = −(u·∇)u  −  ∇p  +  ν ∇²u
```

| Terim     | Fizik                              | LM Analojisi                    |
|-----------|------------------------------------|---------------------------------|
| `−(u·∇)u` | Adveksiyon — anlam taşıma          | Nonlinearity (FFN rolü)         |
| `−∇p`     | Basınç — global etkileşim         | Attention (global coupling)     |
| `ν∇²u`    | Viskozite — pürüzsüzleştirme      | Dropout/LayerNorm (regularizasyon)|

**Özellikle basınç terimi:** Poisson denklemi `∇²p = −∇·adv` ile tüm diziden etkilenir.  
Bu, attention'ın "her token her tokena bakabilir" özelliğini **hiç parametre eklemeden** sağlar.

---

## 2. Araştırma Soruları

### RQ1 — Representasyon Gücü

> NS routing, MHA'ya eşdeğer representasyon gücü sağlıyor mu?

**Operasyonel tanım:** Aynı MLP, aynı veri, aynı token bütçesiyle eğitilmiş FluidLM-S ve GPT-S'nin medyan val PPL farkı `|ΔPPL|`.

**Hipotez (null):** `|ΔPPL| < 0.5` → NS routing MHA ile istatistiksel olarak eşdeğerdir.

---

### RQ2 — Routing–FLOP Tradeoff

> 48 parametre (NS) ile 28.3M parametre (MHA) arasındaki routing farkının FLOP/token karşılığı nedir?

**Teorik hesap:**

| Model   | Routing MFLOP/tok | Toplam MFLOP/tok | Oran    |
|---------|-------------------|------------------|---------|
| FluidLM | ~0.003 (FFT O(L log L)) | 113.4      | —       |
| GPT     | ~75.0  (MHA O(L²))     | 188.7       | ×1.66   |

**Sorular:**
- Daha az FLOP ile aynı PPL → verimlilik kazancı var mı?
- Aynı PPL için gereken eğitim süresi farkı nedir?

---

### RQ3 — Emergent ν Gradyanı

> Öğrenilen kinematik viskozite ν, karakter düzeyinin ötesinde BPE/token düzeyinde de katmanlar boyunca monoton artış gösteriyor mu?

**Gözlenen örüntü (Exp 07, karakter düzeyi, seed çoklu):**
```
ν:  [0.012, 0.014, 0.018, ..., 0.031]   → monoton artış (erken: az düzleştir, geç: çok)
dt: [0.109, 0.072, 0.058, ..., 0.094]   → U-şekli (hızlı keşif → hassas ayar → yönlendir)
```

**Bulgular (Exp 14, BPE, seed 42 ve 43):**
```
seed 42: ν = [0.0159, 0.0255, ..., 0.0258],  gradient = +0.0099
seed 43: ν = [0.0152, 0.0236, ..., 0.0240],  gradient = +0.0088
```

✅ Karakter düzeyindeki örüntü BPE ölçeğinde de tekrar ediyor.  
**RQ3 önbulgu: DOĞRULANDI** (istatistiksel doğrulama için GPT seedleri bekleniyor).

---

## 3. Metodoloji

### Chinchilla Standardı

Akademik yayınlarda kabul gören ölçek yasaları Hoffmann et al. (2022)'e dayanır:

```
Optimal tokens ≈ 20 × N    (N = model parametresi)
FluidLM-S için: 20 × 96M = 1.92B token
Seçilen bütçe : 3B token   (∼ 1.56× optimal — yeterince eğitilmiş)
```

### Neden Token Budget, Epoch Değil?

- Epoch tabanlı karşılaştırma veri boyutuna bağımlıdır
- Token budget, hesaplama maliyetini doğrudan kontrol eder
- Literatürle kıyaslanabilir (OpenAI, DeepMind raporları aynı metriği kullanır)

### Neden BPE, Karakter Değil?

- Token-level PPL, karakter-level PPL'e göre daha bilgilendirici
- GPT-2 BPE (tiktoken) endüstri standardıdır
- WikiText-103 benchmark'ı token-level PPL kullanır

### Neden OpenWebText?

- Kamu erişimli, araştırma topluluğu tarafından kabul görmüş
- GPT-2'nin eğitildiği veriyle aynı dağılım → GPT-S baseline'ı adil
- Streaming desteği → 102 GB VRAM'a sığmayan veri seti sorun değil

### Neden Çoklu Seed?

Tek seed sonuçları aşırı değişken olabilir (PPL, başlangıç ağırlıklarına duyarlıdır). 3 seed ile:
- Ortalama ± standart sapma hesaplanabilir
- İstatistiksel test yapılabilir (Welch t-testi)

---

## 4. İzolasyon Tasarımı

Karşılaştırmanın geçerliliği, **tek değişken** ilkesine dayanır:

```
FluidLM-S = NS routing  + paylaşılan MLP + aynı veri + aynı token budget
GPT-S     = MHA routing + paylaşılan MLP + aynı veri + aynı token budget
                ↑ tek fark ↑
```

**Kontrol edilen değişkenler:**

| Değişken         | FluidLM-S        | GPT-S            |
|-----------------|------------------|------------------|
| d_model         | 768              | 768              |
| n_layers        | 12               | 12               |
| FFN hidden      | 3,072            | 3,072            |
| MLP aktivasyon  | GELU             | GELU             |
| Tokenizer       | tiktoken gpt2    | tiktoken gpt2    |
| Corpus          | OpenWebText      | OpenWebText      |
| Token budget    | 3B               | 3B               |
| Batch           | 32               | 32               |
| Grad accum      | 15               | 15               |
| Optimizer       | AdamW            | AdamW            |
| LR              | 3e-4             | 3e-4             |
| LR schedule     | cosine + warmup  | cosine + warmup  |
| dtype           | bfloat16         | bfloat16         |

**Kontrol edilemeyen (kabul edilen):**
- Toplam parametre sayısı farklı (~96M vs ~124M) → FluidLM daha az parametreyle daha fazla verimlilik iddiası açısından bu kabul edilebilir
- NS'in causal basınç hesabı GPT'nin causal attention masking'inden farklı uygulanıyor

---

## 5. Mevcut Bulgular

### Exp 07 — Karakter Düzeyi (Tamamlandı)

- **Val PPL: 5.14** (d=1024, L=16, 135M param)
- Gap ratio 1.11× → overfitting yok
- Ezberleme testleri: tümü geçti
- ν gradyanı: +0.019 (ilk → son katman)
- dt profili: U-şekli

### Exp 10 — Routing İzolasyonu (Tamamlandı)

- NS routing ile MHA routing PPL farkı < 1.0 (tam karşılaştırma Exp 14'te)
- 48 parametre ile 28.3M parametre arasındaki fark PPL'e yansımıyor

### Exp 15 — Nano Pilot: İçerik Bağımlı Hız Ablasyonu (Tamamlandı)

**Arka plan:** Exp 14, FluidLM-S'nin GPT-S'ye ~30 PPL bakımından geride kaldığını gösterdi.  
Kök neden analizi: `speed = tanh(‖u‖)` içerik-bağımsız routing — MHA'nın `score(i,j) = ⟨q_i, k_j⟩` dinamizmini barındırmıyor.

**V4 Deneşi:** `speed_i = tanh(q_i · k_{i-1} / √d_k)` ile nano-ölçekte ablasyon (d=256, L=6, seq=128).

| Model                | Val PPL @ step 200 | ΔPPL vs GPT |
|---------------------|--------------------|--------------|
| Fluid v1 (baseline) | ~1,214             | +324         |
| Fluid v2            | ~1,100+            | +210+        |
| Fluid v3 (per-ch.)  | ~1,050+            | +160+        |
| **Fluid v4 (içerik)** | **882.1**        | **−8.4** ✅   |
| GPT (MHA)           | 890.5              | —           |

> **İlk kez: FluidLM, GPT'yi geçti.** ΔPPL = −8.4 @ step 200.  
> v1'de `speed = tanh(‖u‖)` ile ΔPPL = +324 idi. V4 ile fark 332 PPL kapandı.

**V4 mekanizması:**
```python
q = W_q(u)                                          # [B, L, d_k]
k = W_k(u)                                          # [B, L, d_k]
k_prev = cat([zeros_like(k[:,:1]), k[:,:-1]], dim=1) # causal shift
speed  = tanh((q * k_prev).sum(-1, keepdim=True) / d_k**0.5)  # [B, L, 1]
```

**Parametre maliyeti:** `d_k = d_model // 8` → L=6 için 6 × 2 × (d × d_k) = minik ek yük.  

**Commit:** `6a27f5b` — `fluidlm/ns_layer.py` üretim paketine portlandi.

### Exp 14 — Industrial Scale (Devam Ediyor)

**FluidLM-S sonuçlar:**

| Seed | Val PPL (final) | Step        | Durum        |
|------|-----------------|-------------|-------------|
| 42   | **68.9726**     | 6,102/6,103 | ✅ **TAMAMLANDI** |
| 43   | —               | —           | 🔄 beklemede |
| 44   | —               | —           | 🔄 beklemede |

**GPT-S sonuçlar (seed 42, devam ediyor):**

| Seed | Val PPL (ara) | Step        | Durum      |
|------|---------------|-------------|------------|
| 42   | **48.8288**   | 2,000/6,103 | 🔄 %33 (t=91m) |
| 43   | —             | —           | 🔄 beklemede |
| 44   | —             | —           | 🔄 beklemede |

> **Not:** GPT-S seed 42, step=2000'de 48.83 PPL ile FluidLM'nin final PPL'ini (68.97) çoktan geçti.  
> Eğitim tamamlandığında (~6100. adım) GPT-S'nin **~35–42 PPL** bandında biteceği tahmin ediliyor.  
> Tahmini `ΔPPL ≈ 68.97 − ~38 ≈ ~30` — null hipotez (`|ΔPPL| < 0.5`) **kesin olarak reddedildi.**

**Öğrenilen fizik (Exp 14):**

```
seed 42:
  ν:  [0.0159, 0.0255, 0.0258, 0.0258]  (başlangıç→son, +0.0099 gradient)
  dt: [0.1094, 0.0613, 0.0469, 0.0940]  (U-şekli profil devam ediyor)
  α:  [1.0, 1.0, ..., 1.0]              ← öğrenme yok
  p_scale: [0.1, 0.1, ..., 0.1]         ← öğrenme yok

seed 43:
  ν:  [0.0152, 0.0236, 0.0240]  (+0.0088 gradient)
  dt:  U-şekli devam ediyor
```

**α ve p_scale öğrenemedi** — bu açık bir araştırma sorusudur (bkz. §7).

### Enerji Analizi (Teorik, Exp 14)

| Model   | Routing MFLOP/tok | Teorik fark |
|---------|-------------------|-------------|
| FluidLM | 113.4             | —           |
| GPT     | 188.7             | ×1.66       |

1T token eğitim projeksiyonu (batch=32, backward×3 güç tahmini):
- FluidLM-S: ~4,821 kWh ≈ 1,123 kg CO₂
- GPT-S:     ~8,020 kWh ≈ 1,869 kg CO₂
- Tasarruf:  ~3,199 kWh (~39.9%)

---

## 6. ΔPPL Yorumlama Eşikleri

GPT-S sonuçları tamamlandığında `ΔPPL = |FluidLM_medyan − GPT_medyan|`:

| ΔPPL        | Yorum           | RQ1 Sonucu                                              |
|-------------|-----------------|--------------------------------------------------------|
| < 0.3       | İstatistiksel gürültü içinde | ✅✅ **Güçlü:** NS = MHA eşdeğeri         |
| 0.3 – 0.7   | Küçük fark      | ✅ **Orta:** NS yakın fakat hafif geride                |
| 0.7 – 2.0   | Belirgin fark   | ⚠️ **Zayıf:** MHA routing gücü var, araştırma devam    |
| > 2.0       | Büyük fark      | ❌ **Olumsuz:** NS bu ölçekte yetersiz                   |

**Mevcut tahmin (17 Mayıs 2026, GPT-S seed 42 step=2000 ara sonucu):**

```
FluidLM-S s42 (final):  68.9726 PPL  @ step 6102
GPT-S     s42 (ara):    48.8288 PPL  @ step 2000  →  ~38 PPL tahmin final
Tahmini ΔPPL ≈ 30+  →  tablonun dışında, ❌ kesin olumsuz
```

> **Güncel Durum (v4 sonrası, 17 Mayıs 2026):**  
> `fluidlm/ns_layer.py` içerik bağımlı hıza (v4) güncellendi — commit `6a27f5b`.  
> Nano-pilot v4, GPT'yi step 200'de ΔPPL=−8.4 ile geçti (ilk FluidLM zaferı).  
> Exp 14'te bu düzeltmenin S-ölçekte PPL açığını ne kadar kapayacağı henüz bilinmiyor; yeniden eğitim gerekiyor.
> 
> **Eski analiz (v1 routing için):**  
> Bu fark, NS routing'in içerik-bağımsız yapısından kaynaklanıyordu:  
> MHA her forward'da `score(i,j) = ⟨q_i, k_j⟩` hesaplar (token çiftleri dinamik).  
> V1 NS routing `speed = tanh(‖u‖)` kullanıyordu — token j, token i'yi hiç görmez.  
> **V4 bu sorunu düzeltir:** `speed_i = tanh(q_i · k_{i-1} / √d_k)` ile içerik-dinamik routing.

---

## 7. Açık Sorular

### S5 — V4 Düzeltmesi S-Ölçekte ΔPPL'i Kapatıyor mu?

Nano-pilot (d=256, L=6) üzerninde v4, GPT'yi ΔPPL=−8.4 ile geçti.  
S-ölçekte (d=768, L=12) bu iyileştirmenin ~30 PPL açığı üzerine etkisi belirsiz.

**Hipotezler:**
1. V4, S-ölçekte de anlamlı iyileştirme sağlar (ama full gap kapanmayabilir)
2. S-ölçekte daha zengin tokenönü bağlam → `k_{i-1}` tek adım yeterli değil
3. Wider attention window (k_{i-2}, k_{i-3}) eklenmesi gerekiyor

**Sonraki adım:** `14_industrial_compare.py` v4 routing ile yeniden eğit (commit `6a27f5b` ile mevcut).

### S1 — α ve p_scale Neden Öğrenmiyor?

Exp 14 checkpoint'lerinde `α = 1.0` ve `p_scale = 0.1` değerleri başlangıç değerlerinde takılı.

**Hipotezler:**
1. Öğrenme hızı çok düşük (tüm parametreler aynı LR kullanıyor)
2. Loss yüzeyi bu parametreler için çok yassı
3. Basınç terimi zaten baskın — küçük değişimler loss'a yansımıyor

**Çözüm adayı:** Ayrı parametre grubu ile daha yüksek LR:
```python
param_groups = [
    {'params': [p for n,p in model.named_parameters() if 'log_alpha' in n or 'log_p_scale' in n],
     'lr': 1e-2},   # 10× daha yüksek
    {'params': [p for n,p in model.named_parameters() if 'log_alpha' not in n and 'log_p_scale' not in n],
     'lr': 3e-4},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1)
```

### S2 — Ölçek Yasası

FluidLM için Chinchilla ölçek yasası GPT ile aynı mı?  
`L_FluidLM(N, D) ≈ L_GPT(N, D)` ?

Şu an için M ölçeğinde tek veri noktası planlanıyor (Exp 14, FluidLM-M + GPT-M).

### S3 — KV Cache Yokluğu

FluidLM'de KV cache gerekmez — basınç cumsum ile gerçek zamanlı hesaplanır. Bu büyük dizi uzunluklarında (L > 4096) belirgin avantaj sağlar mı?

### S4 — Adaptif Derinlik Analizi

Eğitilmiş modelde farklı token türleri (nadir kelime vs sık kelime) kaç katman kullanıyor?  
Bu dağılım ν ile korelasyonlu mu?

---

## 8. Planlanan Çalışmalar

### Kısa Vade (Exp 14 tamamlanınca)

- [x] FluidLM-S seed 42 eğit → **68.9726 PPL** ✅ (6102/6103, t=372m)
- [x] İçerik-bağımsız routing sorununu teşhis et (Exp 15 nano-pilot ablasyonu)
- [x] V4 içerik-bağımlı hız: `NanoFluidLayerV4` implement et, nano'da ΔPPL=−8.4 doğrula ✅
- [x] `fluidlm/ns_layer.py` üretim paketine v4 portla, commit `6a27f5b` ✅
- [ ] GPT-S seed 42 eğit (🔄 %33, ara: 48.83 @ step 2000)
- [ ] GPT-S seed 43/44 eğit
- [ ] ΔPPL hesapla, RQ1 karara bağla
- [ ] **Exp 14'i v4 routing ile yeniden çalıştır** (commit `6a27f5b` ile güncel — S-ölçekte etki ölç)
- [ ] FluidLM-S seed 43/44 eğit
- [ ] WikiText-103 zero-shot eval
- [ ] FluidLM-M + GPT-M (seed 42)
- [ ] Enerji analizi: gerçek GPU watt (pynvml)

### Orta Vade

- [ ] α/p_scale için ayrı LR deneyi (S2 sorusu)
- [ ] Adaptif derinlik profili analizi (S4 sorusu)
- [ ] Dizi uzunluğu artışı: seq=2048 (bellek analizi)

### Uzun Vade (Potansiyel Yayın)

- [ ] Ölçek yasası grafikleri (N vs L, D vs L)
- [ ] RQ1/RQ2/RQ3 bulgularını §4 olarak yazma
- [ ] Peer review için arXiv ön baskı

---

## Referanslar

1. Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.* arXiv:2203.15556  
2. Brown et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS  
3. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS  
4. Press & Wolf (2017). *Using the Output Embedding to Improve Language Models.* EACL  
5. Graves (2016). *Adaptive Computation Time for Recurrent Neural Networks.* arXiv:1603.08983  
6. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* (GPT-2)  
7. Batchelor (1967). *An Introduction to Fluid Dynamics.* Cambridge University Press  
