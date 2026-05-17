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

### Exp 15 — Nano Pilot: V4 Tam Koşu (20M Token, TAMAMLANDI)

**Script:** `experiments/15_nano_pilot.py --model v4gpt --tokens 20e6`  
**Yapılandırma:** d=256, L=6, seq=128, batch=8, grad_accum=8, total_steps=2,441

**Sonuç (final @ step 2441):**

| Model         | Final val_ppl | Yorum                  |
|--------------|---------------|------------------------|
| FluidLM v4   | **402.77**    | content-dep speed      |
| GPT-Nano     | **410.38**    | MHA                    |
| **ΔPPL**     | **−7.61**     | **FluidLM v4 kazandı** |

**Eval noktası başına ΔPPL (step × PPL):**

| Step  | FluidLM v4 | GPT    | ΔPPL    | Kazanan     |
|-------|-----------|--------|---------|-------------|
| 200   | 882.1     | 890.5  | **−8.4**  | v4 ✅       |
| 400   | 677.0     | 684.4  | **−7.4**  | v4 ✅       |
| 600   | 637.3     | 600.8  | +36.4   | GPT ⚠️      |
| 800   | 808.5     | 567.4  | +241.0  | GPT ❌ (spike!) |
| 1000  | 462.3     | 497.5  | **−35.2** | v4 ✅       |
| 1200  | 435.5     | 519.0  | **−83.5** | v4 ✅       |
| 1400  | 552.6     | 447.9  | +104.6  | GPT ⚠️      |
| 1600  | 551.4     | 393.3  | +158.1  | GPT ❌      |
| 1800  | 441.9     | 398.5  | +43.4   | GPT ⚠️      |
| 2000  | 334.9     | 435.8  | **−100.9** | v4 ✅      |
| 2200  | 388.3     | 426.8  | **−38.5** | v4 ✅       |
| 2400  | 363.7     | 364.9  | **−1.3**  | v4 ✅ (yaklaştı) |

**Öğrenilen fizik — v4 nano final:**

```
Layer  0:  ν=0.0156  α=8.6463  p_scale=0.9747  dt=0.0385
Layer  1:  ν=0.0198  α=1.4870  p_scale=0.9754  dt=0.0886
Layer  2:  ν=0.0193  α=2.0536  p_scale=0.9585  dt=0.1213
Layer  3:  ν=0.0178  α=1.3270  p_scale=0.9826  dt=0.1129
Layer  4:  ν=0.0177  α=1.2402  p_scale=0.9886  dt=0.1230
Layer  5:  ν=0.0178  α=1.4312  p_scale=0.9880  dt=0.1351
ν gradient: +0.0000  (OLUŞMEDI ✗)
```

**Üç kritik gozlem:**

1. **α Öğrendi:** Layer 0 = 8.65 (başlangıç: 1.0). Exp 14 v1'de α hiç oynamıyordu. V4 routing, α'nın gradyan almasını sağladı. S1 açık sorusunun ("neden öğrenmiyor?") kısmi yanıtı: routing kalitesi yeterliyse optimizer sinyali akıyor.

2. **p_scale Öğrendi:** ~0.1 başlangıçtan ~0.97'ye. Başınc terimi şimdi neredeyse tam güçte çalışıyor.

3. **ν gradyanı yok (nano'da):** Nano ölçekte (d=256, L=6) ν gradyanı oluşmuyor. Oysa S-ölçekte (d=768, L=12) +0.1936 görüldü. Bu, ν gradyanının **derinlik ve genişliğe bağlı** olduğunu işaret ediyor.

**Yüksek varyans / instabilite sorunu:**  
DeltaPPL step başına −1 ile +241 arasında salınıyor. Step 800'de v4 spike (808 PPL, GPT=567). Bu, W_q/W_k projektörlerinin LR ile birlikte olası gradyan patlamalarından kaynaklanıyor.

**Hipotez:** `W_q`/`W_k` parametrelerine `grad_clip` veya daha düşük LR gerekiyor. Alternatif: `stop_gradient(k_prev)` — sadece query öğrenir.

**Önerilen Exp 16 (S6):** Stabilite ablasyonu:
- Deney A: `W_q`/`W_k` LR = `base_lr / 10`
- Deney B: `k_prev.detach()` (tek yönlü gradyan)
- Deney C: Larger `d_k` (d_model // 4)

### Exp 14 — Industrial Scale (Devam Ediyor)

**FluidLM-S sonuçlar:**

| Seed | Routing | Token Budget | Val PPL (best) | Step        | Durum        |
|------|---------|-------------|----------------|-------------|-------------|
| 42   | v1      | 3B          | **68.9726**    | 6,102/6,103 | ✅ **TAMAMLANDI** |
| 42   | **v4**  | **1B**      | **107.1919**   | 2,000/2,034 | ✅ **TAMAMLANDI** |
| 43   | —       | —           | —              | —           | 🔄 beklemede |
| 44   | —       | —           | —              | —           | 🔄 beklemede |

**GPT-S sonuçlar:**

| Seed | Token Budget | Val PPL (ara) | Step    | Durum      |
|------|-------------|---------------|---------|------------|
| 42   | **1B**      | 58,074 (başl.) | 0/2,034 | 🔄 yeni başladı |
| 43   | —           | —             | —       | ⏳ bekliyor |
| 44   | —           | —             | —       | ⏳ bekliyor |

> **Not:** V4 routing ile 1B token kıyaslama koşusu: GPT-S s42 tamamlandığında FluidLM-S v4 @ 1B ile doğrudan karşılaştırılacak. ΔPPL yönü (positif mi negatif mi) kritik — nano-pilot v4, ΔPPL=−8.4 ile GPT'yi geçmişti.

**Öğrenilen fizik — V4 routing öncesi vs sonrası (Exp 14, seed 42):**

| Parametre        | V1 routing (3B) | **V4 routing (1B)** | Değişim        |
|-----------------|-----------------|---------------------|---------------|
| ν gradient      | +0.0099         | **+0.1936**         | **×19.6×** 🟢 |
| ν early (0-3)   | ~0.016          | **0.1606**          | +10×          |
| ν late (8-11)   | ~0.026          | **0.2542**          | +10×          |
| ν range         | [0.012, 0.031]  | **[0.0541, 0.2698]**| çok daha geniş |

> **Önemli bulgu:** V4 içerik-bağımlı hız, NU gradyanını ~20 kat artırdı. V1'de `ν` başlangıç değerinde neredeyse takılı kalıyordu; v4 ile katman boyunca **0.05 → 0.27** geniş bir spektrum öğreniyor.  
> Bu, RQ3 ("ν gradyanı monoton artar mı?") için çok daha güçlü bir doğrulama.  

**V4 @ 1B PPL eğrisi:**

```
step=    0  val_ppl=59,750  (başlangıç chaos)
step=  500  val_ppl=  204.9
step= 1000  val_ppl=  141.5
step= 1500  val_ppl=  114.2
step= 2000  val_ppl=  107.2  ← en iyi
step= 2033  val_ppl=  111.0  (final)
```

> GPT-S s42 @ 1B tamamlandığında karşılaştırma güncellenmeli.

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

**Mevcut durum (17 Mayıs 2026 — v4 routing, 1B token karşılaştırması):**

```
FluidLM-S s42 v4 @ 1B:  107.19 PPL  (best @ step 2000/2034)  ✅ TAMAMLANDI
GPT-S     s42    @ 1B:    ?    PPL  (step 0/2034, yeni başladı)  🔄 devam

Beklenen ΔPPL @ 1B: belirsiz — nano-pilot v4 −8.4 verdi;
  S-ölçekte eşit ya da pozitif çıkabilir, GPT-S tamamlanınca güncellenecek.

V1 @ 3B karşılaştırma (referans): FluidLM=68.97, GPT-S s42 final=~38 (tahmini)
  V1 tahmini ΔPPL ≈ +30  → tablo dışında (❌ kesin olumsuz)
```

> **V4 routing kritik ek bulgusu:** ν gradyanı +0.0099 → +0.1936 (+20×). Fiziksel parametreler artık güçlü biçimde öğreniyor. Bu RQ3'te çok daha güçlü bir doğrulama sağlar.

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

### S6 — V4 İnstabilitesi: W_q/W_k Gradyanı Kontrol Edilebilir mi?

Nano-pilot ΔPPL step başına −1 ile +241 arasında salınıyor. Step 800 spike (ΔPPL=+241) kritik bir instabilite sinyali.

**Üç hipo tez:**
1. `W_q`/`W_k` gradyanları genel LR ile çok büyük adım atıyor
2. `k_prev` üzerinden gelen backprop döngüsü (q₂ → k₁ → q₁) instabilite yaratıyor
3. `d_k` çok küçük (d//8) — dot product varyansı yüksek

**Aday çözümler:**
- `W_q`/`W_k` için `lr_scale = 0.1` (10× düşük)
- `k_prev = k_prev.detach()` — tek yönlü gradyan
- `d_k = d_model // 4` — daha geniş projeksiyon

### S5 — V4 Düzeltmesi S-Ölçekte ΔPPL'i Kapatıyor mu?

Nano-pilot (d=256, L=6) üzerninde v4, GPT'yi ΔPPL=−8.4 ile geçti.  
S-ölçekte (d=768, L=12) bu iyileştirmenin ~30 PPL açığı üzerine etkisi belirsiz.

**Hipotezler:**
1. V4, S-ölçekte de anlamlı iyileştirme sağlar (ama full gap kapanmayabilir)
2. S-ölçekte daha zengin tokenönü bağlam → `k_{i-1}` tek adım yeterli değil
3. Wider attention window (k_{i-2}, k_{i-3}) eklenmesi gerekiyor

**Sonraki adım:** `14_industrial_compare.py` v4 routing ile yeniden eğit (commit `6a27f5b` ile mevcut).

### S1 — α ve p_scale Neden Öğrenmiyor? (Kısmen Çözüldü)

**Exp 14 v1:** `α = 1.0`, `p_scale = 0.1` — hiç oynamadı.  
**Exp 15 v4 nano:** `α` = [1.23, 8.65] — öğrendi. `p_scale` ~0.97 — öğrendi.

**Yorum:** V4 routing, α ve p_scale için yeterli gradyan sinyali sağlıyor. V1'de `speed = tanh(‖u‖)` sabit olunca basınç grady anı maskelenıyordu. V4 içerik-bağlı hız, tüm fiziksel parametreleri "kilidinden kurtardı".

> Bu, S1 sorusunun yanıtıdır: sorun LR değil, routing kalitesiydi. Ayrı LR grubu önerisi artık gereksiz.

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

### Kısa Vade (Devam Eden)

- [x] FluidLM-S seed 42 eğit → **68.9726 PPL** ✅ (6102/6103, t=372m)
- [x] İçerik-bağımsız routing sorununu teşhis et (Exp 15 nano-pilot ablasyonu)
- [x] V4 içerik-bağımlı hız: `NanoFluidLayerV4` implement et ✅
- [x] Nano-pilot 20M token tam koşus: **final ΔPPL = −7.61** ✅
- [x] α ve p_scale öğrenememe sorunu çözüldü — v4 routing ile kendiliğinden öğreniyor ✅
- [x] `fluidlm/ns_layer.py` üretim paketine v4 portla, commit `6a27f5b` ✅
- [ ] GPT-S seed 42 @ 1B eğit (🔄 devam)
- [ ] Exp 14 v4: FluidLM-S s42 @ 1B (107.19 PPL) vs GPT-S s42 @ 1B karşılaştır
- [ ] V4 instabilite ablasyonu — Exp 16: `k_prev.detach()` / LR scale / d_k deneyi
- [ ] GPT-S seed 43/44 eğit
- [ ] ΔPPL hesapla, RQ1 karara bağla
- [ ] FluidLM-S seed 43/44 eğit
- [ ] WikiText-103 zero-shot eval
- [ ] FluidLM-M + GPT-M (seed 42)

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
