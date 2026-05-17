# FluidLM Terimler Sözlüğü

> Her terimin **doğal dil karşılığı**, **gerçekte ne yaptığı** ve **diğer terimlerle nasıl etkileştiği** açıklanıyor.

---

## Temel Varlıklar

### `u` — Hız Alanı (Velocity Field)

**Doğal dil:** Bir tokenın "ne kadar ve hangi yönde hareket ettiği".

**Gerçekte ne yapar:** Modelin her adımında her tokenın temsili bir `D`-boyutlu vektördür. Başlangıçta `token_embedding + position_embedding` ile başlatılır. Katmanlar boyunca Navier-Stokes denklemiyle güncellenir. Son katmanın çıktısı olan `u`, `lm_head` matrisine çarpılarak bir sonraki tokenın olasılık dağılımına dönüşür.

**Boyutu:** `[B, L, D]` — B=batch, L=dizi uzunluğu (token pozisyonları), D=embedding boyutu.

**Etkileşimler:**
- `gradient(u)` → adveksiyonun hammaddesi
- `laplacian(u)` → viskozite teriminin hammaddesi
- `divergence(u)` → basınç çözücüsüne girdi
- Her katmandan geçince güncellenir; `delta_KE` ile ne kadar değiştiği ölçülür
- `norm(u)` → `tanh` ile kırpılıp adveksiyon hızına dönüşür

---

### `p` — Basınç Alanı (Pressure Field)

**Doğal dil:** "Tüm tokenların birbirini ne kadar itip çektiği" — Transformer'daki attention matrisinin fizikten türeyen karşılığı.

**Gerçekte ne yapar:** Her katmanda, adveksiyon teriminin ıraksıklığından (`div(adv)`) Poisson denklemi çözülerek hesaplanır. `p[i]` değeri, `i`. tokenın tüm diğer tokenlardan biriktirdiği "itme/çekme sinyali"dir. Bu sinyal `∇p` (basınç gradyanı) olarak `u`'ya geri beslenir: basıncın yüksek olduğu yerden alçak olduğu yere akış olur.

**İki çözüm modu:**
| Mod | Formül | Kapsam |
|-----|--------|--------|
| Causal (LM eğitimi) | `p[i] = cumsum(-div(adv))[:i] / α` | Yalnızca geçmiş tokenlar |
| Non-causal (analiz) | FFT ile Poisson çözümü: `p̂_k = f̂_k / λ_k` | Tüm dizi, tam global |

**Etkileşimler:**
- Girdisi: `divergence(adv)` — adveksiyonun yerel ıraksama bilgisi
- Çıktısı: `gradient(p)` → `p_scale` ile ölçeklenip `u`'ya eklenir
- `α` (Helmholtz sabiti) → basıncın etkili menzilini belirler
- `p_scale` → basıncın `u` üzerindeki toplam ağırlığını ayarlar

---

### `adv` — Adveksiyon (Advection)

**Doğal dil:** "Anlam bir tokendan komşusuna taşınıyor" — yerel, doğrusal olmayan etkileşim.

**Gerçekte ne yapar:** `u`'nun kendi kendini taşıması. Formül: `adv = tanh(‖u‖) · ∂u/∂x`. Her tokenın embedding'i, kendisiyle komşusu arasındaki farka (`gradient`) kendi büyüklüğünün (tanh ile sınırlandırılmış versiyonu) katı kadar sürüklenir. Transformer'daki FFN (MLP) katmanının doğrusal olmayanlık rolünü üstlenir.

**`tanh(‖u‖)` neden var:** Hızın büyüklüğünü `(0, 1)` aralığına sıkıştırır. Bu "Reynolds sayısını" sınırlar — embedding değerleri patlamaz, sayısal kararlılık korunur.

**Etkileşimler:**
- Girdisi: `u` ve `gradient(u)` — her ikisi de mevcut katman durumu
- Çıktısı: `divergence(adv)` → basınç çözücüsüne beslenir
- `−adv` olarak nihai türeve katkıda bulunur (self-advection negatif işaretli)
- Causal modda `gradient` sıfır-dolgulu geriye fark kullanır → gelecek token görülmez

---

## Diferansiyel Operatörler

### `gradient(u)` — Gradyan ∂u/∂x

**Doğal dil:** "Bu token, solundaki tokenden ne kadar farklı?"

**Gerçekte ne yapar:** Dizi boyunca komşu tokenlar arasındaki farkı hesaplar.

| Mod | Formül | Sezgisel |
|-----|--------|---------|
| Causal | `u[i] − u[i−1]` (sola sıfır dolgu) | Sadece geçmiş |
| Non-causal | `(u[i+1] − u[i−1]) / 2` | Merkezi fark, O(h²) |

**Etkileşimler:**
- `adv = speed · gradient(u)` → adveksiyon için kullanılır
- `p_grad = gradient(p)` → basınç geri beslemesi için kullanılır
- `laplacian(u) = gradient(gradient(u))` değil; ayrı hesaplanır (daha verimli)

---

### `laplacian(u)` — Laplacian ∂²u/∂x²

**Doğal dil:** "Bu token, her iki komşusunun ortalamasından ne kadar sapıyor?"

**Gerçekte ne yapar:** İkinci türev — yerel pürüzlülüğü ölçer. Sonuç negatifse token ortalamanın üstünde (tepe), pozitifse ortalamanın altında (çukur). Viskozite terimi `ν · laplacian(u)` bu sapmayı küçülterek komşuları birbirine yakınlaştırır.

| Mod | Formül |
|-----|--------|
| Causal | `u[i] − 2u[i−1] + u[i−2]` (iki sola sıfır dolgu) |
| Non-causal | `u[i+1] − 2u[i] + u[i−1]` (periyodik) |

**Etkileşimler:**
- `ν · laplacian(u)` → nihai türevde `+visc` olarak eklenir
- `ν` büyükse laplacian'ın etkisi artar → güçlü düzleştirme
- Gradyan gibi causal modda gelecek tokenlara bakmaz

---

### `divergence(u)` — Iraksama ∇·u

**Doğal dil:** "Bu tokendan çevresine ne kadar "akış çıkıyor" — komşularından uzaklaşıyor mu, yaklaşıyor mu?"

**Gerçekte ne yapar:** `gradient(u)`'nun D boyutu üzerindeki ortalaması. `[B, L, D]` → `[B, L]` skaler alan. Pozitif divergence: token komşularından uzaklaşıyor (kaynak). Negatif: yaklaşıyor (yutak).

**Etkileşimler:**
- `divergence(adv)` → basınç Poisson denkleminin sağ tarafı
- Causal modda `gradient`'in causal versiyonu kullanılır
- NOT: FluidLM `∇·u = 0` (sıkıştırılamazlık) zorlamaz — basınç coupling için divergence kullanılır ama alan serbest bırakılır

---

## Öğrenilebilir Fiziksel Parametreler

> Bunlar her NS katmanının öğrendiği 4 skalerdir. Tüm modelde toplam `4 × n_layers` parametre.

### `ν` (nu) — Viskozite

**Doğal dil:** "Bu katman komşu tokenleri birbirine ne kadar benzetsin?"

**Gerçekte ne yapar:** `ν · laplacian(u)` terimi, embedding uzayındaki pürüzleri düzler. Yüksek ν → agresif düzleştirme → token temsilleri birbirine yakınlaşır → soyut, uzun menzilli özellikler. Düşük ν → keskin, lokal ayrışma korunur → ince sözdizimsel ayrımlar.

**İç temsil:** `log_nu` parametresi saklanır, `ν = exp(log_nu)` ile kullanılır → her zaman pozitif, sıfıra düşemez.

**Öğrenilen değerler (10_faircompare, epoch 171):**
- Erken katmanlar (0–3): ν ≈ 0.011 — düşük, yerel/keskin
- Orta katmanlar (4–7): ν ≈ 0.013
- Geç katmanlar (8–11): ν ≈ 0.016 — yüksek, soyut/pürüzsüz

Model bu hiyerarşiyi hiç söylenmeden keşfetti. Transformer interpretability literatürünün yıllarca el yordamıyla bulduğu "erken katmanlar sözdizimsel, geç katmanlar semantik" örüntüsü burada fizik yasasından çıkıyor.

**Etkileşimler:**
- `ν` büyüdükçe `visc = ν · laplacian(u)` büyür → `u` daha hızlı düzlenir
- `Δt` ile çarpıldığında etkili viskozite `ν · Δt`'dir (diffusion number)
- `α` ile doğrudan etkileşmez ama ikisi birlikte katmanın "rejimi"ni belirler

---

### `Δt` (dt) — Zaman Adımı

**Doğal dil:** "Bu katman `u`'yu ne kadar ileri taşısın?"

**Gerçekte ne yapar:** NS denkleminin bir adım büyüklüğü. `u_new = u + Δt · F(u)` (Euler) veya RK4'te her k için kullanılır. Büyük Δt → her katman daha büyük değişim → daha az katmana ihtiyaç (ama kararsızlık riski). Küçük Δt → küçük adımlar → daha stabil ama daha yavaş yakınsama.

**İç temsil:** `log_dt` olarak saklanır, `Δt = exp(log_dt)`.

**Öğrenilen değerler:**
- Erken: Δt ≈ 0.052 — küçük adımlar, ince düzeltmeler
- Geç: Δt ≈ 0.060 — büyüyen adımlar, katmanlar derinleştikçe daha cesur

**Etkileşimler:**
- Euler: `u ← u + Δt · (−adv − ∇p + visc)`
- RK4: aynı ama 4 ara değerlendirme → daha doğru
- `ΔKE = ‖u_new − u‖² ∝ Δt · ‖F(u)‖²` — adaptif durdurma için eşik karşılaştırması Δt'ye duyarlı

---

### `α` (alpha) — Helmholtz Sabiti

**Doğal dil:** "Basınç ne kadar uzağa etki etsin?"

**Gerçekte ne yapar:** FFT Poisson çözücüsünde frekans alanı paydasını düzenler: `λ_k = 2(cos(2πk/L) − 1) − α²`. `α = 0` → tam global etki (her token her tokena eşit şekilde etki eder). `α > 0` → uzak tokenların etkisi üstel sönümlenir, etki menzili `∝ 1/α`. Matematiksel amacı: `k = 0` (DC mod) paydasının sıfıra düşmesini engeller → basınç patlaması olmaz.

**Causal modda** Poisson çözücüsü kullanılmaz, yerine `cumsum / α` kullanılır — burada `α` bölüm faktörü olarak birikim büyüklüğünü ölçekler.

**Öğrenilen değerler:** Tüm katmanlarda ≈ 1.0000 (başlangıç değerinde kaldı). Ablasyon deneyi 12 bunu test ediyor: frozen α ≈ serbest α → α gereksiz olabilir.

**Etkileşimler:**
- Poisson'da: `p̂_k = f̂_k / λ_k(α)` — α λ'yı küçültür → p büyür → uzak etki artar
- Causal'da: `p = cumsum(-div) / (α + ε)` — α büyüdükçe basınç küçülür
- `p_scale` ile birlikte basıncın nihai etkisi `p_scale / α` ile orantılı

---

### `p_scale` — Basınç Ağırlığı

**Doğal dil:** "Basınç terimi `u`'yu ne kadar etkilesin?"

**Gerçekte ne yapar:** `p_grad = p_scale · gradient(p)` — basınç gradyanı `u`'ya eklenmeden önce bu katsayıyla ölçeklenir. Sıfıra yakın değer: basınç etkisiz, model yerel (adveksiyon + viskozite dominant). Büyük değer: global coupling dominant, tüm dizi birbirini etkiler.

**İç temsil:** `log_p_scale`, başlangıç değeri `log(0.1)` → `p_scale = 0.1`.

**Öğrenilen değerler:** Tüm katmanlarda ≈ 0.1006 (başlangıç değerinde kaldı, α gibi).

**Etkileşimler:**
- `α` ile birlikte basıncın etkili gücü: `p_scale / (α + ε)`
- Çok büyük olursa basınç dominant olur → adveksiyon ve viskozite anlamsızlaşır
- 10× LR ile eğitilir (NS parametrelerine özel) ama yine de oynamadı

---

## Entegrasyon Şemaları

### Euler İntegratörü

**Doğal dil:** "Bir adım ilerle, nerede durduğuna bak."

**Formül:** `u_new = u + Δt · F(u)`

**Gerçekte ne yapar:** F(u) = NS sağ tarafı (`−adv − ∇p + ν∇²u`) bir kez hesaplanır, doğrusal ekstrapolasyon yapılır. 1. mertebeden doğruluk. Hızlı, basit, ama büyük Δt'de hata birikir.

**Etkileşimler:** `u`, `Δt`, `F(u)` → `u_new`. PyTorch'un autograd'ı bu işlemi tam olarak farklılaştırır — adjoint yöntemi gerekmez.

---

### RK4 İntegratörü (Runge-Kutta 4. Mertebe)

**Doğal dil:** "Önce kısa adım at, orada ne göreceğini tahmin et, orta noktayı ölç, sonra tam adımı yap — dört tahminin ağırlıklı ortalamasını al."

**Formül:**
```
k₁ = F(u)
k₂ = F(u + Δt/2 · k₁)
k₃ = F(u + Δt/2 · k₂)
k₄ = F(u + Δt   · k₃)
u_new = u + Δt/6 · (k₁ + 2k₂ + 2k₃ + k₄)
```

**Gerçekte ne yapar:** F(u)'yu 4 farklı noktada değerlendirir, Simpson benzeri ağırlıklarla birleştirir. 4. mertebeden doğruluk: aynı Δt için Euler'den çok daha doğru, veya aynı doğruluk için daha büyük Δt kullanılabilir → daha az katman yeterli.

**Maliyet:** Euler'in 4 katı hesaplama (4× `_rhs` çağrısı). Üretim için önerilir.

**Etkileşimler:** Her `k_i` hesabı yeni bir `gradient`, `laplacian`, `divergence`, `solve_poisson` çağrısı yapar. `Δt` tüm adımlarda paylaşılır.

---

## Adaptif Derinlik

### `ΔKE` — Kinetik Enerji Değişimi

**Doğal dil:** "Bu katmandan geçmek ne kadar işe yaradı?"

**Gerçekte ne yapar:** `ΔKE = mean((u_new − u_old)²)` — bir katmanın `u`'yu ne kadar değiştirdiğini ölçer. Büyük ΔKE: katman hâlâ önemli değişimler yapıyor. Küçük ΔKE: akış stabilleşti, daha fazla katman anlamsız.

**Etkileşimler:**
- Her `FluidLayer.forward()` çağrısı ΔKE döndürür
- `FluidLM.forward()` bunu `convergence_threshold` ile karşılaştırır
- `min_steps` kadar katman her zaman çalışır (kararlılık için)
- `Δt` büyüdükçe ΔKE büyür → eşiğe daha erken ulaşılır → daha az katman

---

### `convergence_threshold`

**Doğal dil:** "Ne zaman 'yeterince iyi' diyip duruyoruz?"

**Gerçekte ne yapar:** Adaptif erken durdurma eşiği. `ΔKE < threshold` olunca kalan katmanlar atlanır. Basit cümleler 3–4 katmanda yakınsar, karmaşık yapılar 10–12 ister. Her girdi için farklı hesap maliyeti → enerji verimli.

**Etkileşimler:**
- `min_steps` ile birlikte çalışır: en az `min_steps` katman, en fazla `n_layers` katman
- Çok küçük eşik → neredeyse hiç erken durmaz, sabit derinlik gibi davranır
- Çok büyük eşik → çok erken durur, kalite düşer

---

## Spektral Çözücü

### FFT Poisson Çözücüsü

**Doğal dil:** "Tüm tokenların etkisini frekans uzayında tek seferde çöz."

**Gerçekte ne yapar:** Poisson denklemi `(∇² − α²)p = rhs`'yi Fourier dönüşümüyle çözer. Uzamsal konvolüsyon, Fourier uzayında nokta çarpımına döner: her `k` frekansı için `p̂_k = f̂_k / λ_k` — bölme işlemi. `irfft` ile geri dönüşüm yapılır. Tüm işlem O(L log L), attention'ın O(L²)'sine karşı.

**Adımlar:**
1. `rhs` → `fft.rfft` → frekans alanı
2. Her frekans için `p̂_k = f̂_k / λ_k(α)` — paralel, O(L) iş
3. `irfft` → uzamsal basınç alanı `p`

**Etkileşimler:**
- Girdisi: `divergence(adv)` — adveksiyonun ıraksama alanı
- `α` → `λ_k = 2(cos(2πk/L) − 1) − α²` paydasını belirler
- Non-causal modda kullanılır; causal modda `cumsum` kullanılır (FFT periyodik sınır koşulu causal değil)
- Gradient-compatible: `autograd` FFT üzerinden tam türev alır

---

## Model Düzeyinde Kavramlar

### `token_emb` — Token Gömme (Embedding)

**Doğal dil:** "Her karakterin/kelimenin başlangıç hız vektörü."

**Gerçekte ne yapar:** `vocab_size × D` boyutlu bir lookup tablosu. Her token ID'si D-boyutlu bir vektöre dönüştürülür. Bu vektör, NS denkleminin başlangıç koşulu `u₀`'dır. Eğitim boyunca güncellenir.

**Weight Tying:** `lm_head.weight = token_emb.weight` — giriş ve çıkış aynı matrisi paylaşır. ~vocab×D parametre tasarrufu, Press & Wolf 2017'den.

---

### `pos_emb` — Konum Gömme (Positional Embedding)

**Doğal dil:** "Bu token dizinin neresinde?"

**Gerçekte ne yapar:** `max_seq_len × D` lookup tablosu. Token embedding'ine eklenerek `u₀`'ı konuma duyarlı kılar. FluidLM'de NS operatörleri sıra farkını (`u[i] − u[i−1]`) zaten kullanır, bu yüzden `pos_emb`'nin etkisi Transformer'a kıyasla daha küçük — ama yine de çıkarılırsa PPL kötüleşir.

---

### `lm_head` — Dil Modeli Başlığı

**Doğal dil:** "Son katmanın çıktısından hangi kelime gelecek?"

**Gerçekte ne yapar:** `D → vocab_size` doğrusal dönüşüm (bias yok). Son `u`'yu kelime olasılıklarına çevirir. `weight_tying` ile token_emb ile aynı matris → ayrı parametre yok.

---

### `LayerNorm`

**Doğal dil:** "Çıkmadan önce vektörü normalize et."

**Gerçekte ne yapar:** Tüm NS katmanlarından sonra, lm_head'den önce uygulanır. `u`'nun D boyutu üzerinde ortalama=0, std=1 yapılır + öğrenilebilir ölçek/kaydırma. NS dinamikleri bazen embedding değerlerini büyütür veya küçültür — LayerNorm bu dağılımı sabitler.

---

### `MLP / FFN` — İleri Besleme Ağı

**Doğal dil:** "Her tokenı bağımsız olarak dönüştür."

**Gerçekte ne yapar:** Her NS katmanının içinde, akış adımından sonra bir `Linear → GELU → Linear` bloğu bulunur. `D → 4D → D` boyut genişletme. Parametre sayısının %99.5'ini oluşturur — `2 × D × 4D × 2 = 8D²` her katmanda. NS routing yalnızca 4 skaler; büyük kapasiteyi MLP taşır.

---

## Eğitim Kavramları

### PPL — Perplexity (Çelişki)

**Doğal dil:** "Model bir sonraki tokeni tahmin ederken ne kadar şaşırıyor?"

**Gerçekte ne yapar:** `PPL = exp(cross_entropy_loss)`. PPL=4.59 → model ortalamada 4.59 olası eşit olasılıklı seçenek arasında "şaşırıyor" gibi davranıyor. Düşük PPL = daha iyi dil modeli. PPL=1 teorik minimum (mükemmel tahmin).

**Karşılaştırma notu:** PPL'ler yalnızca aynı tokenizasyon ve veri üzerinde karşılaştırılabilir. Char-level (07) ile BPE (08) PPL'leri doğrudan karşılaştırılamaz.

---

### `ΔKE` (Eğitim değil, mimari) — bkz. Adaptif Derinlik bölümü.

---

### `cross_entropy_loss`

**Doğal dil:** "Tahmin ettiğin şey ne kadar yanlış?"

**Gerçekte ne yapar:** Doğru token için tahmin edilen log-olasılığının negatifi. Causal LM için: her pozisyonda `t`, modelden `logits[t]` alınır, `t+1` tokenının log-olasılığı hesaplanır, ortalaması loss olur. Backprop bu loss'u `lm_head → NS katmanları → embedding`'e kadar yayar.

---

### `autograd` / Backpropagation

**Doğal dil:** "Her parametrenin loss'u nasıl etkilediğini bul, tersi yönde güncelle."

**FluidLM'e özgü not:** NS denklemindeki tüm operatörler (gradient, laplacian, cumsum, FFT) PyTorch'un autograd'ıyla tam diferansiyel alınabilir. Adjoint yöntemi gerekmez çünkü katman sayısı sabit (10–12) — Neural ODE gibi binlerce adım değil.

---

## Terimler Arası Etkileşim Haritası

```
input_ids
    │
    ▼
token_emb + pos_emb
    │
    ▼ u₀ [B, L, D]
    │
    ├─── FluidLayer(ν, Δt, α, p_scale) ──────────────────────┐
    │         │                                               │
    │    gradient(u) ──► speed·grad = adv                    │
    │         │                │                             │
    │    laplacian(u)           │                             │
    │         │           divergence(adv)                    │
    │         │                │                             │
    │    ν·lap(u) = visc        ▼                             │
    │                    cumsum(-div)/α = p (causal)          │
    │                          │                             │
    │                    gradient(p)·p_scale = p_grad        │
    │                          │                             │
    │         F(u) = −adv − p_grad + visc                    │
    │                          │                             │
    │    Δt ile integre: RK4 veya Euler                      │
    │                          │                             │
    │         u_new = u + Δt·(F)                             │
    │                          │                             │
    │         ΔKE = ‖u_new−u‖²  ◄─ convergence_threshold?   │
    │                          │                             │
    │                          ▼                             │
    └─── (N kez tekrar veya ΔKE < eşik → dur) ───────────────┘
    │
    ▼ u_T [B, L, D]
    │
LayerNorm
    │
lm_head (= token_emb.weight^T)
    │
    ▼ logits [B, L, vocab]
    │
cross_entropy_loss
    │
    ▼
backprop → ∂L/∂ν, ∂L/∂Δt, ∂L/∂α, ∂L/∂p_scale, ∂L/∂emb, ...
```

---

## Transformer Karşılaştırması

| FluidLM Terimi | Transformer Karşılığı | Temel Fark |
|---------------|----------------------|------------|
| `u` (hız alanı) | hidden state `h` | NS denklemle güncellenir, matris çarpımıyla değil |
| `∇p` (basınç gradyanı) | `softmax(QKᵀ/√d)·V` | Fizik yasasından türer, öğrenilmez |
| `adv` (adveksiyon) | FFN (MLP) | Self-interaction, quadratic değil |
| `ν` (viskozite) | Dropout oranı | Öğrenilebilir, katman başı, fiziksel anlam var |
| `Δt` (zaman adımı) | Residual ölçeği | Öğrenilebilir adım büyüklüğü |
| `ΔKE < eşik` | Sabit N katman | Adaptif, içeriğe göre farklı hesap |
| FFT Poisson | Multi-head attention | O(L log L) vs O(L²), global ama daha hızlı |
| `α` | Head sayısı / attention skalası | Etkileşim menzilini kontrol eder |
