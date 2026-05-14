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
├── src/
│   ├── fluid_ops.py     # ∇, ∇², div, Poisson çözücü (FFT tabanlı)
│   ├── ns_layer.py      # Navier-Stokes katmanı (Euler / RK4)
│   ├── fluid_lm.py      # Tam dil modeli + adaptif derinlik + üretim
│   └── __init__.py
├── experiments/
│   ├── 01_1d_diffusion.py    # 1D difüzyon — temel kavramlar
│   ├── 02_ns_layer_test.py   # NS katmanı analizi — basınç/attention karşılaştırması
│   └── 03_toy_lm_train.py    # Karakter düzeyi eğitim
├── configs/
│   └── base_config.yaml
└── requirements.txt
```

---

## Kurulum

```bash
pip install -r requirements.txt
```

## Deneyleri Çalıştır

```bash
# Deney 1: 1D difüzyon — "token = sıcaklık alanı" metaforu
python experiments/01_1d_diffusion.py

# Deney 2: NS katmanı — basıncın attention gibi nasıl global etki yaptığı
python experiments/02_ns_layer_test.py

# Deney 3: Gerçek eğitim — karakter düzeyi dil modeli
python experiments/03_toy_lm_train.py
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
