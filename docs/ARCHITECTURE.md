# FluidLM — Mimari Dokümantasyonu

> **Hedef kitle:** Modeli kodla anlamak isteyen araştırmacılar ve geliştiriciler.  
> Matematiksel sezgi için → [`MATH_FOR_EVERYONE.md`](../MATH_FOR_EVERYONE.md)  
> Terim açıklamaları için → [`GLOSSARY.md`](../GLOSSARY.md)

---

## İçindekiler

1. [Tasarım Felsefesi](#1-tasarım-felsefesi)
2. [Veri Akışı — İleri Geçiş](#2-veri-akışı--ileri-geçiş)
3. [NS Katmanı — Detaylı Matematik](#3-ns-katmanı--detaylı-matematik)
4. [Nedensellik (Causality)](#4-nedensellik-causality)
5. [Adaptif Derinlik](#5-adaptif-derinlik)
6. [Parametre Sayısı Karşılaştırması](#6-parametre-sayısı-karşılaştırması)
7. [Entegrasyon Yöntemleri](#7-entegrasyon-yöntemleri)
8. [Modül Bağımlılık Grafiği](#8-modül-bağımlılık-grafiği)
9. [Bellek Analizi](#9-bellek-analizi)
10. [Ölçek Konfigürasyonları](#10-ölçek-konfigürasyonları)

---

## 1. Tasarım Felsefesi

Standart Transformer'da token etkileşimi **elle tasarlanmış** bir mekanizmaya dayanır:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

Bu formül, araştırmacıların inductive bias'larını yansıtır — neden bu formül başka bir şey değil?

FluidLM'nin sorusu şudur: **Etkileşim mekanizması fizik yasalarından türetilebilir mi?**

Yanıt: 1-D sıkıştırılamaz Navier-Stokes denklemi.

```
∂u/∂t = −(u·∇)u  −  ∇p  +  ν ∇²u
            ↓           ↓       ↓
        Adveksiyon   Basınç  Viskozite
      (yerel, NL)   (global) (düzeltme)
```

Her terim bilinen bir işlevi karşılar:

| NS Terimi    | Denklem          | Transformer Karşılığı        | Kapsam |
|-------------|-----------------|------------------------------|--------|
| Adveksiyon  | `−(u·∇)u`        | FFN / nonlinearity           | Yerel  |
| Basınç      | `−∇p`            | Attention matrisi            | Global |
| Viskozite   | `ν∇²u`           | Dropout + LayerNorm          | Yerel  |

**Kritik fark:** Transformer'da attention `O(L²)` parametre kullanır (QKV matrisleri). FluidLM'de basınç `O(L log L)` hesaplama karmaşıklığıyla FFT üzerinden çözülür ve **katman başı 4 skaler** dışında ek parametre gerektirmez.

---

## 2. Veri Akışı — İleri Geçiş

```
Input token IDs  [B, L]
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Embedding                                           │
│  u = token_emb(ids) + pos_emb(arange(L))  [B, L, D] │
│  u = Dropout(u)                                      │
└────────────────────┬────────────────────────────────┘
                     │  u₀ [B, L, D]
        ┌─────────────────────────────────────────────────────┐
        │  FluidLayer 1                                  │
        │  u1 = norm1(u0)                                │
        │  u_ns = u1 + dt·F(u1)   ← NS sublayer         │
        │  u  = u0 + (u_ns − u1)  ← residual            │
        │  ΔKE = ‖u_ns − u1‖² / ‖u1‖²                    │
        │  u  = u + mlp(norm2(u))  ← MLP sublayer        │
        └─────────────────────────────────────────────────────┘
                     │  u₁, ΔKE₁
        ┌─────────────────────────────────────────────────────┐
        │  FluidLayer 2                                  │
        │  (aynı yapı: norm1 → NS → residual + ΔKE, norm2 → MLP) │
        └─────────────────────────────────────────────────────┘
                     │  ⋮
        ┌─────────────────────────────────────────────────────┐
        │  FluidLayer N                                  │
        │  [adaptif: ΔKE < ε → dur]                      │
        └─────────────────────────────────────────────────────┘
                     │  u_T [B, L, D]
        ┌────────────▼────────────┐
        │  LayerNorm(u_T)          │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  LM Head (weight-tied)  │
        │  logits = u_T @ Wᵀ_emb  │
        └────────────┬────────────┘
                     │
               logits [B, L, V]
```

**Weight Tying:** LM head ağırlığı `token_emb.weight` ile paylaşılır (Press & Wolf 2017). Bu `vocab_size × d_model` parametreyi tasarruf eder — 50257 × 768 ≈ 38.6M parametre.

---

## 3. NS Katmanı — Detaylı Matematik

Her `FluidLayer`, şu hesabı yapar:

### 3.1 Adveksiyon (v4 — İçerik Bağımlı Hız)

```
# Projeksiyon (öğrenilen, d_k = d_model // 8)
q[b,i]      = W_q · u[b,i]              # [B, L, d_k]
k[b,i]      = W_k · u[b,i]              # [B, L, d_k]

# Causal dot-product: token i'nin hızı, i-1'inci tokenin içeriğine bağlı
speed[b,i]  = tanh( (q[b,i] · k[b,i-1]) / √d_k )   # [B, L, 1]

grad_u[b,i] = ∂u/∂x[b,i]               # [B, L, D] — yerel gradyan
adv[b,i]    = speed[b,i] · grad_u[b,i]  # [B, L, D] — taşınan değişim
```

**Neden içerik bağımlı?**  
Eski `speed = tanh(‖u‖)` yalnızca token i'nin kendi normuna bakıyordu — komşu tokenları hiç görmüyordu.  
V4 ile `speed_i = tanh(q_i · k_{i-1} / √d_k)`: token i'nin taşıma hızı, önceki tokenin içeriğine göre dinamik biçimde ayarlanır. Bu, attention'ın `score(i,j) = ⟨W_Q·uᵢ, W_K·uⱼ⟩` etkileşimini O(L) maliyetle yaklaşık olarak modellemektedir.  

> **Fiziksel yorum:** `speed_i`, Reynolds sayısının içerik gated versiyonu — akış hızı, yukarı akışın bağlamına göre değişir (gerçek türbülansta da bu böyledir).  

`tanh(·)` ile hız `(−1, 1)` aralığında kalır — geri akış (negatif hız) teorik olarak mümkündür; bu, bağlama göre "geri dönüp bak" davranışını sağlar.

### 3.2 Basınç (Causal Mod)

Causal modda gelecek tokenlar görülemez. FFT tabanlı global Poisson çözücü yerine **kümülatif toplam** (cumsum) kullanılır:

```
div_adv[b,i] = mean_d(∂adv_d/∂x[b,i])         # [B, L] — ıraksama skaleri
p[b,i]       = −cumsum(div_adv[b,:])[:i] / α   # [b,i] yalnızca j ≤ i görür
∇p[b,i]      = ∂p/∂x[b,i]                      # [B, L] → broadcast → [B, L, D]
```

**Non-causal modda** (analiz için) tam spektral Poisson çözücü:

```
Fourier uzayında: λ_k = 2(cos(2πk/L) − 1) − α²
p̂_k = f̂_k / λ_k      # f = −div(adv)
p = IFFT(p̂)
```

`α > 0` terimi (Helmholtz sabiti) düşük frekanslı modların patlamasını önler.

### 3.3 Viskozite

```
visc[b,i] = ν · ∇²u[b,i]    # [B, L, D] — ikinci türev: yumuşatma
```

Causal modda:
```
∇²u[b,i] ≈ u[b,i] − 2·u[b,i-1] + u[b,i-2]   # (sıfır-dolgu ile)
```

### 3.4 NS Sağ El Tarafı

```
F(u) = −adv − p_scale · ∇p + ν · ∇²u
```

### 3.5 İleri Geçiş (Pre-norm + Residual + MLP)

`forward(u)` iki sublayer çalıştırır:

```
# --- Sublayer 1: NS routing ---
u1    = norm1(u)                        # pre-norm (u1 cache’lenir)
u_ns  = u1 + dt · F(u1)               # Euler  (ya da RK4)
u     = u + (u_ns − u1)               # residual bağlantısı
ΔKE   = ‖u_ns − u1‖² / ‖u1‖²          # NS değişimi (MLP dahil değil)

# --- Sublayer 2: MLP (knowledge storage) ---
u     = u + mlp(norm2(u))              # pre-norm + residual
```

**ΔKE, MLP’den önce hesaplanır.** Adaptif derinlik için NS routing'in ne kadar değişimi götrdüğü ölçülmek istenir, MLP'ın katkısı değil.

### 3.6 RK4 Entegrasyonu

Sublayer 1 içinde Euler yerine RK4:

```
k1 = F(u)
k2 = F(u + dt/2 · k1)
k3 = F(u + dt/2 · k2)
k4 = F(u + dt   · k3)
u_new = u + dt/6 · (k1 + 2k2 + 2k3 + k4)
```

RK4, Euler'e göre daha kararlı (yüksek ν veya büyük dt için önerilir) ama 4× daha yavaş.

---

## 4. Nedensellik (Causality)

Dil modellemesi autoregressive'dir: `p(x_t | x_1, …, x_{t-1})`. Gelecek tokenlara erişim **kaçak bilgi** (data leakage) oluşturur.

FluidLM'de nedensellik **üç operatörde** uygulanır:

| Operatör           | Non-causal (default)            | Causal (LM için)                |
|--------------------|---------------------------------|---------------------------------|
| `gradient(u)`      | `(u[i+1] − u[i-1]) / 2`        | `u[i] − u[i-1]`  (zero-pad)    |
| `laplacian(u)`     | `u[i+1] − 2u[i] + u[i-1]`      | `u[i] − 2u[i-1] + u[i-2]`     |
| `solve_poisson(f)` | FFT → global, tüm dizi          | `cumsum(−div_adv) / α`          |

`FluidLayer(causal=True)` (varsayılan) ile tüm operatörler causal çalışır.  
`FluidLM(causal=True)` tüm katmanlara `causal=True` geçirir.

---

## 5. Adaptif Derinlik

Her katmandan sonra **kinetik enerji değişimi** hesaplanır:

```
ΔKE = mean_over_batch( ‖u_new − u_old‖² )
```

Eğer `ΔKE < convergence_threshold` ve en az `min_steps` katman çalıştıysa → **erken dur**.

```python
# FluidLM.forward() içinde
for i, layer in enumerate(self.layers):
    u, delta_ke = layer(u)         # ΔKE = NS routing değişimi (MLP dahil değil)
    if adaptive and i >= min_steps - 1:
        if delta_ke < convergence_threshold:
            break
```

**Sonuç:** Basit tokenlar (noktalama, sık kelimeler) 3–4 adımda yakınsarken karmaşık içerik 10–12 adım kullanabilir. Bu, sabit derinlikli Transformer'lara göre daha verimli bir hesaplama profili sağlar.

---

## 6. Parametre Sayısı Karşılaştırması

### FluidLM-S (d=768, L=12)

| Bileşen             | Parametre Sayısı    | Not                                      |
|---------------------|---------------------|------------------------------------------|
| Token embedding     | 50,257 × 768 = 38.6M| LM head ile paylaşılır (weight tying)    |
| Positional embedding| 1,024 × 768 = 0.8M  |                                          |
| NS routing / katman | 4 skaler             | log_nu, log_dt, log_alpha, log_p_scale   |
| NS routing toplam   | 12 × 4 = **48**     | Tüm katmanlar için                       |
| MLP / katman        | 4 × 768² × 2 ≈ 4.7M | FFN (hidden=4×d)                         |
| MLP toplam          | 12 × 4.7M = 56.6M   |                                          |
| LayerNorm vb.       | ~0.2M               |                                          |
| **Toplam**          | **~96M**            |                                          |

### GPT-S (aynı d=768, L=12)

| Bileşen             | Parametre Sayısı    | Not                                      |
|---------------------|---------------------|------------------------------------------|
| Token + pos embedding| 39.4M             | FluidLM ile aynı                         |
| MHA / katman        | 4 × 768² = 2.4M     | Q, K, V, O projeksiyonları               |
| MHA toplam          | 12 × 2.4M = **28.3M**|                                         |
| MLP / katman        | 4.7M               | FluidLM ile **aynı**                     |
| MLP toplam          | 56.6M              |                                          |
| **Toplam**          | **~124M**           |                                          |

**Routing farkı:** 48 parametre (NS) vs 28,311,552 parametre (MHA) — **589,824× fark**.  
MLP aynı tutulduğunda bu fark, iki modelin karşılaştırmasında izole edilmiş değişkeni oluşturur.

---

## 7. Entegrasyon Yöntemleri

```python
FluidLayer(integrator='euler')  # Hızlı, prototyping için
FluidLayer(integrator='rk4')    # Kararlı, üretim için (varsayılan)
```

| Özellik           | Euler           | RK4              |
|-------------------|-----------------|------------------|
| Sıra              | O(h)            | O(h⁴)            |
| RHS değerlendirme | 1×              | 4×               |
| Bellek            | 1×              | ~4×              |
| Kullanım          | Debug/hız       | Eğitim/üretim    |

---

## 8. Modül Bağımlılık Grafiği

```
fluid_ops.py
  ├── gradient()         — ∂u/∂x (causal veya non-causal)
  ├── laplacian()        — ∂²u/∂x² (causal veya non-causal)
  ├── divergence()       — Σ_d ∂u_d/∂x → skaler [B,L]
  └── solve_poisson()    — FFT tabanlı (∇² − α²)p = f

ns_layer.py
  └── FluidLayer(nn.Module)
        ├── Parametreler: log_nu, log_dt, log_alpha, log_p_scale
        ├── norm1, norm2   — pre-norm (NS sublayer ve MLP sublayer)
        ├── mlp            — Linear → GELU → Dropout → Linear → Dropout
        ├── _rhs(u)        — NS sağ el tarafı, fluid_ops kullanır
        └── forward(u)     — norm1→NS→residual + ΔKE, norm2→MLP→residual

fluid_lm.py
  └── FluidLM(nn.Module)
        ├── token_emb, pos_emb, emb_drop
        ├── layers: [FluidLayer × N]
        ├── norm, lm_head (weight-tied)
        ├── forward()    — adaptif derinlik, logits döner
        ├── generate()   — autoregressive üretim
        └── physical_params() — ν, dt, α, p_scale değerleri
```

---

## 9. Bellek Analizi

### Eğitim (bfloat16, batch=32, seq=1024, grad_accum=15)

| Bileşen                      | S Scale (~96M)    | M Scale (~285M)    |
|------------------------------|-------------------|--------------------|
| Model parametreleri (bf16)   | ~0.19 GB          | ~0.57 GB           |
| Optimizer state (fp32, ×3)   | ~1.15 GB          | ~3.42 GB           |
| Gradyanlar (bf16)            | ~0.19 GB          | ~0.57 GB           |
| Aktivasyonlar (B=32, seq=1024)| ~8 GB            | ~18 GB             |
| **Toplam**                   | **~10 GB**        | **~23 GB**         |

RTX PRO 6000 (102 GB VRAM) her iki ölçeği de rahatça taşır.

### Çıkarım (batch=1, seq=1024)

| Bileşen                      | S Scale           | M Scale            |
|------------------------------|-------------------|--------------------|
| Model parametreleri          | ~0.19 GB          | ~0.57 GB           |
| KV cache (yok — NS modeli)   | **0 GB**          | **0 GB**           |
| Aktivasyonlar                | ~0.1 GB           | ~0.2 GB            |
| **Toplam**                   | **~0.3 GB**       | **~0.8 GB**        |

> **Not:** FluidLM'de KV cache yoktur — basınç hesabı cumsum üzerinden yapılır, geçmiş aktivasyon saklanmaz. Çıkarımda bellek avantajı vardır.

---

## 10. Ölçek Konfigürasyonları

| Ölçek | d_model | n_layers | MLP hidden | seq_len | Vocab    | Yaklaşık param |
|-------|---------|----------|------------|---------|----------|----------------|
| Toy   | 128     | 4        | 512        | 256     | 50,257   | ~10M           |
| S     | 768     | 12       | 3,072      | 1,024   | 50,257   | ~96M           |
| M     | 1,024   | 24       | 4,096      | 1,024   | 50,257   | ~285M          |
| L     | 1,280   | 36       | 5,120      | 2,048   | 50,257   | ~750M          |
| XL    | 2,048   | 24       | 8,192      | 2,048   | 50,257   | ~950M          |

Exp 08'deki ~950M parametre modeli XL ölçeğine karşılık gelir (d=2048, L=24).

---

## Referanslar

- Navier-Stokes denklemi: Batchelor, G.K. (1967). *An Introduction to Fluid Dynamics.*
- Spectral Poisson: Canuto et al. (1987). *Spectral Methods in Fluid Dynamics.*
- Weight tying: Press & Wolf (2017). *Using the Output Embedding to Improve Language Models.*
- Adaptif hesaplama: Graves (2016). *Adaptive Computation Time for Recurrent Neural Networks.*
- Chinchilla ölçek yasaları: Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.*
