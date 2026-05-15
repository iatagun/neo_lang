# FluidLM — API Referansı

> Tüm public sınıf ve fonksiyonların imzaları, parametreleri ve dönüş değerleri.

---

## İçindekiler

- [`fluidlm.fluid_ops`](#fluidlmfluid_ops)
- [`fluidlm.ns_layer.FluidLayer`](#fluidlmns_layerfluidlayer)
- [`fluidlm.fluid_lm.FluidLM`](#fluidlmfluid_lmfluidlm)

---

## `fluidlm.fluid_ops`

Tüm operatörler `[B, L, D]` tensörler üzerinde çalışır (B=batch, L=seq, D=embed).

---

### `gradient(u, causal=False)`

Birinci türev `∂u/∂x` dizi ekseni boyunca.

```python
gradient(u: torch.Tensor, causal: bool = False) -> torch.Tensor
```

| Parametre | Tip    | Varsayılan | Açıklama |
|-----------|--------|------------|----------|
| `u`       | Tensor | —          | `[B, L, D]` |
| `causal`  | bool   | `False`    | `True` → geriye fark (dil modeli), `False` → merkezi fark (analiz) |

**Dönüş:** `[B, L, D]` — `u` ile aynı şekil.

**Causal=False (merkezi fark, periyodik BC):**
```
∂u/∂x[i] = (u[i+1] − u[i-1]) / 2
```

**Causal=True (geriye fark, sıfır-dolgu):**
```
∂u/∂x[i] = u[i] − u[i-1]    (u[-1] = 0)
```

---

### `laplacian(u, causal=False)`

İkinci türev `∂²u/∂x²`.

```python
laplacian(u: torch.Tensor, causal: bool = False) -> torch.Tensor
```

| Parametre | Tip    | Varsayılan | Açıklama |
|-----------|--------|------------|----------|
| `u`       | Tensor | —          | `[B, L, D]` |
| `causal`  | bool   | `False`    | Nedensellik modu |

**Dönüş:** `[B, L, D]`

**Formüller:**
```
Non-causal: u[i+1] − 2u[i] + u[i-1]
Causal:     u[i]   − 2u[i-1] + u[i-2]   (iki sıfır-dolgu)
```

---

### `divergence(u, causal=False)`

Ortalama ıraksama skaleri.

```python
divergence(u: torch.Tensor, causal: bool = False) -> torch.Tensor
```

**Dönüş:** `[B, L]` — `mean_d(∂u_d/∂x)`.

---

### `solve_poisson(rhs, alpha=1.0)`

Helmholtz–Poisson denklemi `(∇² − α²) p = rhs` için spektral çözücü.

```python
solve_poisson(rhs: torch.Tensor, alpha: float = 1.0) -> torch.Tensor
```

| Parametre | Tip    | Varsayılan | Açıklama |
|-----------|--------|------------|----------|
| `rhs`     | Tensor | —          | `[B, L]` — sağ el tarafı |
| `alpha`   | float  | `1.0`      | Helmholtz sabiti ≥ 0; 0 → standart Poisson |

**Dönüş:** `[B, L]` — basınç alanı `p`.

**Fourier uzayında:**
```
λ_k = 2(cos(2πk/L) − 1) − α²
p̂_k = f̂_k / λ_k
p    = IFFT(p̂)
```

`α > 0` her `k` için `λ_k < −α² < 0` garantisi verir → bölme sıfıra yaklaşmaz.

**Fiziksel yorum:**
- `α = 0` → tam global etkileşim (attention analog)
- `α → ∞` → yalnızca yerel etkileşim (local smoothing)
- FluidLM'de `α = exp(log_alpha)` öğrenilen parametredir

---

## `fluidlm.ns_layer.FluidLayer`

Tek Navier-Stokes zaman adımı olarak learnable katman.

```python
class FluidLayer(nn.Module)
```

### Yapıcı

```python
FluidLayer(
    d_model:    int,
    nu:         float = 0.01,
    dt:         float = 0.1,
    alpha:      float = 1.0,
    integrator: str   = 'rk4',
    causal:     bool  = True,
)
```

| Parametre    | Tip   | Varsayılan | Açıklama |
|-------------|-------|------------|----------|
| `d_model`   | int   | —          | Embedding boyutu D |
| `nu`        | float | `0.01`     | Kinematik viskozite başlangıç değeri |
| `dt`        | float | `0.1`      | Zaman adımı başlangıç değeri |
| `alpha`     | float | `1.0`      | Helmholtz sabiti başlangıç değeri |
| `integrator`| str   | `'rk4'`    | `'euler'` veya `'rk4'` |
| `causal`    | bool  | `True`     | Dil modeli için `True` (autoregressive) |

### Learnable Parametreler

| Parametre   | Şekil  | Kısıt     | Açıklama |
|------------|--------|-----------|----------|
| `log_nu`   | skaler | exp → > 0 | Kinematik viskozite |
| `log_dt`   | skaler | exp → > 0 | Zaman adımı |
| `log_alpha`| skaler | exp → > 0 | Poisson etkileşim menzili |
| `log_p_scale`| skaler | exp → > 0 | Basınç katkısı ağırlığı |

### Properties

```python
layer.nu       # exp(log_nu)      — anlık kinematik viskozite
layer.dt       # exp(log_dt)      — anlık zaman adımı
layer.alpha    # exp(log_alpha)   — anlık Helmholtz sabiti
layer.p_scale  # exp(log_p_scale) — anlık basınç ölçeği
```

### `forward(u)`

```python
forward(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

| Parametre | Tip    | Açıklama |
|-----------|--------|----------|
| `u`       | Tensor | `[B, L, D]` — giriş hız alanı |

**Dönüş:** `(u_new, delta_ke)`
- `u_new`: `[B, L, D]` — güncellenmiş hız alanı
- `delta_ke`: skaler Tensor — `mean(‖u_new − u‖²)`

**Örnek:**
```python
layer = FluidLayer(d_model=768, nu=0.01, dt=0.1)
u = torch.randn(2, 128, 768)
u_new, dke = layer(u)
print(f"ΔKE: {dke.item():.6f}")
```

---

## `fluidlm.fluid_lm.FluidLM`

Navier-Stokes dinamiğine dayalı tam dil modeli.

```python
class FluidLM(nn.Module)
```

### Yapıcı

```python
FluidLM(
    vocab_size:             int,
    d_model:                int   = 256,
    n_layers:               int   = 12,
    max_seq_len:            int   = 512,
    nu:                     float = 0.01,
    dt:                     float = 0.1,
    alpha:                  float = 1.0,
    integrator:             str   = 'rk4',
    convergence_threshold:  float = 1e-4,
    min_steps:              int   = 3,
    dropout:                float = 0.1,
    causal:                 bool  = True,
)
```

| Parametre               | Tip   | Varsayılan | Açıklama |
|------------------------|-------|------------|----------|
| `vocab_size`           | int   | —          | Kelime dağarcığı boyutu |
| `d_model`              | int   | `256`      | Embedding / hız alanı boyutu |
| `n_layers`             | int   | `12`       | Maksimum NS katmanı sayısı |
| `max_seq_len`          | int   | `512`      | Positional embedding için maksimum uzunluk |
| `nu`                   | float | `0.01`     | Tüm katmanlar için başlangıç viskozitesi |
| `dt`                   | float | `0.1`      | Tüm katmanlar için başlangıç zaman adımı |
| `alpha`                | float | `1.0`      | Tüm katmanlar için başlangıç Helmholtz sabiti |
| `integrator`           | str   | `'rk4'`    | `'euler'` veya `'rk4'` |
| `convergence_threshold`| float | `1e-4`     | Adaptif durdurma için ΔKE eşiği |
| `min_steps`            | int   | `3`        | Adaptif modda minimum katman sayısı |
| `dropout`              | float | `0.1`      | Embedding dropout oranı |
| `causal`               | bool  | `True`     | Tüm katmanlara nedensellik modu |

### `forward(input_ids, adaptive=False, return_stats=False)`

```python
forward(
    input_ids:    torch.Tensor,
    adaptive:     bool = False,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]
```

| Parametre      | Tip    | Varsayılan | Açıklama |
|---------------|--------|------------|----------|
| `input_ids`   | Tensor | —          | `[B, L]` — token indeksleri |
| `adaptive`    | bool   | `False`    | ΔKE < eşik → erken dur |
| `return_stats`| bool   | `False`    | ΔKE geçmişi ve adım sayısını döndür |

**Dönüş (return_stats=False):** `logits [B, L, vocab_size]`

**Dönüş (return_stats=True):** `(logits, stats)` — `stats = {'n_steps': int, 'delta_kes': List[float]}`

**Örnek:**
```python
model = FluidLM(vocab_size=50257, d_model=768, n_layers=12)
ids   = torch.randint(0, 50257, (2, 128))

# Standart ileri geçiş
logits = model(ids)                                     # [2, 128, 50257]

# Adaptif derinlik + istatistik
logits, stats = model(ids, adaptive=True, return_stats=True)
print(f"Kullanılan katman: {stats['n_steps']}")
print(f"ΔKE değerleri: {stats['delta_kes']}")
```

### `generate(prompt_ids, max_new_tokens=64, temperature=1.0, top_k=50, adaptive=True)`

Autoregressive token üretimi.

```python
@torch.no_grad()
generate(
    prompt_ids:     torch.Tensor,
    max_new_tokens: int   = 64,
    temperature:    float = 1.0,
    top_k:          int   = 50,
    adaptive:       bool  = True,
) -> torch.Tensor
```

| Parametre       | Tip    | Varsayılan | Açıklama |
|----------------|--------|------------|----------|
| `prompt_ids`   | Tensor | —          | `[B, L_prompt]` |
| `max_new_tokens`| int   | `64`       | Üretilecek token sayısı |
| `temperature`  | float  | `1.0`      | < 1 → keskin, > 1 → düz dağılım |
| `top_k`        | int    | `50`       | İlk k adayla örnekleme; 0 → kısıtsız |
| `adaptive`     | bool   | `True`     | Üretimde adaptif derinlik kullan |

**Dönüş:** `[B, L_prompt + max_new_tokens]`

**Örnek:**
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
prompt = torch.tensor([enc.encode("The theory of")]).cuda()

output = model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50)
print(enc.decode(output[0].tolist()))
```

### `num_parameters(trainable_only=True)`

```python
num_parameters(trainable_only: bool = True) -> int
```

**Örnek:**
```python
print(f"Toplam parametre: {model.num_parameters():,}")
# → Toplam parametre: 95,947,825
```

### `physical_params()`

Her katmanın öğrenilen fiziksel parametrelerini döndürür.

```python
physical_params() -> dict[str, list[float]]
```

**Dönüş:**
```python
{
    'nu':      [0.0159, 0.0255, ..., 0.0258],  # viskozite (artış eğilimi)
    'dt':      [0.1094, 0.0613, ..., 0.0940],  # zaman adımı (U-şekli)
    'alpha':   [1.0000, 1.0000, ..., 1.0000],  # Helmholtz sabiti
    'p_scale': [0.1000, 0.1000, ..., 0.1000],  # basınç ağırlığı
}
```

**Örnek:**
```python
params = model.physical_params()
nu_vals = params['nu']
print(f"ν gradyanı: {nu_vals[-1] - nu_vals[0]:+.4f}")
# → ν gradyanı: +0.0099  (erken katmanlar daha az düzleştirir)
```

---

## Hızlı Başlangıç

```python
import torch
from fluidlm import FluidLM

# Model oluştur
model = FluidLM(
    vocab_size=50257,   # tiktoken gpt2
    d_model=768,
    n_layers=12,
    max_seq_len=1024,
    integrator='rk4',
    causal=True,
).cuda().bfloat16()

print(f"Parametre: {model.num_parameters():,}")  # ~96M

# Eğitim adımı
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

input_ids  = torch.randint(0, 50257, (4, 256)).cuda()    # [B=4, L=256]
target_ids = torch.randint(0, 50257, (4, 256)).cuda()

logits = model(input_ids)                                 # [4, 256, 50257]
loss   = torch.nn.functional.cross_entropy(
    logits.view(-1, 50257),
    target_ids.view(-1)
)
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Loss: {loss.item():.4f}")

# Metin üretimi
import tiktoken
enc    = tiktoken.get_encoding("gpt2")
prompt = torch.tensor([enc.encode("The study of artificial")]).cuda()
out    = model.generate(prompt, max_new_tokens=50, temperature=0.8)
print(enc.decode(out[0].tolist()))

# Fiziksel parametreler
params = model.physical_params()
print("ν değerleri:", [f"{v:.4f}" for v in params['nu']])
```

---

## Checkpoint Kaydetme ve Yükleme

```python
# Kaydet
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'step': current_step,
    'best_val_ppl': best_ppl,
    'config': {
        'd_model': 768, 'n_layers': 12,
        'vocab_size': 50257, 'max_seq_len': 1024,
    }
}, 'checkpoints/my_model.pt')

# Yükle
ckpt = torch.load('checkpoints/my_model.pt', map_location='cuda')
cfg  = ckpt['config']
model = FluidLM(**cfg).cuda().bfloat16()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Step {ckpt['step']}, Val PPL: {ckpt['best_val_ppl']:.4f}")
```
