# FluidLM Nasıl Çalışır?
### Hiçbir şey bilmiyorum, ama merak ediyorum — diye başlayan biri için

> Burada sadece **toplama, çıkarma, çarpma, bölme** var. Başka hiçbir şey yok.  
> Okurken kalem-kağıt al, sayıları kendin hesapla.

---

## Önce şunu anlayalım: Bilgisayar kelimeyi nasıl "görür"?

Bilgisayar metni okuyamaz. Sadece sayıları anlayabilir.

Diyelim ki alfabemizde 4 harf var: **A, B, C, D**

Biz şöyle bir sözlük yapalım:

```
A = 1
B = 2
C = 3
D = 4
```

"BAD" yazdığımızda bilgisayar bunu görür: **[2, 1, 4]**

Tamam. Ama bu sayılar çok küçük, tek boyutlu. Bilgisayar bunlarla pek bir şey yapamaz.  
O yüzden her harfi **daha uzun bir sayı listesine** çeviririz.

---

## Adım 1 — Harfi Vektöre Çevir (Embedding)

"A" harfi için şöyle bir liste üretelim *(bu listeyi model öğreniyor, biz şimdilik sabit verelim)*:

```
A = [0.2,  0.8,  0.1]
B = [0.9,  0.3,  0.7]
C = [0.1,  0.5,  0.9]
```

"BAC" kelimesini girersek, elimizde şu tablo var:

```
Token 0 (B) = [0.9,  0.3,  0.7]
Token 1 (A) = [0.2,  0.8,  0.1]
Token 2 (C) = [0.1,  0.5,  0.9]
```

Bu tabloya **u** diyoruz. Boyutu: **[3 token × 3 boyut]**

> Fizik benzetmesi: Bu tabloyu bir nehrin kesiti gibi düşün.  
> Her satır bir nokta, her sütun o noktadaki "akış hızı".  
> FluidLM bu tabloyu nehir gibi akıtır. Katmanlar = zamanın geçmesi.

---

## Adım 2 — Konum Bilgisi Ekle (Position)

Bilgisayar "BAC" ile "CAB"'ın farklı olduğunu nereden bilecek? Sırayı bilmesi lazım.

Her pozisyon için ayrı bir sayı listesi hazırlıyoruz:

```
Pozisyon 0 = [0.01, 0.02, 0.01]
Pozisyon 1 = [0.02, 0.01, 0.03]
Pozisyon 2 = [0.03, 0.03, 0.02]
```

Bunu harf listesiyle **toplayıp** yeni u'yu elde ediyoruz:

```
Token 0: [0.9+0.01,  0.3+0.02,  0.7+0.01] = [0.91, 0.32, 0.71]
Token 1: [0.2+0.02,  0.8+0.01,  0.1+0.03] = [0.22, 0.81, 0.13]
Token 2: [0.1+0.03,  0.5+0.03,  0.9+0.02] = [0.13, 0.53, 0.92]
```

Bu bizim başlangıç **u₀**'ımız. Artık her katmanda bu tabloyu güncelleyeceğiz.

---

## Adım 3 — Gradyan: "Komşumdan Ne Kadar Farklıyım?"

Şimdi eğlenceli kısım başlıyor.

Her tokenın, **solundaki tokenden ne kadar farklı olduğunu** hesaplıyoruz.

```
Gradyan[0] = u[0] - 0          = [0.91, 0.32, 0.71]   ← solda kimse yok, sıfırla çıkar
Gradyan[1] = u[1] - u[0]       = [0.22-0.91, 0.81-0.32, 0.13-0.71] = [-0.69,  0.49, -0.58]
Gradyan[2] = u[2] - u[1]       = [0.13-0.22, 0.53-0.81, 0.92-0.13] = [-0.09, -0.28,  0.79]
```

Bu işlemin adı matematikçiler arasında **"türev"** ya da **"∂u/∂x"**.  
Ama burada yaptığımız şey şu: **çıkarma**.  
"Sağındakinden solundakini çıkar." Bu kadar.

> Önemli: Token 0 için solda kimse olmadığından sıfır koyuyoruz.  
> Bu **"geçmişe bak, geleceğe bakma"** kuralını sağlıyor.  
> Yoksa model henüz yazmadığı harfleri "görürdü" — hile olurdu.

---

## Adım 4 — Hız (Speed): "Ne Kadar Büyüktüm?"

Her tokenın ne kadar büyük olduğunu bir sayıyla özetliyoruz.

Token 0: `[0.91, 0.32, 0.71]`  
Büyüklük = `0.91² + 0.32² + 0.71²` = `0.828 + 0.102 + 0.504` = **1.434**  
Karekök = **1.198**

*(Karekök bilmiyorsan: "1.2 × 1.2 = 1.44, 1.198 × 1.198 ≈ 1.434" diye bulunur)*

Sonra bu büyüklüğü **0 ile 1 arasına sıkıştırıyoruz**. Bunun adı tanh.  
Tam formülü şimdi öğrenmene gerek yok. Sadece şunu bil:

```
Büyüklük çok küçükse → hız ≈ 0  (neredeyse durmuş)
Büyüklük büyükse    → hız ≈ 1  (tam hızda ama 1'i geçemiyor)
```

Diyelim ki Token 0'ın hızı = **0.83** çıktı.

---

## Adım 5 — Adveksiyon: "Hareket Taşıma"

Şimdi gradyanı hızla çarpıyoruz. Adı **adveksiyon**.

```
Adveksiyon[0] = hız[0] × Gradyan[0]
              = 0.83 × [0.91, 0.32, 0.71]
              = [0.755, 0.266, 0.589]

Adveksiyon[1] = hız[1] × Gradyan[1]
              = 0.71 × [-0.69, 0.49, -0.58]
              = [-0.490, 0.348, -0.412]

Adveksiyon[2] = hız[2] × Gradyan[2]
              = 0.88 × [-0.09, -0.28, 0.79]
              = [-0.079, -0.246, 0.695]
```

> Bu nedir? Her token, **ne kadar hızlıysa o kadar** komşusundan etkileniyor.  
> Hızlı token → büyük komşu etkisi  
> Durmuş token → komşu onu pek etkilemiyor  
>
> Transformer'da bu işin adı "FFN (MLP)". Orada büyük matrisler var.  
> Burada sadece çarpma işlemi.

---

## Adım 6 — Basınç: "Tüm Zinciri Birbirine Bağla"

Şimdiye kadar yaptıklarımız hep **yerel** — her token yalnızca komşusuna bakıyor.  
Ama "BAC" yazarken A, hem B'den hem C'den etkilenmeli.

Bunun için **basınç** hesaplıyoruz. Bu adım, tüm tokenlara **aynı anda** bakıyor.

### Adım 6a — Iraksama (Divergence): "Her tokendan ne kadar yayılıyor?"

Her tokenın adveksiyonunun ortalama büyüklüğünü alıyoruz:

```
div[0] = (0.755 + 0.266 + 0.589) / 3 = 1.610 / 3 = 0.537
div[1] = (-0.490 + 0.348 + (-0.412)) / 3 = -0.554 / 3 = -0.185
div[2] = (-0.079 + (-0.246) + 0.695) / 3 = 0.370 / 3 = 0.123
```

### Adım 6b — Birikimli Toplam: "Etki birikiyor"

Şimdi bu değerleri **soldan sağa biriktiriyoruz** (cumsum):

```
p[0] = -div[0]                    = -0.537
p[1] = -div[0] + (-div[1])        = -0.537 + 0.185 = -0.352
p[2] = -div[0] + (-div[1]) + (-div[2]) = -0.352 + (-0.123) = -0.475
```

Sonra α ile bölüyoruz (α = 1.0 diyelim, yani bölme işlemi değiştirilmiyor):

```
p[0] = -0.537 / 1.0 = -0.537
p[1] = -0.352 / 1.0 = -0.352
p[2] = -0.475 / 1.0 = -0.475
```

### Adım 6c — Normalize et: "Çok büyük çıkmasın"

Standart sapmasını hesapla (bu biraz uzun ama tek bir "büyüklük ölçeği"):

```
ortalama = (-0.537 + (-0.352) + (-0.475)) / 3 = -0.455
farklar² = (0.082² + 0.103² + 0.020²) = 0.007 + 0.011 + 0.000 ≈ 0.018
std ≈ √(0.018/3) ≈ 0.077
```

Her p değerini std ile bölüyoruz:

```
p[0] = -0.537 / 0.077 = -6.97
p[1] = -0.352 / 0.077 = -4.57
p[2] = -0.475 / 0.077 = -6.17
```

*(Normalize sayılar büyük görünüyor ama önemli olan aralarındaki **fark**.)*

### Adım 6d — Basınç Gradyanı: "Basınç nereden nereye akıyor?"

Şimdi p'nin gradyanını alıyoruz (yine çıkarma):

```
p_grad[0] = p[0] - 0       = -6.97
p_grad[1] = p[1] - p[0]    = -4.57 - (-6.97) = +2.40
p_grad[2] = p[2] - p[1]    = -6.17 - (-4.57) = -1.60
```

Bu değeri `p_scale` ile çarpıyoruz (p_scale = 0.1):

```
p_grad_scaled[0] = -6.97 × 0.1 = -0.697
p_grad_scaled[1] = +2.40 × 0.1 = +0.240
p_grad_scaled[2] = -1.60 × 0.1 = -0.160
```

> Token 1 pozitif basınç gradyanı aldı: "komşusuna doğru itildi."  
> Token 0 negatif aldı: "komşusundan uzaklaştı."  
> Bu, Transformer'daki **attention**'ın fizikten türeyen karşılığı.  
> Yani attention elle yazılmadı — basınç denklemi onu kendi üretti.

---

## Adım 7 — Viskozite: "Komşularımın Ortalamasına Yaklaş"

Viskozite, her tokenı komşularının ortalamasına yaklaştırır. Pürüzleri siler.

İkinci gradyan (laplacian). Yine çıkarma ama bu sefer iki adım:

```
laplacian[0] = u[0] - 2×0    + 0        = u[0]  (solda iki sıfır)
laplacian[1] = u[1] - 2×u[0] + 0        (solda bir sıfır)
laplacian[2] = u[2] - 2×u[1] + u[0]
```

Token 2 için hesaplayalım (sadece ilk boyut):

```
laplacian[2][0] = 0.13 - 2×0.22 + 0.91
               = 0.13 - 0.44 + 0.91
               = 0.60
```

Bunu ν ile çarpıyoruz (ν = 0.011 diyelim):

```
visc[2][0] = 0.011 × 0.60 = 0.0066
```

> Yüksek ν → tokenlerin temsilleri birbirine benziyor → soyut, genel anlam  
> Düşük ν → tokenlerin temsilleri farklı kalıyor → ince sözdizimsel ayrımlar  
>
> Model bunu eğitim sırasında kendi keşfetti:  
> Erken katmanlar → ν küçük (farklılıkları koru)  
> Geç katmanlar → ν büyük (benzeştir, soyutlaştır)

---

## Adım 8 — Hepsini Topla: NS Denklemi

Şimdiye kadar üç şey hesapladık:

```
1. Adveksiyon  (−adv)       : yerel anlam taşıma
2. Basınç      (−p_grad)    : global long-range bağlantı
3. Viskozite   (+visc)      : düzleştirme, soyutlaştırma
```

Bunları topla → bu katmanın "değişim miktarı" F(u):

```
F(u)[i] = -adv[i] - p_grad[i] + visc[i]
```

Token 2, ilk boyut için hesaplayalım:

```
F(u)[2][0] = -(-0.079)  -  (-0.160)  +  0.0066
           =   0.079    +   0.160    +  0.0066
           =   0.246
```

---

## Adım 9 — Güncelle: Yeni u

Zaman adımı Δt ile çarpıp u'ya ekliyoruz:

```
u_new[2][0] = u[2][0] + Δt × F(u)[2][0]
            = 0.13    + 0.05 × 0.246
            = 0.13    + 0.0123
            = 0.142
```

Bu kadar. **Bu bir katmanın çıktısı.**  
Sonraki katman bu yeni u'yu alır, aynı işlemleri tekrar yapar.  
12 katman boyunca bu devam eder.

---

## Adım 10 — Ne Zaman Duruyoruz? (ΔKE)

Her katmandan sonra şunu hesaplıyoruz:

```
Değişim = u_new - u_old
ΔKE = (tüm değişim değerlerinin karelerinin ortalaması) / (u'nun büyüklüğü)
```

Token 2, ilk boyut:

```
Değişim = 0.142 - 0.13 = 0.012
Kare    = 0.012² = 0.000144
```

Tüm tokenlar ve tüm boyutlar için bu hesabı yap, ortalamasını al.  
Bu sayı çok küçük oldu mu? **Dur. Akış stabilleşti. Daha fazla katman gerekmez.**

Basit cümleler → erken katmanda stabilleşir → 3-4 katman yeter  
Karmaşık cümleler → geç katmanda stabilleşir → 10-12 katman gerekir

---

## Adım 11 — Skor Hesapla: Hangi Harf En Olası?

12 katmandan sonra elimizde final u var. Boyutu hâlâ [3 token × 3 boyut].

Her token için vocab sözlüğündeki tüm harflerle bir **benzerlik skoru** hesaplıyoruz.  
Bu da çarpma ve toplama:

```
skor("A") = u[2][0] × 0.2  +  u[2][1] × 0.8  +  u[2][2] × 0.1
          = 0.142  × 0.2   +  0.53   × 0.8   +  0.92   × 0.1
          = 0.028  + 0.424 + 0.092
          = 0.544

skor("B") = 0.142 × 0.9  +  0.53 × 0.3  +  0.92 × 0.7
          = 0.128 + 0.159 + 0.644
          = 0.931

skor("C") = 0.142 × 0.1  +  0.53 × 0.5  +  0.92 × 0.9
          = 0.014 + 0.265 + 0.828
          = 1.107
```

En yüksek skor → en olası sonraki harf → **C**

Model "BAC"'tan sonra "C" gelir diyor. Doğruysa kayıp düşük, yanlışsa kayıp yüksek, güncelle.

---

## Her Şeyin Özeti Tek Sayfada

```
GİRDİ: "BAC"
         │
         ▼
[1] Sayıya çevir     B→[0.9,0.3,0.7]  A→[0.2,0.8,0.1]  C→[0.1,0.5,0.9]
         │
[2] Konum ekle       her satıra küçük sayılar topla
         │
         ▼  u₀ = başlangıç tablosu
         │
    ┌────┴──── KATMAN 1 ──────────────────────────────────────────────┐
    │                                                                  │
    │  [3] Gradyan   = her satır - solundaki satır   (çıkarma)        │
    │  [4] Hız       = büyüklük → 0-1 arasına sıkıştır               │
    │  [5] Adveksiyon = hız × gradyan                (çarpma)         │
    │  [6] Basınç    = kümülatif toplam / α          (toplama,bölme)  │
    │  [7] Viskozite = ν × (u - 2×sol + solsol)      (çarpma)         │
    │  [8] F(u)      = -adv - basınç + viskozite      (toplama)        │
    │  [9] u_yeni    = u + Δt × F(u)                 (toplama,çarpma) │
    │  [10] ΔKE      = değişim küçük mü? → dur       (bölme)          │
    └────────────────────────────────────────────────────────────────┘
         │
    ┌────┴──── KATMAN 2 ────┐
    │  aynı işlemler...     │
    └───────────────────────┘
         │
         ⋮  (12 katmana kadar veya ΔKE küçülene kadar)
         │
         ▼  u_final
         │
    [11] Skoru hesapla: u_final × harf tablosu → en yüksek = tahmin
         │
         ▼
    ÇIKTI: sonraki harf tahmini
```

---

## 4 Parametrenin Basit Anlamı

Model eğitim sırasında şu 4 değeri her katman için ayrı ayrı öğreniyor:

| Parametre | Yaptığı iş | Basit benzetme |
|-----------|-----------|---------------|
| **ν (nu)** | Tokenleri birbirine yaklaştır | Sünger: sıkarsan her şey aynı oluyor |
| **Δt** | Her katmanda ne kadar değişsin | Gaz pedalı: küçük = ağır bas, büyük = bas |
| **α** | Basıncın ne kadar uzağa etki etsin | Megafon açma/kapama |
| **p_scale** | Basıncı ne kadar ciddiye al | Basıncın mikrofon sesi |

Model şunu keşfetti: erken katmanlarda ν küçük olmalı, geç katmanlarda büyük.  
Hiç söylemeden, sadece hata küçülsün diye milyonlarca kez deneyerek buldu.

---

## Transformer ile Karşılaştırma (Çok Basit)

| Transformer | FluidLM | Fark |
|-------------|---------|------|
| Büyük matrisler × büyük matrisler | Sayı listeleri + 4 işlem | FluidLM çok daha az parametre |
| Q, K, V = elle tasarlanmış | Adveksiyon + Basınç = fizikten çıktı | Kim daha iyi? Araştırıyoruz |
| Her katman sabit hesap | Stabilleşince dur | FluidLM basit cümle için daha hızlı |
| ~28M parametre/katman (routing) | 4 sayı/katman (routing) | 7 milyon kat fark |

Sonuç: FluidLM biraz daha az doğru (PPL 4.59 vs 4.27) ama çok daha küçük.  
Bu fark kabul edilebilir mi? Araştırmanın sorusu bu.

---

*FluidLM'i daha ayrıntılı öğrenmek istiyorsan sıradaki adım: [`GLOSSARY.md`](GLOSSARY.md)*  
*Mimariyi görsel olarak görmek istiyorsan: [`assets/13_architecture_3d.png`](assets/13_architecture_3d.png)*
