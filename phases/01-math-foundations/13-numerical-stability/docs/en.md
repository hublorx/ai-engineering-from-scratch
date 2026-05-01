# Stabilnosc numeryczna

> Floating point to przeciekawa abstrakcja. Ugryzie cie podczas treningu i nie zobaczysz tego nadchodzacego.

**Typ:** Zbuduj to
**Jezyk:** Python
**Wymagania wstepne:** Faza 1, Lekcje 01-04
**Czas:** ~120 minut

## Cele uczenia sie

- Implementuj numerycznie stabilny softmax i log-sum-exp uzywajac triku z odejmowaniem maksimum
- Zidentyfikuj overflow, underflow i katastrofalna cancelation w obliczeniach floating-point
- Zweryfikuj analityczne gradienty wzgledem numerycznych gradientow za pomoca scentrowanych roznic skonczonych
- Wyjasnij, dlaczego bfloat16 jest preferowany nad float16 do treningu i jak skalowanie loss zapobiega underflow gradientu

## Problem

Twoj model trenujesz przez trzy godziny, a potem loss staje sie NaN. Dodajesz instrukcje print. Logity sa w porzadku w kroku 9000. W kroku 9001 sa `inf`. W kroku 9002 kazdy gradient jest `nan` i trening martwy.

Albo: twoj model trenuje do konca, ale dokladnosc jest o 2% gorsza niz twierdzi artykul. Sprawdzasz wszystko. Architektura sie zgadza. Hiperparametry sie zgadzaja. Dane sie zgadzaja. Problem polega na tym, ze artykul uzyl float32, a ty uzyles float16 bez wlasciwego skalowania. Trzydzieci dwa bity skumulowanego bledu zaokraglenia cicho zjadly twoja dokladnosc.

Albo: implementujesz cross-entropy loss od zera. Dziala na malych logitach. Gdy logity przekraczaja 100, zwraca `inf`. Softmax overflowowal bo `exp(100)` jest wiekszy niz moze reprezentowac float32. Kazdy framework ML obsluguje to dwulinijkowym trikiem. Nie wiediales, ze ten trik istnieje.

Stabilnosc numeryczna to nie teoretyczny problem. To roznica miedzy treningiem, ktory sie udaje, a takim, ktory cicho sie nie udaje. Kazdy powazny bug ML, ktory bedziesz debugowac, ostatecznie sprowadza sie do floating point.

## Koncepcja

### IEEE 754: Jak Komputery Przechowuja Liczby Rzeczywiste

Komputery przechowuja liczby rzeczywiste jako wartosci floating point zgodne ze standardem IEEE 754. Float ma trzy czesci: bit znaku, wykladnik i mantysa (significand).

```
float32 uklad (32 bity calkowicie):
[1 znak] [8 wykladnik] [23 mantysa]

Wartosc = (-1)^znak * 2^(wykladnik - 127) * 1.mantysa
```

Mantysa determinuje precyzje (ile istotnych cyfr). Wykladnik determinuje zakres (jak duza lub mala moze byc liczba).

```
Format     Bity   Wykladnik  Mantysa  Cyfry dzies.  Zakres (przyblizony)
float64    64     11         52       ~15-16         +/- 1.8e308
float32    32     8          23       ~7-8           +/- 3.4e38
float16    16     5          10       ~3-4           +/- 65,504
bfloat16   16     8          7        ~2-3           +/- 3.4e38
```

float32 daje okolo 7 cyfr dziesietnych precyzji. To znaczy, ze moze rozroznisc 1.0000001 i 1.0000002, ale nie 1.00000001 i 1.00000002. Po 7 cyfrach wszystko jest szumem zaokraglenia.

float16 daje okolo 3 cyfr. Najwieksza liczba, jaka moze reprezentowac to 65,504. To zatrwazajaco mala wartosc dla ML, gdzie logity, gradienty i aktywacje regularnie przekraczaja to.

bfloat16 to odpowiedz Google na problem zakresu float16. Ma ten sam 8-bitowy wykladnik co float32 (ten sam zakres, do 3.4e38), ale tylko 7 bitow mantysy (mniejsza precyzja niz float16). Dla trenowania sieci neuronowych zakres jest wazniejszy niz precyzja, wiec bfloat16 zazwyczaj wygrywa.

### Dlaczego 0.1 + 0.2 != 0.3

Liczba 0.1 nie moze byc dokladnie reprezentowana w binarnym floating point. W podstawie 2 jest to powtarzajaca sie ulamek:

```
0.1 w binarnie = 0.0001100110011001100110011... (powtarza sie w nieskonczonosc)
```

float32 obcina to do 23 bitow mantysy. Przechowywana wartosc to w przyblizeniu 0.100000001490116. Podobnie, 0.2 jest przechowywane jako w przyblizeniu 0.200000002980232. Ich suma to 0.300000004470348, nie 0.3.

```
W Pythonie:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

To ma znaczenie dla ML bo:

1. Porownania loss jak `if loss < threshold` moga dac zle odpowiedzi
2. Akumulowanie wielu malych wartosci (aktualizacje gradientu przez tysiecy krokow) dryfuje od prawdziwej sumy
3. Suma kontrolna i testy odtwarzalnosci failuja jesli porownujesz float z `==`

Rozwiazanie: nigdy nie porownuj float z `==`. Uzywaj `abs(a - b) < epsilon` lub `math.isclose()`.

### Katastrofalna Cancelation

Gdy odejmujesz dwie niemal rowne liczby floating point, istotne cyfry sie znosza i zostajesz z szumem zaokraglenia promowanym do wiodacych cyfr.

```
a = 1.0000001    (przechowywane jako 1.00000011920929 w float32)
b = 1.0000000    (przechowywane jako 1.00000000000000 w float32)

Prawdziwa roznica:  0.0000001
Obliczona:          0.00000011920929

Blad wzgledny: 19.2%
```

To jest 19% bled wzgledny z jednego odejmowania. W ML, to sie dzieje za kazdym razem gdy:

- Obliczasz wariancje danych z duza srednia: `E[x^2] - E[x]^2` gdy E[x] jest duze
- Odejmujesz niemal rowne log-prawdopodobienstwa
- Obliczasz gradienty przez skonczone rozницы ze zbyt malym epsilon

Rozwiazanie: przearanizuj wzory, zeby uniknac odejmowania duzych, niemal rownych liczb. Dla wariancji, uzyj algorytmu Welforda lub scentruj dane najpierw. Dla log-prawdopodobienstw, pracuj w przestrzeni log przez caly czas.

### Overflow i Underflow

Overflow wystepuje gdy wynik jest zbyt duzy do reprezentacji. Underflow wystepuje gdy jest zbyt maly (blizszy zero niz najmniejsza reprezentowalna dodatnia liczba).

```
Granice float32:
  Maksimum:  3.4028235e+38
  Minimum dodatnie (normalne): 1.175e-38
  Minimum dodatnie (denorm): 1.401e-45
  Overflow:  cokolwiek > 3.4e38 staje sie inf
  Underflow: cokolwiek < 1.4e-45 staje sie 0.0
```

Funkcja `exp()` jest glownym zrodlem overflow w ML:

```
exp(88.7)  = 3.40e+38   (z trudem miesci sie w float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (z trudem powyzej underflow)
exp(-104)  = 0.0         (underflow do zera)
```

Funkcja `log()` uderza w drugim kierunku:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (w porzadku)
log(1e-46) = -inf        (wejscie underflowowalo do 0, potem log(0) = -inf)
```

W ML, `exp()` pojawia sie w softmax, sigmoid i obliczeniach prawdopodobienstwa. `log()` pojawia sie w cross-entropy, log-likelihoods i KL divergence. Kombinacja `log(exp(x))` to pole minowe bez wlasciwych sztuczek.

### Trik Log-Sum-Exp

Obliczanie `log(sum(exp(x_i)))` bezposrednio jest numerycznie niebezpieczne. Jesli jakies `x_i` jest duze, `exp(x_i)` overflowuje. Jesli wszystkie `x_i` sa bardzo ujemne, kazde `exp(x_i)` underflowuje do zera i `log(0)` to `-inf`.

Trik: odejmij wartosc maksymalna przed potegowaniem.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Dlaczego to dziala: po odjeciu `max(x)`, najwiekszy wykladnik to `exp(0) = 1`. Zadnego overflow mozliwe. Co najmniej jeden skladnik sumy to 1, wiec suma to co najmniej 1, a `log(1) = 0`. Zadnego underflow do `-inf` mozliwe.

Dowod:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (dodaj i odejmij c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (wyczynnij exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Ustaw `c = max(x)` i overflow jest wyeliminowany.

Ten trik pojawia sie wszedzie w ML:
- Normalizacja softmax
- Obliczanie cross-entropy loss
- Sumowanie log-prawdopodobienstw w modelach sekwencyjnych
- Mieszanina Gaussianow
- Wnioskowanie wariacyjne

### Dlaczego Softmax Potrzebuje Triku z Odejmowaniem Maksimum

Softmax konwertuje logity na prawdopodobienstwa:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Bez triku, logity [100, 101, 102] powoduja overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

Czy to overflowuje float32 (max ~3.4e38)? Nie, 2.69e43 < 3.4e38? Wlasciwie:
exp(88.7) jest juz na granicy float32.
exp(100) = inf w float32.
```

Z trikiem, odejmij max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Prawdopodobienstwa sa identyczne. Obliczenia sa bezpieczne. To nie jest optymalizacja. To wymog poprawnosci.

### NaN i Inf: Wykrywanie i Zapobieganie

`nan` (Not a Number) i `inf` (nieskonczonosc) rozprzestrzeniaja sie wiralnie przez obliczenia. Jeden `nan` w aktualizacji gradientu czyni wage `nan`, co czyni kazdy nastepny output `nan`. Trening martwy w jednym kroku.

Jak pojawia sie `inf`:
- `exp()` duzej liczby dodatniej
- Dzielenie przez zero: `1.0 / 0.0`
- `float32` overflow w akumulacjach

Jak pojawia sie `nan`:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` liczby ujemnej
- `log()` liczby ujemnej
- Jakakolwiek arytmetyka z istniejacym `nan`

Wykrywanie:

```python
import math

math.isnan(x)       # True jesli x jest nan
math.isinf(x)       # True jesli x jest +inf lub -inf
math.isfinite(x)    # True jesli x nie jest ani nan ani inf
```

Strategie zapobiegania:

1. Zaciskaj wejscia do `exp()`: `exp(clamp(x, -80, 80))`
2. Dodaj epsilon do mianownikow: `x / (y + 1e-8)`
3. Dodaj epsilon wewnatrz `log()`: `log(x + 1e-8)`
4. Uzywaj stabilnych implementacji (log-sum-exp, stable softmax)
5. Gradient clipping, zeby zapobiec eksplozji wag
6. Sprawdzaj `nan`/`inf` po kazdym forward pass podczas debugowania

### Sprawdzanie Gradientu Numerycznego

Analityczne gradienty (z backpropagation) moga miec bugi. Sprawdzanie gradientu numerycznego weryfikuje je przez obliczanie gradientow ze skonczonymi roznicami.

Wzor scentrowanej rozницы:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

To jest dokladne O(h^2), znacznie lepsze niz roznica do przodu `(f(x+h) - f(x)) / h`, ktora jest tylko O(h).

Wybieranie h: za duze i przyblizenie jest zle. Za male i katastrofalna cancelation niszczy odpowiedz. `h = 1e-5` do `1e-7` jest typowe.

Sprawdzanie: oblicz bled wzgledny miedzy analitycznymi i numerycznymi gradientami.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Zasady kciukowe:
- relative_error < 1e-7: perfekcyjnie, gradient jest poprawny
- relative_error < 1e-5: akceptowalne, prawdopodobnie poprawne
- relative_error > 1e-3: cos jest nie tak
- relative_error > 1: gradient jest calkowicie zly

Zawsze sprawdzaj gradienty przy implementacji nowej warstwy lub funkcji loss. PyTorch dostarcza `torch.autograd.gradcheck()` do tego.

### Trening Mieszanej Precyzji

Nowoczesne GPU maja wyspecjalizowany sprzet (Tensor Cores), ktory oblicza mnozenie macierzy float16 2-8x szybciej niz float32. Trening mieszanej precyzji to wykorzystuje:

```
1. Utrzymuj kopię wag w float32 jako master
2. Forward pass w float16 (szybki)
3. Oblicz loss w float32 (zapobiega overflow)
4. Backward pass w float16 (szybki)
5. Skaluj gradienty do float32
6. Aktualizuj master wagi float32
```

Problem z czystym treningiem float16: gradienty sa czesto bardzo male (1e-8 lub mniejsze). Float16 underflowuje cokolwiek ponizej ~6e-8 do zera. Twoj model przestaje sie uczyc bo wszystkie aktualizacje gradientu sa zerowe.

Rozwiazanie to skalowanie loss:

```
1. Pomnoz loss przez duzy czynnik skalowania (np. 1024)
2. Backward pass oblicza gradienty (loss * 1024)
3. Wszystkie gradienty sa 1024x wieksze (wepchniete powyzej underflow float16)
4. Podziel gradienty przez 1024 przed aktualizacja wag
5. Efekt netto: ta sama aktualizacja, ale bez underflow
```

Dynamiczne skalowanie loss automatycznie dostosowuje czynnik skalowania. Zaczynaj od duzej wartosci (65536). Jesli gradienty overflowuja do `inf`, zmniejsz o polowe. Jesli N krokow przejdzie bez overflow, podwaj.

### bfloat16 kontra float16: Dlaczego bfloat16 Wygrywa dla Treningu

```
float16:   [1 znak] [5 wykladnik]  [10 mantysa]
bfloat16:  [1 znak] [8 wykladnik]  [7 mantysa]
```

float16 ma wieksza precyzje (10 bitow mantysy vs 7) ale ograniczony zakres (max ~65,504). bfloat16 ma mniejsza precyzje ale ten sam zakres co float32 (max ~3.4e38).

Dla trenowania sieci neuronowych:

- Aktywacje i logity regularnie przekraczaja 65,504 podczas skokow treningowych. float16 overflowuje; bfloat16 sobie radzi.
- Skalowanie loss jest wymagane z float16 ale zazwyczaj zbedne z bfloat16 bo jego zakres pokrywa spektrum wielkosci gradientow.
- bfloat16 to proste obciecie float32: upusc dolne 16 bitow mantysy. Konwersja jest trywialna i bezstratna w wykladniku.

float16 jest preferowany dla inferencji gdzie wartosci sa ograniczone i precyzja ma wieksze znaczenie. bfloat16 jest preferowany dla treningu gdzie zakres ma wieksze znaczenie. Dlatego TPU i nowoczesne GPU NVIDIA (A100, H100) maja natywna obsluge bfloat16.

### Gradient Clipping

Eksplodujace gradienty wystepuja gdy gradienty rosna wykladniczo przez wiele warstw (czeste w RNN, glebokich sieciach i transformerach). Pojedynczy duzy gradient moze skorumpowac wszystkie wagi w jednym kroku.

Dwa typy clipping:

**Clip by value:** zaciskaj kazdy element gradientu niezaleznie.

```
grad = clamp(grad, -max_val, max_val)
```

Proste ale moze zmienic kierunek wektora gradientu.

**Clip by norm:** skaluj caly wektor gradientu tak, zeby jego norma nie przekraczala progu.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Zachowuje kierunek gradientu. To jest to, co robi `torch.nn.utils.clip_grad_norm_()`. To jest standardowy wybor.

Typowe wartosci: `max_norm=1.0` dla transformerow, `max_norm=0.5` dla RL, `max_norm=5.0` dla prostszych sieci.

Gradient clipping to nie hack. To mechanizm bezpieczenstwa. Bez niego, pojedynczy outlier batch moze wyprodukowac gradient wystarczajaco duzy, zeby zniszczyc tygodnie treningu.

### Warstwy Normalizacji jako Stabilizatory Numeryczne

Batch normalization, layer normalization i RMS normalization sa zwykle prezentowane jako regularizatory, ktore pomagaja treningowi zbiegac. Sa takze stabilizatorami numerycznymi.

Bez normalizacji, aktywacje moga rosnac lub kurczyc sie wykladniczo przez warstwy:

```
Warstwa 1: wartosci w [0, 1]
Warstwa 5: wartosci w [0, 100]
Warstwa 10: wartosci w [0, 10,000]
Warstwa 50: wartosci w [0, inf]
```

Normalizacja przesuwa srodek i skaluje aktywacje przy kazdej warstwie:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (typowo 1e-5) zapobiega dzieleniu przez zero gdy wszystkie aktywacje sa identyczne. Nauczane parametry `gamma` i `beta` pozwalaja sieci przywrocic dowolna skale, jaka potrzebuje.

To utrzymuje wartosci w numerycznie bezpiecznym zakresie przez cala siec, zapobiegajac zarowno overflow w forward pass jak i eksplozji gradientu w backward pass.

### Typowe Bugi Numeryczne w ML

**Bug: Loss to NaN po kilku epokach.**
Przyczyna: logity zrosly zbyt duze, softmax overflowowal. Albo learning rate jest zbyt wysoki i wagi rozeszly sie.
Rozwiazanie: uzyj stable softmax (odejmowanie maksimum), zmniejsz learning rate, dodaj gradient clipping.

**Bug: Loss utknal na log(num_classes).**
Przyczyna: wyjscia modelu sa bliskie jednorodnym prawdopodobienstwom. Czesto znaczy, ze gradienty znikaja albo model w ogole sie nie uczy.
Rozwiazanie: sprawdz, ze etykiety danych sa poprawne, zweryfikuj funkcje loss, sprawdz martwe ReLU.

**Bug: Wartosc walidacji jest nizsza niz oczekiwano o 1-3%.**
Przyczyna: mieszana precyzja bez wlasciwego skalowania loss. Gradient underflow cicho zeruje male aktualizacje.
Rozwiazanie: wlacz dynamiczne skalowanie loss, albo przełącz na bfloat16.

**Bug: Normy gradientow sa 0.0 dla niektorych warstw.**
Przyczyna: martwe neurony ReLU (wszystkie wejscia ujemne), albo float16 underflow.
Rozwiazanie: uzyj LeakyReLU lub GELU, uzyj skalowania gradientu, sprawdz inicjalizacje wag.

**Bug: Model dziala na jednym GPU ale daje inne wyniki na innym.**
Przyczyna: nie-deterministyczna kolejnosc akumulacji floating point. Redukcje rownolegle GPU sumuja w roznych kolejnosciach na roznym sprzecie, a dodawanie floating point nie jest laczne.
Rozwiazanie: zaakceptuj male roznice (1e-6), albo ustaw `torch.use_deterministic_algorithms(True)` i zaakceptuj karne w postaci szybkosci.

**Bug: `exp()` zwraca `inf` w obliczeniu loss.**
Przyczyna: surowe logity przekazane do `exp()` bez triku z odejmowaniem maksimum.
Rozwiazanie: uzyj `torch.nn.functional.log_softmax()`, ktory implementuje log-sum-exp wewnetrznie.

**Bug: Trening rozchodzi sie po przejsciu z float32 na float16.**
Przyczyna: float16 nie moze reprezentowac wielkosci gradientow ponizej 6e-8 lub aktywacji powyzej 65,504.
Rozwiazanie: uzyj mieszanej precyzji ze skalowaniem loss (AMP), albo uzyj bfloat16 zamiast tego.

## Zbuduj to

### Krok 1: Zademonstruj limity precyzji floating point

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Krok 2: Implementuj naiwny vs stabilny softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Krok 3: Implementuj stabilny log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Krok 4: Implementuj stabilny cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Krok 5: Sprawdzanie gradientu

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Uzyj tego

### Symulacja mieszanej precyzji

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Wykrywanie NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Zobacz `code/numerical.py` dla kompletnych implementacji ze wszystkimi przypadkami brzegowymi zademonstrowanymi.

## Dostarcz to

Ta lekcja produkuje:
- `code/numerical.py` ze stabilnym softmax, log-sum-exp, cross-entropy, sprawdzaniem gradientu i symulacja mieszanej precyzji
- `outputs/prompt-numerical-debugger.md` do diagnozowania problemow NaN/Inf i numerycznych w treningu

Te stabilne implementacje powracaja w Fazie 3 przy budowaniu treningowego loopa i w Fazie 4 przy implementacji mechanizmow attention.

## Cwiczenia

1. **Katastrofalna cancelation.** Oblicz wariancje [1000000.0, 1000001.0, 1000002.0] uzywajac naiwnego wzoru `E[x^2] - E[x]^2` w float32. Nastepnie oblicz to uzywajac online algorytmu Welforda. Porownaj bledy z prawdziwa wariancja (0.6667).

2. **Poszukiwanie precyzji.** Znajdz najmniejsza dodatnia wartosc float32 `x` taka, ze `1.0 + x == 1.0` w Pythonie. To jest machine epsilon. Zweryfikuj, ze pasuje do `numpy.finfo(numpy.float32).eps`.

3. **Przypadki brzegowe log-sum-exp.** Przetestuj swoja funkcje `logsumexp_stable` z: (a) wszystkimi rownymi wartosciami, (b) jedna wartoscia duzo wieksza niz reszta, (c) wszystkimi bardzo ujemnymi wartosciami (-1000). Zweryfikuj, ze daje poprawne wyniki tam, gdzie wersja naiwna fails.

4. **Sprawdzanie gradientu warstwy sieci neuronowej.** Implementuj pojedyncza warstwe liniowa `y = Wx + b` i jej analityczny backward pass. Uzyj `numerical_gradient` do zweryfikowania poprawnosci dla macierzy wag 3x2.

5. **Eksperyment ze skalowaniem loss.** Symuluj trening z float16: stworz losowe gradienty w zakresie [1e-9, 1e-3], przekonwertuj do float16, i zmierz jaka czesc staje sie zerowa. Nastepnie zastosuj skalowanie loss (pomnoz przez 1024), przekonwertuj do float16, skaluj z powrotem, i zmierz znowu czesc zerowa.

## Kluczowe pojecia

| Pojecie | Co ludzie mowia | Co to faktycznie oznacza |
|---------|-----------------|----------------------|
| IEEE 754 | "Standard floatow" | Miedzynarodowy standard definiujacy binarne formaty floating point, reguly zaokraglania i wartosci specjalne (inf, nan). Kazdy nowoczesny CPU i GPU go implementuje. |
| Machine epsilon | "Limit precyzji" | Najmniejsza wartosc e taka, ze 1.0 + e != 1.0 w danym formacie float. Dla float32, to okolo 1.19e-7. |
| Katastrofalna cancelation | "Utrata precyzji z odejmowania" | Gdy odejmujemy niemal rowne liczby floating point, istotne cyfry sie znosza i szum zaokraglenia dominuje wynik. |
| Overflow | "Liczba za duza" | Wynik przekracza maksymalna reprezentowalna wartosc i staje sie inf. exp(89) overflowuje float32. |
| Underflow | "Liczba za mala" | Wynik jest blizszy zero niz najmniejsza reprezentowalna dodatnia liczba i staje sie 0.0. exp(-104) underflowuje float32. |
| Trik log-sum-exp | "Odejmij maksimum najpierw" | Obliczanie log(sum(exp(x))) przez wyciecie exp(max(x)) zeby zapobiec overflow i underflow. Uzywany w softmax, cross-entropy i matematyce log-prawdopodobienstw. |
| Stable softmax | "Softmax ktory nie eksploduje" | Odejmowanie max(logits) przed potegowaniem. Wynik numerycznie identyczny, zadnego overflow mozliwe. |
| Sprawdzanie gradientu | "Zweryfikuj swoj backprop" | Porownywanie analitycznych gradientow z backpropagation z numerycznymi gradientami ze skonczonych roznic, zeby zlapac bugi implementacji. |
| Mieszana precyzja | "Float16 forward, float32 backward" | Uzywanie floatow nizszej precyzji dla operacji krytycznych czasowo i wyzszej precyzji dla operacji numerycznie wrazliwych. Typowe przyspieszenie to 2-3x. |
| Skalowanie loss | "Zapobiegaj underflow gradientu" | Mnozenie loss przez duza stala przed backprop, zeby gradienty pozostaly w reprezentowalnym zakresie float16, potem dzielenie przez te sama stala przed aktualizacja wag. |
| bfloat16 | "Brain floating point" | Format Google 16-bitowy z 8 bitami wykladnika (ten sam zakres co float32) i 7 bitami mantysy (mniejsza precyzja niz float16). Preferowany dla treningu. |
| Gradient clipping | "Ogranicz norme gradientu" | Skalowanie wektora gradientu tak, zeby jego norma nie przekraczala progu. Zapobiega eksplodujacym gradientom niszczyc wagi. |
| NaN | "Not a Number" | Specjalna wartosc float z niezdefiniowanych operacji (0/0, inf-inf, sqrt(-1)). Rozprzestrzenia sie przez wszelka nastepna arytmetyke. |
| Inf | "Nieskonczonosc" | Specjalna wartosc float z overflow lub dzielenia przez zero. Moze laczyc sie produkujac NaN (inf - inf, inf * 0). |
| Gradient numeryczny | "Brute force derivative" | Przyblizanie pochodnej przez ewaluacje f(x+h) i f(x-h) i dzielenie przez 2h. Wolne ale niezawodne dla weryfikacji. |

## Dalsza lektura

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- definitywne odniesienie, geste ale kompletne
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- artykul NVIDIA, ktory wprowadzil skalowanie loss dla treningu float16
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- praktyczny przewodnik po mieszanej precyzji w PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- dlaczego Google wybralo ten format dla TPU
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- algorytm redukujacy blad zaokraglenia w sumach floating point