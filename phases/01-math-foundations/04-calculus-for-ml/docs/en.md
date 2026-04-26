# Rachunek różniczkowy dla uczenia maszynowego

> Pochodne mówią ci, w którą stronę jest w dół. To wszystko, czego sieć neuronowa potrzebuje, żeby się uczyć.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-03
**Time:** ~60 minutes

## Cele uczenia się

- Obliczaj numeryczne i analityczne pochodne dla typowych funkcji ML (x^2, sigmoid, cross-entropy)
- Implementuj gradient descent od zera, żeby zminimalizować funkcję straty w 1D i 2D
- Wyprowadzaj gradient modelu regresji liniowej i trenuj go przez ręczne aktualizacje wag
- Wyjaśnij macierz Hessian, aproksymacje Taylora i ich związek z metodami optymalizacji

## Problem

Masz sieć neuronową z milionami wag. Każda waga to pokrętło. Musisz wymyślić, w którą stronę obrócić każde pokrętło, żeby model był trochę mniej błędny. Rachunek różniczkowy daje ci ten kierunek.

bez rachunku różniczkowego, trenowanie sieci neuronowej oznaczałoby próbowanie losowych zmian i nadzieję na najlepsze. Z pochodnymi wiesz dokładnie, jak każda waga wpływa na błąd. Obracasz każde pokrętło we właściwym kierunku, za każdym razem.

## Koncepcja

### Czym jest pochodna?

Pochodna mierzy tempo zmian. Dla funkcji y = f(x), pochodna f'(x) mówi ci: jeśli przesuniesz x o odrobinę, jak bardzo y się zmieni?

Geometrycznie, pochodna to nachylenie linii tangens w punkcie.

**f(x) = x^2:**

| x | f(x) | f'(x) (nachylenie) |
|---|------|---------------|
| 0 | 0    | 0 (płaskie, na dole) |
| 1 | 1    | 2 |
| 2 | 4    | 4 (nachylenie linii tangens w tym punkcie) |
| 3 | 9    | 6 |

W x=2, nachylenie wynosi 4. Jeśli przesuniesz x odrobinę w prawo, y wzrasta o około 4 razy tę wartość. W x=0, nachylenie wynosi 0. Jesteś na dnie misy.

Formalna definicja:

```
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

W kodzie pomijasz granicę i po prostu używasz bardzo małego h. To pochodna numeryczna.

### Pochodne cząstkowe: jedna zmienna na raz

Rzeczywiste funkcje mają wiele wejść. Funkcja straty sieci neuronowej zależy od tysięcy wag. Pochodna cząstkowa trzyma wszystkie zmienne stałe oprócz jednej, a następnie bierze pochodną względem tej jednej.

```
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     (traktuj y jako stałą)
df/dy = 3x + 2y     (traktuj x jako stałą)
```

Każda pochodna cząstkowa odpowiada na pytanie: jeśli przesunę tylko tę jedną wagę, jak zmieni się strata?

### Gradient: wektor wszystkich pochodnych cząstkowych

Gradient zbiera każdą pochodną cząstkową w jeden wektor. Dla funkcji f(x, y, z), gradient to:

```
grad f = [ df/dx, df/dy, df/dz ]
```

Gradient wskazuje w kierunku największego wzrostu. Aby zminimalizować funkcję, idź w przeciwnym kierunku.

**Wykres konturowy f(x,y) = x^2 + y^2:**

Funkcja tworzy kształt misy z okręgami jako liniami konturowymi. Minimum находится в (0, 0).

| Punkt | grad f | -grad f (kierunek spadku) |
|-------|--------|----------------------------|
| (1, 1) | [2, 2] (wskazuje pod górę, od minimum) | [-2, -2] (wskazuje w dół, do minimum) |
| (0, 0) | [0, 0] (płaskie, w minimum) | [0, 0] |

To jest gradient descent na obrazie. Oblicz gradient, zneguj go, zrób krok.

### Związek z optymalizacją

Trenowanie sieci neuronowej to optymalizacja. Masz funkcję straty L(w1, w2, ..., wn), która mierzy, jak bardzo model się myli. Chcesz ją zminimalizować.

```
Reguła aktualizacji gradient descent:

  w_new = w_old - learning_rate * dL/dw

Dla każdej wagi:
  1. Oblicz pochodną cząstkową straty względem tej wagi
  2. Odejmij jej małą wielokrotność od wagi
  3. Powtórz
```

Learning rate kontroluje rozmiar kroku. Za duży i przeskakujesz. Za mały i czołgasz się.

**Powierzchnia strat (przekrój 1D):**

Funkcja straty L(w) tworzy krzywą z szczytami i dolinami, gdy waga w się zmienia.

| Cecha | Opis |
|---------|-------------|
| Globalne minimum | Najniższy punkt na całej krzywej -- najlepsze rozwiązanie |
| Lokalne minimum | Dolina niższa od sąsiadów, ale nie najniższa ogółem |
| Nachylenie | Gradient descent podąża za nachyleniem w dół od dowolnego punktu startu |

Gradient descent podąża za nachyleniem w dół. Może utknąć w lokalnych minimach, ale w przestrzeniach wysokowymiarowych (miliony wag) to rzadko praktyczny problem.

### Pochodne numeryczne vs analityczne

Istnieją dwa sposoby obliczania pochodnej.

Analityczne: zastosuj reguły rachunku ręcznie. Dla f(x) = x^2, pochodna to f'(x) = 2x. Dokładne. Szybkie.

Numeryczne: przybliż używając definicji. Oblicz f(x+h) i f(x-h) dla bardzo małego h, potem użyj różnicy.

```
Numeryczna (różnica centralna):

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 dobrze działa w praktyce
```

Pochodne numeryczne są wolniejsze, ale działają dla dowolnej funkcji. Pochodne analityczne są szybkie, ale wymagają wyprowadzenia wzoru. Frameworki sieci neuronowych używają trzeciego podejścia: automatycznej dyferencjacji, która oblicza dokładne pochodne mechanicznie. Zobaczysz to w Fazie 3.

### Pochodne ręcznie dla prostych funkcji

To są pochodne, które będziesz widywać w kółko w ML.

```
Funkcja        Pochodna       Używana w
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Funkcje straty (MSE)
f(x) = wx + b  f'(w) = x        Warstwa liniowa (gradient względem wagi)
                f'(b) = 1        Warstwa liniowa (gradient względem biasu)
                f'(x) = w        Warstwa liniowa (gradient względem wejścia)
f(x) = e^x     f'(x) = e^x     Softmax, attention
f(x) = ln(x)   f'(x) = 1/x     Funkcja straty cross-entropy
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Aktywacja sigmoid
```

Dla f(x) = x^2:

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   znaczenie
  -2    4      -4      nachylenie w lewo (malejące)
  -1    1      -2      nachylenie w lewo (malejące)
   0    0       0      płaskie (minimum!)
   1    1       2      nachylenie w prawo (rosnące)
   2    4       4      nachylenie w prawo (rosnące)
```

Dla f(w) = wx + b z x=3, b=1:

```
f(w) = 3w + 1    f'(w) = 3

Pochodna względem w to just x.
Jeśli x jest duże, mała zmiana w powoduje dużą zmianę w wyjściu.
```

### Reguła łańcuchowa

Gdy funkcje są składane, reguła łańcuchowa mówi, jak je różniczkować.

```
Jeśli y = f(g(x)), to dy/dx = f'(g(x)) * g'(x)

Przykład: y = (3x + 1)^2
  outer: f(u) = u^2       f'(u) = 2u
  inner: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

Sieci neuronowe to łańcuchy funkcji: wejście -> liniowa -> aktywacja -> liniowa -> aktywacja -> strata. Backpropagation to reguła łańcuchowa stosowana wielokrotnie od wyjścia do wejścia. To cały algorytm.

### Macierz Hessian

Gradient mówi ci nachylenie. Hessian mówi ci krzywiznę.

Hessian to macierz drugich pochodnych cząstkowych. Dla funkcji f(x1, x2, ..., xn), element (i, j) Hessian to:

```
H[i][j] = d^2f / (dx_i * dx_j)
```

Dla funkcji dwóch zmiennych f(x, y):

```
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```

**Co Hessian mówi ci w punkcie krytycznym (gdzie gradient = 0):**

| Właściwość Hessian | Znaczenie | Przykładowa powierzchnia |
|-----------------|---------|-----------------|
| Określony dodatnio (wszystkie wartości własne > 0) | Lokalne minimum | Misa w górę |
| Określony ujemnie (wszystkie wartości własne < 0) | Lokalne maksimum | Misa w dół |
| Nieokreślony (mieszane wartości własne) | Punkt siodłowy | Kształt siodła |

**Przykład:** f(x, y) = x^2 - y^2 (funkcja siodłowa)

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Wartości własne: 2 i -2 (jedna dodatnia, jedna ujemna)
--> Punkt siodłowy w (0, 0)
```

Porównaj z f(x, y) = x^2 + y^2 (misa):

```
H = | 2  0 |
    | 0  2 |

Wartości własne: 2 i 2 (obie dodatnie)
--> Lokalne minimum w (0, 0)
```

**Dlaczego Hessian ma znaczenie w ML:**

Metoda Newtona używa Hessian do robienia lepszych kroków optymalizacji niż gradient descent. Zamiast just following the slope, uwzględnia krzywiznę:

```
Aktualizacja Newtona:    w_new = w_old - H^(-1) * gradient
Gradient descent:        w_new = w_old - lr * gradient
```

Metoda Newtona zbiega szybciej, bo Hessian "przeskaluje" gradient -- strome kierunki dostają mniejsze kroki, płaskie kierunki dostają większe.

Złowrogość: dla sieci neuronowej z N parametrami, Hessian ma N x N. Model z 1 milionem parametrów potrzebowałby macierzy z 1 bilionem wpisów. Dlatego używamy aproksymacji.

| Metoda | Co używa | Koszt | Zbieżność |
|--------|-------------|------|-------------|
| Gradient descent | Tylko pierwsze pochodne | O(N) na krok | Wolna (liniowa) |
| Metoda Newtona | Pełny Hessian | O(N^3) na krok | Szybka (kwadratowa) |
| L-BFGS | Aproksymacja Hessian z historii gradientów | O(N) na krok | Średnia (superliniowa) |
| Adam | Adaptacyjne stopy per-parametr (aproksymacja diagonalna Hessian) | O(N) na krok | Średnia |
| Natural gradient | Macierz informacji Fishera (statystyczny Hessian) | O(N^2) na krok | Szybka |

W praktyce Adam jest domyślnym optimizerem dla deep learning. Aproksymuje informacje drugiego rzędu tanio, śledząc bieżącą średnią i wariancję gradientów per parametr.

### Aproksymacja Taylora

Każda gładka funkcja może być przybliżona lokalnie przez wielomian:

```
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

Im więcej wyrazów włączysz, tym lepsze przybliżenie -- ale tylko blisko punktu x.

**Dlaczego szeregi Taylora mają znaczenie dla ML:**

- **Taylor pierwszego rzędu = gradient descent.** Gdy używasz f(x + h) ~ f(x) + f'(x)*h, robisz liniowe przybliżenie. Gradient descent minimalizuje ten liniowy model, żeby wybrać h = -lr * f'(x).

- **Taylor drugiego rzędu = metoda Newtona.** Używając f(x + h) ~ f(x) + f'(x)*h + (1/2)*f''(x)*h^2, dostajesz model kwadratowy. Minimalizacja daje h = -f'(x)/f''(x) -- krok Newtona.

- **Projektowanie funkcji straty.** MSE i cross-entropy są gładkie, co oznacza, że ich rozwinięcia Taylora są dobrze zachowane. To nie jest przypadek. Gładkie straty czynią optymalizację przewidywalną.

```
Rząd aproksymacji    Co przechwytuje    Metoda optymalizacji
-------------------    -----------------   -------------------
0th order (stała)     Just the value      Random search
1st order (liniowa)   Nachylenie          Gradient descent
2nd order (kwadratowa) Krzywizna         Metoda Newtona
Wyższe rzędy          Drobniejsza struktura     Rzadko używane w ML
```

Kluczowy wgląd: cała optymalizacja oparta na gradientach to w istocie przybliżanie funkcji straty lokalnie i krok do minimum tego przybliżenia.

### Całki w ML

Pochodne mówią ci o tempach zmian. Całki obliczają akumulacje -- pole pod krzywą.

W ML rzadko obliczasz całki ręcznie, ale koncepcja jest wszędzie:

**Prawdopodobieństwo.** Dla ciągłej zmiennej losowej z gęstością p(x):
```
P(a < X < b) = całka od a do b z p(x) dx
```
Pole pod krzywą gęstości prawdopodobieństwa między a i b to prawdopodobieństwo wylądowania w tym zakresie.

**Wartość oczekiwana.** Średni wynik ważony prawdopodobieństwem:
```
E[f(X)] = całka z f(x) * p(x) dx
```
Oczekiwana strata nad dystrybucją danych to całka. Trenowanie minimalizuje empiryczną aproksymację tego.

**Dywergencja KL.** Mierzy, jak bardzo dwie dystrybucje się różnią:
```
KL(p || q) = całka z p(x) * log(p(x) / q(x)) dx
```
Używana w VAEs, destylacji wiedzy i wnioskowaniu Bayesa.

**Stałe normalizacyjne.** We wnioskowaniu Bayesa:
```
p(w | data) = p(data | w) * p(w) / całka z p(data | w) * p(w) dw
```
Mianownik to całka po wszystkich możliwych wartościach parametrów. Jest often intractable, dlatego używamy aproksymacji jak MCMC i wariational inference.

| Pojęcie całki | Gdzie pojawia się w ML |
|-----------------|----------------------|
| Pole pod krzywą | Prawdopodobieństwo z funkcji gęstości |
| Wartość oczekiwana | Funkcje straty, minimalizacja ryzyka |
| Dywergencja KL | VAEs, optymalizacja polityki, destylacja |
| Normalizacja | Posteriory Bayesa, mianownik softmax |
| Likelihood brzegowy | Porównanie modeli, evidence lower bound (ELBO) |

### Reguła łańcuchowa wielu zmiennych w grafie obliczeniowym

Reguła łańcuchowa nie stosuje się tylko do funkcji skalarnych na prostej. W sieci neuronowej zmienne rozgałęziają się i łączą. Oto jak pochodne przepływają przez prosty forward pass:

```mermaid
graph LR
    x["x (input)"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"loss fn"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

Backward pass oblicza gradienty od prawej do lewej:

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

Każda strzałka mnoży przez lokalną pochodną. Gradient dla dowolnego parametru to iloczyn wszystkich lokalnych pochodnych wzdłuż ścieżki od straty do tego parametru. Gdy ścieżki się rozgałęziają i łączą, sumujesz wkłady (wielowariantowa reguła łańcuchowa).

To wszystkim jest backpropagation: reguła łańcuchowa systematycznie stosowana przez graf obliczeniowy, od wyjścia do wejść.

### Macierz Jacobian

Gdy funkcja mapuje wektor do wektora (jak warstwa sieci neuronowej), jej pochodna to macierz. Jacobian zawiera każdą pochodną cząstkową każdego wyjścia względem każdego wejścia.

Dla f: R^n -> R^m, Jacobian J to macierz m x n:

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

Nie będziesz obliczać Jacobianów ręcznie dla sieci neuronowych. PyTorch to obsługuje. Ale wiedza, że istnieje, pomaga zrozumieć kształty w backpropagation: jeśli warstwa mapuje R^n do R^m, jej Jacobian to m x n. Gradient przepływa wstecz przez transpozycję tej macierzy.

### Dlaczego to ma znaczenie dla sieci neuronowych

Każda waga w sieci neuronowej dostaje gradient. Gradient mówi ci, jak dostosować tę wagę, żeby zredukować stratę.

```mermaid
graph LR
    subgraph Forward["Forward Pass"]
        I["input"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["loss"]
    end
```

```mermaid
graph RL
    subgraph Backward["Backward Pass"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

Każda aktualizacja wag:
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

Forward pass oblicza predykcję i stratę. Backward pass oblicza gradient straty względem każdej wagi. Potem każda waga robi mały krok w dół. Powtórz dla milionów kroków. To deep learning.

## Buduj to

### Krok 1: Pochodna numeryczna od zera

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

Pochodna numeryczna matches the analytical one to many decimal places.

### Krok 2: Pochodne cząstkowe i gradienty

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### Krok 3: Gradient descent do znalezienia minimum f(x) = x^2

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

Starting at x=5, each step moves closer to x=0 (the minimum).

### Krok 4: Gradient descent na funkcji 2D

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### Krok 5: Porównanie pochodnych numerycznych i analitycznych

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### Krok 6: Obliczanie Hessian numerycznie

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- mixed signs
print(f"Bowl Hessian:   {H_bowl}")    # [[2, 0], [0, 2]]  -- both positive
```

Hessian funkcji siodłowej ma wartości własne 2 i -2 (mieszane znaki, potwierdzające punkt siodłowy). Misa ma wartości własne 2 i 2 (obie dodatnie, potwierdzające minimum).

### Krok 7: Aproksymacja Taylora w akcji

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

Near x0=0, sin(x) ~ x (pierwszy rząd Taylora). Aproksymacja jest doskonała dla małego h, ale break down for large h. Dlatego gradient descent works best with small learning rates -- każdy krok zakłada, że liniowe przybliżenie jest dokładne.

### Krok 8: Dlaczego to ma znaczenie dla sieci neuronowej

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
```

Every gradient-based training loop follows this pattern: predict, compute loss, compute gradients, update weights.

## Użyj tego

With NumPy, the same operations are faster and more concise:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

Właśnie zbudowałeś gradient descent od zera. PyTorch automatyzuje obliczanie gradientów, ale pętla aktualizacji jest identyczna.

## Ćwiczenia

1. Zaimplementuj `numerical_second_derivative(f, x)` używając `numerical_derivative` called twice. Zweryfikuj, że druga pochodna x^3 przy x=2 wynosi 12.
2. Użyj gradient descent do znalezienia minimum f(x, y) = (x - 3)^2 + (y + 1)^2. Start from (0, 0). Odpowiedź powinna zbiegać do (3, -1).
3. Dodaj momentum do pętli gradient descent: utrzymuj wektor prędkości, który akumuluje przeszłe gradienty. Porównaj szybkość zbieżności z i bez momentum na f(x) = x^4 - 3x^2.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|----------------|----------------------|
| Pochodna | "Nachylenie" | Tempo zmiany funkcji w punkcie. Mówi, jak bardzo wyjście zmienia się na jednostkę zmiany wejścia. |
| Pochodna cząstkowa | "Pochodna jednej zmiennej" | Pochodna względem jednej zmiennej, gdy wszystkie inne są trzymane stałe. |
| Gradient | "Kierunek największego wzrostu" | Wektor wszystkich pochodnych cząstkowych. Wskazuje w kierunku najszybszego wzrostu funkcji. |
| Gradient descent | "Idź w dół" | Odejmij gradient (razy learning rate) od parametrów, żeby zredukować stratę. Rdzeń trenowania sieci neuronowych. |
| Learning rate | "Rozmiar kroku" | Skalar kontrolujący, jak duży każdy krok gradient descent. Za duży: diverguje. Za mały: zbiega wolno. |
| Reguła łańcuchowa | "Pomnóż pochodne" | Reguła różniczkowania złożonych funkcji: df/dx = df/dg * dg/dx. Matematyczna podstawa backpropagation. |
| Jacobian | "Macierz pochodnych" | Gdy funkcja mapuje wektory do wektorów, Jacobian to macierz wszystkich pochodnych cząstkowych wyjść względem wejść. |
| Pochodna numeryczna | "Skończone różnice" | Aproksymacja pochodnej przez ewaluację funkcji w dwóch bliskich punktach i obliczenie nachylenia między nimi. |
| Backpropagation | "Reverse-mode autodiff" | Obliczanie gradientów warstwa po warstwie od wyjścia do wejścia używając reguły łańcuchowej. Jak sieci neuronowe się uczą. |
| Hessian | "Macierz drugich pochodnych" | Macierz wszystkich drugich pochodnych cząstkowych. Opisuje krzywiznę funkcji. Dodatnio określony Hessian w punkcie krytycznym oznacza lokalne minimum. |
| Szereg Taylora | "Aproksymacja wielomianowa" | Aproksymacja funkcji w pobliżu punktu używając jej pochodnych: f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ... Podstawa zrozumienia, dlaczego gradient descent i metoda Newtona działają. |
| Całka | "Pole pod krzywą" | Akumulacja ilości przez zakres. W ML całki definiują prawdopodobieństwa, wartości oczekiwane i dywergencję KL. |

## Dalsze czytanie

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) - wizualna intuicja dla pochodnych, całek i reguły łańcuchowej
- [Stanford CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) - jak gradienty przepływają przez warstwy sieci neuronowej