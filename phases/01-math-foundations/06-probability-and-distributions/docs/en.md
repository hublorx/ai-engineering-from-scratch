# Prawdopodobieństwo i rozkłady

> Prawdopodobieństwo to język, którego AI używa do wyrażania niepewności.

**Typ:** Ucz się
**Język:** Python
**Wymagania wstępne:** Faza 1, Lekcje 01-04
**Czas:** ~75 minut

## Cele uczenia się

- Implementuj PMF i PDF od podstaw dla rozkładów Bernoulliego, kategorialnego, Poissona, jednostajnego i normalnego
- Oblicz wartość oczekiwaną, wariancję i wykorzystaj Centralne Twierdzenie Graniczne, aby wyjaśnić, dlaczego rozkłady Gaussowskie dominują
- Zbuduj funkcje softmax i log-softmax z trickiem stabilności numerycznej (odejmowanie max logita)
- Oblicz stratę entropii krzyżowej z logitów i połącz ją z ujemnym log-wiarygodnością

## Problem

Klasyfikator zwraca `[0.03, 0.91, 0.06]`. Model językowy wybiera następne słowo spośród 50 000 kandydatów. Model dyfuzyjny generuje obrazy przez próbkowanie z nauczonych rozkładów. Wszystkie te przypadki ilustrują działanie prawdopodobieństwa.

Każda predykcja modelu to rozkład prawdopodobieństwa. Każda funkcja straty mierzy, jak daleko przewidziany rozkład jest od prawdziwego. Każdy krok trenowania dostosowuje parametry, aby jeden rozkład bardziej przypominał drugi. Bez znajomości prawdopodobieństwa nie można zrozumieć żadnego artykułu ML, debugować modelu ani wyjaśnić, dlaczego strata trenowania wynosi NaN.

## Koncepcja

### Zdarzenia, Przestrzenie próbek i Prawdopodobieństwo

Przestrzeń próbek S to zbiór wszystkich możliwych wyników. Zdarzenie to podzbiór przestrzeni próbek. Prawdopodobieństwo przypisuje zdarzeniom liczby z przedziału od 0 do 1.

```
Rzut monetą:
  S = {O, R}
  P(O) = 0.5,  P(R) = 0.5

Rzut pojedynczą kostką:
  S = {1, 2, 3, 4, 5, 6}
  P(parzyste) = P({2, 4, 6}) = 3/6 = 0.5
```

Trzy aksjomaty definiują całą teorię prawdopodobieństwa:
1. P(A) >= 0 dla dowolnego zdarzenia A
2. P(S) = 1 (coś zawsze się dzieje)
3. P(A lub B) = P(A) + P(B) gdy A i B nie mogą wystąpić jednocześnie

Wszystko inne (twierdzenie Bayesa, wartości oczekiwane, rozkłady) wynika z tych trzech reguł.

### Prawdopodobieństwo warunkowe i niezależność

P(A|B) to prawdopodobieństwo zdarzenia A pod warunkiem, że zaszło B.

```
P(A|B) = P(A i B) / P(B)

Przykład: talia kart
  P(Król | Karta z obrazem) = P(Król i Karta z obrazem) / P(Karta z obrazem)
                            = (4/52) / (12/52)
                            = 4/12 = 1/3
```

Dwa zdarzenia są niezależne, gdy wiedza o jednym nie daje żadnej informacji o drugim:

```
Niezależne:   P(A|B) = P(A)
Równoważne: P(A i B) = P(A) * P(B)
```

Rzuty monetą są niezależne. Losowanie kart bez zwracania nie jest.

### Funkcje masy prawdopodobieństwa a funkcje gęstości prawdopodobieństwa

Dyskretne zmienne losowe mają funkcję masy prawdopodobieństwa (PMF). Każdy wynik ma określone prawdopodobieństwo, które można odczytać bezpośrednio.

```
PMF: P(X = k)

Sprawiedliwa kostka:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Suma wszystkich prawdopodobieństw = 1
```

Ciągłe zmienne losowe mają funkcję gęstości prawdopodobieństwa (PDF). Gęstość w pojedynczym punkcie nie jest prawdopodobieństwem. Prawdopodobieństwo pochodzi z całkowania gęstości przedziału.

```
PDF: f(x)

P(a <= X <= b) = całka z f(x) od a do b

f(x) może być większe niż 1 (gęstość, nie prawdopodobieństwo)
całka od -inf do +inf z f(x) dx = 1
```

Ta różnica ma znaczenie w ML. Wyniki klasyfikacji to PMF (dyskretne wybory). Przestrzenie ukryte VAE używają PDF (ciągłe).

### Powszechne rozkłady

**Bernoulli:** jedno doświadczenie, dwa wyniki. Modeluje binarną klasyfikację.

```
P(X = 1) = p
P(X = 0) = 1 - p
Średnia = p,  Wariancja = p(1-p)
```

**Categorical:** jedno doświadczenie, k wyników. Modeluje wieloklasową klasyfikację (wyjście softmax).

```
P(X = i) = p_i,  gdzie suma p_i = 1
Przykład: P(kot) = 0.7,  P(pies) = 0.2,  P(ptak) = 0.1
```

**Jednostajny:** wszystkie wyniki równie prawdopodobne. Używany do losowej inicjalizacji.

```
Dyskretny: P(X = k) = 1/n dla k w {1, ..., n}
Ciągły: f(x) = 1/(b-a) dla x w [a, b]
```

**Normalny (Gaussowski):** krzywa dzwonowa. Sparametryzowany przez średnią (mu) i wariancję (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Rozkład standardowy: mu = 0, sigma = 1
  68% danych w odległości 1 sigma
  95% w odległości 2 sigma
  99.7% w odległości 3 sigma
```

**Poisson:** zliczenia rzadkich zdarzeń w ustalonym interwale. Modeluje współczynniki zdarzeń.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Średnia = lambda,  Wariancja = lambda
```

### Wartość oczekiwana i wariancja

Wartość oczekiwana to średnia ważona wyników.

```
Dyskretna:   E[X] = suma x_i * P(X = x_i)
Ciągła: E[X] = całka z x * f(x) dx
```

Wariancja mierzy rozproszenie wokół średniej.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Odchylenie standardowe = sqrt(Var(X))
```

W ML wartość oczekiwana pojawia się jako funkcja straty (średnia strata nad rozkładem danych). Wariancja informuje o stabilności modelu. Wysoka wariancja gradientów oznacza zaszumione trenowanie.

### Rozkłady łączne i brzegowe

Rozkład łączny P(X, Y) opisuje dwie zmienne losowe razem.

Przykład łącznego PMF (X = pogoda, Y = parasol):

| | Y=0 (bez parasola) | Y=1 (parasol) | Rozkład brzegowy P(X) |
|---|---|---|---|
| X=0 (słonecznie) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (deszcz) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Rozkład brzegowy P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Rozkład brzegowy sumuje po drugiej zmiennej:

```
P(X = x) = suma po wszystkich y z P(X = x, Y = y)
```

Sumy wierszy i kolumn w tabeli powyżej to rozkłady brzegowe.

### Dlaczego rozkład normalny pojawia się wszędzie

Centralne twierdzenie graniczne: suma (lub średnia) wielu niezależnych zmiennych losowych dąży do rozkładu normalnego, niezależnie od wyjściowego rozkładu.

```
Rzuć 1 kostkę: rozkład jednostajny (płaski)
Średnia z 2 kostek: trójkątny (z wierzchołkiem)
Średnia z 30 kostek: prawie doskonały dzwon

To działa dla DOWOLNEGO wyjściowego rozkładu.
```

Dlatego:
- Błędy pomiarowe są w przybliżeniu normalne (wiele małych niezależnych źródeł)
- Inicjalizacja wag w sieciach neuronowych używa rozkładów normalnych
- Szum gradientów w SGD jest w przybliżeniu normalny (suma wielu gradientów próbek)
- Rozkład normalny to rozkład o maksymalnej entropii dla danej średniej i wariancji

### Logarytmy prawdopodobieństw

Surowe prawdopodobieństwa powodują problemy numeryczne. Mnożenie wielu małych prawdopodobieństw szybko prowadzi do underflow.

```
P(zdanie) = P(słowo1) * P(słowo2) * ... * P(słowo_n)
           = 0.01 * 0.003 * 0.02 * ...
           -> 0.0 (underflow po ~30 wyrazach)
```

Logarytmy prawdopodobieństw to rozwiązanie. Mnożenie staje się dodawaniem.

```
log P(zdanie) = log P(słowo1) + log P(słowo2) + ... + log P(słowo_n)
              = -4.6 + -5.8 + -3.9 + ...
              -> skończona liczba (brak underflow)
```

Zasady:
- log(a * b) = log(a) + log(b)
- Logarytmy prawdopodobieństw są zawsze <= 0 (ponieważ 0 < P <= 1)
- Bardziej ujemne = mniej prawdopodobne
- Strata entropii krzyżowej to ujemne log-prawdopodobieństwo poprawnej klasy

### Softmax jako rozkład prawdopodobieństwa

Sieci neuronowe zwracają surowe wyniki (logity). Softmax przekształca je w prawidłowy rozkład prawdopodobieństwa.

```
softmax(z_i) = exp(z_i) / suma(exp(z_j) dla wszystkich j)

Właściwości:
  - Wszystkie wyjścia są w (0, 1)
  - Wszystkie wyjścia sumują się do 1
  - Zachowuje względne uporządkowanie wejść
  - exp() wzmacnia różnice między logitami
```

Trick z softmax: odejmij max logit przed wykładniczym przekształceniem, aby uniknąć overflow.

```
z = [100, 101, 102]
exp(102) = overflow

z_przesuniete = z - max(z) = [-2, -1, 0]
exp(0) = 1  (bezpieczne)

Ten sam wynik, bez overflow.
```

Log-softmax łączy softmax i log dla stabilności numerycznej. PyTorch używa tego wewnętrznie dla straty entropii krzyżowej.

### Próbkowanie

Próbkowanie oznacza losowe wybieranie wartości z rozkładu. W ML:
- Dropout losowo wybiera, które neurony wyzerować
- Augmentacja danych próbkuje losowe transformacje
- Modele językowe próbkują następny token z przewidywanego rozkładu
- Modele dyfuzyjne próbkują szum i progresywnie usuwają szum

Próbkowanie z arbitralnych rozkładów wymaga technik jak próbkowanie transformacji odwrotnej, rejection sampling, lub sztuczka reprametrizacji (używana w VAE).

## Zbuduj to

### Krok 1: Podstawy prawdopodobieństwa

```python
import math
import random

def silnia(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def kombinacje(n, k):
    return silnia(n) // (silnia(k) * silnia(n - k))

def prawdopodobienstwo_warunkowe(p_a_i_b, p_b):
    return p_a_i_b / p_b

p_krol_dla_karty_z_obrazem = prawdopodobienstwo_warunkowe(4/52, 12/52)
print(f"P(Król | Karta z obrazem) = {p_krol_dla_karty_z_obrazem:.4f}")
```

### Krok 2: PMF i PDF od podstaw

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / silnia(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Krok 3: Wartość oczekiwana i wariancja

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Kostka: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Krok 4: Próbkowanie z rozkładów

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Krok 5: Softmax i logarytmy prawdopodobieństw

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Krok 6: Demonstracja Centralnego Twierdzenia Granicznego

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Krok 7: Wizualizacja

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Pełne implementacje ze wszystkimi wizualizacjami znajdują się w `code/probability.py`.

## Zastosuj to

Z NumPy i SciPy wszystko powyższe to jednolinijkowce:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Średnia: {np.mean(samples):.4f}, Odchylenie: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Zbudowałeś to od podstaw. Teraz wiesz, co robią wywołania biblioteki.

## Ćwiczenia

1. Zaimplementuj próbkowanie transformacji odwrotnej dla rozkładu wykładniczego. Zweryfikuj, próbkując 10 000 wartości i porównując histogram z prawdziwym PDF.

2. Zbuduj tabelę rozkładu łącznego dla dwóch obciążonych kostek. Oblicz rozkłady brzegowe i sprawdź, czy kostki są niezależne.

3. Oblicz stratę entropii krzyżowej dla klasyfikatora 5-klasowego, który zwraca logity `[2.0, 0.5, -1.0, 3.0, 0.1]` dla poprawnej klasy o indeksie 3. Następnie sprawdź swój wynik z `nn.CrossEntropyLoss` w PyTorch.

4. Napisz funkcję, która przyjmuje listę log-prawdopodobieństw i zwraca najbardziej prawdopodobną sekwencję, całkowite log-prawdopodobieństwo i równoważne surowe prawdopodobieństwo. Przetestuj ją ze zdaniem 50 słów, gdzie każde słowo ma prawdopodobieństwo 0.01.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|----------------|----------------------|
| Przestrzeń próbek | "Wszystkie możliwości" | Zbiór S każdego możliwego wyniku eksperymentu |
| PMF | "Funkcja prawdopodobieństwa" | Funkcja przypisująca dokładne prawdopodobieństwo każdemu dyskretnemu wynikowi, sumująca się do 1 |
| PDF | "Krzywa prawdopodobieństwa" | Funkcja gęstości dla zmiennych ciągłych. Całkuj ją na przedziale, aby uzyskać prawdopodobieństwo |
| Prawdopodobieństwo warunkowe | "Prawdopodobieństwo przy danym założeniu" | P(A\|B) = P(A i B) / P(B). Fundament myślenia bayesowskiego i twierdzenia Bayesa |
| Niezależność | "Nie wpływają na siebie" | P(A i B) = P(A) * P(B). Wiedza o jednym zdarzeniu nie daje informacji o drugim |
| Wartość oczekiwana | "Średnia" | Suma ważona prawdopodobieństwem wszystkich wyników. Funkcja straty jest wartością oczekiwaną |
| Wariancja | "Jak bardzo rozproszone" | Oczekiwana kwadratowa odległość od średniej. Wysoka wariancja = zaszumione, niestabilne oszacowania |
| Rozkład normalny | "Krzywa dzwonowa" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Pojawia się wszędzie z powodu CTG |
| Centralne Twierdzenie Graniczne | "Średnie stają się normalne" | Średnia wielu niezależnych próbek dąży do rozkładu normalnego niezależnie od źródła |
| Rozkład łączny | "Dwie zmienne razem" | P(X, Y) opisuje prawdopodobieństwo każdej kombinacji wyników X i Y |
| Rozkład brzegowy | "Sumujemy drugą zmienną" | P(X) = suma_y P(X, Y). Odtwarza rozkład jednej zmiennej z rozkładu łącznego |
| Log-prawdopodobieństwo | "Logarytm prawdopodobieństwa" | log P(x). Zamienia iloczyny na sumy, zapobiegając numerycznemu underflow w długich sekwencjach |
| Softmax | "Zamienia wyniki w prawdopodobieństwa" | softmax(z_i) = exp(z_i) / suma(exp(z_j)). Odwzorowuje rzeczywiste logity na prawidłowy rozkład prawdopodobieństwa |
| Entropia krzyżowa | "Funkcja straty" | -sum(p_true * log(p_predicted)). Mierzy jak różne są dwa rozkłady. Niższa jest lepsza |
| Logity | "Surowe wyjścia modelu" | Nieznormalizowane wyniki przed softmax. Nazwa pochodzi od funkcji logistycznej |
| Próbkowanie | "Losowanie wartości" | Generowanie wartości według rozkładu prawdopodobieństwa. Jak modele generują wyniki |

## Dalsze czytanie

- [3Blue1Brown: Ale czym jest Centralne Twierdzenie Graniczne?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - wizualny dowód, dlaczego średnie stają się normalne
- [Stanford CS229 Przegląd Prawdopodobieństwa](https://cs229.stanford.edu/section/cs229-prob.pdf) - zwięzłe źródło obejmujące wszystko tutaj i więcej
- [Sztuczka Log-Sum-Exp](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - dlaczego stabilność numeryczna ma znaczenie i jak ją osiągnąć