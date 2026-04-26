# Metody próbkowania

> Próbkowanie to sposób, w jaki AI eksploruje przestrzeń możliwości.

**Typ:** Zbuduj
**Język:** Python
**Wymagania wstępne:** Phase 1, Lessons 06-07 (Prawdopodobieństwo, Twierdzenie Bayesa)
**Czas:** ~120 minut

## Cele uczenia się

- Zaimplementuj od podstaw metodę odwrotnej dystrybuanty, próbkowanie przez odrzucanie oraz próbkowanie ważnościowe, używając tylko jednorodnych liczb losowych
- Zbuduj próbkowanie z temperaturą, top-k oraz top-p (nucleus) do generowania tokenów przez modele językowe
- Wyjaśnij sztuczkę z reprametryzacją i dlaczego umożliwia ona propagację wsteczną przez próbkowanie w VAE
- Uruchom algorytm Metropolis-Hastings MCMC, aby próbkować z nieznormalizowanego rozkładu docelowego

## Problem

Model językowy kończy przetwarzać twoje polecenie i produkuje wektor 50 000 logitów. Jeden dla każdego tokenu w jego słowniku. Teraz musi wybrać jeden. Jak?

Jeśli zawsze wybiera token o najwyższym prawdopodobieństwie, każda odpowiedź jest identyczna. Deterministyczna. Nudna. Jeśli wybiera losowo z rozkładu jednorodnego, wynik to bełkot. Odpowiedź znajduje się gdzieś pomiędzy tymi skrajnościami, a to gdzieś jest kontrolowane przez próbkowanie.

Próbkowanie nie ogranicza się do generowania tekstu. Uczenie ze wzmocnieniem szacuje gradienty polityki poprzez próbkowanie trajektorii. VAE uczą się reprezentacji latentnych poprzez próbkowanie z nauczonych rozkładów i propagację wsteczną przez losowość. Modele dyfuzyjne generują obrazy poprzez próbkowanie szumu i iteracyjne usuwanie szumu. Metody Monte Carlo szacują całki, które nie mają rozwiązania w formie zamkniętej. Algorytmy MCMC eksplorują wysokowymiarowe rozkłady a posteriori, których nie można wyliczyć.

Każdy generatywny system AI jest systemem próbkowania. Strategia próbkowania determinuje jakość, różnorodność i kontrolowalność wyniku. Ta lekcja buduje od podstaw każdą główną metodę próbkowania, zaczynając od jednorodnych liczb losowych i kończąc na technikach, które napędzają nowoczesne LLM i modele generatywne.

## Koncepcja

### Dlaczego próbkowanie ma znaczenie

Próbkowanie pojawia się w czterech fundamentalnych rolach w AI i uczeniu maszynowym:

**Generowanie.** Modele językowe, modele dyfuzyjne i GAN wszystkie produkują wyniki poprzez próbkowanie. Algorytm próbkowania bezpośrednio kontroluje kreatywność, spójność i różnorodność. Próbkowanie z temperaturą, top-k i nucleus to pokrętła, które inżynierowie obracają codziennie.

**Szkolenie.** Stochastyczny spadek gradientu próbkuje minibatche. Dropout próbkuje neurony do deaktywacji. Augmentacja danych próbkuje losowe transformacje. Próbkowanie ważnościowe przeważa próbki, aby zredukować wariancję gradientu w uczeniu ze wzmocnieniem (PPO, TRPO).

**Estymacja.** Wiele wielkości w ML nie ma rozwiązania w formie zamkniętej. Oczekiwana strata nad rozkładem danych, funkcja partycyjna modelu opartego na energii, wiarygodność w wnioskowaniu bayesowskim. Estymacja Monte Carlo przybliża wszystkie te wielkości poprzez uśrednianie po próbkach.

**Eksploracja.** Algorytmy MCMC eksplorują rozkłady a posteriori w wnioskowaniu bayesowskim. Strategie ewolucyjne próbkują zaburzenia parametrów. Próbkowanie Thompsona równoważy eksplorację i eksploatację w problemach bandytów.

Główne wyzwanie: możesz próbkować bezpośrednio tylko z prostych rozkładów (jednorodny, normalny). Dla wszystkiego innego potrzebujesz metody konwersji prostych próbek na próbki z twojego rozkładu docelowego.

### Jednorodne próbkowanie losowe

Każda metoda próbkowania zaczyna się tutaj. Generator jednorodnych liczb losowych produkuje wartości w [0, 1), gdzie każdy podprzedział równej długości ma równe prawdopodobieństwo.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

Aby próbkować jednorodnie ze zbioru dyskretnego n elementów, wygeneruj U i zwróć floor(n * U). Aby próbkować z ciągłego przedziału [a, b], oblicz a + (b - a) * U.

Kluczowy wgląd: jedna jednorodna liczba losowa zawiera dokładnie właściwą ilość losowości, aby wyprodukować jedną próbkę z dowolnego rozkładu. Sztuczka polega na znalezieniu właściwej transformacji.

### Metoda odwrotnej dystrybuanty (Inverse Transform Sampling)

Dystrybuanta (CDF) mapuje wartości na prawdopodobieństwa:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Odwrotna dystrybuanta mapuje prawdopodobieństwa z powrotem na wartości. Jeśli U ~ Uniform(0, 1), to X = F_inverse(U) wynika z rozkładu docelowego.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Przykład rozkładu wykładniczego:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

To działa doskonale, gdy możesz zapisać F_inverse w formie zamkniętej. Dla rozkładu normalnego nie ma formy zamkniętej odwrotnej dystrybuanty, więc używamy innych metod (Box-Muller lub aproksymacja numeryczna).

**Wersja dyskretna:** Dla rozkładów dyskretnych zbuduj CDF jako sumę skumulowaną, wygeneruj U i znajdź pierwszy indeks, gdzie suma skumulowana przekracza U. Tak działa `sample_categorical` w Lesson 06.

### Próbkowanie przez odrzucanie

Gdy nie możesz odwrócić CDF, ale możesz wyliczyć docelową PDF ze stałą, próbkowanie przez odrzucanie działa.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Im ciaśniejsze wiązanie M, tym wyższy współczynnik akceptacji. W niskich wymiarach (1-3), próbkowanie przez odrzucanie działa dobrze. W wysokich wymiarach współczynnik akceptacji spada wykładniczo, ponieważ większość objętości proposal jest odrzucana. To jest przekleństwo wymiarowości dla próbkowania przez odrzucanie.

**Przykład: próbkowanie z obciętego rozkładu normalnego.** Użyj uniform proposal nad obciętym zakresem. Obwiednia M to maksimum PDF normalnego w tym zakresie.

**Przykład: próbkowanie z półkola.** Proponuj jednorodnie w prostokącie ograniczającym. Akceptuj, jeśli punkt mieści się w półkolu. Tak Monte Carlo oblicza pi: współczynnik akceptacji równa się stosunkowi pól pi/4.

### Próbkowanie ważnościowe

Czasem nie potrzebujesz próbek z rozkładu docelowego p(x). Potrzebujesz oszacować oczekiwanie pod p(x), a masz próbki z innego rozkładu q(x).

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

To jest kluczowe w uczeniu ze wzmocnieniem. W PPO (Proximal Policy Optimization) zbierasz trajektorie pod starą polityką pi_old, ale chcesz optymalizować nową politykę pi_new. Waga ważności to pi_new(a|s) / pi_old(a|s). PPO przycina te wagi, aby zapobiec zbyt dużemu odchyleniu nowej polityki od starej.

Wariancja estymatora próbkowania ważnościowego zależy od tego, jak podobne jest q do p. Jeśli q bardzo różni się od p, kilka próbek z ogromnymi wagami dominuje oszacowanie. Samonormalizowane próbkowanie ważnościowe dzieli przez sumę wag, aby zredukować ten problem:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Estymacja Monte Carlo

Estymacja Monte Carlo przybliża całki poprzez uśrednianie losowych próbek. Prawo wielkich liczb gwarantuje zbieżność.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

Tempo błędu nie zależy od wymiaru. Dlatego metody Monte Carlo dominują w wysokich wymiarach, gdzie całkowanie oparte na siatce jest niemożliwe.

**Szacowanie pi:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Szacowanie oczekiwań:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC konstruuje łańcuch Markowa, którego rozkład stacjonarny to rozkład docelowy p(x). Po wystarczającej liczbie kroków próbki z łańcucha są (w przybliżeniu) próbkami z p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

Dla symetrycznych propozycji (q(x'|x) = q(x|x')), stosunek upraszcza się do p(x')/p(x). To jest oryginalny algorytm Metropolis.

**Dlaczego to działa.** Reguła akceptacji zapewnia szczegółową równowagę: prawdopodobieństwo bycia w x i przejścia do x' równa się prawdopodobieństwu bycia w x' i przejścia do x. Szczegółowa równowaga implikuje, że p(x) jest rozkładem stacjonarnym łańcucha.

**Praktyczne rozważania:**
- Burn-in: odrzuć wczesne próbki przed osiągnięciem równowagi przez łańcuch
- Thinning: zachowuj co k-tą próbkę, aby zredukować autokorelację
- Skala propozycji: zbyt mała i łańcuch porusza się wolno (wysoka akceptacja, wolna eksploracja); zbyt duża i większość propozycji jest odrzucana (niska akceptacja, utknięcie w miejscu)
- Optymalny współczynnik akceptacji dla propozycji Gaussowskiej w wysokich wymiarach wynosi około 0.234

### Próbkowanie Gibbsa

Próbkowanie Gibbsa to szczególny przypadek MCMC dla rozkładów wielowymiarowych. Zamiast proponować ruch we wszystkich wymiarach naraz, aktualizuje jedną zmienną na raz z jej rozkładu warunkowego.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Próbkowanie Gibbsa wymaga, abyś mógł próbkować z każdego rozkładu warunkowego p(x_i | x_{-i}). To jest proste dla wielu modeli:
- Sieci bayesowskie: warunkowe wynikają ze struktury grafu
- Mieszaniny Gaussowskie: warunkowe są Gaussowskie
- Modele Isinga: warunkowa każdego spinu zależy tylko od jego sąsiadów

Współczynnik akceptacji jest zawsze 1 (każda propozycja jest akceptowana), ponieważ próbkowanie z dokładnego rozkładu warunkowego automatycznie spełnia szczegółową równowagę.

**Ograniczenie.** Gdy zmienne są silnie skorelowane, próbkowanie Gibbsa miesza się powoli, ponieważ aktualizacja jednej zmiennej na raz nie może wykonywać dużych ruchów diagonalnych przez rozkład.

### Próbkowanie z temperaturą (używane w LLM)

Modele językowe wyprowadzają logity z_1, ..., z_V dla każdego tokenu w słowniku. Softmax konwertuje je na prawdopodobieństwa. Temperatura skaluje logity przed softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Dlaczego to działa.** Dzielenie logitów przez T < 1 wzmacnia różnice między logitami. Jeśli z_1 = 2 i z_2 = 1, dzielenie przez T = 0.5 daje z_1/T = 4 i z_2/T = 2, powiększając lukę. Po softmax token z najwyższym logitem dostaje znacznie większy udział.

**W praktyce:**
- T = 0.0: dekodowanie zachłanne, najlepsze dla factual Q&A
- T = 0.3-0.7: lekko kreatywne, dobre do generowania kodu
- T = 0.7-1.0: zbalansowane, dobre do ogólnej konwersacji
- T = 1.0-1.5: kreatywne pisanie, burza mózgów
- T > 1.5: coraz bardziej losowe, rzadko użyteczne

Temperatura nie zmienia, które tokeny są możliwe. Zmienia masę prawdopodobieństwa przydzieloną każdemu tokenowi.

### Próbkowanie Top-k

Próbkowanie top-k ogranicza zbiór kandydatów do k tokenów o najwyższych prawdopodobieństwach, następnie renormalizuje i próbkuje z tego ograniczonego zbioru.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k zapobiega wybieraniu przez model skrajnie nieprawdopodobnych tokenów (literówek, nonsensu), które istnieją w długim ogonie rozkładu słownika. Problem: k jest stałe niezależnie od kontekstu. Gdy model jest pewny (jeden token ma prawdopodobieństwo 95%), k = 40 nadal pozwala na 39 alternatyw. Gdy model jest niepewny (prawdopodobieństwo rozłożone na 1000 tokenów), k = 40 odcina prawdopodobne opcje.

### Próbkowanie Top-p (Nucleus)

Próbkowanie top-p dynamicznie dostosowuje rozmiar zbioru kandydatów. Zamiast zachowywać stałą liczbę tokenów, zachowuje najmniejszy zbiór tokenów, których skumulowane prawdopodobieństwo przekracza p.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

Gdy model jest pewny, nucleus sampling zachowuje kilka tokenów (może 2-3). Gdy model jest niepewny, zachowuje wiele (może 200). Ta adaptacyjna cecha sprawia, że nucleus sampling generalnie produkuje lepszy tekst niż top-k.

**Typowe kombinacje:**
- Temperatura 0.7 + top-p 0.9: dobra uniwersalna konfiguracja
- Temperatura 0.0 (zachłanna): najlepsza dla deterministycznych zadań
- Temperatura 1.0 + top-k 50: konfiguracja z oryginalnej pracy Fan et al. (2018)

Top-k i top-p można łączyć. Najpierw zastosuj top-k, potem top-p na pozostałym zbiorze.

### Sztuczka z reprametryzacją (używana w VAE)

Autoenkodery wariacyjne (VAE) uczą się poprzez enkoding inputów do rozkładu w przestrzeni latentnej, próbkowanie z tego rozkładu i dekodowanie próbki z powrotem. Problem: nie możesz propagować wstecznie przez operację próbkowania.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Sztuczka z reprametryzacją oddziela losowość od parametrów:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

To działa, ponieważ N(mu, sigma^2) ma ten sam rozkład co mu + sigma * N(0, 1). Kluczowy wgląd: przenieś losowość do źródła bez parametrów (epsilon), następnie wyraź próbkę jako różniczkowalną transformację parametrów.

**W pętli treningowej VAE:**
1. Enkoder wyprowadza mu i log(sigma^2) dla każdego inputu
2. Próbkuj epsilon ~ N(0, 1)
3. Oblicz z = mu + sigma * epsilon
4. Dekoduj z, aby zrekonstruować input
5. Propaguj wstecznie przez kroki 4, 3, 2, 1 (możliwe, bo krok 3 jest różniczkowalny)

Bez sztuczki z reprametryzacją VAE nie mogą być trenowane standardową propagacją wsteczną. Ten pojedynczy wgląd uczynił VAE praktycznymi.

### Gumbel-Softmax (różniczkowalne próbkowanie kategorialne)

Sztuczka z reprametryzacją działa dla rozkładów ciągłych (Gaussowski). Dla dyskretnych rozkładów kategorialnych potrzebujemy innego podejścia. Gumbel-Softmax dostarcza różniczkowalną aproksymację próbkowania kategorialnego.

**Sztuczka Gumbel-Max (nieróżniczkowalna):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (różniczkowalna aproksymacja):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax produkuje ciągłą relaksację dyskretnej próbki. Wynik to wektor prawdopodobieństwa (miękki one-hot) zamiast twardego one-hot. Gradienty przepływają przez softmax. Podczas passy forward w treningu możesz użyć "straight-through" estymatora: użyj twardego argmax dla passy forward, ale miękkich gradientów Gumbel-Softmax dla passy backward.

**Zastosowania:**
- Dyskretne zmienne latentne w VAE
- Neural architecture search (wybieranie dyskretnych operacji)
- Hard attention mechanisms
- Uczenie ze wzmocnieniem z dyskretnymi akcjami

### Próbkowanie warstwowe

Standardowe próbkowanie Monte Carlo może pozostawiać luki w przestrzeni próbek przez przypadek. Próbkowanie warstwowe wymusza równomierne pokrycie poprzez dzielenie przestrzeni na warstwy i próbkowanie z każdej.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Próbkowanie warstwowe zawsze ma niższą lub równą wariancję w porównaniu do standardowego Monte Carlo:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Zastosowania:**
- Całkowanie numeryczne (quasi-Monte Carlo)
- Podziały danych treningowych (zapewnienie równowagi klas w każdym foldzie)
- Próbkowanie ważnościowe ze stratyfikacją (łączenie obu technik)
- NeRF (Neural Radiance Fields) używa próbkowania warstwowego wzdłuż promieni kamery

### Połączenie z modelami dyfuzyjnymi

Modele dyfuzyjne generują obrazy poprzez proces próbkowania. Proces forward dodaje szum Gaussowski do obrazu przez T kroków, aż stanie się czystym szumem. Proces odwrotny uczy się usuwać szum, odzyskując oryginalny obraz krok po kroku.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

Połączenie z metodami z tej lekcji:
- Każdy krok usuwania szumu używa sztuczki z reprametryzacją (próbkuj szum, zastosuj deterministyczną transformację)
- Harmonogram szumu {alpha_t} kontroluje formę wyżarzania temperatury
- Trening używa estymacji Monte Carlo do aproksymacji ELBO (evidence lower bound)
- Ancestral sampling w modelach dyfuzyjnych jest łańcuchem Markowa (każdy krok zależy tylko od aktualnego stanu)

Cały proces generowania obrazu to iteracyjne próbkowanie: zacznij od szumu i na każdym kroku próbkuj nieznacznie mniej zaszumioną wersję warunkową na nauczonym modelu usuwania szumu.

## Zbuduj to

### Krok 1: Jednorodne próbkowanie i odwrotna dystrybuanta

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Wygeneruj 10 000 próbek wykładniczych i sprawdź, czy średnia wynosi 1/lambda.

### Krok 2: Próbkowanie przez odrzucanie

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Użyj próbkowania przez odrzucanie, aby rysować z rozkładu obciętego normalnego. Sprawdź kształt poprzez histogramowanie próbek.

### Krok 3: Próbkowanie ważnościowe

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Oszacuj E[X^2] pod rozkładem normalnym używając uniform proposal. Porównaj ze znaną odpowiedzią (mu^2 + sigma^2).

### Krok 4: Estymacja Monte Carlo pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Krok 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Próbkuj z rozkładu bimodalnego (mieszanina dwóch rozkładów Gaussowskich). Wizualizuj trajektorię łańcucha.

### Krok 6: Próbkowanie Gibbsa

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Krok 7: Próbkowanie z temperaturą

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Pokaż, jak temperatura zmienia rozkład wynikowy dla zestawu logitów tokenów.

### Krok 8: Próbkowanie Top-k i Top-p

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Krok 9: Sztuczka z reprametryzacją

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Zademonstruj, że gradienty przepływają przez reprametryzowaną próbkę, ale nie przez bezpośrednie próbkowanie.

### Krok 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Pokaż, jak zmniejszanie temperatury sprawia, że wynik zbliża się do wektora one-hot.

Pełne implementacje ze wszystkimi wizualizacjami znajdują się w `code/sampling.py`.

## Użyj tego

Z NumPy i SciPy wersje produkcyjne:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

Dla MCMC na skali użyj dedykowanych bibliotek:
- PyMC: pełne modelowanie bayesowskie z NUTS (adaptive HMC)
- emcee: ensemble MCMC sampler
- NumPyro/JAX: GPU-accelerated MCMC

Zbudowałeś to od podstaw. Teraz wiesz, co wywołania biblioteczne robią pod maską.

## Ćwiczenia

1. Zaimplementuj próbkowanie odwrotnej dystrybuanty dla rozkładu Cauchy'ego. Dystrybuanta to F(x) = 0.5 + arctan(x)/pi. Wygeneruj 10 000 próbek i wykreśl histogram przeciwko prawdziwej PDF. Zwróć uwagę na ciężkie ogony (wartości skrajne daleko od centrum).

2. Użyj próbkowania przez odrzucanie, aby wygenerować próbki z rozkładu Beta(2, 5) używając propozycji Uniform(0, 1). Wykreśl zaakceptowane próbki przeciwko prawdziwej PDF Beta. Jaki jest teoretyczny współczynnik akceptacji?

3. Oszacuj całkę sin(x) od 0 do pi używając Monte Carlo z 1 000, 10 000 i 100 000 próbkami. Porównaj błąd na każdym poziomie. Zweryfikuj, że błąd skaluje się jak O(1/sqrt(N)).

4. Zaimplementuj Metropolis-Hastings, aby próbkować z rozkładu 2D p(x, y) proporcjonalnego do exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Wykreśl próbki i trajektorię łańcucha. Eksperymentuj z różnymi odchyleniami standardowymi propozycji.

5. Zbuduj kompletną demonstrację generowania tekstu: mając słownik 10 słów z logitami, generuj sekwencje 20 tokenów używając (a) zachłannego, (b) temperatura=0.7, (c) top-k=3, (d) top-p=0.9. Porównaj różnorodność wyników w 5 uruchomieniach.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|----------------|----------------------|
| Próbkowanie | "Rysowanie losowych wartości" | Generowanie wartości zgodnie z rozkładem prawdopodobieństwa. Mechanizm stojący za całym generatywnym AI |
| Rozkład jednorodny | "Wszystkie jednakowo prawdopodobne" | Każda wartość w [a, b] ma jednakową gęstość prawdopodobieństwa 1/(b-a). Punkt startowy dla wszystkich metod próbkowania |
| Odwrotna dystrybuanta | "Transformacja probabilistyczna" | F_inverse(U) konwertuje próbkę jednorodną na próbkę z dowolnego rozkładu o znanej dystrybuancie. Dokładna i efektywna |
| Próbkowanie przez odrzucanie | "Proponuj i akceptuj/odrzucaj" | Generuj z prostego proposal, akceptuj z prawdopodobieństwem proporcjonalnym do stosunku target/proposal. Dokładne, ale marnuje próbki |
| Próbkowanie ważnościowe | "Przeważaj próbki" | Oszacuj oczekiwania pod p(x) używając próbek z q(x) przez przeważanie każdej próbki przez p(x)/q(x). Kluczowe dla PPO w RL |
| Monte Carlo | "Uśredniaj losowe próbki" | Aproksymuj całki jako średnie próbek. Błąd O(1/sqrt(N)) niezależnie od wymiaru |
| MCMC | "Losowy spacer, który zbiega" | Konstruuj łańcuch Markowa, którego rozkład stacjonarny to target. Metropolis-Hastings to fundamentalny algorytm |
| Metropolis-Hastings | "Akceptuj pod górę, czasem w dół" | Proponuj ruchy, akceptuj na podstawie stosunku gęstości. Szczegółowa równowaga zapewnia zbieżność do rozkładu docelowego |
| Próbkowanie Gibbsa | "Jedna zmienna na raz" | Aktualizuj każdą zmienną z jej rozkładu warunkowego przy stałych pozostałych. 100% współczynnik akceptacji |
| Temperatura | "Pokrętło pewności" | Dzieli logity przez T przed softmax. T<1 wyostrza (bardziej pewny), T>1 spłaszcza (bardziej różnorodny) |
| Próbkowanie Top-k | "Zachowaj k najlepszych" | Wyzeruj wszystko oprócz k tokenów o najwyższym prawdopodobieństwie, renormalizuj, próbkuj. Stały rozmiar zbioru kandydatów |
| Próbkowanie Nucleus (top-p) | "Zachowaj prawdopodobne" | Zachowaj najmniejszy zbiór tokenów, których skumulowane prawdopodobieństwo przekracza p. Adaptacyjny rozmiar zbioru kandydatów |
| Sztuczka z reprametryzacją | "Przenieś losowość na zewnątrz" | Zapisz z = mu + sigma * epsilon gdzie epsilon ~ N(0,1). Sprawia, że próbkowanie jest różniczkowalne. Niezbędne do treningu VAE |
| Gumbel-Softmax | "Miękkie próbkowanie kategorialne" | Różniczkowalna aproksymacja próbkowania kategorialnego używająca szumu Gumbel + softmax z temperaturą |
| Próbkowanie warstwowe | "Wymuszone pokrycie" | Podziel przestrzeń próbek na warstwy, próbkuj z każdej. Zawsze niższa wariancja niż naiwne Monte Carlo |
| Burn-in | "Okres rozgrzewki" | Początkowe próbki MCMC odrzucane przed osiągnięciem przez łańcuch rozkładu stacjonarnego |
| Szczegółowa równowaga | "Warunek odwracalności" | p(x) * T(x->y) = p(y) * T(y->x). Warunek wystarczający, aby p był rozkładem stacjonarnym łańcucha Markowa |
| Próbkowanie dyfuzyjne | "Iteracyjne usuwanie szumu" | Generuj dane zaczynając od szumu i stosując nauczone kroki usuwania szumu. Każdy krok to warunkowa operacja próbkowania |

## Dalsza lektura

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - szczegółowy poradnik o fundamentach MCMC
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - oryginalna praca o Gumbel-Softmax
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - praca o próbkowaniu nucleus (top-p)
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - praca o VAE wprowadzająca sztuczkę z reprametryzacją
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM łączy próbkowanie z generowaniem obrazów