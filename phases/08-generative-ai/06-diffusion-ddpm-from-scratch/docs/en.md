# Modele dyfuzyjne — DDPM od podstaw

> Ho, Jain, Abbeel (2020) dali dziedzinie przepis, którego nie mogła porzucić. Zniszcz dane szumem w tysiącu małych kroków. Wytrenuj jedną sieć neuronową, aby przewidywała szum. Odwróć proces podczas wnioskowania. Dziś każdy główny model obrazu, wideo, 3D i muzyki działa w tej pętli, prawdopodobnie z dopasowaniem przepływu (flow matching) lub sztuczkami spójności na wierzchu.

**Typ:** Konstruowanie
**Języki:** Python
**Wymagania wstępne:** Faza 3 · 02 (Backprop), Faza 8 · 02 (VAE)
**Czas:** ~75 minut

## Problem

Chcesz próbnik (sampler) dla `p_data(x)`. GAN-y grają w grę minimax, która często się rozbiega. VAE produkują rozmyte próbki z dekodera Gaussowskiego. Naprawdę chcesz celu trenowania, który jest (a) jedną stabilną stratą (bez punktów siodłowych, bez minimax), (b) dolnym ograniczeniem dla `log p(x)` (więc masz wiarygodności), i (c) próbkami dorównującymi jakości SOTA.

Sohl-Dickstein et al. (2015) mieli teoretyczną odpowiedź: zdefiniuj łańcuch Markowa `q(x_t | x_{t-1})`, który stopniowo dodaje szum Gaussowski, i wytrenuj odwrotny łańcuch `p_θ(x_{t-1} | x_t)`, aby usuwać szum. Ho, Jain, Abbeel (2020) pokazali, że strata może być uproszczona do jednej linii — przewiduj szum — i wyczyścili matematykę. W 2020 była to osobliwość. W 2021 produkowała próbki najnowocześniejsze. W 2022 stała się Stable Diffusion. W 2026 jest podłożem.

## Koncepcja

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**Proces prosty `q`.** Dodaj szum Gaussowski w `T` małych krokach. Forma zamknięta — powód, dla którego matematyka jest rozwiązywalna — polega na tym, że krok skumulowany również jest Gaussowski:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

gdzie `α̅_t = ∏_{s=1..t} (1 - β_s)` dla harmonogramu `β_t`. Wybierz `β_t` z 1e-4 do 0.02 liniowo przez T=1000 kroków i `x_T` jest w przybliżeniu `N(0, I)`.

**Proces odwrotny `p_θ`.** Naucz sieć neuronową `ε_θ(x_t, t)`, która przewiduje dodany szum. Mając `x_t`, usuń szum przez:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

gdzie `σ_t` to albo `sqrt(β_t)` albo nauczona wariancja. Wyrażenie jest brzydkie, ale to tylko algebra — rozwiązujemy dla `x_{t-1}` mając rozkład a posteriori `q(x_{t-1} | x_t, x_0)` i podstawiamy `x_0` jego oszacowaniem przez przewidywanie szumu.

**Funkcja straty treningowej.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

Próbkuj `x_0` z danych, wybierz losowe `t`, próbkuj `ε ~ N(0, I)`, oblicz zaszumione `x_t` jednym strzałem przez formę zamkniętą, i regresuj na szumie. Jedna strata, bez minimax, bez KL, bez sztuczek z reparametryzacją.

**Próbkowanie.** Zacznij od `x_T ~ N(0, I)`. Iteruj krok odwrotny od `t = T` do `1`. Gotowe.

## Dlaczego to działa

Trzy intuicje:

1. **Usuwanie szumu jest łatwe; generowanie jest trudne.** Przy `t=T`, dane to czysty szum — sieć musi rozwiązać trywialny problem. Przy `t=0`, sieć musi tylko posprzątać kilka pikseli. Przy pośrednich `t`, problem jest trudny, ale sieć ma wiele gradientów płynących przez te same wagi z każdego poziomu szumu.

2. **Dopasowanie do score w przebraniu.** Vincent (2011) udowodnił, że przewidywanie szumu jest równoważne estymowaniu `∇_x log q(x_t | x_0)`, czyli *score*. Odwrotny SDE wykorzystuje ten score, aby wspiąć się po gradiencie gęstości — prowadzony losowy spacer w kierunku regionów wysokiego prawdopodobieństwa.

3. **ELBO redukuje się do prostego MSE.** Pełne wariacyjne ograniczenie dolne ma termin KL na każdy krok czasowy. Z parametryzacją DDPM te terminy KL upraszczają się do MSE na przewidywaniu szumu ze specyficznymi współczynnikami; Ho porzucił współczynniki (nazywając to stratą „prostą") i jakość się *poprawiła*.

## Zbuduj to

`code/main.py` implementuje 1-wymiarowy DDPM. Dane to mieszanka dwumodalna. „Sieć" to maleńki MLP, który przyjmuje `(x_t, t)` i wyprowadza przewidywany szum. Trening to strata jednolinijkowa. Próbkowanie iteruje odwrotny łańcuch.

### Krok 1: harmonogram prosty (forma zamknięta)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Krok 2: próbkuj `x_t` jednym strzałem

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Krok 3: jeden krok treningowy

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Krok 4: próbkowanie odwrotne

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

Dla problemu 1-W z 40 krokami czasowymi i MLP 24-jednostkowym, to uczy się mieszanki dwumodalnej w ~200 epokach.

## Warunkowanie czasowe

Sieć musi wiedzieć, który krok czasowy usuwa szum. Dwie standardowe opcje:

- **Osadzenie sinusoidalne.** Jak kodowanie pozycyjne Transformera. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. Przepuść przez MLP, rozprowadź do sieci.
- **Warunkowanie FiLM / group-norm.** Projektuj osadzenie na skale/odchylenie na kanał (FiLM) w każdym bloku.

Nasz kod demonstracyjny używa sinusoidalnego → konkatenacji. Produkcyjne U-Nety używają FiLM.

## Pułapki

- **Harmonogram ma ogromne znaczenie.** Liniowe `β` to domyślne DDPM, ale harmonogram cosinusowy (Nichol & Dhariwal, 2021) daje lepszy FID przy tym samym obliczeniu. Zmień harmonogramy, jeśli jakość się zatrzymuje.
- **Osadzenie kroku czasowego jest delikatne.** Przekazywanie surowego `t` jako float działa dla demonstracyjnego 1-D, ale zawodzi dla obrazów; zawsze używaj właściwego osadzenia.
- **V-predykcja vs ε-predykcja.** Dla wąskich reżimów (bardzo małe lub bardzo duże t), `ε` ma słaby stosunek sygnału do szumu. V-predykcja (`v = α·ε - σ·x`) jest bardziej stabilna; SDXL, SD3 i Flux jej używają.
- **Wskazówka bez klasyfikatora (CFG).** Podczas wnioskowania oblicz zarówno warunkowy jak i bezwarunkowy `ε`, następnie `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` z `w ≈ 3-7`. Omówione w Lekcji 08.
- **1000 kroków to dużo.** Produkcja używa DDIM (20-50 kroków), DPM-Solver (10-20 kroków) lub destylacji (1-4 kroki). Zobacz Lekcję 12.

## Zastosowanie

| Rola | Typowy stos w 2026 |
|------|-----------------------|
| Dyfuzja przestrzeni pikseli obrazu (mała, demonstracyjna) | DDPM + U-Net |
| Latentna dyfuzja obrazu | Koder VAE + U-Net lub DiT (Lekcja 07) |
| Latentna dyfuzja wideo | Przestrzenno-czasowy DiT (Sora, Veo, WAN) |
| Latentna dyfuzja audio | Encodec + transformator dyfuzyjny |
| Nauka (cząsteczki, białka, fizyka) | Równoważna dyfuzja (EDM, RFdiffusion, AlphaFold3) |

Dyfuzja to uniwersalny generatywny szkielet. Dopasowanie przepływu (Lekcja 13) to konkurent 2024-2026, który zwykle wygrywa w szybkości wnioskowania przy tej samej jakości.

## Wyślij to

Zapisz `outputs/skill-diffusion-trainer.md`. Umiejętność przyjmuje zbiór danych + budżet obliczeniowy i wyprowadza: harmonogram (liniowy/cosinusowy/sigmoid), cel predykcji (ε/v/x), liczbę kroków, skale wskazówki, rodzinę próbnika i protokół ewaluacji.

## Ćwiczenia

1. **Łatwe.** Zmień T z 40 na 10 w `code/main.py`. Jak pogarsza się jakość próbek (wizualny histogram wyjść)? Przy jakim T struktura dwumodalna się załamuje?
2. **Średnie.** Przełącz z ε-predykcji na v-predykcję. Wyprowadź ponownie krok odwrotny. Porównaj końcową jakość próbek.
3. **Trudne.** Dodaj wskazówkę bez klasyfikatora. Warunkuj na etykiecie klasy `c ∈ {0, 1}`, porzuć ją 10% czasu podczas treningu, i w czasie próbkowania użyj `ε = (1+w)·ε_cond - w·ε_uncond`. Zmierz wskaźnik trafień w trybie warunkowym przy `w = 0, 1, 3, 7`.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Proces prosty | „Dodawanie szumu" | Stały łańcuch Markowa `q(x_t | x_{t-1})`, który niszczy dane. |
| Proces odwrotny | „Usuwanie szumu" | Nauczony łańcuch `p_θ(x_{t-1} | x_t)`, który rekonstruuje dane. |
| Harmonogram β | „Drabina szumu" | Wariancja na krok; liniowa, cosinusowa lub sigmoidalna. |
| α̅ | „Alpha bar" | Iloczyn skumulowany `∏(1 - β)`; daje formę zamkniętą `x_t` z `x_0`. |
| Strata prosta | „MSE na szumie" | `||ε - ε_θ(x_t, t)||²`; wszystkie wariacyjne wyprowadzenia redukują się do tego. |
| ε-predykcja | „Przewiduj szum" | Wyjście to dodany szum; standardowe DDPM. |
| V-predykcja | „Przewiduj prędkość" | Wyjście to `α·ε - σ·x`; lepsze warunkowanie przez t. |
| DDPM | „Artykuł" | Ho et al. 2020; liniowe β, 1000 kroków, U-Net. |
| DDIM | „Deterministyczny próbnik" | Próbnik niemarkowski, 20-50 kroków, ten sam cel treningowy. |
| Wskazówka bez klasyfikatora | „CFG" | Mieszaj warunkowe i bezwarunkowe predykcje szumu, aby wzmocnić warunkowanie. |

## Uwaga produkcyjna: wnioskowanie dyfuzyjne to problem liczby kroków

Artykuł DDPM wykonuje T=1000 kroków odwrotnych. Nikt tego nie wysyła produkcyjnie. Każdy prawdziwy stos wnioskowania wybiera jedną z trzech strategii — i każda mapuje się czysto na produkcyjne ramowanie „skąd bierze się latencja":

1. **Szybszy próbnik, ten sam model.** DDIM (20-50 kroków), DPM-Solver++ (10-20), UniPC (8-16). Zamiennik pętli odwrotnej bez zmian; wytrenowane wagi `ε_θ` pozostają nietknięte. Redukuje latencję 20-50×.
2. **Destylacja.** Trenuj studenta, aby dopasował nauczyciela w mniejszej liczbie kroków: Progressive Distillation (2 → 1), Consistency Models (dowolne → 1-4), LCM, SDXL-Turbo, SD3-Turbo. Redukuje latencję kolejne 5-10×, wymaga przeuczenia.
3. **Cache i kompilacja.** `torch.compile(unet, mode="reduce-overhead")`, backendy dyfuzyjne TensorRT-LLM, `xformers`/SDPA attention, wagi bf16. Redukuje latencję na krok ~2×. Składa się z (1) i (2).

Dla produkcyjnego serwera dyfuzyjnego rozmowa o budżecie jest taka sama jak opisana w literaturze produkcyjnej dla LLM: latencja to `num_steps × step_cost + VAE_decode`, przepustowość to `batch_size × (num_steps × step_cost)^-1`. TTFT jest mała (jeden krok); odpowiednik TPOT to pełny czas odpowiedzi, ponieważ generowanie obrazu jest „wszystko na raz" z perspektywy użytkownika.

## Dalsze czytanie

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — artykuł o dyfuzji, wyprzedzający swoją epokę.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, mniej kroków.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — harmonogram cosinusowy, nauczona wariancja.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — wskazówka klasyfikatora.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — zunifikowana notacja, najczystszy przepis.