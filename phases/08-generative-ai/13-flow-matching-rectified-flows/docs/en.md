# Flow Matching i Rectified Flows

> Modele dyfuzyjne potrzebują 20-50 kroków próbkowania, ponieważ podążają zakrzywioną ścieżką od szumu do danych. Flow matching (Lipman i in., 2023) oraz rectified flow (Liu i in., 2022) trenowały proste ścieżki. Prostsze ścieżki oznaczają mniej kroków, co oznacza szybszą inferencję. Stable Diffusion 3, Flux.1 i AudioCraft 2 wszystkie przeszły na flow matching w 2024 roku.

**Typ:** Zbuduj to
**Języki:** Python
**Wymagania wstępne:** Faza 8 · 06 (DDPM), Faza 1 · Rachunek różniczkowy
**Czas:** ~45 minut

## Problem

Odwrotny proces DDPM to 1000-krokowy losowy marsz od `N(0, I)` z powrotem do rozkładu danych. DDIM skrócił go do 20-50 deterministycznych kroków. Chcesz mniej kroków — idealnie jeden. Przeszkodą jest to, że ODE rozwiązujące proces odwrotny jest sztywne; ścieżka jest zakrzywiona.

Gdybyś mógł tak trenować model, aby ścieżka od szumu do danych była *linią prostą*, pojedynczy krok Eulera od `t=1` do `t=0` by zadziałał. Flow matching buduje to bezpośrednio: zdefiniuj interpolację liniową od `x_1 ∼ N(0, I)` do `x_0 ∼ dane`, trenuj pole wektorowe `v_θ(x, t)`, aby dopasować jego pochodną czasową, całkuj podczas inferencji.

Rectified flow (Liu 2022) idzie dalej: iteracyjnie prostuje ścieżki za pomocą procedury reflow, która tworzy progresywnie bardziej liniowe ODE. Po dwóch iteracjach reflow, sampler 2-krokowy dorównuje jakości 50-krokowego DDPM.

## Koncepcja

![Flow matching: interpolacja liniowa między szumem a danymi](../assets/flow-matching.svg)

### Prosta ścieżka przepływu

Zdefiniuj:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

gdzie `x_0 ~ dane` i `x_1 ~ N(0, I)`. Pochodna czasowa wzdłuż tej prostej jest stała:

```
dx_t / dt = x_1 - x_0
```

Zdefiniuj neuronowe pole wektorowe `v_θ(x_t, t)` i trenuj je, aby dopasować tę pochodną:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

To jest **warunkowa funkcja straty flow matching** (Lipman 2023). Trening jest wolny od symulacji: nigdy nie rozwijaj ODE. Po prostu próbkuj `(x_0, x_1, t)` i regresuj.

### Próbkowanie

Podczas inferencji, całkuj nauczone pole wektorowe *wstecz* w czasie:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

Zacznij od `x_1 ~ N(0, I)`, krok Eulera w dół do `t=0`.

### Rectified flow (Liu 2022)

Prosta ścieżka przepływu działa, ale nauczone ścieżki *nie są faktycznie proste* — zakrzywiają się, ponieważ wiele `x_0` może być mapowanych na to samo `x_1`. Krok reflow rectified flow:

1. Trenuj model przepływu v_1 z losowymi parami.
2. Próbkuj N par `(x_1, x_0)` przez całkowanie v_1 od `x_1` do jego docelowego `x_0`.
3. Trenuj v_2 na tych sparowanych przykładach. Ponieważ pary są teraz "dopasowane przez ODE", interpolacja liniowa między nimi jest rzeczywiście bardziej płaska.
4. Powtórz.

W praktyce 2 iteracje reflow pozwalają osiągnąć niemal liniowy przepływ, umożliwiając inferencję w 2-4 krokach. SDXL-Turbo, SD3-Turbo, LCM to wszystko modele zdestylowane z flow matching.

### Dlaczego to wygrało dla obrazów w 2024 roku

Trzy powody:

1. **Trening wolny od symulacji** — brak rozwijania ODE podczas treningu, trywialne do wdrożenia.
2. **Lepsza geometria funkcji straty** — proste ścieżki mają spójny stosunek sygnału do szumu, podczas gdy funkcja straty ε DDPM ma zły SNR na brzegach harmonogramu.
3. **Szybsza inferencja** — 4-8 kroków w jakości SDXL-Turbo; 1 krok z destylacją spójności.

## Flow matching vs DDPM — dokładne połączenie

Flow matching ze ścieżką warunkową Gaussa to dyfuzja *z określonym harmonogramem szumu*. Wybierz harmonogram `x_t = α(t) x_0 + σ(t) x_1` i flow matching odzyskuje dyfuzję sformułowaną w sensie Stratonowicza z `v = α'·x_0 - σ'·x_1`. Te dwa są algebraicznie równoważne dla ścieżek Gaussa.

Co dodał flow matching: *jasność* celu (zwykła prędkość), czystsza funkcja straty i swoboda eksperymentowania z interpolantami niewymiarowymi Gaussa.

## Zbuduj to

`code/main.py` implementuje 1-wymiarowy flow matching na dwumodalnej mieszaninie Gaussa. Pole wektorowe `v_θ(x, t)` to mały MLP trenowany z prostoliniowym celem. Podczas inferencji, całkuj 1, 2, 4 i 20 kroków Eulera i porównaj jakość próbek.

### Krok 1: funkcja straty treningu

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Krok 2: wieloetapowa inferencja

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Krok 3: porównaj liczbę kroków

Oczekuj, że sampler 4-krokowy już dorówna jakości 20-krokowego — to duża sprawa dla opóźnień.

## Pułapki

- **Parametryzacja czasu.** Flow matching używa `t ∈ [0, 1]` z `t=0` przy danych, `t=1` przy szumie. DDPM używa `t ∈ [0, T]` z `t=0` przy danych, `t=T` przy szumie. Ten sam kierunek, inna skala. Artykuły stale to mylą.
- **Wybór harmonogramu.** Prosta linia rectified flow to "ten" harmonogram flow matching, ale możesz użyć cosinusowego lub logit-normalnego próbkowania t (SD3 to robi) dla lepszego pokrycia skali.
- **Koszt reflow.** Generowanie sparowanego zbioru danych dla reflow to pełny przebieg inferencji na próbkę. Rób reflow tylko gdy naprawdę potrzebujesz inferencji w 1-2 krokach.
- **Classifier-free guidance nadal obowiązuje.** Po prostu zamień ε na v w kombinacji liniowej: `v_cfg = (1+w) v_cond - w v_uncond`.

## Użyj tego

| Przypadek użycia | Stos technologiczny 2026 |
|----------|-----------|
| Tekst-do-obrazu, najlepsza jakość | Flow matching: SD3, Flux.1-dev |
| Tekst-do-obrazu, 1-4 kroki | Zdestylowany flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Inferencja w czasie rzeczywistym | Destylacja spójności z bazy flow-matched (LCM, PCM) |
| Generowanie audio | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Generowanie wideo | Flow matching zmieszany z dyfuzją (Sora, Veo, Stable Video) |
| Nauka / fizyka (trajektorie cząstek, cząsteczki) | Flow matching + equiwariantne pole wektorowe |

Za każdym razem, gdy artykuł mówi "szybszy niż dyfuzja" w 2025-2026, prawie zawsze jest to flow matching + destylacja.

## Wyślij to

Zapisz `outputs/skill-fm-tuner.md`. Skill przyjmuje specyfikację modelu w stylu dyfuzji i konwertuje ją na konfigurację treningową flow matching: wybór harmonogramu, rozkład próbkowania czasu (jednostajny / logit-normalny), optymalizator, plan reflow, docelowa liczba kroków, protokół ewaluacji.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` i porównaj MSE 1-krokowe vs 20-krokowe vs prawdziwy rozkład danych.
2. **Średnie.** Przełącz z jednostajnego próbkowania t na logit-normalne (koncentruje próbkowanie w połowie t). Czy jakość modelu się poprawia?
3. **Trudne.** Zaimplementuj jedną iterację reflow: generuj sparowane (x_0, x_1) przez całkowanie pierwszego modelu, trenuj drugi model na parach i porównaj jakość próbek 1-krokowych.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Flow matching | "Prostoliniowa dyfuzja" | Trenuj `v_θ(x, t)`, aby dopasować `x_1 - x_0` wzdłuż interpolantu. |
| Rectified flow | "Reflow" | Iteracyjna procedura prostująca nauczone przepływy. |
| Pole prędkości | "v_θ" | Wyjście modelu — kierunek ruchu `x_t`. |
| Interpolant prostoliniowy | "Ścieżka" | `x_t = (1-t)·x_0 + t·x_1`; trywialna pochodna celu. |
| Sampler Eulera | "Solver ODE 1. rzędu" | Najprostszy integrator; dobrze działa gdy ścieżki są proste. |
| Logit-normalne t | "Próbkowanie SD3" | Koncentruj próbkowanie `t` ku wartościom pośrednim, gdzie gradienty są najsilniejsze. |
| Destylacja spójności | "Sampler 1-krokowy" | Trenuj studenta, aby mapował dowolne `x_t` bezpośrednio na `x_0`. |
| CFG z prędkością | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; ten sam trik, nowa zmienna. |

## Uwaga produkcyjna: Flux.1-schnell to flow matching w najszybszej wersji

Produkcyjny sukces flow matching to Flux.1-schnell — DiT z flow matching, zdestylowany do 1-4 kroków inferencji przy zachowaniu jakości Flux-dev. Notes Nielsa "Uruchom Flux na maszynie 8GB" to referencyjna recepta wdrożeniowa: kodowanie T5 + CLIP, kwantyzowany MMDiT denoise (w 4 krokach dla schnell vs 50 dla dev), dekodowanie VAE. Księgowanie kosztów:

| Wariant | Kroki | Opóźnienie przy 1024² na L4 | Całkowite FLOPS (względne) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (surowy) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× szybciej) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

Reguła produkcyjna: **bazowy model flow-matched + destylacja = domyślny wybór 2026 dla szybkiego tekstu-do-obrazu.** Każdy główny dostawca wysyła tę kombinację: SD3-Turbo (SD3 + flow + destylacja), Flux-schnell (Flux-dev + prostowanie rectified-flow), CogView-4-Flash. Czyste bazy dyfuzyjne istnieją tylko dla legacy checkpointów.

## Dalsze czytanie

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow.
- [Lipman i in. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching.
- [Esser i in. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, rectified flow na skali.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — ogólna ramówka obejmująca FM + dyfuzję.
- [Song i in. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — destylacja 1-krokowa dyfuzji / przepływu.
- [Sauer i in. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — wariant turbo.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — flow matching w produkcji.