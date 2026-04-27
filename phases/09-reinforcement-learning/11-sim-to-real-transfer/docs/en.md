# Sim-to-Real Transfer

> Polityka wytrenowana w symulatorze, która zawodzi na sprzęcie, to polityka, która zapamiętała symulator. Domain randomization, domain adaptation i system identification to trzy narzędzia pozwalające przeprowadzić nauczone sterowniki przez przepaść rzeczywistości.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 9 · 08 (PPO), Faza 2 · 10 (Bias/Wariancja)
**Czas:** ~45 minut

## Problem

Trenowanie prawdziwego robota jest wolne, niebezpieczne i kosztowne. Biped potrzebuje milionów epizodów treningowych, żeby nauczyć się chodzić; prawdziwy biped, który choć raz przewróci się, niszczy sprzęt. Symulacja daje nieograniczone reseci, deterministyczną powtarzalność, równoległe środowiska i brak uszkodzeń fizycznych.

Ale symulatory są błędne. Łożyska mają większe tarcie niż modele MuJoCo. Kamery mają dystorsję soczewki, której symulator nie uwzględnia. Silniki mają opóźnienia, luz i nasycenie, które 99% modeli symulacyjnych pomija. Wiatr, kurz i zmienne oświetlenie sabotują politykę wytrenowaną na sterylnym renderingu. **Przepaść rzeczywistości** — systematyczna różnica między dystrybucją symulacji a dystrybucją rzeczywistą — to centralny problem wdrożonego RL dla robotyki.

Potrzebujesz polityki, która jest **odporna na sym-to-real distribution shift**. Trzy historyczne podejścia: randomizacja symulatora (domain randomization), adaptacja polityki z niewielką ilością danych rzeczywistych (domain adaptation / fine-tuning), lub identyfikacja parametrów rzeczywistego systemu i dopasowanie ich (system identification). W 2026 dominujący przepis łączy wszystkie trzy z masywną symulacją równoległą (Isaac Sim, Isaac Lab, Mujoco MJX na GPU).

## Koncepcja

![Trzy reżimy sim-to-real: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR).** Tobin et al. 2017, Peng et al. 2018. Podczas treningu randomizuj każdy parametr symulacji, który może różnić się na prawdziwym robocie: masy, współczynniki tarcia, wzmocnienia PD silników, szum czujników, pozycję kamery, oświetlenie, tekstury, modele kontaktu. Polityka uczy się warunkowego rozkładu "w której symulacji jest dziś" i uogólnia na całym zakresie. Jeśli prawdziwy robot mieści się w przestrzeni treningowej envelope, polityka działa.

- **Zaleta:** niepotrzebne dane rzeczywiste. Jeden przepis, wiele robotów.
- **Wada:** nadmiernie zrandomizowany trening produkuje "uniwersalną", ale zbyt ostrożną politykę. Zbyt dużo szumu ≈ zbyt duża regularizacja.

**System Identification (SI).** Dopasuj parametry symulatora do danych ze świata rzeczywistego przed treningiem. Jeśli możesz zmierzyć tarcie w przegubie ramienia na prawdziwym robocie, wstaw to do symulacji. Następnie trenuj politykę, która oczekuje tych wartości. Wymaga dostępu do prawdziwego systemu, ale bezpośrednio zmniejsza przepaść rzeczywistości.

- **Zaleta:** precyzyjny, niskoszumowy cel treningowy.
- **Wada:** residualny błąd modelu jest niewidoczny dla polityki; małe niezidentyfikowane efekty (np. martwa strefa silnika) nadal psują wdrożenie.

**Domain Adaptation.** Trenuj w symulacji, dostrajaj z niewielką ilością danych rzeczywistych. Dwa warianty:

- **Real2Sim2Real:** naucz się rezydualnego symulatora `f(s, a, z) - f_sim(s, a)` używając prawdziwych rollouts, trenuj w skorygowanym symulatorze. Zamyka przepaść bez dużych danych rzeczywistych.
- **Observation adaptation:** trenuj politykę, która mapuje rzeczywiste obs → sym-like obs przez nauczony extractor cech (np. GAN pixel-to-pixel). Kontroler pozostaje w symulacji.

**Privileged learning / teacher-student.** Miki et al. 2022 (ANYmal quadruped). Trenuj *nauczyciela* w symulacji, który ma dostęp do uprzywilejowanych informacji (prawdziwe tarcie, wysokość terenu, dryft IMU). Distiluj *studenta*, który widzi tylko rzeczywiste obserwacje czujników. Student uczy się wnioskować uprzywilejowane cechy z historii, jest odporny na zmiany parametrów fizycznych.

**Masowa symulacja równoległa.** 2024–2026. Isaac Lab, Mujoco MJX, Brax wszystkie uruchamiają tysiące równoległych robotów na jednym GPU. PPO z 4,096 równoległymi humanoidami zbiera lata doświadczenia w godzinach. "Przepaść rzeczywistości" kurczy się wraz z poszerzaniem dystrybucji treningowej; DR staje się niemal darmowe, gdy każde z tych 4,096 envs ma różne zrandomizowane parametry.

**Przepis na rzeczywistość 2026 (przykład chodzenia czworonoga):**

1. Masowa symulacja równoległa z domain-randomized grawitacją, tarciem, wzmocnieniami silników, ładunkiem.
2. Polityka-nauczyciel trenowana z uprzywilejowanymi informacjami (mapa terenu, prawdziwa prędkość ciała).
3. Polityka-student distilowana z nauczyciela używając tylko propriocepcji (enkodery przegubów nóg).
4. Opcjonalna adaptacja obserwacji przez autoencoder na prawdziwym IMU.
5. Wdrożenie. Zero-shot na 10+ środowiskach. Jeśli zawodzi, wykonaj minuty fine-tuningu w świecie rzeczywistym z PPO z ograniczeniami bezpieczeństwa.

## Zbuduj to

Kod tej lekcji to malutka demonstracja domain randomization na GridWorld z *szumnymi* przejściami. Trenujemy politykę, która doświadcza zrandomizowanych prawdopodobieństw poślizgu w "sym" i ewaluujemy na "real" z poziomem poślizgu, którego nigdy nie widziała podczas treningu. Kształt mapuje bezpośrednio na transfer MuJoCo-to-hardware.

### Krok 1: sparametryzowany sym

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` to parametr, który symulator udostępnia. W prawdziwej robotyce może to być tarcie, masa, wzmocnienie silnika — cokolwiek, co różni się między sym a real.

### Krok 2: trenuj z DR

Na początku każdego epizodu próbkuj `slip ~ Uniform[0.0, 0.4]`. Trenuj PPO / Q-learning / cokolwiek. Rób to przez wiele epizodów.

### Krok 3: ewaluuj zero-shot na "rzeczywistych" poślizgach

Ewaluuj na `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Pierwsze cztery są w zakresie treningowym; `0.5` i `0.7` są poza. Polityka trenowana z DR powinna pozostać blisko optymalna w zakresie i graceful degradować poza nim. Polityka trenowana ze stałym slip będzie krucha poza jej treningowym slip.

### Krok 4: porównaj z wąskim treningiem

Trenuj drugą politykę tylko z `slip = 0.0`. Ewaluuj na tym samym zakresie slip. Powinieneś zobaczyć katastrofalny spadek, gdy tylko rzeczywisty slip > 0.

## Pułapki

- **Zbyt duża randomizacja.** Trenuj na `slip ∈ [0, 0.9]`, a twoja polityka jest tak awersyjna do ryzyka, że nigdy nie próbuje optymalnej ścieżki. Dopasuj *oczekiwaną* dystrybucję rzeczywistą, nie "wszystko może się wydarzyć."
- **Zbyt mała randomizacja.** Trenuj na cienkim wycinku, a polityka nie może w ogóle uogólniać. Użyj adaptive curriculum (Automatic Domain Randomization), która poszerza dystrybucję w miarę poprawy polityki.
- **Błędnie zidentyfikowana przestrzeń parametrów.** Randomizuj złą rzecz (odcień kamery, gdy rzeczywista przepaść to opóźnienie silnika), a DR nie pomoże. Profiluj najpierw prawdziwego robota.
- **Przeciek uprzywilejowanych informacji.** Nauczyciel, który używa globalnego stanu do akcji, a nie tylko obserwacji, może wyprodukować studenta, który nie może nadążyć. Upewnij się, że polityka nauczyciela jest realizowalna przez studenta przy danym hstory obserwacji.
- **Sim-to-sim transfer failure.** Jeśli twoja polityka nie jest odporna na trudniejszy wariant sym, nie będzie odporna na świat rzeczywisty też. Zawsze testuj na held-out wariancie sym przed wdrożeniem.
- **Brak bezpiecznego zakresu w świecie rzeczywistym.** Polityka, która działa w sym i "działa w real" bez niskopoziomowej tarczy bezpieczeństwa, nadal może złamać sprzęt. Dodaj limity szybkości, limity momentu obrotowego, limity przegubów w nienauczonym kontrolerze.

## Użyj tego

Stos sim-to-real 2026:

| Domena | Stos |
|--------|------|
| Lokomocja krocząca (ANYmal, Spot, humanoid) | Isaac Lab + DR + uprzywilejowany teacher / student |
| Manipulacja (chwytne dłonie, pick-and-place) | Isaac Lab + DR + DR-GAN for vision |
| Autonomiczna jazda | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| Wyścigi dronów | RotorS / Flightmare + DR + online adaptation |
| Manipulacja palcami / in-hand | OpenAI Dactyl (DR na bezprecedensową skalę) |
| Ramiona przemysłowe | MuJoCo-Warp + SI + small real fine-tune |

Dla sterowania na wszystkich skalach, workflow jest spójny: dopasuj sym jak najlepiej, randomizuj to, czego nie możesz dopasować, trenuj ogromne polityki, distilluj, wdrażaj z tarczą bezpieczeństwa.

## Wyślij to

Zapisz jako `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Zaplanuj pipeline transferu sim-to-real dla danej platformy robota + zadania, obejmujący DR, SI i bezpieczeństwo.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Ćwiczenia

1. **Łatwe.** Trenuj agenta Q-learning na fixed-slip GridWorld (slip=0.0). Ewaluuj na slip ∈ {0.0, 0.1, 0.3, 0.5}. Wykreśl return vs slip.
2. **Średnie.** Trenuj agenta DR Q-learning próbkując `slip ~ Uniform[0, 0.3]`. Ewaluuj na tym samym zakresie. Ile DR daje przy slip=0.5 (out-of-distribution)?
3. **Trudne.** Zaimplementuj curriculum: zacznij od slip=0.0, poszerzaj zakres DR za każdym razem, gdy polityka osiągnie 90% optymalnej. Zmierz całkowitą liczbę kroków środowiska do osiągnięcia slip=0.3 zero-shot vs. fixed DR baseline.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Reality gap | "Różnica sim-to-real" | Distribution shift między fizyką/sensoringiem treningowym a wdrożeniowym. |
| Domain randomization (DR) | "Trenuj na różnych sym" | Randomizuj parametry sym podczas treningu, żeby polityka uogólniała. |
| System identification (SI) | "Zmierz real i dopasuj sym" | Oszacuj rzeczywiste parametry fizyczne; ustaw sym, żeby pasował. |
| Domain adaptation | "Fine-tune na danych real" | Mały fine-tune w świecie rzeczywistym po treningu w sym; może adaptować obs lub dynamikę. |
| Privileged info | "Prawda dla nauczyciela" | Informacje, które ma tylko sym; student musi je wnioskować z historii obs. |
| Teacher/student | "Distill privileged -> observable" | Nauczyciel trenowany ze skrótami; student uczy się naśladować bez nich. |
| ADR | "Automatic Domain Randomization" | Curriculum, które poszerza zakresy DR w miarę poprawy polityki. |
| Real2Sim | "Zamknij przepaść danymi real" | Naucz się residuum, żeby sym naśladował rzeczywiste rollouts. |

## Dalsza lektura

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — oryginalny artykuł DR (wizja dla robotyki).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR dla dynamiki, lokomocja czworonoga.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, ADR na skalę.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — teacher-student dla ANYmal.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — masowa symulacja równoległa, która napędza wdrożenia 2025–2026.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — metoda curriculum ADR.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — ramy Dyna (używaj modelu do planowania + rollouts), które stanowią podstawę nowoczesnych pipeline'ów sim-to-real.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — taksonomia metod sim-to-real z wynikami benchmarków.