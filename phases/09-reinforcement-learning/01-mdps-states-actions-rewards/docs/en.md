# MDP-y, Stany, Akcje i Nagrody

> Proces decyzyjny Markova to pięć elementów: stany, akcje, przejścia, nagrody, dyskonto. Wszystko w RL — Q-learning, PPO, DPO, GRPO — optymalizuje się nad tym kształtem. Naucz się tego raz, a przeczytasz resztę reinforcement learning za darmo.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 1 · 06 (Prawdopodobieństwo i rozkłady), Faza 2 · 01 (Taksonomia ML)
**Szacowany czas:** ~45 minut

## Problem

Piszesz bota do szachów. Albo planistę zapasów. Albo agenta tradingowego. Albo pętlę PPO, która trenuje model reasoning. Cztery różne domeny, jeden zaskakujący fakt: wszystkie cztery sprowadzają się do tego samego obiektu matematycznego.

Supervised learning daje ci pary `(x, y)` i prosi o dopasowanie funkcji. Reinforcement learning nie daje żadnych etykiet — tylko strumień stanów, akcje które wykonałeś, i skalar nagrodę. Czy ruch wygrał partię? Czy decyzja o uzupełnieniu zapasów zaoszczędziła pieniądze? Czy trade przyniósł zysk? Czy token, który właśnie wygenerował LLM dla modelu w języku angielskim, prowadził do wyższej nagrody od sędziego?

Nie możesz się uczyć z tego strumienia, dopóki go nie sformalizujesz. „Co widziałem", „co zrobiłem", „co stało się dalej", „jak dobre to było" — każde z nich musi stać się obiektem, o którym możesz myśleć. Ta formalizacja to proces decyzyjny Markova. Każdy algorytm RL w tej fazie, w tym pętle RLHF i GRPO na końcu, optymalizuje nad tym kształtem.

## Koncepcja

![Proces decyzyjny Markova: stany, akcje, przejścia, nagrody, dyskonto](../assets/mdp.svg)

**Pięć obiektów.**

- **Stany** `S`. Wszystko, czego agent potrzebuje do podjęcia decyzji. W GridWorld, komórka. W szachach, szachownica, a w LLM, okno kontekstowe plus dowolna pamięć.
- **Akcje** `A`. Wybory. Ruch góra/dół/lewo/prawo. Wykonaj ruch. Emituj token.
- **Przejścia** `P(s' | s, a)`. Mając stan `s` i akcję `a`, rozkład nad następnym stanem. Deterministyczne w szachach, stochastyczne w zarządzaniu zapasami, a prawie-deterministyczne w dekodowaniu LLM.
- **Nagrody** `R(s, a, s')`. Sygnał skalarny. Wygrana = +1, przegrana = -1. Przychód minus koszt. Wyraz ilorazu logarytmicznego prawdopodobieństwa w GRPO.
- **Dyskonto** `γ ∈ [0, 1)`. Ile przyszła nagroda liczy się w porównaniu z obecną. `γ = 0.99` kupuje horyzont ~100 kroków; `γ = 0.9` kupuje ~10.

**Własność Markova** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. Przyszłość zależy tylko od obecnego stanu. Jeśli tak nie jest, reprezentacja stanu jest niekompletna — to nie porażka metody, to porażka stanu.

**Polityki i zwroty.** Polityka `π(a | s)` mapuje stany na rozkłady akcji. Zwrot `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` to zdyskontowana suma przyszłych nagród. Wartość `V^π(s) = E[G_t | s_t = s]` to oczekiwany zwrot startując z `s` pod polityką `π`. Q-wartość `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` to oczekiwany zwrot startując z konkretną akcją. Każdy algorytm RL szacuje jedną z tych dwóch, potem poprawia π, odpowiednio.

**Równania Bellmana.** Równania punktu stałego, których wszystko w tej fazie używa:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Te równania rozbijają oczekiwany zwrot na "nagrodę tego kroku" plus "zdyskontowaną wartość miejsca, gdzie lądujesz." Rekurencyjne. Każdy algorytm w Fazie 9 albo iteruje to równanie do zbieżności (dynamic programming), albo próbkuje z niego (Monte Carlo), albo bootstrapuje o jeden krok (temporal difference).

## Zbuduj to

### Krok 1: maleńki deterministyczny MDP

GridWorld 4×4. Agent startuje w lewym górnym rogu, stan terminalny w prawym dolnym rogu, nagroda -1 za krok, akcje `{up, down, left, right}`. Zobacz `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Pięć linii. To jest całe środowisko. Deterministyczne przejścia, stała kara za krok, absorbujący stan terminalny.

### Krok 2: roll out polityki

Polityka to funkcja ze stanu do rozkładu akcji. Najprostsza: równomiernie losowa.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Uruchom losową politykę 1000 razy. Średni zwrot to około -60 do -80 dla tej planszy 4×4. Optymalny zwrot to -6 (prosta ścieżka w dół-prawo). Zamykanie tej luki to wszystko w Fazie 9.

### Krok 3: oblicz `V^π` dokładnie przez równanie Bellmana

Dla małych MDP równanie Bellmana to układ równań liniowych. Wylicz stany, zastosuj oczekiwanie, iteruj aż wartości przestaną się zmieniać.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

To jest iteracyjna ewaluacja polityki. To jest pierwszy algorytm w Sutton & Barto i teoretyczny fundament każdej metody RL, która następuje.

### Krok 4: `γ` to hiperparametr z fizycznym znaczeniem

Efektywny horyzont to w przybliżeniu `1 / (1 - γ)`. `γ = 0.9` → 10 kroków. `γ = 0.99` → 100 kroków. `γ = 0.999` → 1000 kroków.

Zbyt niska wartość, a agent działa krótkowzrocznie. Zbyt wysoka, a przypisanie zasług staje się nieznośne, bo wiele wczesnych kroków dzieli odpowiedzialność za odległą przyszłą nagrodę. LLM RLHF typowo używa `γ = 1`, bo epizody są krótkie i ograniczone. Zadania control używają `0.95–0.99`. Gry strategiczne z dalekim horyzontem używają `0.999`.

## Pułapki

- **Nie-Markowski stan.** Jeśli potrzebujesz trzech ostatnich obserwacji do decyzji, "stan" to nie tylko obecna obserwacja. Fix: stackuj ramki (DQN na Atari stackuje 4) albo użyj rekurencyjnego stanu (LSTM/GRU nad obserwacjami).
- **Rzadkie nagrody.** Nagrody tylko za wygraną sprawiają, że nauka jest prawie niemożliwa w dużych przestrzeniach stanów. Kształtuj nagrody (sygnał pośredni) albo bootstrapuj z imitacji (Faza 9 · 09).
- **Hackowanie nagród.** Optymalizacja proxy reward często produkuje patologiczne zachowanie. Agent wyścigów łodzi od OpenAI kręcił się w kółko zbierając powerupy wiecznie, zamiast finiszować wyścig. Zawsze definiuj reward z docelowego wyniku, nie z proxy.
- **Błędne dyskonto.** `γ = 1` na zadaniu nieskończonego horyzontu sprawia, że każda wartość jest nieskończona. Zawsze capuj albo skończonym horyzontem albo `γ < 1`.
- **Skala nagrody.** Nagrody {+100, -100} vs {+1, -1} dają identyczne optymalne polityki, ale radykalnie różne wielkości gradientu. Normalizuj do `[-1, 1]` przed włożeniem do PPO/DQN.

## Użyj tego

Stack 2026 redukuje każdy pipeline RL do MDP przed dotknięciem kodu:

| Sytuacja | Stan | Akcja | Nagroda | γ |
|-----------|------|--------|--------|---|
| Control (locomotion, manipulation) | Kąty + prędkości stawów | Ciągłe momenty | Kształtowana pod zadanie | 0.99 |
| Gry (szachy, Go, poker) | Plansza + historia | Legalny ruch | Wygrana=+1 / przegrana=-1 | 1.0 (skończony) |
| Inventory / pricing | Zapasy + popyt | Ilość zamówienia | Przychód - koszt | 0.95 |
| RLHF dla LLM | Tokeny kontekstu | Następny token | Wynik reward model na końcu | 1.0 (epizod ~200 tokenów) |
| GRPO dla reasoning | Prompt + częściowa odpowiedź | Następny token | Weryfikator 0/1 na końcu | 1.0 |

Napisz pięć krotek przed napisaniem jakiejkolwiek pętli trenowania. Większość raportów "RL nie działa" sięga do MDP formulation, który był zepsuty na papierze.

## Wyślij to

Zapisz jako `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj GridWorld 4×4 i rollout z losową polityką w `code/main.py`. Uruchom 10,000 epizodów. Raportuj średnią i std zwrotu. Porównaj do optymalnego zwrotu (-6).
2. **Średnie.** Uruchom `policy_evaluation` z `γ ∈ {0.5, 0.9, 0.99}` dla polityki równomiernie losowej. Wydrukuj `V` jako siatkę 4×4 dla każdej. Wyjaśnij, dlaczego wartości stanów blisko terminala rosną szybciej z większym `γ`.
3. **Trudne.** Zrób GridWorld stochastycznym: każda akcja slipuje do sąsiedniego kierunku z prawdopodobieństwem `p = 0.1`. Przeewaluuj politykę równomiernie losową. Czy `V[start]` robi się lepsze czy gorsze? Dlaczego?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| MDP | "Ustawienie reinforcement learning" | Krotka `(S, A, P, R, γ)` spełniająca własność Markova. |
| Stan | "Co agent widzi" | Dostateczna statystyka dla przyszłych dynamik pod wybraną klasą polityk. |
| Polityka | "Zachowanie agenta" | Warunkowy rozkład `π(a | s)` albo deterministyczne mapowanie `s → a`. |
| Zwrot | "Całkowita nagroda" | Zdyskontowana suma `Σ γ^t r_t` od obecnego kroku. |
| Wartość | "Jak dobry jest stan" | Oczekiwany zwrot pod `π` startując z `s`. |
| Q-wartość | "Jak dobra jest akcja" | Oczekiwany zwrot pod `π` startując z `s` z pierwszą akcją `a`. |
| Równanie Bellmana | "Rekurencja dynamic programming" | Dekompozycja punktu stałego wartości / Q na jedno-krokową nagrodę plus zdyskontowaną wartość następnika. |
| Dyskonto `γ` | "Przyszłość vs obecność" | Geometryczna waga na odległą przyszłą nagrodę; efektywny horyzont `~1/(1-γ)`. |

## Dalsze czytanie

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — podręcznik. Rozdz. 3 obejmuje MDP i równania Bellmana; Rozdz. 1 motywuje hipotezę nagrody, która leży u podstaw każdej kolejnej lekcji.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — źródło równania Bellmana.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — zwięzły primer MDP z głębokiego kąta RL.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — opsresearchowa referencja na MDP i dokładne metody rozwiązywania.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — najczystsza derivacja MDP jako specjalizacja dynamic programming.