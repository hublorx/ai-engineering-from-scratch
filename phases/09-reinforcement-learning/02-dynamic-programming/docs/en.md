# Programowanie dynamiczne — Iteracja polityki i Iteracja wartości

> Programowanie dynamiczne to RL z oszustwem. Znasz już funkcje przejścia i nagrody; wystarczy iterować równanie Bellmana, aż `V` lub `π` przestaną się zmieniać. To jest benchmark, do którego dąży każda metoda oparta na próbkowaniu.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 9 · 01 (MDP)
**Szacowany czas:** ~75 minut

## Problem

Masz MDP ze znanym modelem: możesz odpytywać `P(s'|s,a)` i `R(s,a,s')` dla dowolnej pary stan-akcja. Kierownik magazynu zna rozkład popytu. Gra planszowa ma deterministyczne przejścia. Gridworld to cztery linie Pythona. Masz *model*.

Model-free RL (Q-learning, PPO, REINFORCE) zostały wynalezione dla przypadku, gdy nie masz modelu — możesz tylko próbkować ze środowiska. Ale gdy go masz, istnieją szybsze, lepsze metody: programowanie dynamiczne. Bellman zaprojektował je w 1957 roku. Wciąż definiują poprawność: gdy ludzie mówią "optymalna polityka dla tego MDP," mają na myśli politykę, którą zwróci DP.

Potrzebujesz ich w 2026 roku z trzech powodów. Po pierwsze, każde tabularne środowisko w badaniach RL (GridWorld, FrozenLake, CliffWalking) jest rozwiązywane za pomocą DP, aby wygenerować politykę referencyjną. Po drugie, dokładne wartości pozwalają *debugować* metody próbkowania: jeśli Twoje oszacowanie Q-learningu dla `V*(s_0)` różni się od odpowiedzi DP o 30%, że Twój Q-learning ma błąd. Po trzecie, nowoczesne metody offline RL i planowania (MCTS, wyszukiwanie AlphaZero, model-based RL w Fazie 9 · 10) wszystkie iterują backup Bellmana przez nauczony lub dany model.

## Koncepcja

![Iteracja polityki i iteracja wartości, obok siebie](../assets/dp.svg)

**Dwa algorytmy, oba to iteracja punktu stałego na równaniu Bellmana.**

**Iteracja polityki.** Na przemian wykonuje dwa kroki, aż polityka przestanie się zmieniać.

1. *Ewaluacja:* mając politykę `π`, oblicz `V^π` poprzez wielokrotne stosowanie `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`, aż do zbieżności.
2. *Poprawa:* mając `V^π`, uczyń `π` zachłanną względem `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Zbieżność jest gwarantowana, ponieważ (a) każdy krok poprawy albo zachowuje `π` bez zmian, albo ściśle zwiększa `V^π` dla pewnego stanu; (b) przestrzeń deterministycznych polityk jest skończona. Zwykle zbiega w ~5–20 zewnętrznych iteracjach, nawet dla dużych przestrzeni stanów.

**Iteracja wartości.** Łączy ewaluację i poprawę w jednym przejściu. Stosuje *optymalności* równanie Bellmana:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Powtarzaj, aż `max_s |V_{new}(s) - V(s)| < ε`. Wyodrębnij politykę na końcu, biorąc zachłanną akcję. Ściśle szybsza na iterację —, ale bez wewnętrznej pętli ewaluacji — zazwyczaj potrzebuje więcej iteracji do zbieżności.

**Uogólniona iteracja polityki (GPI).** Ujednolucające sformułowanie. Funkcja wartości i polityka są zamknięte w dwukierunkowej pętli poprawy; każda metoda, która prowadzi obie do wzajemnej zgodności (asynchroniczna iteracja wartości, zmodyfikowana iteracja polityki, Q-learning, actor-critic, PPO), jest jej instancją.

**Dlaczego `γ < 1` ma znaczenie.** Operator Bellmana jest `γ`-kontrakcją w normie sup: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Kontrakcja implikuje unikalny punkt stały i zbieżność geometryczną. Usuń `γ < 1`, a stracisz gwarancję — potrzebujesz horyzontu skończonego lub absorbującego stanu terminalnego.

## Zbuduj to

### Krok 1: zbuduj model MDP GridWorld

Użyj tego samego 4×4 GridWorld z Lekcji 01. Dodajemy wariant stochastyczny: z prawdopodobieństwem `0.1` agent poślizguje się w losowym kierunku prostopadłym.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` zwraca listę `(s', r, p)`. To jest cały model.

### Krok 2: ewaluacja polityki

Mając politykę `π(s) = {action: prob}`, iteruj równanie Bellmana, aż `V` przestanie się zmieniać:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Krok 3: poprawa polityki

Zastąp `π` polityką zachłanną względem `V`. Jeśli `π` nie zmieniła się, zwróć — jesteśmy w optimum.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Krok 4: połącz je razem

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Typowa zbieżność na 4×4: 4–6 zewnętrznych iteracji. Zwraca `V*(0,0) ≈ -6` i politykę, która ściśle zmniejsza liczbę kroków.

### Krok 5: iteracja wartości (wersja jednopętlowa)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Ten sam punkt stały, mniej linii kodu.

## Pułapki

- **Zapominanie o obsłudze stanów terminalnych.** Jeśli zastosujesz Bellmana do stanu absorbującego, nadal wybiera "najlepszą akcję", która niczego nie zmienia. Chroń się: `if s == terminal: V[s] = 0`.
- **Norma sup vs zbieżność L2.** Używaj `max |V_new - V|`, nie średniej. Gwarancja teoretyczna dotyczy normy sup.
- **Aktualizacje w miejscu vs synchroniczne.** Aktualizacja `V[s]` w miejscu (Gauss-Seidel) zbiega szybciej niż osobny słownik `V_new` (Jacobi). Kod produkcyjny używa aktualizacji w miejscu.
- **Remisy polityki.** Jeśli dwie akcje mają równą wartość Q, `argmax` może zerwać remisy różnie w każdej iteracji, powodując oscylację sprawdzenia "polityka stabilna". Używaj stabilnego rozwiązywania remisów (pierwsza akcja w ustalonej kolejności).
- **Eksplozja przestrzeni stanów.** DP to `O(|S| · |A|)` na przejście. Działa do ~10⁷ stanów. Powyżej tego potrzebujesz aproksymacji funkcji (Faza 9 · 05 i dalej).

## Zastosuj to

W 2026 roku DP jest polityką bazową poprawności i wewnętrzną pętlą planistów:

| Przypadek użycia | Metoda |
|----------|--------|
| Rozwiąż małe tabularne MDP dokładnie | Iteracja wartości (prostsza) lub iteracja polityki (mniej zewnętrznych kroków) |
| Zweryfikuj implementację Q-learningu / PPO | Porównaj z DP-optymalnym V* na środowisku testowym |
| Model-based RL (Faza 9 · 10) | Backup Bellmana na nauczonym modelu przejścia |
| Planowanie w AlphaZero / MuZero | Monte Carlo Tree Search = asynchroniczny backup Bellmana |
| Offline RL (CQL, IQL) | Conservative Q-iteration — DP z karą za akcje OOD |

Za każdym razem, gdy ktoś mówi "optymalna funkcja wartości," ma na myśli "punkt stały DP." Gdy widzisz `V*` lub `Q*` w artykule, wyobraź sobie tę pętlę.

## Wyślij to

Zapisz jako `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Rozwiąż małe tabularne MDP dokładnie za pomocą iteracji polityki lub iteracji wartości. Zgłoś zachowanie zbieżności.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. Jak ten punkt odniesienia będzie używany do debugowania/oceny metod opartych na próbkowaniu.

Odmawiaj uruchomienia DP dla przestrzeni stanów > 10⁷. Odmawiaj twierdzenia o zbieżności bez sprawdzenia normy sup. Oznaczaj każde γ ≥ 1 w zadaniu o horyzoncie nieskończonym jako naruszenie gwarancji.
```

## Ćwiczenia

1. **Łatwe.** Uruchom iterację wartości na 4×4 GridWorld z `γ ∈ {0.9, 0.99}`. Ile przejść do `max |ΔV| < 1e-6`? Wydrukuj `V*` jako siatkę 4×4.
2. **Średnie.** Porównaj iterację polityki vs iterację wartości na *stochastycznym* GridWorld (prawdopodobieństwo poślizgu `0.1`). Policz: przejścia, czas zegara ściennego, końcowe `V*(0,0)`. Który zbiega szybciej w iteracjach? W czasie zegara ściennego?
3. **Trudne.** Zbuduj zmodyfikowaną iterację polityki: w kroku ewaluacji wykonaj tylko `k` przejść zamiast do zbieżności. Wykreśl błąd `V*(0,0)` vs `k` dla `k ∈ {1, 2, 5, 10, 50}`. Co krzywa mówi Ci o kompromisie ewaluacji/poprawy?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Iteracja polityki | "Algorytm DP" | Naprzemienna ewaluacja (`V^π`) i poprawa (zachłanna `π` względem `V^π`) aż polityka przestanie się zmieniać. |
| Iteracja wartości | "Szybsze DP" | Backup optymalności Bellmana zastosowany w jednym przejściu; zbiega do `V*` geometrycznie. |
| Operator Bellmana | "Rekursja" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; `γ`-kontrakcja w normie sup. |
| Kontrakcja | "Dlaczego DP zbiega" | Każdy operator `T` z `||T x - T y|| ≤ γ ||x - y||` ma unikalny punkt stały. |
| GPI | "Wszystko jest DP" | Generalized Policy Iteration: każda metoda prowadząca `V` i `π` do wzajemnej zgodności. |
| Aktualizacja synchroniczna | "Styl Jacobi" | Używaj starego `V` przez całe przejście; czysto analizowalne, ale wolniejsze. |
| Aktualizacja w miejscu | "Styl Gauss-Seidel" | Używaj `V` podczas gdy jest aktualizowane; zbiega szybciej w praktyce. |

---

**Podsumowanie wprowadzonych poprawek:**

| # | Lokalizacja | Przed | Po |
|---|--------------|-------|-----|
| 1 | Linia 16 | "o 30%, Twoój Q-learning" | "o 30%, że Twój Q-learning" |
| 2 | Linia ~38 | "(a) ..., (b)" | "(a) ...; (b)" |
| 3 | Linia ~40 | "stanów nawet" | "stanów, nawet" |
| 4 | Linia ~42 | "— ale zazwyczaj" | "—, ale zazwyczaj" |
| 5 | Linia ~85 | "się, zwróć" | "się, zwróć" (przecinek był) |
| 6 | Frontmatter | "description: Solve a small..." | "description: Rozwiąż małe tabularne..." |