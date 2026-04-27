# Metody Monte Carlo — Uczenie się z kompletnych epizodów

> Programowanie dynamiczne potrzebuje modelu. Monte Carlo potrzebuje tylko epizodów. Uruchom politykę, obserwuj zwroty, uśrednij je. Najprostsza idea w RL — i ta, która odblokowuje wszystko dalej.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 9 · 01 (MDP), Faza 9 · 02 (Programowanie dynamiczne)
**Czas:** ~75 minut

## Problem

Programowanie dynamiczne jest eleganckie, ale zakłada, że możesz odpytać `P(s' | s, a)` dla każdego stanu i akcji. Prawie nic w realnym świecie tak nie działa. Robot nie może analitycznie obliczyć rozkładu nad pikselami kamery po momencie obrotowym. Algorytm cenowy nie może całkować po każdej możliwej reakcji klienta. LLM nie może wyliczyć wszystkich możliwych kontynuacji po tokenie.

Potrzebujesz metody, która wymaga tylko możliwości *próbkowania* ze środowiska. Uruchom politykę. Otrzymaj trajektorię `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Użyj jej do oszacowania wartości. To jest Monte Carlo.

Przesunięcie z DP do MC jest filozoficznie ważne: przechodzimy od *znanego modelu + dokładnego backupa* do *próbnych uruchomień + uśrednionego zwrotu*. Wariancja gwałtownie rośnie, ale zastosowanie eksploduje. Każdy algorytm RL po tej lekcji — TD, Q-learning, REINFORCE, PPO, GRPO — jest w gruncie rzeczy estymatorem Monte Carlo, czasem z nałożoną warstwą bootstrapowania.

## Koncepcja

![Monte Carlo: rollout, oblicz zwroty, uśrednij; first-visit vs every-visit MC](../assets/monte-carlo.svg)

**Główna idea w jednym zdaniu:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` gdzie `G^{(i)}(s)` to obserwowane zwroty po wizytach w `s` pod polityką `π`.

**First-visit vs every-visit MC.** Mając epizod, który odwiedza stan `s` wielokrotnie, first-visit MC liczy tylko zwrot z pierwszej wizyty; every-visit MC liczy wszystkie wizyty. Oba są nieobciążone w granicy. First-visit jest prostszy w analizie (próbki iid). Every-visit wykorzystuje więcej danych na epizod i typowo szybciej zbiega w praktyce.

**Przyrostowa średnia.** Zamiast przechowywać wszystkie zwroty, aktualizuj uruchomioną średnią:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Przeorganizuj: `V_new = V_old + α · (target - V_old)` z `α = 1/n`. Zamień `1/n` na stały rozmiar kroku `α ∈ (0, 1)` i otrzymasz niestacjonarny estymator MC, który śledzi zmiany w `π`. Ten ruch to całe przejście od MC do TD do każdego nowoczesnego algorytmu RL.

**Eksploracja teraz jest problemem.** DP dotykało każdego stanu przez enumerację. MC widzi tylko stany, które odwiedza polityka. Jeśli `π` jest deterministyczna, całe regiony przestrzeni stanów nigdy nie są próbkowane, a ich oszacowania wartości pozostają na zero na zawsze. Trzy poprawki, w kolejności historycznej:

1. **Exploring starts.** Rozpocznij każdy epizod od losowej pary (s, a). Gwarantuje pokrycie; nierealistyczne w praktyce (nie możesz "zresetować" robota w dowolny stan).
2. **ε-greedy.** Działaj zachłannie względem aktualnego Q, ale z prawdopodobieństwem `ε` wybierz losową akcję. Wszystkie pary stan-akcja są próbkowane asymptotycznie.
3. **Off-policy MC.** Zbieraj dane pod polityką behawioralną `μ`, ucz się o docelowej polityce `π` przez importance sampling. Wysoka wariancja, ale to most do metod z buforem powtórek jak DQN.

**Monte Carlo Control.** Ewaluuj → ulepszaj → ewaluuj, jak w iteracji polityki, ale ewaluacja jest próbkowana:

1. Uruchom `π`, otrzymaj epizod.
2. Zaktualizuj `Q(s, a)` z obserwowanych zwrotów.
3. Zrób `π` ε-greedy względem `Q`.
4. Powtórz.

Zbiega do `Q*` i `π*` z prawdopodobieństwem 1 przy łagodnych warunkach (każda para odwiedzana nieskończenie często, `α` spełnia Robbins-Monro).

## Zbuduj to

### Krok 1: rollout → lista (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Bez modelu, tylko `env.reset()` i `env.step(s, a)`. Ten sam interfejs co środowisko gym, ale okrojony.

### Krok 2: oblicz zwroty (wsteczne przejście)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Jeden przebieg, `O(T)`. Wsteczna rekurencja `G_t = r_{t+1} + γ G_{t+1}` unika ponownego sumowania.

### Krok 3: first-visit MC evaluation

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Trzy linie robią robotę: oznacz stan jako odwiedzony przy pierwszej wizycie, inkrementuj licznik, aktualizuj uruchomioną średnią.

### Krok 4: ε-greedy MC control (on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Krok 5: porównaj do DP gold standard

Twoje oszacowanie MC `V^π` powinno zgadzać się z wynikiem DP z Lekcji 02, gdy epizody → ∞. W praktyce: 50 000 epizodów na GridWorld 4×4 daje ci dokładność w granicach `~0.1` od odpowiedzi DP.

## Pułapki

- **Nieskończone epizody.** MC wymaga, żeby epizody *kończyły się*. Jeśli twoja polityka może zapętlać się na zawsze, ustaw limit `max_steps` i traktuj ten limit jako domyślne niepowodzenie. GridWorld z losową polityką rutynowo przekracza limit — to normalne, po prostu upewnij się, że liczysz to poprawnie.
- **Wariancja.** MC używa pełnych zwrotów. Przy długich epizodach wariancja jest ogromna — jedno nieszczęśliwe nagroda na końcu przesuwa `V(s_0)` o tę samą wartość. Metody TD (Lekcja 04) to obcinają przez bootstrapowanie.
- **Pokrycie stanów.** Chciwy MC na świeżym Q z remisy będzie próbował tylko jednej akcji. Musisz eksplorować (ε-greedy, exploring starts, UCB).
- **Niestacjonarne polityki.** Jeśli `π` się zmienia (jak w MC control), stare zwroty pochodzą z innej polityki. MC ze stałym-α to obsługuje; MC z próbkową średnią nie.
- **Off-policy importance sampling.** Wagi `π(a|s)/μ(a|s)` mnożą się przez trajektorię. Wariancja eksploduje z horyzontem. Ogranicz przez per-decision weighted IS lub przełącz na TD.

## Użyj tego

Rola metod Monte Carlo w 2026:

| Przypadek użycia | Dlaczego MC |
|------------------|-------------|
| Gry o krótkim horyzoncie (blackjack, poker) | Epizody naturalnie się kończą; zwroty są czyste. |
| Offline evaluation polityki z logów | Średnie zdyskontowanych zwrotów po zapisanych trajektoriach. |
| Monte Carlo Tree Search (AlphaZero) | MC rollouts z liści drzewa kierują selekcją. |
| Ewaluacja RL LLM | Oblicz średnią nagrodę po próbkowanych kontynuacjach dla danej polityki. |
| Estymacja baseline w PPO | Cel przewagi `A_t = G_t - V(s_t)` używa MC `G_t`. |
| Nauczanie RL | Najprostszy algorytm, który faktycznie działa — odarcie z bootstrapowania pokazuje sedno. |

Nowoczesne algorytmy deep-RL (PPO, SAC) interpolują między czystym MC (pełne zwroty) a czystym TD (jednokrokowy bootstrap) przez *n*-step returns lub GAE. Oba końce są instancjami tego samego estymatora.

## Wyślij to

Zapisz jako `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Ewaluuj politykę przez Monte Carlo rollouts i produkuj raport zbieżności z porównaniem do DP jeśli dostępne.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj first-visit MC evaluation dla polityki uniform-random na GridWorld 4×4. Uruchom 10 000 epizodów. Wykreśl `V(0,0)` jako funkcję liczby epizodów wobec odpowiedzi DP.
2. **Średnie.** Zaimplementuj ε-greedy MC control z `ε ∈ {0.01, 0.1, 0.3}`. Porównaj średni zwrot po 20 000 epizodów. Jak wygląda krzywa? Gdzie żyje kompromis bias-wariancja?
3. **Trudne.** Zaimplementuj *off-policy* MC z importance sampling: zbieraj dane pod polityką uniform-random `μ`, oszacuj `V^π` dla deterministycznej optymalnej polityki `π`. Porównaj plain IS vs per-decision IS vs weighted IS. Który ma najniższą wariancję?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Monte Carlo | "Losowe próbkowanie" | Oszacowuj oczekiwania przez uśrednianie po iid próbkach z rozkładu. |
| Return `G_t` | "Przyszła nagroda" | Suma zdyskontowanych nagród od kroku `t` do końca epizodu: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "Licz każdy stan raz" | Tylko pierwsza wizyta w epizodzie przyczynia się do oszacowania wartości. |
| Every-visit MC | "Użyj wszystkich wizyt" | Każda wizyta przyczynia się; lekko obciążony ale bardziej próbkowo efektywny. |
| ε-greedy | "Szum eksploracji" | Wybierz zachłanną akcję z prawdopodobieństwem `1-ε`; losową akcję z prawdopodobieństwem `ε`. |
| Importance sampling | "Korygowanie próbkowania z niewłaściwego rozkładu" | Zważ zwroty przez `π(a|s)/μ(a|s)` iloczyny, żeby oszacować `V^π` z danych `μ`. |
| On-policy | "Ucz się z moich własnych danych" | Polityka docelowa = polityka behawioralna. Vanilla MC, PPO, SARSA. |
| Off-policy | "Ucz się z czyichś innych danych" | Polityka docelowa ≠ polityka behawioralna. Importance-sampled MC, Q-learning, DQN. |

## Dalsze czytanie

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — kanoniczne opracowanie.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — analiza first-visit vs every-visit.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC i kontrola wariancji.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — nowoczesne estymatory IS o niskiej wariancji.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — pierwsza na dużą skalę empiryczna demonstracja MC/TD self-play zbiegającego do nadludzkiej gry; konceptualny prekursor każdej lekcji w drugiej połowie tej fazy.