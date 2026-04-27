# Temporal Difference — Q-Learning i SARSA

> Monte Carlo czeka, aż epizod się zakończy. TD aktualizuje po każdym kroku, bootstrapując następny oszacowany szacunek wartości. Q-learning jest off-policy i optymistyczny; SARSA jest on-policy i ostrożna. Oba to jedna linijka kodu. Oba stanowią fundament każdej deep-RL w tej fazie.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Szacowany czas:** ~75 minut

## Problem

Monte Carlo działa, ale ma dwa kosztowne wymagania. Potrzebuje epizodów, które się kończą, i aktualizuje dopiero po finale return. Jeśli Twój epizod ma 1,000 kroków, MC czeka 1,000 kroków zanim zaktualizuje cokolwiek. Ma wysoką wariancję, niskie obciążenie i jest wolna w praktyce.

Dynamic programming ma profil przeciwny — zerowa wariancja, bootstrap backupi — ale wymaga znanego modelu.

Temporal difference (TD) learning rozbija różnicę na pół. Z jednego przejścia `(s, a, r, s')`, tworzymy jednokrokowy cel `r + γ V(s')` i popychamy `V(s)` w jego kierunku. Bez modelu. Bez pełnych epizodów. Obciążenie z użycia przybliżonego `V` po prawej stronie, ale dramatycznie niższa wariancja niż MC i aktualizacje online od pierwszego kroku.

To jest punkt zwrotny, na którym opiera się cała współczesna RL — DQN, A2C, PPO, SAC. Reszta Phase 9 to warstwy przybliżenia funkcji i triki zbudowane na szczycie jednokrokowej aktualizacji TD, którą napiszesz w tej lekcji.

## Koncepcja

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**Aktualizacja TD(0) dla V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

Wyrażenie w nawiasie to TD error `δ = r + γ V(s') - V(s)`. To online odpowiednik `G_t - V(s_t)` w MC. Zbieżność wymaga `α` spełniającego Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) i nieskończenie częstych odwiedzin wszystkich stanów.

**Q-learning.** Off-policy TD metoda dla kontroli:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` zakłada, że *greedy* polityka będzie podążana od `s'` w dal, niezależnie od tego, jaką akcję agent faktycznie wykonuje. Ta decyplacja sprawia, że Q-learning uczy się `Q*` podczas gdy agent eksploruje przez ε-greedy. Mnih et al. (2015) przekonwertowali to na deep Q-learning na Atari (Lesson 05).

**SARSA.** On-policy TD metoda:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

Nazwa to krotka `(s, a, r, s', a')`. SARSA używa akcji `a'`, którą agent faktycznie wykonuje następnie, nie greedy `argmax`. Zbiega do `Q^π` dla jakiejkolwiek ε-greedy `π` jest uruchomiona, co w granicy `ε → 0` staje się `Q*`.

**Różnica na cliff-walking.** Na klasycznym zadaniu cliff-walking (spadek z klifu = nagroda -100), Q-learning uczy się optymalnej ścieżki wzdłuż krawędzi klifu, ale czasami ponosi karę podczas eksploracji. SARSA uczy się bezpieczniejszej ścieżki oddalonej o jeden krok od klifu, ponieważ uwzględnia szum eksploracyjny w swojej wartości Q. Z treningiem oba osiągają optimum przy `ε → 0`. W praktyce to ma znaczenie: kiedy eksploracja faktycznie zachodzi podczas deploymentu, zachowanie SARSA jest bardziej konserwatywne.

**Expected SARSA.** Zastąp `Q(s', a')` jej wartością oczekiwaną pod `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Niższa wariancja niż SARSA (brak próbki `a'`), ten sam on-policy cel. Często domyślny w nowoczesnych podręcznikach.

**n-step TD i TD(λ).** Interpoluj między TD(0) a MC czekając `n` kroków przed bootstrapowaniem. `n=1` to TD, `n=∞` to MC. TD(λ) uśrednia po wszystkich `n` z wagami geometrycznymi `(1-λ)λ^{n-1}`. Większość deep-RL używa `n` między 3 a 20.

## Zbuduj to

### Krok 1: SARSA na ε-greedy policy

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Osiem linijek. *Jedyna* różnica od Q-learning to linijka target.

### Krok 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` decypluje target od zachowania. Ten jeden symbol to różnica między on-policy a off-policy.

### Krok 3: krzywe uczenia

Śledź średni return na 100 epizodów. Q-learning zbiega szybciej na prostym deterministycznym GridWorld; SARSA jest bardziej konserwatywna na cliff-walking. Na 4×4 GridWorld w `code/main.py`, oba są blisko-optymalne po ~2,000 epizodach z `α=0.1, ε=0.1`.

### Krok 4: porównaj do DP truth

Uruchom value iteration (Lesson 02) żeby dostać `Q*`. Sprawdź `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. Zdrowy tabularyczny agent TD ląduje w `~0.5` na 4×4 GridWorld po 10,000 epizodach.

## Pułapki

- **Początkowe wartości Q mają znaczenie.** Optymistyczna inicjalizacja (`Q = 0` dla zadania z ujemną nagrodą) zachęca do eksploracji. Pesymistyczna inicjalizacja może uwięzić greedy policy na zawsze.
- **Harmonogram α.** Stałe `α` jest w porządku dla niestacjonarnych problemów. Zmniejszające się `α_n = 1/n` daje zbieżność w teorii, ale jest za wolne w praktyce — przypnij `α` w `[0.05, 0.3]` i monitoruj krzywą uczenia.
- **Harmonogram ε.** Zacznij wysoko (`ε=1.0`), zmniejszaj do `ε=0.05`. "GLIE" (greedy in the limit with infinite exploration) to warunek zbieżności.
- **Max bias w Q-learning.** Operator `max` jest obciążony w górę, gdy `Q` jest zaszumiony. Prowadzi do przeceniania — Hasselt's Double Q-learning (używane przez DDQN w Lesson 05) naprawia to dwoma tabelami Q.
- **Niezakończone epizody.** TD może uczyć się bez terminali, ale musisz albo ograniczyć kroki, albo obsłużyć bootstrap poprawnie na granicy. Standard: traktuj limit jako nie-terminalny, kontynuuj bootstrapowanie.
- **Hashowanie stanu.** Jeśli stany są krotkami/tensorami, używaj hashowalnego klucza (tuple, nie list; tuple zaokrąglonych floatów, nie surowych).

## Użyj tego

Landscape TD w 2026:

| Zadanie | Metoda | Powód |
|---------|--------|-------|
| Małe tabularyczne środowiska | Q-learning | Uczy się optymalnej polityki bezpośrednio. |
| On-policy safety-critical | SARSA / Expected SARSA | Konserwatywna podczas eksploracji. |
| Wysoko-wymiarowe stany | DQN (Phase 9 · 05) | Neural-net Q-function z replay i target net. |
| Ciągłe akcje | SAC / TD3 (Phase 9 · 07) | TD update na Q-network; policy net emituje akcje. |
| LLM RL (reward-model-based) | PPO / GRPO (Phase 9 · 08, 12) | Actor-critic z TD-style advantage przez GAE. |
| Offline RL | CQL / IQL (Phase 9 · 08) | Q-learning z konserwatywną regularyzacją. |

Dziewięćdziesiąt procent "RL", którą czytasz w artykułach z 2026, to jakaś elaboracja Q-learning lub SARSA. Zrozum tabularyczną aktualizację w palcach, zanim będziesz czytać głębiej.

## Wyślij to

Zapisz jako `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Wybierz między Q-learning, SARSA, Expected SARSA dla tabularycznego lub małego-feature RL zadania.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj Q-learning i SARSA na 4×4 GridWorld. Wyrysuj krzywe uczenia (średni return na 100 epizodów) dla 2,000 epizodów. Kto zbiega szybciej?
2. **Średnie.** Zbuduj środowisko cliff-walking (4×12, ostatni wiersz to klif z nagrodą -100 i resetem do startu). Porównaj finalne polityki Q-learning i SARSA. Screenshotuj ścieżki, które każda bierze. Która jest bliżej klifu?
3. **Trudne.** Zaimplementuj Double Q-learning. Na GridWorld z szumem nagród (Gaussowski szum σ=5 dodany do per-step reward), pokaż że Q-learning przecenia `V*(0,0)` o znaczącą ilość, podczas gdy Double Q-learning nie.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| TD error | "Sygnał aktualizacji" | `δ = r + γ V(s') - V(s)`, zbootstrappowany residuum. |
| TD(0) | "Jednokrokowy TD" | Aktualizacja po każdym przejściu używając tylko oszacowania następnego stanu. |
| Q-learning | "Off-policy RL 101" | TD update z `max` po akcjach następnego stanu; uczy się `Q*` niezależnie od behavior policy. |
| SARSA | "On-policy Q-learning" | TD update używający faktycznej następnej akcji; uczy się `Q^π` dla current ε-greedy π. |
| Expected SARSA | "Niska-wariancja SARSA" | Zastąp próbkowaną `a'` jej oczekiwaniem pod π. |
| GLIE | "Poprawny harmonogram eksploracji" | Greedy in the Limit with Infinite Exploration; potrzebne dla zbieżności Q-learning. |
| Bootstrapping | "Używanie current estimate w target" | To, co odróżnia TD od MC. Źródło obciążenia, ale masywna redukcja wariancji. |
| Maximization bias | "Q-learning przecenia" | `max` po zaszumionych oszacowaniach jest obciążony w górę; naprawione przez Double Q-learning. |

## Dalsze czytanie

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — oryginalny paper i dowód zbieżności.
- [Sutton & Barto (2018). Rozdz. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — poprawka na maximization bias.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — motywacja expected SARSA.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — paper, który spopularyzował SARSA (wtedy nazywany "modified connectionist Q-learning").
- [Sutton & Barto (2018). Rozdz. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — uogólnia TD(0) do TD(n), ścieżka od Q-learning do eligibility traces i, później, GAE w PPO.