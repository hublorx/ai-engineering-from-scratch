# Actor-Critic — A2C i A3C

> REINFORCE jest zaszumiony. Dodaj krytyka, który uczy się `V̂(s)`, odejmij go od zwrotu, i otrzymasz przewagę, która ma takie same oczekiwanie, ale znacznie niższą wariancję. To jest actor-critic. A2C uruchamia go synchronicznie; A3C uruchamia go w wielu wątkach. Obie są mentalnym modelem dla każdej nowoczesnej metody deep-RL.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Szacowany czas:** ~75 minut

## Problem

Vanilla REINFORCE działa, ale jego wariancja jest straszna. Zwroty Monte Carlo `G_t` mogą się wahać o czynnik 10 między epizodami. Mnożenie tego szumu przez `∇ log π` i uśrednianie produkuje estymator gradientu, który potrzebuje tysięcy epizodów, żeby przesunąć politykę na tę samą odległość, co znacznie mniej aktualizacji DQN.

Wariancja pochodzi od używania surowych zwrotów. Jeśli odejmiesz baseline `b(s_t)` — dowolną funkcję stanu, w tym nauczoną wartość — oczekiwanie pozostaje niezmienione, a wariancja spada. Najlepszym dającym się śledzić baseline jest `V̂(s_t)`. Teraz wielkość mnożąca `∇ log π` to *przewaga*:

`A(s, a) = G - V̂(s)`

Akcja jest dobra, jeśli wygenerowała zwrot powyżej średniej; zła, jeśli poniżej. REINFORCE z nauczonym krytykiem to *actor-critic*. Krytyk daje aktorowi nauczyciela o niskiej wariancji. To jest podstawa każdej głębokiej metody polityki po 2015 (A2C, A3C, PPO, SAC, IMPALA).

## Koncepcja

![Actor-critic: sieć polityki plus sieć wartości, TD residual jako przewaga](../assets/actor-critic.svg)

**Dwie sieci, jeden wspólny loss:**

- **Aktor** `π_θ(a | s)`: polityka. Próbkowana do działania. Trenowana z policy gradient.
- **Krytyk** `V_φ(s)`: estymuje oczekiwany zwrot ze stanu. Trenowany, żeby zminimalizować `(V_φ(s) - target)²`.

**Przewaga.** Dwie standardowe formy:

- *MC advantage:* `A_t = G_t - V_φ(s_t)`. Nieobciążona, wyższa wariancja.
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Obciążona (używa `V_φ`), znacznie niższa wariancja. Nazywana też *TD residual* `δ_t`.

**n-step advantage.** Interpolacja między nimi:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` to czyste TD. `n = ∞` to MC. Większość implementacji używa `n = 5` dla Atari, `n = 2048` dla PPO na MuJoCo.

**Generalized Advantage Estimation (GAE).** Schulman et al. (2016) zaproponowali wykładniczo ważoną średnią wszystkich n-step advantages:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

z `λ ∈ [0, 1]`. `λ = 0` to TD (niska wariancja, wysokie obciążenie). `λ = 1` to MC (wysoka wariancja, nieobciążona). `λ = 0.95` to domyślna wartość w 2026 — dostrój, aż dial obciążenie/wariancja będzie tam, gdzie chcesz.

**A2C: synchroniczny advantage actor-critic.** Zbierz `T` kroków z `N` równoległych środowisk. Oblicz advantage dla każdego kroku. Zaktualizuj aktora i krytyka na połączonym batchu. Powtórz. Prostszy, bardziej skalowalny odpowiednik A3C.

**A3C: asynchroniczny advantage actor-critic.** Mnih et al. (2016). Utwórz `N` wątków roboczych, każdy uruchamia env. Każdy wątek oblicza gradienty lokalnie na własnym rollout, a potem asynchronicznie aplikuje je do współdzielonego serwera parametrów. Buffer reply nie jest potrzebny — wątki dekorelują przez uruchamianie różnych trajektorii. A3C udowodniło, że można trenować na CPU w skali. W 2026 GPU-based A2C (batched parallel envs) dominuje, bo GPU chcą dużych batchy.

**Łączny loss.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

Trzy terminy: policy-gradient loss, regresja wartości, bonus entropii. `c_v ~ 0.5`, `c_e ~ 0.01` to kanoniczne punkty wyjścia.

## Zbuduj to

### Krok 1: krytyk

Liniowy krytyk `V_φ(s) = w · features(s)` aktualizowany z MSE:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

Na tabularycznym env krytyk zbiega w kilkaset epizodów. Na Atari zastąp liniowego krytyka wspólnym trunkiem CNN + value head.

### Krok 2: n-step advantage

Mając rollout długości `T` i boostrappowany końcowy `V(s_T)`:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` to cel krytyka. `advantages` to to, co mnoży `∇ log π`.

### Krok 3: połączona aktualizacja

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy, jeden rollout na aktualizację, oddzielne learning rates dla aktora i krytyka.

### Krok 4: równoległość (A3C vs A2C)

- **A3C:** utwórz `N` wątków. Każdy uruchamia swój env i swój forward pass. Okresowo pushuj aktualizacje gradientów do wspólnego mastera. Bez locków na masterze — wyścigi są ok, po prostu dodają szum.
- **A2C:** uruchom `N` instancji env w jednym procesie, stackuj obserwacje w batch `[N, obs_dim]`, batched forward pass, batched backward pass. Wyższe wykorzystanie GPU, deterministyczne, łatwiejsze do rozumowania. Domyślne w 2026.

Nasz przykładowy kod jest single-threaded dla przejrzystości; przepisanie do batched A2C to trzy linie numpy.

## Pułapki

- **Obciążenie krytyka przed gradientem aktora.** Jeśli krytyk jest losowy, jego baseline jest nieinformacyjny i trenujesz na czystym szumie. Rozgrzej krytyka przez kilkaset kroków przed włączeniem policy gradient, albo użyj wolnego learning rate aktora.
- **Normalizacja advantage.** Normalizuj advantage do zerowej średniej/jednostkowej std per batch. Stabilizuje trening masywnie przy niemal zerowym koszcie.
- **Wspólny trunk.** Użyj wspólnego extractora cech dla aktora i krytyka na obrazkach. Oddzielne heads. Wspólne features korzystają z obu lossów.
- **On-policy contract.** A2C używa danych dokładnie raz na aktualizację. Więcej i Twój gradient jest obciążony (importance-sampling correction to PPO).
- **Zapadnięcie entropii.** Bez `c_e > 0` polityka staje się niemal-deterministic po kilkuset aktualizacjach i przestaje eksplorować.
- **Skala nagród.** Wielkość advantage zależy od skali nagrody. Normalizuj nagrody (np. dzielenie przez running-std) dla spójnych wielkości gradientu między zadaniami.

## Użyj tego

A2C/A3C rzadko są ostatecznym wyborem w 2026, ale są architekturą, którą wszystko późniejsze udoskonala:

| Metoda | Relacja do A2C |
|--------|----------------|
| PPO | A2C + clipped importance ratio dla multi-epoch updates |
| IMPALA | A3C + V-trace off-policy correction |
| SAC (Phase 9 · 07) | Off-policy A2C z soft-value critic (następna lekcja) |
| GRPO (Phase 9 · 12) | A2C bez krytyka — group-relative advantage |
| DPO | A2C złożone w preference-ranking loss, bez samplingu |
| AlphaStar / OpenAI Five | A2C z league training + imitation pre-training |

Jeśli widzisz "advantage" w artykule z 2026, myśl actor-critic.

## Wyślij to

Zapisz jako `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Ćwiczenia

1. **Łatwe.** Trenuj actor-critic z MC advantage (`G_t - V(s_t)`) na 4×4 GridWorld. Porównaj efektywność próbkowania z REINFORCE z running-mean-baseline z Lesson 06.
2. **Średnie.** Przełącz na TD-residual advantage (`r + γ V(s') - V(s)`). Zmierz wariancję batchy advantage. O ile spadła?
3. **Trudne.** Zaimplementuj GAE(λ). Przesiej `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`. Wykreśl finalny zwrot vs efektywność próbkowania. Gdzie jest sweet spot obciążenie/wariancja dla tego zadania?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Aktor | "Sieć polityki" | `π_θ(a|s)`, aktualizowana przez policy gradient. |
| Krytyk | "Sieć wartości" | `V_φ(s)`, aktualizowany przez regresję MSE do zwrotów / celów TD. |
| Przewaga | "Jak dużo lepsze niż średnia" | `A(s, a) = Q(s, a) - V(s)` lub jego estymatory. Mnożnik dla `∇ log π`. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; jednokrokowy estymator advantage. |
| GAE | "Pokrętło interpolacji" | Wykładniczo ważona suma n-step advantages, sparametryzowana przez `λ`. |
| A2C | "Synchroniczny actor-critic" | Batched przez envs; jeden krok gradientu na rollout. |
| A3C | "Async actor-critic" | Wątki robocze pushują gradienty do współdzielonego serwera parametrów. Oryginalny paper; mniej popularny w 2026. |
| Bootstrap | "Użyj V na horyzoncie" | Obetnij rollout, dodaj `γ^n V(s_{t+n})` żeby zamknąć sumę. |

## Dalsze czytanie

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, oryginalny async actor-critic paper.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — podstawy; sparuj to z Ch. 9 o aproksymacji funkcji gdy krytyk jest siecią neuronową.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — skalowalny rozproszony actor-critic z V-trace off-policy correction.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — produkcyjne implementacje A2C/PPO warte przeczytania.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — fundamentalny wynik zbieżności dla dekompozycji actor-critic w dwóch skalach czasowych.