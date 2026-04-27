# Policy Gradient — REINFORCE od zera

> Przestań szacować wartość. Sparametryzuj politykę bezpośrednio, oblicz gradient oczekiwanej nagrody, idź pod górę. Williams (1992) zapisał to w jednym twierdzeniu. Dlatego istnieją PPO, GRPO i każda pętla RL dla LLM-ów.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 3 · 03 (Backpropagation), Faza 9 · 03 (Monte Carlo), Faza 9 · 04 (TD Learning)
**Szacowany czas:** ~75 minut

## Problem

Q-learning i DQN sparametryzują *funkcję wartości*. Wybierasz akcje przez `argmax Q`. To jest w porządku dla dyskretnych akcji i dyskretnych stanów. Łamie się, gdy akcje są ciągłe (co to `argmax` nad 10-wymiarowym momentem obrotowym?) albo gdy chcesz stochastyczną politykę (`argmax` jest z definicji deterministyczny).

Policy gradients sparametryzuj zamiast tego *politykę*. `π_θ(a | s)` to neural net, który outputuje rozkład nad akcjami. Próbkuj z niego, aby działać. Oblicz gradient oczekiwanej nagrody względem `θ`. Idź pod górę. Bez `argmax`. Bez rekursji Bellmana. Po prostu gradient ascent na `J(θ) = E_{π_θ}[G]`.

Twierdzenie REINFORCE (Williams 1992) mówi ci, że ten gradient jest obliczalny: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Uruchom epizod. Oblicz return. Pomnóż przez `∇ log π_θ(a | s)` w każdym kroku. Uśrednij. Gradient-ascent. Gotowe.

Każdy algorytm LLM-RL w 2026 — PPO, DPO, GRPO — to ulepszenie REINFORCE. Zrozumienie go w palcach to warunek wstępny dla reszty tej fazy i dla Fazy 10 · 07 (implementacja RLHF) i Fazy 10 · 08 (DPO).

## Koncepcja

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Twierdzenie o policy gradient.** Dla każdej polityki `π_θ` sparametryzowanej przez `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

gdzie `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` to zdyskontowany return od kroku `t`. Oczekiwanie jest nad pełnymi trajektoriami `τ` próbkowanymi z `π_θ`.

**Dowód jest krótki.** Różniczkuj `J(θ) = Σ_τ P(τ; θ) G(τ)` pod znakiem oczekiwania. Użyj `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (log-derivative trick). Rozłóż `log P(τ; θ) = Σ log π_θ(a_t | s_t) + terms environement that do not depend on θ`. Termsy środowiskowe znikają. Dwa wiersze algebry dają ci twierdzenie.

**Tricki redukcji wariancji.** Vanilla REINFORCE ma morderczą wariancję — returns są noisy, `∇ log π` jest noisy, ich iloczyn jest bardzo noisy. Dwa standardowe fixy:

1. **Odejmowanie baseline.** Zastąp `G_t` przez `G_t - b(s_t)` dla dowolnego baseline `b(s_t)`, który nie zależy od `a_t`. Nieobciążony, bo `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Typowy wybór: `b(s_t) = V̂(s_t)` nauczony przez critic → actor-critic (Lekcja 07).
2. **Reward-to-go.** Zastąp `Σ_t G_t · ∇ log π_θ(a_t | s_t)` przez `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Tylko przyszłe returns liczą się dla danej akcji — przeszłe nagrody wnoszą szum o średniej zero.

Połączone, dostajesz:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

co jest REINFORCE z baseline — bezpośrednim przodkiem A2C (Lekcja 07) i PPO (Lekcja 08).

**Parametryzacja softmax policy.** Dla dyskretnych akcji, standardowy wybór:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

gdzie `f_θ` to dowolny neural net, który outputuje score per action. Gradient ma czystą formę:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

tj. score podjętej akcji minus jej oczekiwana wartość pod polityką.

**Gaussian policy dla ciągłych akcji.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` ma zamkniętą formę. To wszystko, czego potrzebuje SAC z Fazy 9 · 07.

## Zbuduj to

### Krok 1: softmax policy network

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Użyj linear policy (jeden wektor wag per action) dla tabular env. Dla Atari, wstaw CNN i zachowaj softmax head.

### Krok 2: próbkowanie i log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Krok 3: rollout z przechwyconymi log-probs

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Krok 4: REINFORCE update

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient `∇ log π(a|s) = e_a - π(·|s)` (onehot z `a` minus prawdopodobieństwa) to serce softmax policy gradients. Wpal to w pamięć mięśniową.

### Krok 5: baselines

Running mean z `G` over recent episodes to wystarczająca redukcja wariancji, żeby uruchomić 4×4 GridWorld; potrzeba ~500 epizodów do konwergencji. Ulepsz baseline do nauczonego `V̂(s)` i dostajesz actor-critic.

## Pułapki

- **Exploding gradients.** Returns mogą być ogromne. Zawsze normalizuj `G` do `~N(0, 1)` across the batch przed mnożeniem przez `∇ log π`.
- **Entropy collapse.** Polityka konwerbuje do near-deterministic action zbyt wcześnie, przestaje eksplorować, utyka. Fix: dodaj entropy bonus `β · H(π(·|s))` do obiektywu.
- **High variance.** Vanilla REINFORCE potrzebuje tysięcy epizodów. Critic baseline (Lekcja 07) albo trust region TRPO/PPO (Lekcja 08) to standardowy fix.
- **Sample inefficiency.** On-policy oznacza, że wyrzucasz każdy transition po jednym update. Off-policy corrections via importance sampling przywracają dane, kosztem wariancji (PPO's ratio to clipped IS weight).
- **Non-stationary gradients.** Ten sam gradient z 100 epizodów temu używa starego `π`. On-policy methods update co kilka rolloutów z tego powodu.
- **Credit assignment.** Bez reward-to-go, przeszłe nagrody wnoszą szum. Zawsze używaj reward-to-go.

## Użyj tego

W 2026, REINFORCE jest rzadko uruchamiany bezpośrednio, ale jego wzór na gradient jest wszędzie:

| Przypadek użycia | Wyprowadzona metoda |
|----------|---------------|
| Ciągła kontrola | PPO / SAC z Gaussian policy |
| LLM RLHF | PPO z KL penalty, działający na token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE z group-relative baseline, bez critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Dyskretna akcja robotyka | A2C, A3C, PPO |
| Ustawienia tylko z preferencjami | DPO — REINFORCE przepisany jako preference-likelihood loss, bez próbkowania |

Gdy czytasz `loss = -advantage * log_prob` w 2026 training script, to jest REINFORCE z baseline. Całe prace (DPO, GRPO, RLOO) to triczi redukcji wariancji na szczycie tej jednej linii.

## Wyślij to

Zapisz jako `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj REINFORCE na 4×4 GridWorld z linear softmax policy. Trenuj przez 1,000 epizodów bez baseline. Wykreśl krzywą uczenia; zmierz wariancję (std of returns).
2. **Średnie.** Dodaj running-mean baseline. Trenuj ponownie. Porównaj sample efficiency i wariancję z vanilla run. O ile baseline redukuje kroki do konwergencji?
3. **Trudne.** Dodaj entropy bonus `β · H(π)`. Przeskanuj `β ∈ {0, 0.01, 0.1, 1.0}`. Wykreśl final return i policy entropy. Gdzie jest sweet spot na tym zadaniu?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to tak naprawdę oznacza |
|------|-----------------|-----------------------|
| Policy gradient | "Trenuj politykę bezpośrednio" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`; wyprowadzone z log-derivative trick. |
| REINFORCE | "Oryginalny algorytm PG" | Williams (1992); Monte Carlo returns pomnożone przez log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; czyni gradienty oczekiwań tractable. |
| Baseline | "Redukcja wariancji" | Dowolny `b(s)` odejmowany od `G`; nieobciążony, bo `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Liczą się tylko przyszłe returns" | `G_t^{from t}` zamiast pełnego `G_0`; poprawne i lower-variance. |
| Entropy bonus | "Zachęcaj do eksploracji" | `+β · H(π(·|s))` term utrzymuje politykę przed kolapsem. |
| On-policy | "Trenuj na tym, co właśnie widziałeś" | Gradient expectation jest w.r.t. current policy — nie można bezpośrednio reuse old data. |
| Advantage | "O ile lepsze od średniej" | `A(s, a) = G(s, a) - V(s)`; signed quantity, którą REINFORCE-with-baseline mnoży. |

## Dalsze czytanie

- Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning — oryginalny artykuł REINFORCE.
- Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation — nowoczesne twierdzenie o policy gradient z aproksymacją funkcji.
- Sutton & Barto (2018). Rozdz. 13 — Policy Gradient Methods — podręcznikowa prezentacja.
- OpenAI Spinning Up — VPG / REINFORCE — przejrzysta pedagogiczna prezentacja z kodem PyTorch.
- Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients — redukcja wariancji i widok naturalnego gradienta, który łączy REINFORCE z rodziną trust-region (TRPO, PPO).