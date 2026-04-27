# Proximal Policy Optimization (PPO)

> A2C odrzuca każdy rollout po jednej aktualizacji. PPO opakowuje gradient polityki w obcięty współczynnik ważkości, dzięki czemu można wykonać 10+ epok na tych samych danych bez eksplozji polityki. Schulman i in., 2017. Wciąż domyślny algorytm gradientu polityki w 2026 roku.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Szacowany czas:** ~75 minut

## Problem

A2C (Lekcja 07) jest algorytmem on-policy: gradient `E_{π_θ}[A · ∇ log π_θ]` wymaga danych próbkowanych z *bieżącej* `π_θ`. Wykonaj jedną aktualizację, a `π_θ` się zmienia; dane, których użyłeś, są teraz off-policy. Używając ich ponownie, Twój gradient jest obciążony.

Rollouty są kosztowne. W Atari, jeden rollout na 8 envs × 128 kroków = 1024 przejścia i kilkanaście sekund czasu środowiskowego. Wyrzucanie tego po jednym kroku gradientu jest marnotrawstwem.

Trust Region Policy Optimization (TRPO, Schulman 2015) było pierwszym rozwiązaniem: ogranicz każdą aktualizację, aby dywergencja KL między starą i nową polityką pozostała poniżej `δ`. Teoretycznie czyste, ale wymaga rozwiązania metodą sprzężonych gradientów na każdą aktualizację. Nikt nie używa TRPO w 2026 roku.

PPO (Schulman i in. 2017) zastępuje twarde ograniczenie trust region prostym obciętym celem. Jedna dodatkowa linia kodu. Dziesięć epok na rollout. Brak sprzężonych gradientów. Wystarczające gwarancje teoretyczne. Dziewięć lat później wciąż jest domyślnym algorytmem gradientu polityki dla wszystkiego — od MuJoCo po RLHF.

## Koncepcja

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**Współczynnik ważkości.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

To stosunek wiarogodności nowej polityki do polityki, która zebrała dane. `r_t = 1` oznacza brak zmiany. `r_t = 2` oznacza, że nowa polityka jest dwukrotnie bardziej skłonna podjąć `a_t` niż stara.

**Obcięty surroga.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Dwa wyrazy:

- Jeśli przewaga `A_t > 0`, a współczynnik próbuje rosnąć powyżej `1 + ε`, obcięcie spłaszcza gradient — nie wciskaj dobrej akcji dalej niż `+ε` powyżej starej probabilności.
- Jeśli przewaga `A_t < 0`, a współczynnik próbuje rosnąć powyżej `1 - ε` (co oznacza, że zrobilibyśmy złą akcję bardziej prawdopodobną w porównaniu z jej obciętą redukcją), obcięcie ogranicza gradient — nie wciskaj złej akcji poniżej `-ε`.

`min` obsługuje drugi kierunek: jeśli współczynnik przesunął się w *korzystnym* kierunku, wciąż otrzymujesz gradient (brak obcinania po stronie, która by Cię zraniła).

Typowe `ε = 0.2`. Wykreśl cel jako funkcję `r_t`: funkcja kawałkami liniowa z płaskim dachem po "dobrej stronie" i płaską podłogą po "złej stronie."

**Pełna strata PPO.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Ta sama struktura actor-critic co A2C. Trzy współczynniki, zwykle `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**Pętla treningowa.**

1. Zbierz `N × T` przejść przez `N` równoległych envs przez `T` kroków każde.
2. Oblicz przewagi (GAE), zamroź je jako stałe.
3. Zamroź `π_{θ_old}` jako migawkę bieżącej `π_θ`.
4. Przez `K` epok, dla każdego minibatcha `(s, a, A, V_target, log π_old(a|s))`:
   - Oblicz `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Zastosuj `L^{CLIP}` + strata wartości + entropia.
   - Krok gradientu.
5. Odrzuć rollout. Wróć do kroku 1.

`K = 10` i minibatche rozmiaru 64 to standardowy zestaw hiperparametrów. PPO jest odporne: dokładne liczby rzadko mają znaczenie w granicach ±50%.

**Wariant z karą KL.** Oryginalny artykuł zaproponował alternatywę z adaptacyjną karą KL: `L = L^{PG} - β · KL(π_θ || π_old)` z `β` dostosowywanym na podstawie obserwowanego KL. Wersja z obcinaniem stała się dominująca; wariant z karą KL przetrwał w RLHF (gdzie KL do polityki referencyjnej jest osobnym ograniczeniem, którego zawsze chcesz).

## Zbuduj To

### Krok 1: przechwyć `log π_old(a | s)` w czasie rollout

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Migawka jest robiona raz, w czasie rollout. Nie zmienia się podczas epok aktualizacji.

### Krok 2: oblicz przewagi GAE (Lekcja 07)

Tak samo jak A2C. Normalizuj przez batch.

### Krok 3: obcięta aktualizacja surrogata

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Wzorzec "obcięte → zero gradient" to serce PPO. Jeśli nowa polityka już zbyt mocno odbiegła w korzystnym kierunku, aktualizacja się zatrzymuje.

### Krok 4: wartość i entropia

Dodaj standardowe MSE do celu critic i bonus entropii dla aktora, tak samo jak A2C.

### Krok 5: diagnostyka

Trzy rzeczy do obserwacji przy każdej aktualizacji:

- **Średnie KL** `E[log π_old - log π_θ]`. Powinno pozostać w `[0, 0.02]`. Jeśli przekroczy `0.1`, zmniejsz `K_EPOCHS` lub `LR`.
- **Frakcja obcięć** — frakcja próbek, których współczynnik leży poza `[1-ε, 1+ε]`. Powinna wynosić `~0.1-0.3`. Jeśli `~0`, obcięcie nigdy się nie uruchamia → zwiększ `LR` lub `K_EPOCHS`. Jeśli `~0.5+`, nadmiernie dopasowujesz rollout → zmniejsz je.
- **Wyjaśniona wariancja** `1 - Var(V_target - V_pred) / Var(V_target)`. Metryka jakości critic. Powinna rosnąć w kierunku 1 w miarę uczenia się critic.

## Pułapki

- **Źle dostrojony współczynnik obcięcia.** `ε = 0.2` to de facto standard. Zejście do `0.1` sprawia, że aktualizacje są zbyt ostrożne; `0.3+` zaprasza niestabilność.
- **Zbyt wiele epok.** `K > 20` rutynowo destabilizuje, bo polityka odbiega daleko od `π_old`. Ogranicz epoki, szczególnie dla dużych sieci.
- **Brak normalizacji nagrody.** Duże skale nagrody zabierają część zakresu obcinania. Normalizuj nagrody (bieżące odchylenie) przed obliczaniem przewag.
- **Zapomniana normalizacja przewagi.** Normalizacja zero-średnia/jednostkowe-odchylenie na batch jest standardowa. Pominięcie jej psuje PPO na większości benchmarków.
- **Stała stopa uczenia nie jest redukowana.** PPO korzysta z liniowego spadku LR do zera. Stały LR często jest gorszy.
- **Błędy matematyczne we współczynniku ważkości.** Zawsze `exp(log_new - log_old)` dla stabilności numerycznej, nie `new / old`.
- **Zły znak gradientu.** Maksymalizuj surrogat = *minimalizuj* `-L^{CLIP}`. Odwrócony znak to najczęstszy bug PPO.

## Użyj To

PPO to domyślny algorytm RL 2026 w zaskakująco wielu domenach:

| Przypadek użycia | Wariant PPO |
|------------------|-------------|
| MuJoCo / sterowanie robotyki | PPO z polityką Gaussowską, GAE(0.95) |
| Atari / gry dyskretne | PPO z polityką kategoryczną, rolling 128-step rollouts |
| RLHF dla LLM | PPO z karą KL do modelu referencyjnego, nagroda od RM na końcu odpowiedzi |
| Agenty gier na dużą skalę | IMPALA + PPO (AlphaStar, OpenAI Five) |
| LLM-y do wnioskowania | GRPO (Lekcja 12) — wariant PPO bez critic |
| Dane tylko preferencyjne | DPO — obcięcie PPO+KL w formie zamkniętej, brak próbkowania online |

Kształt *straty PPO* — obcięty surrogat + wartość + entropia — to szkielet dla DPO, GRPO i niemal każdego pipeline'u RLHF.

## Wyślij To

Zapisz jako `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Ćwiczenia

1. **Łatwe.** Uruchom PPO na 4×4 GridWorld z `ε=0.2, K=4`. Porównaj efektywność próbkowania z A2C (jedna epoka na rollout) przy dopasowanych krokach środowiskowych.
2. **Średnie.** Przeskanuj `K ∈ {1, 4, 10, 30}`. Wykreślł zwrot vs kroki środowiskowe i śledź średnie KL na aktualizację. Przy jakim `K` KL eksploduje na tym zadaniu?
3. **Trudne.** Zastąp obcięty surrogat adaptacyjną karą KL (`β` podwajane jeśli `KL > 2·target`, zmniejszane jeśli `KL < target/2`). Porównaj końcowy zwrot, stabilność i brak obcinania.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`; odchylenie od polityki, która zebrała dane. |
| Clipped surrogate | "Główny trick PPO" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; płaski gradient po obcięciu po korzystnej stronie. |
| Trust region | "Intencja TRPO / PPO" | Ogranicz każdą aktualizację KL, aby zagwarantować monoticzną poprawę. |
| KL penalty | "Miękki trust region" | Alternatywne PPO: `L - β · KL(π_θ || π_old)`. Adaptacyjne `β`. |
| Clip fraction | "Jak często obcinanie się uruchamia" | Diagnostyka — powinno być 0.1-0.3; poza tym oznacza źle dostrojone. |
| Multi-epoch training | "Ponowne użycie danych" | K epok na każdym rolloucie; koszt wariancji w zamian za efektywność próbkowania. |
| On-policy-ish | "Głównie on-policy" | PPO jest nominalnie on-policy, ale K>1 epok używa lekko off-policy danych bezpiecznie. |
| PPO-KL | "Inne PPO" | Wariant z karą KL; używany w RLHF gdzie KL-do-referencji jest już ograniczeniem. |

## Dalsze czytanie

- [Schulman i in. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — artykuł.
- [Schulman i in. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, poprzednik PPO.
- [Andrychowicz i in. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — każdy hiperparametr PPO poddany ablacji.
- [Ouyang i in. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; recepta PPO w RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — czyste nowoczesne przedstawienie z PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — referencyjny jednoplikowy PPO używany przez wiele artykułów.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — produkcyjna recepta dla PPO na modelach językowych; czytaj obok Lekcji 09 (RLHF).
- [Engstrom i in. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — artykuł o "37 optymalizacjach na poziomie kodu"; które tricki PPO są kluczowe, a które to folklor.