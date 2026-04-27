`). Przez tysiące kroków, średnia długość odpowiedzi rośnie z ~100 do ~10 000 tokenów, a wyniki benchmarków matematycznych wspięły się do poziomów bliskich o1-preview. Model uczy się wnioskować od zera. Minus: jego łańcuchy myśli są często nieczytelne, mieszają języki i brakuje im stylistycznego poloru.
- **R1.** Naprawiamy problemy czytelności R1-Zero czteroetapowym pipeline'em:
  1. **Cold-start SFT.** Zbieramy kilka tysięcy demonstracji długiego-CoT z czystym formatowaniem. Supervised-finetune bazowego modelu na nich. To daje czytelny punkt startowy.
  2. **GRPO zorientowane na wnioskowanie.** Aplikujemy GRPO z nagrodami accuracy+format plus *language-consistency reward*, aby zapobiec code-switching.
  3. **Rejection sampling + SFT runda 2.** Próbkujemy ~600K trajektorii wnioskowania z checkpointa RL, zachowujemy tylko te z poprawnymi końcowymi odpowiedziami i czytelnym CoT, i łączymy z ~200K przykładami SFT bez wnioskowania (pisanie, QA, self-cognition). Fine-tune bazowego modelu ponownie.
  4. **GRPO pełnego spektrum.** Jeszcze jedna runda RL obejmująca zarówno wnioskowanie (nagrody oparte na regułach), jak i ogólne alignment (nagrody oparte na preferencjach helpfulness/harmlessness).

Wynik dorównuje o1 na AIME i MATH-500 przy otwartych wagach i jest wystarczająco mały, by go destylować. Ta sama praca wydaje również sześć destylowanych gęstych modeli (Qwen-1.5B przez Llama-70B) przez SFT na śladach wnioskowania R1 — bez RL na studencie. Destylacja silnego nauczyciela RL konsekwentnie bije RL od zera przy skali studenta.

**Dlaczego GRPO zamiast PPO dla wnioskowania.** Trzy powody w pracy DeepSeekMath (luty 2024): (1) brak sieci wartości do trenowania, połowa pamięci; (2) grupowy baseline naturalnie obsługuje rzadką nagrodę end-of-trajectory, którą produkują zadania wnioskowania; (3) normalizacja per-prompt czyni advantages porównywalnymi między problemami o radykalnie różnej trudności, czego krytyk PPO nie może.

**Search-free vs search-based.** Gry się rozeszły:

- *Gry z perfekcyjną informacją z długimi horyzontami* (Go, szachy): nadal search-based. AlphaZero / MuZero dominują.
- *LLM reasoning*: brak MCTS jeszcze w produkcji; GRPO na pełnych rollouts, best-of-N dla inference compute. Process reward models (PRMs) sugerują, że step-level search może zostać dodany z powrotem.

## Zbuduj to

Kod w `code/main.py` implementuje **GRPO w miniaturze** — bandytę z wieloma grupami próbek. Algorytm jest taki sam jak na LLM; tylko polityka i środowisko są prostsze. Uczy *straty* i *group-relative advantage*, które jest innowacją 2025.

### Krok 1: tiny środowisko weryfikatora

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

W prawdziwym GRPO weryfikator uruchamia testy jednostkowe lub sprawdza równość matematyczną.

### Krok 2: polityka: softmax nad K tokenami odpowiedzi na prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Odpowiednik outputu ostatniej warstwy LLM warunkowanego na prompt.

### Krok 3: grupowe próbkowanie i group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Group-relative advantage to trick DeepSeek z 2024. Brak krytyka. "Baseline" to średnia grupy, a normalizacja używa std grupy.

### Krok 4: porównaj do REINFORCE baseline (value-free)

Ta sama konfiguracja, ten sam compute, plain REINFORCE. GRPO converges szybciej i stabilniej.

### Krok 5: obserwuj entropię i KL

Te same diagnostyki co RLHF: średni KL do referencji, entropia polityki, nagroda w czasie. Gdy te się ustabilizują, trening jest gotowy.

## Pułapki

- **Reward hacking przez gaming weryfikatora.** GRPO dziedziczy ryzyko RLHF: jeśli weryfikator jest błędny lub exploatowalny, LLM znajdzie exploit. Odporne weryfikatory (wiele przypadków testowych, formal proofs) mają znaczenie.
- **Grupowy rozmiar za mały.** Wariancja grupowego baseline idzie jak `1/√G`. Poniżej `G = 4`, sygnał advantage jest zaszumiony; standard to `G = 8` do `64`.
- **Length bias.** LLM completions o różnych długościach mają różne log-probabilities. Normalizuj przez liczbę tokenów, albo użyj sequence-level log-prob, albo przycinaj do max długości.
- **Czyste self-play cycles.** Trening w stylu AlphaZero może utknąć w pętlach dominacji na ogólnych-sum games. Mitigowane przez różnorodne pule przeciwników (league play, Lekcja 10).
- **Search-policy mismatch.** AlphaZero trenuje politykę, żeby naśladować output search. Jeśli policy net jest za mały, żeby reprezentować dystrybucję search, trening stoi.
- **Compute floor.** MuZero / AlphaZero potrzebują masywnego compute. Pojedyncze ablation często wymaga setek GPU-hours. Miniature demos istnieją (np. AlphaZero na Connect Four) do nauki.
- **Verifier coverage.** Testy jednostkowe, które przechodzą dla błędnego rozwiązania, wzmacniają bug. Projektuj weryfikatory, które łapią edge cases.

## Użyj to

Krajobraz game-RL 2026, według domeny:

| Domena | Dominująca metoda |
|--------|-------------------|
| Dwuosobowe gry o sumie zero na planszy (Go, szachy, shogi) | AlphaZero / MuZero / KataGo |
| Gry karciane z niedoskonałą informacją (poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / gry pixel | Muesli / MuZero / IMPALA-PPO |
| Duże wieloosobowe strategie (Dota, StarCraft) | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM math/code reasoning | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM alignment | DPO / RLHF-PPO (nie GRPO; weryfikator to preferencja nie weryfikowalna) |
| Robotyka | PPO + DR (nie game-RL, ale używa tych samych narzędzi policy-gradient) |
| Problemy kombinatoryczne | Warianty AlphaZero (AlphaTensor, AlphaDev) |

**Przepis** — self-play, search-augmented improvement, policy distillation — rozciąga się na tekst, piksele i kontrolę fizyczną. GRPO to najmłodszy przypadek; więcej nadchodzi.

## Wyślij to

Zapisz jako `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Zaprojektuj pipeline treningowy game-RL lub reasoning-RL (AlphaZero / MuZero / GRPO) dla danej domeny.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Znane reguły? Markov? Stochastyczne? Multiagent? Informuje AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT z learned prior), Gumbel-sampled, best-of-N, lub none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Odmów AlphaZero na imperfect-info games (kieruj do CFR). Odmów GRPO bez zaufanego weryfikatora. Odmów każdemu pipeline'owi game-RL bez ustalonego zbioru przeciwników baseline (self-play ELO jest niekalibrowane w przeciwnym razie).
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj GRPO bandytę w `code/main.py`. Trenuj na 2 promptech × 4 tokeny odpowiedzi każdy. Zbiegnij w < 1 000 updates z `G=8`.
2. **Średnie.** Podłącz PPO (clipped) i vanilla REINFORCE. Porównaj sample efficiency i wariancję nagrody do GRPO na tym samym bandycie.
3. **Trudne.** Rozszerz do length-2 "reasoning chain": agent emituje dwa tokeny, a weryfikator nagradza parę. Zmierz, jak GRPO radzi sobie z credit assignment przez dwu-step sequences. (Wskazówka: oblicz group advantage per *pełną sekwencję*, propsaguj do obu pozycji tokenów.)

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| MCTS | "Tree search with learned net" | Monte Carlo Tree Search; wybór UCB1/PUCT z learned `(p, v)` priors. |
| AlphaZero | "Self-play + MCTS" | Policy-value net trenowana, żeby naśladować odwiedziny MCTS i wynik gry. |
| MuZero | "Learned-model AlphaZero" | Ta sama pętla, ale w przestrzeni ukrytej przez learned dynamics. |
| GRPO | "Critic-free PPO" | Group Relative Policy Optimization; REINFORCE z group-mean baseline + KL. |
| PUCT | "AlphaZero's UCB" | `Q + c · p · √N / (1 + N_a)` — balansuje oszacowanie wartości z prior. |
| Self-play | "Agent vs past self" | Standard dla zero-sum; symetryczny sygnał treningowy. |
| League play | "Population-based self-play" | Przeszłe + obecne + exploiters próbkowane jako przeciwnicy. |
| Verifier reward | "Verifiable RL" | Nagroda pochodzi z deterministycznego checkera (testy przechodzą, odpowiedź się zgadza). |
| Process reward | "PRM" | Ocenia każdy krok wnioskowania, nie tylko końcową odpowiedź. |

## Dalsze czytanie

- [Silver i in. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver i in. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser i in. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals i in. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — praca, która wprowadziła GRPO i group-relative baseline.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — pełny czteroetapowy przepis R1 plus ablation R1-Zero.
- [Brown i in. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + deep-learning na skali.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — praca, która to wszystko zaczęła.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — produkcyjna referencja dla aplikowania GRPO z custom reward functions.
- [Qwen Team (2024). Qwen2.5-Math — replikacja GRPO](https://github.com/QwenLM/Qwen2.5-Math) — otwarta replikacja przepisu R1 w wielu skalach.
- [Sutton & Barto (2018). Rozdz. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — podręcznikowe ujęcie self-play, search i "designed reward", które R1 instantiates na skali LLM.