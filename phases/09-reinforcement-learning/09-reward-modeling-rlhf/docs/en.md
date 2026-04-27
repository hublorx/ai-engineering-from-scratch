# Modelowanie nagród i RLHF

> Ludzie nie mogą napisać funkcji nagrody dla „dobrej odpowiedzi asystenta", ale potrafią porównać dwie odpowiedzi i wybrać lepszą. Dopasuj model nagród do tych porównań, a następnie zastosuj RL wobec modelu językowego. Christiano 2017. InstructGPT 2022. Przepis, który zamienił GPT-3 w ChatGPT. W 2026 roku jest w większości zastępowany przez DPO — ale mentalny model zostaje.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 05 (Sentiment), Faza 9 · 08 (PPO)
**Szacowany czas:** ~45 minut

## Problem

Trenowałeś model językowy na zadaniu przewidywania następnego tokena. Pisze poprawną gramatycznie angielszczyznę. Kłamie jednak, plecie i odmawia odmowy. Nie możesz tego naprawić kolejnym pretrainingiem — tekst z internetu jest problemem, nie rozwiązaniem.

Chcesz *skalarne wartości nagrody*, które powiedzą „odpowiedź A jest lepsza niż odpowiedź B dla instrukcji X." Zapisanie tej funkcji nagrody ręcznie jest niemożliwe. „Pomocność" nie jest wyrażeniem w zamkniętej formie nad tokenami. Ludzie potrafią jednak porównać dwa wyniki i zaznaczyć preferencję. Jest to tanie do zbierania na skalę.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) konwertuje preferencje na model nagród, a następnie optymalizuje LM poprzez PPO wobec tej nagrody. W trzech krokach: SFT → RM → PPO. To jest przepis, który dostarczył ChatGPT, Claude, Gemini i każdego innego wyrównanego LLM w latach 2023–2025.

W 2026 roku krok PPO jest w większości zastępowany przez DPO (Faza 10 · 08), ponieważ jest tańszy i niemal równie dobry do dostrajania wyrównania. Ale element *modelu nagród* nadal stanowi podstawę każdego samplera Best-of-N, każdego potoku RL-from-verifiable-rewards i każdego modelu rozumowania używającego process reward model. Zrozum RLHF, a zrozumiesz cały stos wyrównania.

## Koncepcja

![Trójstopniowy RLHF: SFT, trening RM na parach preferencji, PPO z penalizacją KL](../assets/rlhf.svg)

**Etap 1: Supervised Fine-Tuning (SFT).** Zacznij od pretrained base model. Dostrój na ludzkich przykładachdemonstracji docelowego zachowania (odpowiedzi podążające za instrukcjami, pomocne odpowiedzi, itp.). Wynik: model `π_SFT`, który jest *obciążony ku dobrym zachowaniom*, ale wciąż ma nieograniczoną przestrzeń akcji.

**Etap 2: Trening modelu nagród.**

- Zbierz pary odpowiedzi `(y_+, y_-)` na prompty `x`, oznaczone przez ludzi jako „y_+ jest preferowane nad y_-."
- Trenuj model nagród `R_φ(x, y)`, aby przypisywał wyższe wyniki do `y_+`.
- Funkcja strat: **Bradley-Terry pairwise logistic**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ to sigmoida. Różnica w nagrodzie implikuje log-odds preferencji. BT jest standardem od 1952 (Bradley-Terry) i dominującym wyborem w nowoczesnym RLHF.

- `R_φ` jest zwykle inicjalizowany z modelu SFT z dodatkową głowicą skalarną. Ten sam transformer backbone; pojedyncza warstwa liniowa wyprowadza nagrodę.

**Etap 3: PPO wobec RM z penalizacją KL.**

- Inicjalizuj trenowalną politykę `π_θ` z `π_SFT`. Trzymaj zamrożonego *referencyjnego* `π_ref = π_SFT`.
- Nagroda na końcu odpowiedzi `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  Kara KL zapobiega dryfowaniu `π_θ` dowolnie od `π_SFT` — jest to *regularizer*, nie twardy trust region. `β` typowo `0.01`-`0.05`.
- Uruchom PPO (Lekcja 08) z tą nagrodą. Przewagi są obliczane na poziomie token trajectory, ale RM ocenia tylko pełną odpowiedź.

**Dlaczego KL?** Bez niego PPO chętnie znajdzie strategie reward-hackingu — RM był trenowany tylko na completions in-distribution. Odpowiedź out-of-distribution może mieć wyższy wynik niż jakakolwiek napisana przez człowieka. KL utrzymuje `π_θ` blisko manifold, gdzie RM był trenowany. To jest najważniejszy knob w RLHF.

**Status w 2026:**

- **DPO** (Rafailov 2023): algebra w zamkniętej formie łączy Etap 2+3 w pojedynczą nadzorowaną funkcję strat na danych preferencji. Bez RM, bez PPO. Ta sama jakość na benchmarkach wyrównania za ułamek compute. Omówione w Faza 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO z grupowo-relatywnym baseline zamiast critic, nagroda z *verifier* (uruchomienia kodu / dopasowania odpowiedzi matematycznej) zamiast human-trained RM. Dominujące dla modeli rozumowania. Omówione w Faza 9 · 12.
- **Process reward models (PRMs):** oceniają częściowe rozwiązania (każdy krok rozumowania), używane zarówno w RLHF jak i wariantach GRPO dla rozumowania.
- **Constitutional AI / RLAIF:** używają wyrównanego LLM do generowania preferencji zamiast ludzi. Skaluje budżet preferencji.

## Zbuduj to

Ta lekcja używa miniaturowych syntetycznych „promptów" i „odpowiedzi" reprezentowanych jako stringi. RM to liniowy scorer nad reprezentacją bag-of-tokens. Żaden prawdziwy LLM — *kształt* potoku ma znaczenie, nie skala. Zobacz `code/main.py`.

### Krok 1: syntetyczne dane preferencji

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

W prawdziwym RLHF jest to zastępowane przez human labelers. Kształt — `(prompt, preferred_response, rejected_response)` — jest identyczny.

### Krok 2: Bradley-Terry reward model

Liniowy wynik: `R(x, y) = w · bag(y)`. Trenuj, aby zminimalizować BT pairwise log-loss:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

Po kilkuset aktualizacjach `w` przypisuje dodatnie wagi dobrym słowom i ujemne złym.

### Krok 3: PPO-like policy na szczycie RM

Nasza toy policy produkuje pojedynczy token ze słownika. Oceniamy token pod RM, obliczamy `log π_θ(token | prompt)`, dodajemy karę KL do referencji i stosujemy obcięty PPO surrogate.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Krok 4: monitoruj KL

Śledź średnie `KL(π_θ || π_ref)` przy każdej aktualizacji. Jeśli wzrośnie powyżej `~5-10`, polityka znacznie odbiegła od `π_SFT` — niższy `β` rośnie lub reward hacking się zaczyna. To jest najważniejsze diagnostyczne w prawdziwym RLHF.

### Krok 5: przepis produkcyjny z TRL

Gdy już zrozumiesz toy pipeline, oto ta sama pętla jako kod pisany przez użytkownika prawdziwej biblioteki. Hugging Face [TRL](https://huggingface.co/docs/trl) to referencyjna implementacja — `RewardTrainer` dla Etapu 2 i `PPOTrainer` (z wbudowanym KL-to-reference) dla Etapu 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Trzy rzeczy, które biblioteka robi za ciebie. `adap_kl_ctrl=True` implementuje adaptacyjny harmonogram β: jeśli obserwowane KL przekracza `target_kl`, β podwaja się; jeśli poniżej połowy, β połowicznie się zmniejsza. Reference model jest zamrożony z konwencji — nie możesz przypadkowo dzielić parametrów z `policy`. A value head żyje na tym samym backbone co policy (`AutoModelForCausalLMWithValueHead` dołącza skalarną warstwę MLP), dlatego TRL raportuje `policy/kl` i `value/loss` oddzielnie.

## Pułapki

- **Nad-optymalizacja / reward hacking.** RM jest niedoskonały; `π_θ` znajduje adversarial completions, które mają wysoki wynik, ale są złe. Objawy: nagroda rośnie w nieskończoność, podczas gdy human eval score płasko lub spada. Naprawa: zatrzymaj wcześnie, podnieś `β`, poszerz dane treningowe RM.
- **Length hacking.** RM trenowane na pomocnych odpowiedziach często implikicie nagradzają długość. Polityka uczy się paddingować odpowiedzi. Remediacja: length-normalized reward, lub RLAIF z length-aware RM.
- **Zbyt mały RM.** RM musi być co najmniej tak duży jak polityka. Mały RM nie może wiernie oceniać outputs polityki.
- **Tuning KL.** Zbyt niskie β → dryf i reward hacking. Zbyt wysokie β → polityka prawie się nie zmienia. Standardowym trikiem jest *adaptacyjne* β, które celuje w ustalone KL na krok.
- **Szum w danych preferencji.** ~30% ludzkich etykiet jest szumowych lub niejednoznacznych. Kalibruj przez trening RM na agreement-filtered data lub użyj temperature na BT.
- **Problemy off-policy.** Dane PPO są lekko off-policy po pierwszej epoce. Monitoruj clip fraction jak w Lekcji 08.

## Użyj tego

RLHF w 2026 jest warstwowe:

| Warstwa | Cel | Metoda |
|--------|-----|--------|
| Podążanie za instrukcjami, pomocność, nieszkodliwość | Wyrównanie | DPO (Faza 10 · 08) preferowane nad RLHF-PPO. |
| Poprawność rozumowania (matematyka, kod) | Zdolność | GRPO z verifier reward (Faza 9 · 12). |
| Długoterminowe wieloetapowe zadania | Agentyczność | PPO / GRPO z process reward models nad krokami. |
| Bezpieczeństwo / zachowanie odmowy | Bezpieczeństwo | RLHF-PPO z oddzielnym safety RM, lub Constitutional AI. |
| Best-of-N przy inference | Szybkie wyrównanie | Użyj RM przy decode time; nie potrzeba treningu polityki. |
| Reward distillation | Inference compute | Trenuj małą „reward head" na zamrożonym LM. |

RLHF był *the* metodą w 2022–2024. W 2026 roku produkcyjne potoki wyrównania są DPO-first, PPO-only dla RM-intensywnych lub safety-critical kroków.

## Wyślij to

Zapisz jako `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Zaprojektuj potok wyrównania RLHF / DPO / GRPO dla modelu językowego, w tym RM, KL i strategię danych.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Ćwiczenia

1. **Łatwe.** Trenuj Bradley-Terry reward model w `code/main.py` na 500 syntetycznych parach preferencji. Zmierz pairwise accuracy na 100 wstrzymanych parach. Powinna przekroczyć 90%.
2. **Średnie.** Uruchom toy PPO-RLHF loop z `β ∈ {0.0, 0.1, 1.0}`. Dla każdego, wykreśl RM score vs KL-to-reference przez aktualizacje. Które uruchomienia reward-hack?
3. **Trudne.** Zaimplementuj DPO (closed-form preference-likelihood loss) na tych samych danych preferencji i porównaj do RLHF-PPO pipeline w użytym compute i osiągniętym końcowym RM score.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| RLHF | „Alignment RL" | Trójstopniowy potok SFT + RM + PPO (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | „The scoring net" | Nauczona funkcja skalarna dopasowana do pairwise preferences przez Bradley-Terry. |
| Bradley-Terry | „Pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; standardowy RM objective. |
| KL penalty | „Stay near the reference" | `β · KL(π_θ || π_ref)` w nagrodzie; anti-reward-hacking regularizer. |
| Reward hacking | „Goodhart's law" | Polityka wykorzystuje wady RM; objawy: nagroda w górę, human eval płasko. |
| RLAIF | „AI-labeled preferences" | RLHF gdzie etykiety pochodzą od innego LM zamiast od ludzi. |
| PRM | „Process Reward Model" | Ocenia częściowe kroki rozumowania; używane w reasoning pipelines. |
| Constitutional AI | „Metoda Anthropic" | AI-generowane preferencje kierowane jawnymi regułami. |

## Dalsze czytanie

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — artykuł, który zapoczątkował RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — przepis za ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — wcześniejsze RLHF dla podsumowywania.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; post-RLHF default w 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF i self-critique loop.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — the HH paper.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — produkcyjny `RewardTrainer` i `PPOTrainer`. Przeczytaj źródło trainera dla adaptive-KL i value-head details.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) autorstwa Lambert, Castricato, von Werra, Havrilla — kanoniczny przewodnik po trójstopniowym potoku z diagramami.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — biblioteka; `examples/` ma end-to-end RLHF scripts dla Llama, Mistral i Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — reward-hypothesis view; niezbędne prerequisite dla myślenia o reward hackingu.