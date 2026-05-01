# Speculative Decoding — Draft, Verify, Repeat

> Autoregressive decoding jest sekwencyjny. Każdy token czeka na poprzedni. Speculative decoding łamie łańcuch: tani model draftuje N tokenów, drogi model weryfikuje wszystkie N w jednym forward pass. Gdy draft jest poprawny, zapłaciłeś jeden duży forward za N generacji.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 07 (GPT Causal LM), Faza 7 · 12 (KV Cache & Flash Attention)
**Czas:** ~60 minut

## Problem

LLM 70B sampling jeden token bierze ~30 ms na H100. Draft model 3B bierze ~3 ms. Jeśli pozwolimy 3B draftować 5 tokenów do przodu, potem uruchomimy 70B *raz* żeby zweryfikować wszystkie 5, całość to `5×3 + 30 = 45 ms` za do 5 zaakceptowanych tokenów — versus `5×30 = 150 ms` za straight-line generation. To jest pełna obietnica speculative-decoding: handluj niewielką ilością dodatkowej pamięci GPU (draft model) za 2–4× niższy decode latency.

Triк musi zachować rozkład. Speculative sampling, wprowadzony przez Leviathan et al. (2023) i przez Chen et al. concurrently, gwarantuje że output sequence jest **identically distributed** do tego co duży model by wyprodukował sam. Żaden kompromis jakościowy. Po prostu szybsze.

Cztery rodziny draft-verifier pairs dominują 2026 inference:

1. **Vanilla speculative (Leviathan 2023).** Separate draft model (np., Llama 3 1B) + verifier (np., Llama 3 70B).
2. **Medusa (Cai 2024).** Wiele decoding heads na verifierze przewidują pozycje `t+1..t+k` równolegle. Żaden separate draft model.
3. **Rodzina EAGLE (Li 2024, 2025).** Lekki draft który reuseuje ukryte stany verifiera; bliższy acceptance rate niż vanilla; 3–4× typowo.
4. **Lookahead decoding (Fu 2024).** Iteracja Jacobiego; nie potrzeba draft modelu w ogóle. Self-speculation. Niszowe ale dependency-free.

Każdy produkcyjny stack inferencji w 2026 wysyła speculative decoding domyślnie. vLLM, TensorRT-LLM, SGLang, i llama.cpp wszystkie wspierają przynajmniej vanilla + EAGLE-2.

## Koncepcja

### Core algorithm

Dany verifier `M_q` i tańszy draft `M_p`:

1. Niech `x_1..x_k` będzie prefix już zdekodowany.
2. **Draft**: użyj `M_p` żeby autoregresywnie propose `d_{k+1}, d_{k+2}, ..., d_{k+N}` z draft probabilities `p_1..p_N`.
3. **Verify in parallel**: uruchom `M_q` raz na `x_1..x_k, d_{k+1}, ..., d_{k+N}`, dostając verifier probabilities `q_1..q_{N+1}` dla pozycji `k+1..k+N+1`.
4. **Accept/reject each draft token left to right**: dla każdego `i`, zaakceptuj z prawdopodobieństwem `min(1, q_i(d_i) / p_i(d_i))`.
5. On first rejection at position `j`: sample `t_j` z "residual" distribution `(q_j - p_j)_+` normalized. Wszystkie drafts po `j` są odrzucane.
6. On accepting all `N`: sample one extra token `t_{N+1}` from `q_{N+1}` (the free bonus token).

Residual distribution trick to matematyczny wgląd który utrzymuje output distributed exactly as if `M_q` had sampled from scratch.

### Co determinuje przyspieszenie

Niech `α` = expected acceptance rate per draft token. Niech `c` = draft-to-verifier cost ratio. Per step:

- Naive generation makes 1 big-model call per token.
- Speculative makes 1 big-model call per `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` tokens when `α` is high.

Typowa reguła kciuka przy `α = 0.75` i `N = 5`: 3× mniej big-model calls. Draft cost to 5× cheap. Całkowity wall-clock spada ~2.5×.

**α depends on:**

- How well the draft approximates the verifier. Same family / same training data boosts α significantly.
- Decoding strategy. Greedy draft against greedy verifier: high α. Temperature sampling: harder to match; acceptance drops.
- Task type. Code and structured output accept more (predictable); free-form creative writing accepts less.

### Medusa — drafts bez draft model

Medusa zastępuje draft model extra output heads na verifierze. Na pozycji `t`:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

Każda głowa outputuje własne logits. Na inference samplujesz z każdej głowy żeby dostać candidate sequence, potem weryfikujesz jednym forward pass używając tree-attention scheme który rozważa wszystkie candidate continuations at once.

Pros: no second model. Cons: adds trainable parameters; potrzebuje supervised fine-tuning stage (~1B tokens); acceptance rate jest trochę niższy niż vanilla speculative z good draft.

### EAGLE — lepszy draft przez reuse ukrytych stanów

EAGLE-1/2/3 (Li et al., 2024–2025) sprawia że draft model to tiny transformer (typowo 1 warstwa) który ingests verifier's last-layer hidden states. Because the draft sees the verifier's feature representation, its predictions correlate strongly with the verifier's output distribution. Acceptance rates climb from ~0.6 (vanilla) to 0.85+.

EAGLE-3 (2025) dodał tree search over candidate continuations. vLLM i SGLang wysyłają EAGLE-2/3 jako domyślna spec pathway dla Llama 3/4 i Qwen 3.

### KV cache dance

Verification feeding `N` draft tokens into the verifier in one forward pass. To extends the verifier's KV cache by `N` entries. Jeśli some drafts are rejected, musisz rollback the cache do accepted prefix length.

Production implementations (vLLM's `--speculative-model`, TensorRT-LLM's LookaheadDecoder) handle this with scratch KV buffers. Write first, commit on acceptance. It's not conceptually hard, ale jest fiddly.

## Zbuduj To

Zobacz `code/main.py`. Implementujemy core speculative-sampling algorithm (rejection step + residual distribution) z:

- "Big model" który jest deterministic-softmax nad hand-coded distribution (żebyśmy mogli zweryfikować acceptance math analytically).
- "Draft model" który jest perturbation big model.
- Acceptance / rejection loop który produkuje ten sam marginal distribution jak direct sampling.

### Krok 1: the rejection step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` to uniform random number. `q_prob` to verifier's probability dla drafted token. `p_prob` to draft model's probability. Leviathan theorem jest że ta Bernoulli decision, nas followed by sampling from the residual on rejection, preserves the verifier's distribution exactly.

### Krok 2: residual distribution

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Odejmij `p` od `q` element-wise, clamp negative values to zero, renormalize. Sample from this on any rejection.

### Krok 3: one speculative step

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Pięć zaakceptowanych → jeden bonus → sześć tokenów wyprodukowanych w jednym verifier pass.

### Krok 4: measure acceptance rate

Uruchom 10,000 speculative steps przy varying draft-quality levels. Plot acceptance rate vs. KL divergence między draft i verifier distributions. Powinieneś zobaczyć czystą monotone relationship.

### Krok 5: verify distribution equivalence

Empirycznie: histogram tokenów wyprodukowanych przez speculative loop powinien matchować histogram wyprodukowany przez sampling directly from the verifier. To jest Leviathan theorem w praktyce. Chi-square test confirm within sampling error.

## Użyj To

Produkcja:

```bash
# vLLM z EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM z vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM ma najszybszą ścieżkę Medusa od połowy 2026. `faster-whisper` wrapuje speculative decoding dla Whisper-large z małym draftem.

**Picking a draft:**

| Strategia | Kiedy wybrać | Przyspieszenie |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | Szybki prototyp, bez treningu | 1.8–2.3× |
| Medusa heads | Możesz fine-tunować verifier | 2–3× |
| EAGLE-2 / 3 | Produkcja, max prędkość | 3–4× |
| Lookahead | Bez draft, bez treningu, bez dodatkowych params | 1.3–1.6× |

**Kiedy NIE spec-decode:**

- Single-sequence generation of 1–5 tokens. Overhead dominates.
- Wildly creative / high-temperature sampling (α drops).
- Memory-constrained deployments (draft model adds VRAM).

## Wyślij To

Zobacz `outputs/skill-spec-decode-picker.md`. Skill wybiera speculative decoding strategy (vanilla / Medusa / EAGLE / lookahead) i tuning parameters (N, draft temperature) dla nowego inference workload.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Potwierdź że speculative token distribution matchuje verifier's direct-sample distribution na 50,000 tokenów within chi-square p > 0.05.
2. **Średnie.** Plot speedup (tokens per big-model forward) jako funkcja `N` dla `α = 0.5, 0.7, 0.85`. Zidentyfikuj optymalne `N` dla każdego α. (Hint: expected tokens per verify call = `(1 - α^{N+1}) / (1 - α)`.)
3. **Trudne.** Zaimplementuj tiny Medusa: weź capstone GPT z Lekcji 14, dodaj 3 extra LM heads które przewidują pozycje t+2, t+3, t+4. Trenuj na tinyshakespeare z joint multi-head loss. Porównaj acceptance rates vs vanilla draft z truncating tego samego modelu.
4. **Trudne.** Zaimplementuj rollback: start z 10-token prefix KV cache, feed 5 draft tokens, symuluj rejection at position 3. Zweryfikuj że twoje cache reads correctly match "prefix + first 2 accepted drafts" at next iteration.

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Draft model | "The cheap one" | Mniejszy model który propose candidate tokens; usually 10–50× cheaper than the verifier. |
| Verifier | "The big one" | Target model whose distribution preserve; runs once per speculative step. |
| Acceptance rate (α) | "How often the draft is right" | Per-token probability że verifier acceptuje draft. 0.7–0.9 typical. |
| Residual distribution | "The rejection fallback" | `(q - p)_+` normalized; sampling from this on rejection preserves the verifier's distribution. |
| Bonus token | "The free one" | Gdy wszystkie N drafts accepted, sample one more from verifier's next-step distribution. |
| Medusa | "Draft-less speculative" | Wiele LM heads na verifierze przewidują pozycje t+1..t+k w parallel. |
| EAGLE | "Hidden-state draft" | Tiny transformer draft conditioned on verifier's last-layer hidden states. |
| Lookahead decoding | "Jacobi iteration" | Self-speculation using fixed-point iteration; no draft model. |
| Tree attention | "Verify many candidates at once" | Branching verification które rozważa kilka draft continuations simultaneously. |
| KV rollback | "Undo rejected drafts" | Scratch KV buffer; commit on acceptance, discard on reject. |

## Dalsze Czytanie

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — core algorithm i equivalence theorem.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — concurrent introduction; clean Bernoulli-rejection proof.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa paper; tree-attention verification.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; hidden-state-conditioned draft.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; dynamic tree depth.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead, no-draft approach.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — canonical production reference z all four strategies wired up.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — reference code dla EAGLE-1/2/3.