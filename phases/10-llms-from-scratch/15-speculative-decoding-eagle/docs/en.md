# Speculative Decoding and EAGLE-3

> Decode is memory-bound. A 70B model spends most of a forward pass waiting for weights to stream out of HBM, producing one token at the end of a ~70 ms wait. Leviathan, Kalai, Matias (2023) showed you can let a tiny draft model guess the next K tokens, verify them in one big-model forward pass, and accept the longest correct prefix — with a provably exact equivalence to sampling from the target. EAGLE-3 (Li et al., NeurIPS 2025) pushes mean accepted tokens per verify to ~4.65 on Spec-Bench, roughly a 2.4× wall-clock speedup on matched output distribution.

**Type:** Build
**Languages:** Python (stdlib only)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## The Problem

Autoregressive decoding is the cost center of production LLM serving. Every token is one forward pass, and the forward pass is dominated by reading the full weight matrix out of HBM — not by arithmetic. On an H100 decoding a 70B BF16 model, ~280 GB of weights have to stream past the compute units to produce one token. At 3.35 TB/s memory bandwidth that is about 83 ms floor, compute-independent.

You cannot shrink the model without changing its distribution. You cannot make memory faster. The only remaining knob is the ratio of tokens produced per forward pass.

Speculative decoding exploits a specific inefficiency: the decode pass is *memory-bound*, so once you have streamed the weights past the compute units you have significant idle arithmetic capacity. A tiny draft model that guesses 4 tokens costs almost nothing compared to one target forward pass. If most of its guesses are right, you amortize one weight read across multiple tokens — and you do it without touching the target's output distribution.

The diagram at `assets/speculative-decoding.svg` shows the full pipeline: the draft's K-step chain, the target's single verify pass, the left-to-right accept/reject loop, the residual-sample on first rejection, and the bonus token on full acceptance. Keep it open while reading the next section.

## The Concept

### The two-model setup

- **Target** M_p: the slow high-quality model. Distribution p. This is what we want samples from.
- **Draft** M_q: a small fast predictor. Distribution q. Typically 5–30× smaller than the target.

Per speculative step, at position t:

1. The draft generates K tokens autoregressively: x_1, …, x_K ~ q.
2. The target runs **one forward pass** over the K+1 positions and returns p(· | prefix + x_1…x_k) for k = 0…K.
3. An accept/reject rule consumes the drafted tokens left to right, accepting the longest matching prefix and either resampling from a corrected distribution on the first rejection or taking one bonus target-sampled token if all K are accepted.

If the draft matches the target perfectly, you produce K+1 tokens per target forward. If the draft is wrong at position 1, you still produce exactly 1 token. You never produce less than one token per verify.

### The exactness rule (Leviathan et al., 2023)

The trick is a modified rejection-sampling test that preserves p exactly. For each drafted token x_k with draft probability q(x_k) and target probability p(x_k):

```
r ~ Uniform(0, 1)
if r < p(x_k) / q(x_k):
    accept x_k
else:
    sample replacement ~ residual(x) = max(p(x) - q(x), 0) / ||max(p - q, 0)||_1
    stop
```

When p = q the accept probability is 1. When p ≠ q the rejected sample comes from the positive-part residual, which is exactly the mass p has that q failed to cover. Combining the two branches gives you a sample drawn from p by construction — no bias, no correction factor, no temperature hack.

The greedy specialization is simpler: accept if and only if `argmax(p) == x_k`, otherwise emit `argmax(p)` and stop.

### Expected speedup

If the per-token accept probability is α (averaged over positions), the expected number of tokens per target forward pass is:

```
E[tokens per verify] = (1 - α^{K+1}) / (1 - α)
```

At α = 0.8 and K = 4 that is 3.36 tokens per target forward. A plain decode produces 1 token per target forward, so the ceiling speedup is 3.36× — minus the cost of the K draft steps, which is negligible when cost(target) ≫ K · cost(draft).

The only real parameter is α. A good draft is everything.

Example values at K = 4:

| α    | E[tokens per verify] | Ceiling speedup |
|------|----------------------|-----------------|
| 0.50 | 1.94                 | 1.94×           |
| 0.70 | 2.83                 | 2.83×           |
| 0.80 | 3.36                 | 3.36×           |
| 0.90 | 4.10                 | 4.10×           |

Raising K past 6–8 rarely helps: the `α^{K+1}` term compounds geometrically, and each extra draft step adds a serial latency beat you cannot hide.

### Training the draft: distillation

A random small model makes a bad draft. The standard recipe is to distill the draft against the target's output distribution:

1. Pick a small draft architecture sharing the target's tokenizer.
2. Run the target over a large corpus, store its next-token distributions.
3. Train the draft with KL divergence against those stored distributions, not against ground-truth labels.

Production acceptance rates land in 0.6–0.8 for chat, 0.7–0.85 for code, and fall sharply for creative writing at high temperature.

### EAGLE: feature reuse and tree drafting

Li, Wei, Zhang, Zhang (2024, "EAGLE") made two observations. First, a separate draft model re-derives features the target already computed during its last verify — the target's final hidden state is a compressed forecast of the next token and can be fed directly into the draft. Second, a linear chain of K draft tokens throws away cheap parallelism: the draft could output a *tree* of candidates, and the target's single verify pass can check all branches in parallel using a tree-shaped attention mask, then accept the longest correct path.

EAGLE-1 makes the draft a single transformer decoder layer whose input is the target's last hidden state plus the last emitted token. EAGLE-2 (EMNLP 2024) adds a dynamic tree that grows wider where the draft is uncertain and stays narrow where it is confident.

EAGLE-3 (NeurIPS 2025, arXiv:2503.01840) replaces feature prediction with direct token prediction and fuses features from multiple layers of the target — which removes the information bottleneck that limited EAGLE-1/2 when training data scaled up. Spec-Bench reports mean accepted tokens per step ≈ 4.65 with EAGLE-3 and roughly a 2.4× wall-clock speedup at batch size 1.

### Tree attention verification

When the draft emits a tree, the target verifies every node in a single forward pass by switching its attention mask from "strictly lower-triangular over the line" to "strictly lower-triangular over the tree topology". Each node attends only to its ancestors. Verify cost grows in the number of tree nodes, not the depth, and the longest correctly-predicted root-to-leaf path is returned.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

If `a,b` are competing first candidates and `c,d,e,f` are continuations, all six positions are verified in one forward pass; the output is the longest prefix along any accepted path.

### When it wins, when it doesn't

**Wins**: code, structured outputs, common chat — high α, small vocab mass where the draft and target disagree. Memory-bound batch sizes (1–8) where the target has idle FLOPs.

**Loses**: creative writing at temperature ≥ 1.0, where α collapses toward 1/|vocab| and the draft overhead dominates. High-concurrency batched serving where continuous batching has already filled the FLOPs.

Production shops report 2–3× on chat, 3–5× on code, near-zero on creative writing.

## Build It

The shipped `code/main.py` is stdlib-only and operates on **toy discrete distributions** (vocab size 16) so the math is visible without a tokenizer or a GPU. It implements:

1. A `target` distribution and a perturbed `draft` distribution on a shared vocab, both built from fixed seeds for reproducibility.
2. `speculative_step(target_dists, draft_model, prefix, K)` — one round of Leviathan sampling. Draft proposes K tokens; target gives p at each drafted position; accept/reject loop runs left-to-right; residual is computed on rejection; bonus sample is taken on full acceptance.
3. A distribution-equivalence test: run speculative decoding N=200k times and plain target sampling N=200k times, bucket the output token at position 0, and assert total-variation distance < 0.01. This is the empirical exactness check the 2023 paper is famous for.
4. A closed-form α-vs-speedup report that walks α ∈ {0.5, 0.7, 0.9} × K ∈ {1…8} and prints the expected tokens-per-verify surface.
5. A per-position acceptance-rate counter so you can see α drift as the prefix grows.

Run:

```
python3 code/main.py
```

Expected output: an acceptance-rate table, a TV-distance line (`TV = 0.0034`-ish on the default seed), the speedup surface, and a `PASS` banner. The whole run finishes in under a second.

## Use It

- **vLLM** exposes speculative decoding as first-class config. Pass `--speculative-model <draft>` and `--num-speculative-tokens K`; add `--speculative-draft-tensor-parallel-size` when the draft is GPU-resident. EAGLE-3 is supported via the `--speculative-method eagle3` branch in v0.7+.
- **SGLang** ships EAGLE-2 and EAGLE-3 support and composes them with RadixAttention prefix caching.
- **NVIDIA TensorRT-LLM** ships Medusa heads and EAGLE trees as two distinct serving modes with tuned kernels for tree-structured attention.
- **Reference drafts**: Llama 3 family ships a 1B draft that is compatible with the 8B/70B/405B checkpoints; Qwen3 ships a 0.6B draft for the 32B target.
- **Medusa** (Cai et al., 2024) is the deployable alternative when you cannot afford a separate draft: K prediction heads on the target itself, trained via self-distillation. Simpler to deploy, slightly lower α than EAGLE.
- **Lookahead decoding** (Fu et al., 2024) is draft-free — it re-uses n-grams generated by the target's own prior Jacobi iterations and verifies them. Works best when the output has repeated phrases (code, structured outputs).

## Ship It

This lesson produces `outputs/skill-spec-decoder.md` — a skill that takes a workload profile (target model size, task mix, temperature, batch size, serving engine) and recommends a speculative-decoding configuration (draft model, K, tree width, temperature policy, and whether to fall back to plain decode).

## Exercises

1. **Exactness under mismatch.** Deliberately mismatch the draft (e.g., scramble its probabilities) and re-run the TV-distance test. Verify that the output still matches plain target sampling within the empirical tolerance. This is what the rejection rule buys you.

2. **Find the optimal K.** For α ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, find the K that maximizes `(1 - α^{K+1}) / (1 - α) / (K · cost_draft + cost_target)` assuming `cost_draft / cost_target ∈ {0.05, 0.1, 0.2}`. Plot.

3. **Tree vs chain.** Extend `main.py` so the draft emits top-2 branches at each of D depths. Build the tree attention mask as an adjacency matrix; verify that the `target` accepts the longest correct path and that the output distribution is still exactly p.

4. **Temperature collapse.** Sweep temperature from 0.1 to 2.0 in the draft and target. Measure α at each setting. Reproduce the collapse that motivates "do not speculate at T ≥ 1.0 for creative writing".

5. **Draft training stub.** On the toy distribution, fit the draft by minimizing KL(target || draft) with gradient descent over a softmax parameterization. Show α rises as KL falls.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Target model | "The big model" | The slow high-quality model; samples from p |
| Draft model | "The speculator" | The small fast predictor; samples from q; 5–30× smaller |
| K (draft length) | "Lookahead" | Number of speculated tokens per verify pass |
| α (acceptance rate) | "Hit rate" | Per-token probability the draft's proposal is accepted |
| Exact rejection rule | "The accept test" | r < p/q compare that keeps the overall sample distributed as p |
| Residual distribution | "Corrected p − q" | max(p − q, 0) / ||max(p − q, 0)||₁ — sampled on rejection |
| Bonus token | "Free sample" | Extra token drawn from p after all K proposals accepted |
| Tree drafting | "Branching speculation" | Draft outputs a tree of candidates verified in one pass |
| Tree attention mask | "Topological mask" | Causal mask encoding tree topology; each node attends only to ancestors |
| Medusa heads | "Parallel heads" | K extra prediction heads on the target; no separate draft |
| EAGLE feature reuse | "Hidden-state draft" | Draft input is the target's last hidden state, not raw tokens |
| Multi-layer feature fusion | "EAGLE-3 trick" | Draft conditions on features from several target layers, not just the top |
| Direct token prediction | "EAGLE-3 head" | EAGLE-3's draft predicts tokens, not features, fixing the scaling cliff |

## Further Reading

- [Leviathan, Kalai, Matias — "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)](https://arxiv.org/abs/2211.17192) — the exact rejection rule and the speedup analysis.
- [Chen, Borgeaud, Irving et al. — "Accelerating Large Language Model Decoding with Speculative Sampling" (DeepMind 2023)](https://arxiv.org/abs/2302.01318) — the concurrent derivation from DeepMind.
- [Cai, Li, Geng, Peng, Lee, Chen, Dao — "Medusa" (2024)](https://arxiv.org/abs/2401.10774) — parallel-heads alternative to a draft model.
- [Li, Wei, Zhang, Zhang — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (ICML 2024)](https://arxiv.org/abs/2401.15077) — feature reuse plus tree drafting.
- [Li, Wei, Zhang, Zhang — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (EMNLP 2024)](https://arxiv.org/abs/2406.16858) — dynamic tree topology.
- [Li, Wei, Zhang, Zhang — "EAGLE-3: Scaling up Inference Acceleration via Training-Time Test" (NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — direct token prediction plus multi-layer feature fusion.
- [Fu, Bailis, Stoica, Zhang — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" (ICML 2024)](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead, the draft-free alternative.
- [Rodionov et al. — "Hogwild! Inference: Parallel LLM Generation via Concurrent Attention" (NeurIPS 2025 Spotlight)](https://arxiv.org/abs/2504.06261) — a parallel-workers alternative to speculative decoding.
- [Kumar, Dao, May — "Speculative Speculative Decoding" (ICLR 2026)](https://arxiv.org/abs/2603.03251) — overlapping draft speculation with verification itself.
- [Oliaro et al. — "SuffixDecoding" (NeurIPS 2025 Spotlight)](https://arxiv.org/abs/2411.04975) — suffix-tree drafting that hybridizes with EAGLE-3 for 2.5× on Spec-Bench.
