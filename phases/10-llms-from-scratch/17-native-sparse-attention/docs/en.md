# Native Sparse Attention

> Three views of the same past: a blurry summary of everything, a sharp crop of the few blocks that matter, and the last few tokens in full detail. A gate fuses them. That is NSA.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10 Lessons 07-08 (transformer attention), Lesson 12 (inference optimization, KV cache)
**Time:** ~90 minutes

## Learning Objectives

- Explain why standard attention becomes the dominant cost at long context and what sparse attention actually skips
- Implement the three parallel branches of NSA (compression, selection, sliding) from scratch in pure Python
- Derive selection block scores from compression attention probabilities instead of a separate scorer
- Reason about the gating mechanism that fuses the three branches and why it must be learned, not averaged
- Compare NSA to MoBA, Gemma 3 interleaved sliding window, and the sparse attention taxonomy from the 2025 literature

## The Problem

A 128k-token context on a 70B model carries a KV cache of roughly 40 GB. Every decode step reads all of it. The model you deployed for long-document analysis now spends 97% of wall time moving keys and values from HBM to the matmul units and 3% doing math. You can quantize the cache, you can shrink the heads with GQA, you can evict old tokens. None of it changes the fact that full attention reads every key for every query.

Sparse attention is the harder fix: stop reading most of the keys. The question is which ones. Random dropout works for training but wrecks generation. Strided patterns miss long-range facts. Eviction throws away tokens a later query will need. Every training-free scheme in the 2025 Sparse Frontier survey hits a wall around 50% sparsity on long-context QA.

DeepSeek and Moonshot both landed on the same answer within 48 hours in February 2025: do not pick one pattern. Run three in parallel. Compress the whole sequence for global reach. Select a handful of full-resolution blocks for precision. Keep a sliding window for local fluency. Fuse the outputs with a learned gate. Train the model with this attention from scratch so the weights cooperate with the sparsity. That is Native Sparse Attention (NSA).

NSA won the ACL 2025 Best Paper award. On a 27B backbone pretrained with 260B tokens at 64k context, it matches full attention on reasoning benchmarks while reading 8.6% of the keys.

## The Concept

### The three branches

For every query position `t`, NSA builds three different key-value views of the prefix `1..t-1` and runs attention three times in parallel:

| Branch | What it reads | Purpose |
|--------|--------------|---------|
| Compression | `~t/d` coarse block summaries | Global reach, cheap |
| Selection | `top_n` blocks at full resolution | Precision where it matters |
| Sliding | Last `w` tokens verbatim | Local syntax and fluency |

A gating MLP emits three scores in `[0,1]`. The final output is a weighted sum of the three branch outputs. Every branch sees the same query but a different slice of the past.

### Branch 1: compression

Group consecutive keys into overlapping blocks of length `l`, striding by `d < l`. A learned MLP with intra-block positional encoding pools each block into a single key vector. Same treatment for values. You now have about `(t - l) / d + 1` compressed key-value pairs standing in for the full sequence.

```
keys:        [k_0 k_1 k_2 k_3 k_4 k_5 k_6 k_7 k_8 k_9 ...]
block l=4:   [---- block 0 ----][---- block 2 ----]
stride d=2:      [---- block 1 ----][---- block 3 ----]
compressed:  [K_0]    [K_1]    [K_2]    [K_3]   ...
```

Attention over these compressed keys is cheap and covers the entire context. It cannot resolve which specific token inside block 5 mattered, but it can tell you that block 5 mattered.

### Branch 2: selection

This is the clever part. Instead of training a separate scorer to pick important blocks, NSA reuses the compression attention probabilities. Each compressed key came from a specific span of raw keys, so the attention weight over a compressed key is also a weight over its source span. Aggregate those weights into selection-block buckets (size `l'`), take the top `n`, and read those blocks at full resolution.

```
compression probs:   [0.01 0.02 0.41 0.03 0.31 0.04 0.01 0.17 ...]
                            ^          ^               ^
selection buckets:   [__bucket 0__][__bucket 1__][__bucket 2__][__bucket 3__]
top-2 buckets:        picks 0 and 2 (highest aggregate mass)
```

No extra parameters, no extra forward passes. The selection branch is parasitic on the compression branch, which means sparsity is learned end-to-end through the same gradient signal that trains the MLP compressor.

Two extras that stop the model from drifting off the anchor points:

- **Initial blocks:** block 0 is always forced into the top-n. Early tokens in a document often carry global structure (system prompt, task framing).
- **Local blocks:** the `num_local_blocks` ending at position `t` are always forced in. Sliding takes the last `w` tokens; these selection slots keep the block just before the window.

For grouped-query attention, the importance scores are summed across query heads in the same group before ranking. All heads in a group read the same selected blocks, so the hardware can issue one contiguous load instead of `H` scattered ones. This is the "hardware-aligned" in the paper title.

### Branch 3: sliding window

Standard windowed attention over the last `w` tokens. Gemma 3 shipped this idea alone at 5:1 ratio against global layers and shaved KV cache from 60% to 15% of VRAM with no perplexity cost. NSA uses it as one of three branches per layer rather than as a whole-layer replacement.

The sliding branch exists because the first two branches are biased toward older tokens. Compression pools future-facing info into block summaries, selection picks a small number of blocks by aggregate mass, and neither loves the freshest 32 tokens where syntax lives. Sliding fixes that without stealing gradient from the other branches.

### The gate

```
g = sigmoid(W_gate @ q)    # shape (3,)
o = g[0] * cmp_out + g[1] * slc_out + g[2] * win_out
```

Three independent sigmoid gates, not a softmax. The branches do not have to compete; the model can turn on all three at once when the query needs all three views. In practice the authors report gate values cluster around different regimes per layer: early layers lean on sliding, middle layers on selection, late layers blend all three.

### Token flow

```
query q_t
   |
   +---- compress K,V into blocks ----> cmp_K, cmp_V --+-- attend --> cmp_out
   |                                                   |
   |                                                   +-- probs --+
   |                                                               v
   +---- aggregate probs into sel buckets, take top_n --> sel_K, sel_V --> slc_out
   |
   +---- slice last w tokens ----> win_K, win_V --> win_out
   |
   +---- W_gate @ q_t --> g
                            \
                             v
                      o = g · (cmp_out, slc_out, win_out)
```

### Why NSA is native, not bolt-on

Training-free sparse methods (H2O, StreamingLLM, Quest) run an existing dense model through a sparse pattern at inference time. The model was trained on full attention; the sparsity corrupts its distribution and accuracy drops at high sparsity. The Sparse Frontier survey (arXiv:2504.17768) confirms this: all six training-free methods degrade sharply above 0.8 sparsity.

NSA trains the attention sparsity from scratch. The compression MLP, the gate, and the selection block size are part of the architecture. Gradients flow through all three branches. At 27B scale with 260B training tokens, NSA matches or beats full attention on MMLU, GSM8K, and HumanEval while reading a fraction of the keys. That is what "native" buys.

## Build It

Full code in `code/main.py`. Stdlib only, one query at a time, so the shapes stay legible.

### Step 1: the three utilities

Dot product, softmax, attention over a list of keys and values. Nothing NSA-specific yet.

    def attention(q, keys, values, scale):
        if not keys:
            return [0.0] * len(q)
        w = softmax([dot(q, k) * scale for k in keys])
        d = len(values[0])
        out = [0.0] * d
        for wi, v in zip(w, values):
            for j in range(d):
                out[j] += wi * v[j]
        return out

### Step 2: compression

Pool each block with positional bias, project with a learned matrix. Stride `d < l` so blocks overlap. This gives more compressed keys than non-overlapping blocks and smoother attention gradients.

    def build_compressed(keys, values, cfg, params):
        ck, cv, spans = [], [], []
        i = 0
        while i + cfg.block_l <= len(keys):
            ck.append(compress_block(keys[i:i + cfg.block_l], params["k_pos"], params["k_proj"]))
            cv.append(compress_block(values[i:i + cfg.block_l], params["v_pos"], params["v_proj"]))
            spans.append((i, i + cfg.block_l))
            i += cfg.stride_d
        return ck, cv, spans

### Step 3: selection scores from compression probs

For each compressed key with attention probability `p`, distribute `p / span_len` to every raw position it covered, then bin by selection-block size `l'`. Top-n blocks plus forced init and local blocks become the selection set.

    def aggregate_scores(comp_probs, spans, num_sel, sel_block):
        scores = [0.0] * num_sel
        for p, (start, end) in zip(comp_probs, spans):
            for pos in range(start, end):
                sb = pos // sel_block
                if sb < num_sel:
                    scores[sb] += p / (end - start)
        return scores

### Step 4: gather and attend, three times

Gather the selected blocks at full resolution, slice the sliding window, then call `attention` on each. Three independent calls, same query.

### Step 5: gate and fuse

    gates = [sigmoid(dot(row, q)) for row in params["gate_w"]]
    fused = [gates[0] * cmp_out[j] + gates[1] * slc_out[j] + gates[2] * win_out[j]
             for j in range(cfg.d_head)]

### Step 6: verify

Run with seq=128, `top_n=4`, `window_w=16`. Print read counts, gates, and the L2 distance to full attention. Scale the config to the DeepSeek production defaults (`l=32, d=16, l'=64, n=16, w=512`) and print asymptotic read counts at 8k, 32k, 64k context. The expected pattern: NSA reads drop from ~25% of full at 8k to ~8.6% at 64k. Sparsity grows with context length, exactly what you want for long-context serving.

## Use It

### What the DeepSeek kernel does that this code does not

- **Triton-fused branches.** The three branches run in one kernel with shared q-tile loads instead of three sequential attentions.
- **GQA head grouping.** All heads in a group read the same selected blocks, so the scatter cost is amortized. This file runs a single head; add an outer loop and a sum-of-probs across heads in the group before ranking to match the paper.
- **Block-aligned sizes.** `l = 32`, `l' = 64`, `d = 16`. The compression stride is half the block length; the selection block is twice the compression block. This alignment lets the aggregate-scores step reduce to a simple reshape instead of a loop.

### Comparison with neighboring ideas

- **MoBA (Moonshot, arXiv:2502.13189).** Same week as NSA. One branch, parameter-less top-k gating over blocks. Simpler to drop in but requires continued training on existing checkpoints and has no coarse branch to guide selection. Deployed in Kimi at 1M context with 6.5x speedup.
- **Gemma 3 interleaved sliding window (arXiv:2503.19786).** 5:1 local-to-global layer ratio, window 1024. Same sliding idea as the NSA branch but used as whole-layer replacement instead of per-layer branch. Cheaper and simpler, no selection mechanism.
- **DeepSeek Sparse Attention (DSA).** The production successor in DeepSeek V4, built on NSA with further hardware tuning.

The Sparse Frontier survey groups methods along four axes: which tokens to score (query-to-key vs block-to-block), where to apply sparsity (prefill vs decode), what budget policy to use (fixed vs adaptive), and whether the model was trained with the sparsity. NSA is block-to-block, both prefill and decode, fixed budget, natively trained. That combination is the reason it holds up at 0.9+ sparsity.

## Ship It

This lesson's artifact is a skill at `outputs/skill-sparse-attn.md` that walks the decision of whether to reach for NSA, MoBA, Gemma-style sliding, or plain full attention on a concrete deployment. The skill forces you to name the context length, the batch size, the target sparsity, and the training budget before recommending anything.

## Exercises

1. **Easy.** Change the gate from three sigmoids to a softmax. Rerun the demo. Why does the paper prefer independent sigmoids?
2. **Medium.** Add a second head and aggregate selection scores across heads (GQA-style). Confirm all heads in the group read the same block ids.
3. **Hard.** Replace the compression MLP with simple mean pooling. Retrain the gate on a toy copy task where the answer is buried 200 tokens back. Show that removing the MLP costs accuracy even when selection is still on.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sparse attention | "Skip some tokens" | Read a deterministic fraction of the KV cache chosen by a learned or fixed pattern |
| Native sparse attention | "Sparse attention that is fast" | Sparse attention whose parameters were trained jointly with the model, not bolted on at inference |
| Compression branch | "A pooling trick" | Overlapping-block MLP that produces cheap surrogate keys whose attention probs drive selection |
| Selection branch | "Top-k KV eviction" | Reads top-n full-resolution blocks picked by aggregating compression probabilities; no extra scorer |
| Sliding branch | "Local attention" | Fixed window over the last `w` tokens, guaranteeing fresh tokens always participate |
| Gate | "Softmax over branches" | Three independent sigmoids so branches can all fire at once |

## Further Reading

- [Native Sparse Attention (Yuan et al., ACL 2025)](https://arxiv.org/abs/2502.11089) — the paper this lesson implements. Read sections 3 and 4 for the full equations.
- [MoBA: Mixture of Block Attention (Lu et al., 2025)](https://arxiv.org/abs/2502.13189) — the Moonshot parallel, same week, different design choice. Good contrast.
- [The Sparse Frontier (Nawrot et al., 2025)](https://arxiv.org/abs/2504.17768) — the taxonomy and evaluation survey. Read before picking a method for production.
- [Gemma 3 Technical Report (Team Gemma, 2025)](https://arxiv.org/abs/2503.19786) — interleaved sliding window, what NSA's window branch looks like as a whole-layer strategy.
