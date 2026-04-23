# Jamba: Hybrid SSM-Transformer

> At 128k context a pure Transformer spends more memory on its KV cache than on its weights. State-space models have O(1) cache but lose at in-context recall. Jamba interleaves one attention layer for every seven Mamba layers and adds MoE on top — recurrence for throughput, attention for recall, experts for capacity. It is the first hybrid that actually matches dense Transformers at scale.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10, Lessons 04, 12, 14 (Pre-training, Inference optimization, Open model architectures)
**Time:** ~45 minutes

## Learning Objectives

- Explain why the Transformer KV cache dominates long-context memory and how an SSM removes that cost
- Describe Mamba's selective state space update and why it is content-aware, unlike S4
- Reconstruct Jamba's 1:7 attention-to-Mamba interleave with MoE every other layer and justify each ratio
- Compute parameter and memory footprints for Jamba at 128k context and compare against a dense Transformer of equivalent active params
- Name the exact failure modes of pure SSM models and why the attention layers are there

## The Problem

You trained a GPT in Lesson 04 and read five different Transformer dialects in Lesson 14. Every one of them pays the same tax: the KV cache. For each token generated, every layer stores a key vector and a value vector for every previous token. That cache grows linearly with sequence length and layer count, so 128k-context serving is KV cache serving. For Llama 3 8B at 128k, the KV cache is roughly 17 GB — larger than the 16 GB of weights themselves.

Recurrent networks do not have this problem. A classical RNN has O(1) state regardless of sequence length — one hidden vector that summarizes everything so far. The catch, until 2021, was that RNNs were impossibly slow to train (sequential, no parallelism) and structurally bad at in-context recall (information is crushed into a fixed-size state).

The state-space family fixed the training problem. S4 (Gu, Goel, Re 2021) showed you could parameterize a linear SSM so it has an equivalent convolutional form, which trains in parallel, while still running as a recurrence at inference. Mamba (Gu and Dao 2023) added input-dependent selectivity so the model could decide to remember or forget each token, closing most of the language-quality gap with Transformers while keeping linear time and constant cache.

Mamba alone still loses on tasks that need precise retrieval from far back in the context — phone-book lookups, in-context learning from many examples, long-range copy. Those tasks want exact content-addressed recall, which attention does natively and SSMs do not. Jamba (Lieber et al. 2024) is the architectural answer: keep Mamba for the bulk of the layers so the KV cache stays tiny, sprinkle in one attention layer every seven layers for the recall, and add Mixture of Experts to pack in capacity without paying for it per token.

## The Concept

### Why the KV cache hurts

For a Transformer decoder, generating token T+1 needs the keys and values of tokens 1..T at every layer. The cache size per sequence is:

```
kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes
```

Every factor except `bytes` is fixed by the model. Only `seq_len` is yours. Doubling context doubles the cache. GQA and MLA (Lesson 14) shrink `num_kv_heads` — they do not change the linear growth.

An SSM layer, by contrast, carries one fixed-size hidden state `h` of shape (state_dim,) regardless of how long the sequence is. If `state_dim` is 16, every SSM layer stores 16 numbers total, not 16 * seq_len. That is the "O(1) cache" advantage.

### Linear state-space models (S4)

A linear SSM maps an input sequence `u` to an output sequence `y` via a hidden state `h`:

```
h'(t) = A h(t) + B u(t)
y(t)  = C h(t) + D u(t)
```

Discretize with step size Δ and it becomes the classic RNN update `h_t = A_bar h_{t-1} + B_bar u_t`. Three properties make this useful:

1. **Recurrent form.** Inference is one matmul per step. Constant cache. Linear in sequence length.
2. **Convolutional form.** The same computation can be expressed as a global convolution `y = K * u` where `K` is a kernel derived from (A, B, C). Trains in parallel across the whole sequence.
3. **Structured A.** S4 uses a HiPPO-based parameterization that gives the continuous system well-behaved long-range memory. Mamba uses a simpler diagonal form but keeps the structure.

S4 ships these two forms as a duality: train as a convolution, infer as a recurrence.

### Mamba's selection mechanism

S4's parameters (A, B, C, Δ) are fixed per channel. Every token hits the same filter. That is why pure S4 struggles on text: it cannot choose which tokens to pay attention to.

Mamba makes (B, C, Δ) input-dependent. For each token `u_t`, the model computes:

```
B_t = Linear_B(u_t)
C_t = Linear_C(u_t)
Δ_t = softplus(Linear_Δ(u_t))
```

The SSM's input matrix, output matrix, and step size are now functions of the current token. If `Δ_t` is large, more of `u_t` is written into `h_t`. If `Δ_t` is small, the state passes through. This is the RNN analog of attention: the model can selectively propagate or ignore information.

The price: the convolutional duality breaks. With time-varying (B, C, Δ) there is no fixed kernel. Mamba replaces the parallel conv with a hardware-aware parallel scan — a custom CUDA kernel that computes the recurrence in parallel using the associativity of the state update. Linear time, parallel training, selective recurrence.

The Mamba block in its entirety:

```
x -> proj -> [conv1d -> SiLU -> selective_ssm] -> gate * proj -> y
```

One projection in, a 1D causal conv (cheap local mixing), a SiLU, the selective SSM, and a gated output projection. No attention anywhere.

### What Mamba loses

Benchmarks where Mamba still underperforms Transformers of equivalent size:

- **Exact recall from far back.** "The password is MAGENTA. ...10,000 tokens of filler... What was the password?" Transformers nail this with attention. Mamba's finite-dim state has to compress the password along with everything else.
- **Induction heads and in-context learning.** Mechanistic interpretability work shows Transformers form induction heads that do pattern copying. SSMs can approximate these but need more parameters to do so.
- **Many-shot in-context examples.** The whole point of 128k context is stuffing examples into the prompt. SSMs summarize examples into state; attention indexes them directly.

Those are exactly the workloads that benefit most from big context. So the answer is not "replace attention with Mamba," it is "keep enough attention to cover the recall tasks, use Mamba for everything else."

### The Jamba block

Jamba's core unit is a block of `l` layers. Each layer is either an attention layer or a Mamba layer, each followed by an MLP. The layer types interleave at a ratio `a : m` (attention : Mamba). MLPs every `e` layers can be swapped for Mixture of Experts.

The released Jamba-v0.1 configuration:

- `l = 8` layers per block
- `a : m = 1 : 7` — one attention layer per seven Mamba layers
- `e = 2` — MoE replaces the MLP on every other layer
- 16 total experts, top-2 active per token
- Block repeats 4 times (32 layers total)

So out of 32 layers: 4 are attention, 28 are Mamba, and 16 use MoE MLPs. Total parameters: 52B. Active per token: 12B. Fits on a single 80GB GPU with 256k context.

### Why 1:7

The Jamba ablations swept attention ratios from pure-Mamba (0:8) up to 1:1. The findings:

- Pure Mamba loses perplexity to a matched Transformer by a few tenths of nats. It specifically fails needle-in-a-haystack.
- Adding one attention layer per block (1:7) closes the gap and passes needle-in-a-haystack at 256k.
- Going denser (1:3, 1:1) gives little additional quality while reimposing the KV cache cost.

The 1:7 ratio is the Pareto point. It is the minimum attention that keeps the recall tasks honest.

### Why no positional embeddings

Attention is permutation-equivariant on its own, which is why Transformers need RoPE or learned positions. Mamba is a recurrence — the order is baked into the scan. In a hybrid stack, the Mamba layers carry the positional signal into the attention layers. Jamba confirms this empirically and ships with no explicit positional encoding at all.

### KV cache at 128k

With 4 attention layers out of 32 at 128k context, 8 KV heads, head_dim 128, BF16:

```
kv_cache = 2 * 4 * 8 * 128 * 131072 * 2 = 2.15 GB
```

Compare to a pure Transformer with the same dims but all 32 attention layers:

```
kv_cache = 2 * 32 * 8 * 128 * 131072 * 2 = 17.2 GB
```

Eight times smaller. That is the whole product thesis.

### Jamba-1.5

Jamba-1.5-Mini is the post-trained version of the original (12B active, 52B total). Jamba-1.5-Large scales the same architecture to 94B active, 398B total, and ships with ExpertsInt8 — an INT8 quantization scheme for the expert weights that fits the whole model on 8x80GB GPUs at 256k context. Same 1:7 interleave. Same MoE every other layer. Different scale.

### Mamba-3 (2026)

Mamba-3 (ICLR 2026 Oral) keeps the hybrid-ready SSM primitive but improves three things inside the SSM block itself: a trapezoidal discretization for a second-order state update, a complex-valued recurrence equivalent to data-dependent RoPE (which restores the state-tracking capability Mamba-2 had regressed on), and a MIMO formulation that increases arithmetic intensity during decode. At 1.5B scale Mamba-3 (MIMO) matches Mamba-2 with half the state size. The Jamba-shaped assembly is unchanged.

### Building block comparison

| Block | KV cache | Recall | FLOPs/token | Notes |
|-------|----------|--------|-------------|-------|
| Transformer (MHA) | O(L * H * d) per layer | excellent | O(L * d) | quadratic attention, full cache |
| Transformer (GQA) | O(L * H/G * d) per layer | excellent | O(L * d) | Llama 3 default |
| Mamba | O(state_dim) per layer | lossy on long recall | O(d * state_dim) | parallel scan, no attention |
| Jamba block | KV only on attention layers | excellent | dominated by Mamba | 1 attention per 7 Mamba |

That table is the whole lesson in five rows.

## Build It

The code for this lesson simulates a Jamba-shaped stack as pure Python. It does not train anything — the point is to compute the cache and parameter budget that justify the architecture. See `code/main.py`.

### Step 1: Represent the interleave pattern

Given `a : m`, a block of `l` layers with MoE every `e` layers, `main.py` produces a layer-by-layer schedule:

```
layers = ["mamba", "mamba", "mamba", "attn", "mamba", "mamba", "mamba", "mamba"]
mlp    = ["moe",   "mlp",   "moe",   "mlp",  "moe",   "mlp",   "moe",   "mlp"]
```

Repeat the block until the target layer count is reached. This mirrors the Jamba-v0.1 config precisely.

### Step 2: Count parameters

Attention layers are standard GQA blocks (Q/K/V/output projections and one MLP per layer). Mamba layers follow Gu and Dao 2023: an input projection, a 1D causal conv, the SSM projections for (B, C, Δ), and an output gate. MoE MLPs multiply the MLP parameter count by `num_experts` for total, but charge only `top_k` experts for active params.

### Step 3: Compute the KV cache at 128k

Only the attention layers contribute. The Mamba state is counted separately as `num_layers * state_dim * hidden_size * bytes`, typically under 100 MB for a 50B-parameter model.

### Step 4: Compare to a pure Transformer

Same hidden dim, same depth, but every layer is attention. The calculator prints side-by-side KV cache numbers and the ratio. For the Jamba-v0.1 config at 128k, the number is an 8x reduction. At 256k it is larger.

### Step 5: Active vs total params

Sum all experts into total params. Sum `top_k` experts per MoE layer plus dense MLPs plus all attention/Mamba params into active params. Print the ratio — you should see the Jamba-v0.1 numbers (52B total, 12B active, ~23% active).

See `code/main.py` for the implementation. Running it on the bundled configs reproduces the published numbers within rounding.

## Use It

Run `python code/main.py` to print the layer schedule, the parameter breakdown, and the KV cache for Jamba-v0.1 alongside an equivalent dense Transformer. Change `attn_to_mamba_ratio` to see what happens at 1:3 or 2:6. Drop `num_experts` to 1 to see the non-MoE variant.

The implementation is intentionally schematic — no tensor ops, no training. Once you believe the budget, you would reach for the official AI21 implementation to actually run weights. The point of this lesson is that the architecture is specified by a handful of ratios and the memory budget falls out of them.

## Ship It

This lesson produces `outputs/skill-jamba.md`, a skill that takes a deployment target (context length, GPU VRAM, latency budget) and decides whether a hybrid SSM-Transformer is the right answer over a pure dense Transformer. It checks the KV cache at the target context, the expected decode throughput from the SSM majority, and the recall-task requirements of the workload.

## Exercises

1. Compute the Jamba-v0.1 KV cache at 32k, 128k, and 1M context. Then compute the Mamba state size at the same three numbers. Plot both in your head. At which context length does the attention KV cache overtake the Mamba state for the first time?

2. Change the interleave from 1:7 to 1:3 at 32 layers. How many attention layers do you now have? Recompute the 128k KV cache. How much of the Jamba advantage remains?

3. The original Jamba paper reported that a pure Mamba model of matched size failed a needle-in-a-haystack test at 256k. Design the minimum experiment (prompt format, what to vary, what to measure) that would convince you the attention layers were the fix, not something else in the stack.

4. Jamba-1.5-Large has 94B active and 398B total parameters. From the 1:7 interleave and MoE-every-other-layer structure, back-calculate a plausible layer count, hidden dim, and number of experts. You should land within 20% of the published spec.

5. Mamba-3 introduces a complex-valued recurrence equivalent to data-dependent RoPE. In a Jamba hybrid, the Mamba layers already carry positional information implicitly. Does swapping Mamba-1 for Mamba-3 change whether you need explicit RoPE on the attention layers? State your prediction and the experiment that would falsify it.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| SSM | "The new RNN" | State-space model: linear recurrence h' = Ah + Bu, y = Ch + Du, with a structured A matrix |
| S4 | "Structured SSM" | Gu/Goel/Re 2021: HiPPO-based A, runs as either a convolution (train) or a recurrence (infer) |
| Selective SSM | "Content-aware Mamba" | (B, C, Δ) become functions of the current input, so the SSM can choose what to remember |
| Mamba block | "SSM layer" | proj -> conv1d -> SiLU -> selective_ssm -> gated_output — one full Mamba layer |
| Parallel scan | "How Mamba trains fast" | Hardware-aware custom kernel that computes a time-varying recurrence in parallel using associativity |
| Jamba block | "The hybrid unit" | l layers mixing attention and Mamba at ratio a:m, with MLP or MoE every e layers |
| 1:7 interleave | "One attention per seven Mamba" | Jamba's canonical ratio: minimum attention that preserves long-range recall |
| MoE in Jamba | "Sparse experts every other layer" | 16 experts, top-2, applied on every second MLP — 52B total params, 12B active |
| ExpertsInt8 | "Jamba-1.5 quantization" | INT8 storage for expert weights, dequantized to BF16 at compute, fits Jamba-1.5-Large on 8x80GB |
| Needle-in-a-haystack | "Long-context recall benchmark" | Hide a sentence inside a long document, ask the model to retrieve it verbatim — the test pure Mamba fails |

## Further Reading

- [Lieber et al., 2024 -- "Jamba: A Hybrid Transformer-Mamba Language Model"](https://arxiv.org/abs/2403.19887) -- the original architecture paper, ablations on interleave ratio and MoE placement
- [Jamba Team, 2024 -- "Jamba-1.5: Hybrid Transformer-Mamba Models at Scale"](https://arxiv.org/abs/2408.12570) -- 94B active / 398B total, ExpertsInt8 quantization, 256k context at serving
- [Gu and Dao, 2023 -- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"](https://arxiv.org/abs/2312.00752) -- selective SSMs, parallel scan, 5x inference throughput over Transformers
- [Gu, Goel, Re, 2021 -- "Efficiently Modeling Long Sequences with Structured State Spaces"](https://arxiv.org/abs/2111.00396) -- S4, the structured SSM that made state-space models competitive
- [Lahoti et al., 2026 -- "Mamba-3: Improved Sequence Modeling using State Space Principles"](https://openreview.net/forum?id=HwCvaJOiCj) -- ICLR 2026 Oral: trapezoidal discretization, complex-valued recurrence, MIMO decode
