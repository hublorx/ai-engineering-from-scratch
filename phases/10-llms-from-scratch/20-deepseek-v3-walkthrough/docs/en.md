# DeepSeek-V3 Architecture Walkthrough

> 671B total parameters, 37B active per token, 61 transformer layers. The most talked-about open weight release since Llama. This lesson reads the tech report line by line and rebuilds the parameter count from scratch. When you are done, you can look at any column of the DeepSeek-V3 config and name the arithmetic behind it.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10, Lessons 04, 05, 11, 12, 14 (Pre-training, Scaling, Quantization, Inference, Open Model Walkthroughs)
**Time:** ~60 minutes

## Learning Objectives

- Rebuild the 671B / 37B headline number from the published hyperparameters
- Explain Multi-head Latent Attention (MLA) and why its KV cache shrinks ~40x against a naive MHA baseline at the same head count
- Explain fine-grained DeepSeekMoE with 256 routed experts plus 1 shared expert and top-8 routing
- Explain auxiliary-loss-free load balancing and why it replaced the aux-loss routers of DeepSeek-V2 and Mixtral
- Explain sequential Multi-Token Prediction (MTP) and why the MTP module adds roughly 14B of extra weights on disk
- Read the DeepSeek-V3 `config.json` and translate every field into a parameter-count contribution

## The Problem

Lesson 14 gave you a six-knob framework for open model architectures. DeepSeek-V3 is the model that pushes every knob to an extreme. It keeps RMSNorm, SwiGLU, and RoPE like the rest of the frontier, but it replaces standard attention with MLA, replaces the standard MoE with a fine-grained version that routes to 8 of 256 experts plus 1 always-on shared expert, drops the auxiliary load-balancing loss that Mixtral and DeepSeek-V2 relied on, and tacks an extra prediction head on the end for sequential multi-token prediction.

None of these are exotic. Each one is a simple substitution you can compute by hand. Put together, they produce a 671B MoE that activates only 37B per token, runs with a KV cache that is more than an order of magnitude smaller than an MHA-equivalent, and was pre-trained for 2.788M H800 GPU hours, which is cheap by frontier standards.

The goal of this lesson is not to train DeepSeek-V3. The goal is to read its architecture spec and recognize every number in it. You should be able to look at `num_attention_heads=128`, `kv_lora_rank=512`, `q_lora_rank=1536`, `moe_intermediate_size=2048`, and reconstruct the total parameter count to within a percent, without running a single forward pass.

## The Concept

### The Headline Numbers

From the DeepSeek-V3 Technical Report (arXiv:2412.19437) and the published `config.json`:

| Field | Value |
|-------|-------|
| Total parameters | 671B |
| Activated per token | 37B |
| Layers | 61 |
| Hidden dim | 7168 |
| Attention heads | 128 |
| MLA KV latent dim | 512 |
| MLA Q latent dim | 1536 |
| Per-head non-RoPE dim | 128 |
| Per-head RoPE dim | 64 |
| Per-head V dim | 128 |
| Dense FFN width (first 3 layers) | 18432 |
| MoE FFN width (per expert) | 2048 |
| Routed experts per MoE block | 256 |
| Shared experts per MoE block | 1 |
| Experts activated per token | 8 |
| MTP depth | 1 |
| Vocabulary | 129280 |
| Context length | 163840 |
| Training tokens | 14.8T |
| Training cost | 2.788M H800 GPU hours |

Every row is a number you can do arithmetic on. This lesson does the arithmetic.

### Layer Mix: 3 Dense + 58 MoE

DeepSeek-V3 stacks 61 transformer blocks. The first 3 use a standard dense SwiGLU MLP with intermediate width 18432. The remaining 58 replace the MLP with a DeepSeekMoE block. The motivation is stability at the start of the stack, where token representations are noisy and routing is brittle. DeepSeek-V2 uses the same 3-layer dense preamble.

### Multi-Head Latent Attention (MLA)

Standard MHA stores one key and one value per head per token in the KV cache. At 128 heads with head dim 128 across 61 layers and 128k context, that is more than 700 GB per sequence at BF16. Unusable.

Grouped-Query Attention (the Llama 3 answer) shrinks this by sharing K and V across groups of Q heads. DeepSeek went further with MLA: compress the entire KV representation into a 512-dimensional latent per token, and re-project it into per-head K and V at compute time. The cache stores only the latent plus a small 64-dim RoPE head. The per-layer per-token cache shrinks from `2 * 128 * 128 = 32768` dimensions to `512 + 64 = 576` dimensions. A factor of roughly 42x compression.

The tradeoff is more math per token. MLA introduces down-projection matrices and two up-projection matrices (one for K-nope, one for V) that did not exist in MHA. The six matrices per layer are:

- `W_DQ`: hidden to Q-latent (7168 x 1536 = 11.0M)
- `W_UQ`: Q-latent to per-head Q (1536 x 128 x (128+64) = 37.7M)
- `W_DKV`: hidden to KV-latent plus RoPE head (7168 x (512+64) = 4.1M)
- `W_UK`: KV-latent to per-head K-nope (512 x 128 x 128 = 8.4M)
- `W_UV`: KV-latent to per-head V (512 x 128 x 128 = 8.4M)
- `W_O`: per-head V to hidden (128 x 128 x 7168 = 117.4M)

Total: ~187M per layer. Compare to a standard 128-head MHA at the same dims which would be roughly `4 * h^2 = 205M` per layer. MLA is actually *cheaper* in parameter count while giving a 42x smaller cache. That is why every DeepSeek model since V2 uses it.

The decoupled RoPE head is the clever bit. You cannot rotate a compressed latent and recover the original attention scores, so DeepSeek carves out a small 64-dim slot per head that carries the positional rotation outside the compression path. The non-RoPE 128 dims per head ride through the latent, rotary-free.

### DeepSeekMoE: Fine-Grained Experts with Shared Expert Isolation

Mixtral 8x7B has 8 experts per block, picks top-2. DeepSeek-V3 has 256 routed experts per block and picks top-8. Same active fraction of the router's output mass, radically different granularity. Smaller experts (2048-wide each, vs 14336 for a dense Llama-style FFN) specialize on finer patterns, and sparsity at scale recovers accuracy lost to single-expert aliasing.

One always-on **shared expert** (also 2048-wide) catches general features that every token needs. This is shared-expert isolation: factor the "common knowledge" MLP out of the expert pool so the routed experts do not waste capacity re-learning it across many experts.

Per expert: SwiGLU MLP with `3 * hidden * moe_ff = 3 * 7168 * 2048 = 44.0M` parameters. A block holds 256 routed plus 1 shared = 257 experts, totalling **11.32B** of expert weights. Plus the router (7168 x 256 = 1.84M), plus MLA attention (187M), plus two RMSNorms. The whole MoE block weighs **11.5B** parameters on disk.

Per forward pass only 9 experts (8 routed + 1 shared) are active, so the active block cost is:

- MLA attention: 187M
- 9 x 44M expert: 396M
- Router + norms: ~2M
- **Active per MoE block: ~585M**

58 MoE blocks x 585M = 33.9B. Add the 3 dense blocks (~583M each = 1.75B), the embedding (927M), the LM head (927M), and final norm, and you land at 37.6B active. Close to the advertised 37B. See `code/main.py` for the exact derivation.

### Auxiliary-Loss-Free Load Balancing

Every MoE before DeepSeek-V3 used an auxiliary load-balancing loss to keep experts from collapsing (routers prefer a handful of experts and the rest die). The aux loss penalizes imbalance at each step. The downside: it degrades the primary language modeling objective. You pay a perplexity tax to keep routing healthy.

DeepSeek-V3 introduces a bias term `b_i` added to each expert's gating logit before the top-k selection. After each step, experts that got too few tokens have their bias bumped up, and experts that got too many have it bumped down. The bias only affects *selection*, not the gating weight used in the output. The result: load balances without any auxiliary gradient. The main LM loss runs unperturbed.

The switch from softmax to sigmoid-with-normalization on expert scores is secondary but worth noting. V2 used softmax over 160 experts. V3 applies sigmoid per expert, then normalizes. This scales better for top-k over large expert pools.

### Node-Limited Routing (NLR)

Training runs across 2048 H800 GPUs in 256 nodes of 8 GPUs. With 256 experts distributed across 64 nodes for expert parallelism, routing any token to any expert means unpredictable cross-node traffic. DeepSeek caps each token to at most 4 nodes. The 256 experts are grouped into 8 groups of 32 (one group per node), the top-4 groups are selected first via a group-level affinity score, then top-k routing happens within those 4 groups. This trades a tiny routing flexibility loss for predictable, bounded communication. It is a systems decision, not an architectural one — but it changes what experts can co-specialize on.

### Sequential Multi-Token Prediction (MTP)

The main model predicts the next token, standard autoregressive cross-entropy. MTP adds a second head that predicts the token after that. At training time the MTP loss is added with a small weight to the main loss, which provides a denser training signal per position and slightly improves data efficiency.

The MTP module is itself a transformer block: it takes the main model's output embedding, projects and RMSNorms it, concatenates the embedding of the next token, runs one more MoE block, and projects through a shared LM head. At `mtp_depth = 1`, this is one MTP module. It reuses the main model's embedding table and LM head (tied), but its transformer block has its own attention and MoE weights. That block weighs about 11.6B parameters.

Combined with the shared embedding and output head, the published weight total on HuggingFace is 685B: 671B for the main model plus 14B for the MTP module.

At inference MTP can be discarded (no MTP, just sample next token like a normal model) or used for speculative decoding: let the MTP head propose the next-next token and verify it with the main model for a 1.8x decoding speedup reported in the paper.

### Node-Limited Routing + FP8 Training = 2.788M H800 Hours

DeepSeek-V3 trained on 14.8T tokens with 16-way pipeline parallelism and 64-way expert parallelism across 256 nodes. They avoided tensor parallelism entirely. FP8 mixed precision plus custom DualPipe overlap drove the reported 2.788M H800 GPU hours — roughly $5.6M at $2/hour. This is an order of magnitude less than prior frontier MoE training costs and is why the paper was a news event, not just a research release.

## Build It

### Step 1: Model the hyperparameter dict

Every field in `V3` corresponds to a line in the DeepSeek-V3 config.

    V3 = {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "first_dense_layers": 3,
        "num_attention_heads": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "experts_per_token": 8,
        "intermediate_size": 18432,
        "vocab_size": 129280,
        "mtp_depth": 1,
    }

### Step 2: Count MLA parameters

Six matrices plus two norms. Computing each line against the table above:

    def mla_attention_params(c):
        h, q, kv = c["hidden_size"], c["q_lora_rank"], c["kv_lora_rank"]
        heads, nope, rope, v = (c["num_attention_heads"],
                                c["qk_nope_head_dim"],
                                c["qk_rope_head_dim"],
                                c["v_head_dim"])
        return (h * q + q * heads * (nope + rope)
                + h * (kv + rope) + kv * heads * nope
                + kv * heads * v + heads * v * h
                + q + kv)

187M per layer. Every layer uses this, dense or MoE.

### Step 3: Count expert and MoE block parameters

One expert SwiGLU has `3 * h * moe_ff` parameters. A MoE block stores all 257 experts. Active per token: 9 of them plus the router.

    def swiglu(h, ff):
        return 3 * h * ff

    def moe_block_total(c):
        expert = swiglu(c["hidden_size"], c["moe_intermediate_size"])
        experts = (c["n_routed_experts"] + c["n_shared_experts"]) * expert
        router = c["hidden_size"] * c["n_routed_experts"] + c["n_routed_experts"]
        return mla_attention_params(c) + experts + router + 2 * c["hidden_size"]

### Step 4: Assemble the full stack

Add embedding, 3 dense blocks (SwiGLU at width 18432), 58 MoE blocks, final RMSNorm, LM head, and MTP module. Compare total-with-MTP and main-model-only against the headline.

Run `python3 code/main.py`. The output prints the per-component breakdown and a headline check:

    MAIN MODEL (excl. MTP)              total=  671.03B  active=   37.55B
    headline check (main model only, MTP reported separately as 14B)
      reported (DeepSeek-V3 tech report): 671B total / 37B active
      calculated                        : 671.0B total / 37.6B active
      delta                             : 0.00% total, 1.49% active

Under 2% on the active side. The total lands exactly on 671B.

### Step 5: Compute the KV cache win

    def kv_cache_bytes(c, seq, dtype=2):
        latent = c["kv_lora_rank"] + c["qk_rope_head_dim"]
        return 2 * c["num_hidden_layers"] * latent * seq * dtype

At 131,072 context, MLA KV cache is 17.2 GB. An equivalent MHA-style model with the same 128 heads and 128+64 head dim would weigh 732 GB per sequence — a 42x compression. This is the reason DeepSeek-V3 can serve long context at all.

## Use It

Feed the `V3` config into the calculator. Observe the breakdown. Flip `n_shared_experts` to 0 and note how little the active parameter count shifts (the shared expert is one of nine active). Flip `experts_per_token` to 2 (Mixtral-style routing) and watch active params drop to ~33.5B — you save 4B of compute per token at the cost of specialization capacity.

Swap `kv_lora_rank` to 128 (DeepSeek-V2's smaller compression) and you save another 4x in KV cache but the up-projection matrices shrink too, costing some per-token expressiveness.

The calculator is a playground for the architectural tradeoffs. Read each number, perturb it, re-run, and see where DeepSeek's choices show up in the budget.

## Ship It

This lesson produces `outputs/skill-dsv3-ref.md`. It is a reference skill that answers "what happens if I replace X in DeepSeek-V3 with Y" questions using the same parameter-count model as `code/main.py`, and grounds every answer in the arithmetic. You can drop it into any agent that needs to reason about MLA, fine-grained MoE, or the 671B/37B tradeoff without hallucinating numbers.

## Exercises

1. **Rebuild the MTP module weight count.** The paper reports 14B for the MTP module. Our calculator gives 11.6B at `mtp_depth=1`. The delta comes from what is shared vs duplicated — embedding table and LM head are tied. Reconcile the two numbers: what extra parameters (projection, norm, reused embedding) must we be double-counting or undercounting?

2. **MLA vs GQA at the same KV cache size.** At 128k context and 17.2 GB cache, MLA uses a 576-dim latent. What GQA configuration (Q heads, KV heads, head dim) would match that cache at the same hidden size? Compute the parameter cost of that GQA attention and compare it to MLA's 187M per layer.

3. **Expert count sensitivity.** Recompute total and active params for: (a) 128 routed experts, top-8, (b) 256 routed experts, top-4, (c) 512 routed experts, top-8. Hold the active expert budget at 8 x 44M. Which configuration gives the most total capacity per active FLOP? Which is hardest to load-balance?

4. **Drop the shared expert.** Remove the always-on shared expert (set `n_shared_experts = 0`) and route to top-9 routed experts instead. Compute the new total and active params. Beyond parameter count, what changes about the routing dynamics and what the 256 routed experts have to learn?

5. **MTP depth 2.** If DeepSeek-V3 used `mtp_depth = 2` (predict next-next and next-next-next tokens), how many more parameters does the model have on disk? Would the training improvement justify doubling the MTP cost if you believe diminishing returns from deeper MTP chains?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| MLA | "DeepSeek's attention" | Multi-Head Latent Attention: compress K and V into a 512-dim latent per token, decompress per head on the fly, stash a small 64-dim RoPE head alongside for positions — 42x KV cache compression |
| Q-latent | "DeepSeek compresses queries too" | DeepSeek-V3 also LoRA-compresses Q via a 1536-dim latent; reduces activation memory during training, parameter count is unchanged versus full-rank Q |
| DeepSeekMoE | "Fine-grained experts" | Replace 8 large experts with 256 small ones plus 1 always-on shared expert; pick top-8 routed per token; more granular specialization than Mixtral-style MoE |
| Shared expert | "Always-on expert" | One expert that every token passes through, carrying common knowledge so the 256 routed experts do not waste capacity re-learning it |
| Auxiliary-loss-free routing | "No load balancing loss" | Maintain a learned bias per expert that nudges routing toward underused experts; leaves the main LM loss untouched, replaces Mixtral/V2's auxiliary loss |
| Node-Limited Routing | "Only route to 4 nodes" | Group 256 experts into 8 groups, pick top-4 groups by aggregate score before top-k inside groups — bounds cross-node communication during training |
| MTP | "Multi-token prediction" | Extra head(s) predicting the token after the next; training-only signal at mtp_depth=1; optionally reused at inference as a speculative decoder |
| Active parameters | "What runs per token" | The params that participate in one forward pass — for V3 this is 37B out of 671B because 248 of 257 experts are skipped per token |
| Latent KV cache | "Store the compressed version" | During inference, keep only the 576-dim per-token latent in the cache, reconstruct per-head K and V when needed — the math of MLA |
| DualPipe | "Overlap comm and compute" | DeepSeek's custom pipeline-parallel schedule that hides AllToAll expert-routing communication under compute; enabled 2.788M H800 hour training cost |

## Further Reading

- [DeepSeek-AI, 2024 — "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437) — the primary source: architecture, auxiliary-loss-free balancing, MTP, training recipe
- [DeepSeek-AI, 2024 — "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"](https://arxiv.org/abs/2405.04434) — MLA is introduced and derived here; read this first if the MLA projection matrices feel magic
- [Guo et al., 2025 — "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning"](https://www.nature.com/articles/s41586-025-09422-z) — R1 uses the V3 base; GRPO reinforcement learning with rule-based rewards, Nature vol. 645 pp. 633-638
- [DeepSeek-AI, 2024 — DeepSeek-V3 model card on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3) — the `config.json` whose fields this calculator consumes verbatim
- [DeepSeek-AI, 2025 — "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"](https://arxiv.org/abs/2505.09343) — follow-up from the V3 team on why the architectural choices map onto 2026 training hardware the way they do
