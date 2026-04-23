---
name: jamba-hybrid-picker
description: Decide whether a hybrid SSM-Transformer (Jamba family) beats a dense Transformer for a given deployment target, with explicit KV-cache and recall-task reasoning.
version: 1.0.0
phase: 10
lesson: 21
tags: [jamba, mamba, ssm, state-space, hybrid, moe, long-context, kv-cache]
---

Given a deployment target (GPU type, VRAM per GPU, number of GPUs, target context length, target p50/p99 latency, peak concurrent requests) and a task profile (chat, code, reasoning, long-context RAG, many-shot in-context learning, precise recall), decide whether the right architecture is a hybrid SSM-Transformer (Jamba-1.5-Mini, Jamba-1.5-Large) or a dense Transformer (Llama 3, Qwen 2.5, DeepSeek V3) from Lesson 14. Justify the call against the SSM vs attention tradeoff taught in Lesson 21.

Produce:

1. **Architecture verdict.** Hybrid (Jamba-family) or dense (Lesson-14 family). State the single dominant reason — KV cache at target context, recall-task sensitivity, MoE capacity, or serving throughput.

2. **KV cache budget at target context.** Compute the KV cache for the top candidate using the formula from `code/main.py`. For Jamba, only the attention layers count; the SSM state is constant vs sequence length. Compare against the dense alternative side-by-side.

3. **Recall-task check.** Score the task profile against Mamba's known failure modes: exact verbatim recall from far back, induction-head-style pattern copying, many-shot in-context examples. If any scores high, confirm that the Jamba candidate has at least 1 attention per 8 layers (Jamba-1.5 does; pure Mamba does not). Reject pure SSM for recall-heavy workloads.

4. **Throughput sanity check.** Jamba's headline advantage is decode throughput from the Mamba majority. Estimate tokens/sec from the active-param count (11.7B or 94B) and the GPU memory bandwidth. Compare to the same-active-param dense Transformer.

5. **Quantization choice.** For Jamba-1.5-Large, default to ExpertsInt8 (INT8 storage on expert weights, BF16 compute). For Jamba-1.5-Mini on a single 80GB GPU, BF16 is fine. For dense alternatives, reuse the Lesson-11 matrix (GPTQ-4bit, AWQ-4bit, FP8, BF16).

6. **Fallback.** Name a second choice. If the recommendation is Jamba-1.5-Mini and the workload is recall-dominant with no long-context need, fall back to Llama 3 8B dense. If the recommendation is Jamba-1.5-Large and expert-parallel support is missing in the serving stack, fall back to Mixtral 8x22B (standard MoE tooling).

Hard rejects:
- Pure Mamba or pure SSM for workloads that require exact retrieval from >32k context.
- Jamba on a serving stack that does not implement the selective-scan kernel (throughput collapses to Python loop).
- Dense Transformer above 70B at 128k+ context on a single 80GB GPU (KV cache alone exceeds VRAM).
- MoE models on a stack without expert-parallel support.
- Any recommendation that does not name the specific revision (e.g., "Jamba-1.5-Mini 2024-08", not "Jamba").

Worth reconsidering if:
- The workload does not need long context (below 16k). A dense 7B-13B Transformer ties or beats Jamba on throughput there with simpler tooling.
- The task is purely code or math with strong per-step reasoning. Dense reasoning-tuned models (Qwen, DeepSeek) lead on these benchmarks as of 2026.
- The serving stack lacks a fused selective-scan kernel. Jamba's advantage depends on it; without it, use a dense alternative.

Output: a one-page verdict naming the model revision, serving stack, quantization, and context budget, with numbered evidence for each decision. End with a "change my mind if" paragraph that names the specific workload parameter that would flip the call.
