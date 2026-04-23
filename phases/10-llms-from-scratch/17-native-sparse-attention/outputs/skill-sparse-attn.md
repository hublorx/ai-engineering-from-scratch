---
name: sparse-attn
description: Pick a sparse attention strategy for a long-context LLM deployment. Rejects choices that need training budget the team does not have.
version: 1.0.0
phase: 10
lesson: 17
tags: [sparse-attention, nsa, moba, sliding-window, long-context, kv-cache, inference]
---

Given a long-context serving target (model family, parameter count, target context length in tokens, peak concurrent batch size, GPU type and count, accuracy floor on the target task, training budget available for architecture changes) and a task profile (chat, code, multi-document retrieval, needle-in-haystack, math reasoning), recommend a sparse attention strategy with explicit reasoning. Never invent model-specific speedup numbers.

Produce:

1. Current cost profile. Compute full-attention KV cache bytes at target context and batch size. Compute reads per decode step under full attention. Reject the entire exercise if KV cache comfortably fits in HBM and decode throughput already meets target (sparsity adds complexity for no gain at short context).

2. Strategy shortlist. Three candidates from the four options below, each with the one-line reason it made the list:
   - Native Sparse Attention (NSA). Three-branch compression plus selection plus sliding window with a learned gate. Requires training from scratch or extensive continued pretraining.
   - MoBA. Single-branch block-sparse with parameter-less top-k gating. Requires continued training on existing checkpoint; no separate coarse branch.
   - Interleaved sliding window (Gemma 3 style). 5:1 local-to-global layer ratio, window 1024. Training-free for the sliding layers, continued training for the interleave pattern.
   - Training-free sparse (H2O, StreamingLLM, Quest). Drop-in at inference; accuracy degrades above 0.8 sparsity on retrieval-heavy tasks.

3. Training budget check. For each shortlisted option, state the minimum training regime required: pretrain-from-scratch (NSA), continued-pretrain at 50-100B tokens (MoBA, Gemma-style interleave), or zero (training-free). Cross off anything the team cannot afford.

4. Accuracy floor check. For the top candidate, cite whether the task profile is represented in the paper's evaluation (reasoning, long-context QA, code). If the paper evaluated only language modeling perplexity and the task is needle retrieval, halt and demand a small-scale validation run before recommending.

5. Read-count sanity check. At the target context length, compute approximate keys read per decode step under the recommended strategy. Reject the recommendation if the sparsity ratio is below 40% of full attention at that context, because the KV-cache I/O savings will not cover the kernel complexity cost.

6. Fallback. Name the next-best option if the top candidate fails training-budget or accuracy-floor checks. Always name one.

Hard rejects:

- NSA on a team that cannot pretrain from scratch or afford 200B+ continued training tokens.
- Any training-free sparse method at 128k+ context with retrieval-critical tasks, unless the team has validated it on their own eval.
- Recommending sparse attention as a fix for OOM during prefill; sparse attention helps decode, not prefill compute.
- Bolting sparse attention onto a model trained with full attention without continued training and claiming the paper's speedup numbers.
- Sliding-window-only at long context if the task needs any information from tokens older than the window.

Output: a one-page recommendation listing strategy, training regime, expected read-count reduction at target context, and the accuracy evidence cited. End with a "worth reconsidering if..." paragraph naming the specific deployment parameter (context length extension, task shift, new paper) that would flip the choice.
