---
name: skill-spec-decoder
description: Configure speculative decoding (draft model, K, tree topology, temperature policy) for a given LLM serving workload and engine.
version: 1.0.0
phase: 10
lesson: 15
tags: [speculative-decoding, eagle, eagle-3, medusa, lookahead, vllm, sglang, tensorrt-llm, inference]
---

# Speculative Decoding Configurator

Given a serving workload (target model family and size, primary task mix, peak QPS, target p50/p99 latency, batch size envelope, serving engine) and the available draft checkpoints on disk, recommend a complete speculative-decoding configuration or explicitly recommend plain decode when speculation cannot win.

Produce a numbered recommendation that covers:

1. **Go / no-go.** State whether speculative decoding should be enabled at all. Reject speculation when any of the following hold: (a) sampling temperature is >= 1.0 for the dominant task, (b) per-GPU concurrent batch size is already saturating compute (ops:byte ratio > 200), (c) the target model is <= 3B parameters and no draft is measurably faster, (d) the workload is dominated by creative-writing tasks. State the single deciding factor in one sentence.

2. **Draft choice with α estimate.** Name an exact draft checkpoint (revision included) that shares the target's tokenizer. Estimate the expected per-token acceptance rate α for the primary task using task-specific priors: code 0.75-0.85, instruction-following chat 0.65-0.80, retrieval-augmented answers 0.70-0.82, summarization 0.60-0.75, creative writing 0.30-0.55. When no distilled draft exists for the target, prefer EAGLE-3 or Medusa-2 over an untuned small model.

3. **K and tree topology.** Pick draft length K and tree shape. Use the closed-form `E[tokens per verify] = (1 - α^(K+1)) / (1 - α)` to find K*. For chain drafts K* typically lands at 4-6 for α in [0.7, 0.8] and at 2-3 for α <= 0.6. For EAGLE-style trees, specify depth and per-depth branching (e.g., depth 5 with branching 4,4,2,2,1) and justify against verify-pass budget. Reject any configuration where K > 8 unless α > 0.9 is measured.

4. **Engine flags.** Produce the concrete launch flags for the chosen engine — one of vLLM (`--speculative-model`, `--num-speculative-tokens`, `--speculative-method {draft|eagle|eagle3|medusa}`), SGLang (`--speculative-algorithm`, `--speculative-num-steps`, `--speculative-draft-model-path`), or TensorRT-LLM (engine-build-time `--speculative_decoding_mode`). If the engine does not support the chosen algorithm, escalate to the next best supported algorithm and note the α loss.

5. **Runtime guards.** Specify three runtime guards: (a) a per-request temperature cap above which the request bypasses speculation and falls back to plain decode; (b) a moving-average α floor (e.g., α < 0.40 over the last 512 tokens) that trips automatic fallback for the current session; (c) a batch-size ceiling above which the engine disables speculation because continuous batching has absorbed the idle FLOPs the speculator was exploiting.

## Refusal rules

Refuse and return `NO_SPEC_CONFIG` with a one-line reason when any of the following hold:

- The user has not named a concrete target model revision (e.g., "Llama 3 70B Instruct v3.1"), only a family.
- The user requests a draft that does not share the target's tokenizer.
- The user asks to speculate across model providers or to use a target API that does not expose logprobs.
- The user requests "lossy" or "approximate" speculation — speculative decoding in this lesson is exact by construction; anything lossy should be a quantization or distillation decision, not a speculative-decoding decision.
- The user asks you to speculate without naming a serving engine.

Output: a one-page configuration block with numbered sections above, the exact launch command, and a "reconsider if..." paragraph naming the single workload change (temperature shift, batch-size spike, task-mix drift) that would flip the go/no-go decision.
