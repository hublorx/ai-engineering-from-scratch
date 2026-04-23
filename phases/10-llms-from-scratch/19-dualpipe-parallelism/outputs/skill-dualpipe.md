---
name: dualpipe-pipeline-advisor
description: Decide between 1F1B, Zero Bubble, DualPipe, or DualPipeV for a specific pipeline-parallel training configuration.
version: 1.0.0
phase: 10
lesson: 19
tags: [pipeline-parallelism, dualpipe, 1f1b, zero-bubble, moe, distributed-training]
---

# DualPipe Pipeline Advisor

Help an engineer choose a pipeline-parallel schedule for a training run. The choice matters: on a 16-stage MoE pipeline the difference between 1F1B and DualPipe is a 1.3x–1.5x wall-clock speedup, but the wrong choice at the wrong scale wastes parameter memory.

## Inputs to gather

Before recommending anything, collect:

1. **Model topology**: dense or Mixture-of-Experts. If MoE: number of experts, top-k, expert parallel size (`EP`).
2. **Pipeline size** (`PP`): number of pipeline stages.
3. **Micro-batch count** (`M`): micro-batches per global step.
4. **Per-chunk timing estimate** (in any unit, ratio is what matters):
   - `F`: forward chunk time
   - `B`: backward-for-input time
   - `W`: backward-for-weights time
   - `comm_ratio`: fraction of chunk time spent in cross-node comm (dispatch + combine + PP sends)
5. **Memory headroom**: bytes per GPU left after weights + optimizer + activations at the `PP` baseline.
6. **Hardware interconnect**: NVLink-only (single node), NVLink + IB (multi-node), or all IB.

## Step 1: Rule out DualPipe immediately

DualPipe is not the default. Skip it and use 1F1B when any of these are true:

- `comm_ratio < 0.15`. Dense models on NVLink-only pods stay compute-bound. The overlap gain is small and the 2× param memory is real.
- Memory headroom is less than the size of one parameter copy. DualPipe needs 2× params across the pipeline; if you cannot pay it, stop here.
- `PP < 4`. With two or three stages the bubble is already tiny; 1F1B interleaved beats DualPipe on overhead.
- `M ≥ 64` and the 1F1B bubble is already under 10%. You will not recover the extra wiring cost.

If none of those fire, DualPipe is on the table.

## Step 2: Estimate bubble for each candidate

Use the closed-form bubbles, in units of chunk time:

| Schedule | Bubble per rank |
|----------|-----------------|
| 1F1B | `(PP - 1) · (F + B)` |
| 1F1B interleaved (v chunks) | `(PP - 1) · (F + B) / v` |
| Zero Bubble ZB-H2 | `≈ 0` if `M ≥ 2·PP`, else `ZB-H1` bubble = `(PP-1)(F+B)/3` |
| DualPipe | `(PP/2 - 1) · ((F&B) + B - 3·W)` |
| DualPipeV | same as DualPipe (V-shape variant, same bubble) |

`F&B` is the fused forward-and-backward time with compute and comm overlapped: approximately `max(F_compute + B_compute, F_comm + B_comm)`.

Divide the bubble by `M · (F + B)` to get a bubble fraction. Plot the four candidates. If DualPipe's bubble fraction is not meaningfully lower than Zero Bubble ZB-H2's bubble fraction, pick ZB-H2 — it costs 1× params.

## Step 3: Estimate memory

| Schedule | Param copies | Activation micro-batches | Extra state |
|----------|--------------|---------------------------|-------------|
| 1F1B | 1 | `PP` | none |
| 1F1B interleaved | 1 | `PP · v` | `v` chunk schedules |
| Zero Bubble | 1 | `~PP` + pending `W` | backward-for-weights queue |
| DualPipe | 2 | `PP + 1` | bidirectional schedule metadata |
| DualPipeV | 1 | `PP + 1` | V-shape device mapping |

If the 2× param copy for DualPipe does not fit, fall back to DualPipeV. If the activation increase from `PP` to `PP+1` is the constraint (rare), fall back to Zero Bubble.

## Step 4: Check the comm-kernel prerequisites

DualPipe's published speedup assumes you have cross-node all-to-all kernels that let you set SM count for the comm path. Without that the comm lane uses general-purpose SMs and steals from the compute lane — at which point the "overlap" becomes a scheduling lie. Before promising DualPipe numbers, confirm one of:

- You are using a framework (e.g., DeepSpeed, Megatron-LM, or DeepSeek's open-source pack) with tuned all-to-all for your interconnect.
- You have measured, not assumed, that `comm_ratio` holds under overlap.

If neither holds, DualPipe will underperform its own bubble math. Use Zero Bubble until you have the kernels.

## Step 5: Recommend

Produce a single recommendation with evidence:

- **Schedule**: 1F1B / 1F1B interleaved / Zero Bubble / DualPipe / DualPipeV.
- **Expected bubble fraction**: computed from Step 2.
- **Memory delta vs 1F1B baseline**: bytes per GPU.
- **Risk flags**: any of "comm kernels unverified", "M too small", "EP too small to amortize 2× params", "PP < 4 so interleaving wins".
- **Fallback**: a second schedule to switch to if measured `comm_ratio` is more than 20% off the estimate used.

## When to reject all pipeline parallelism

If `PP · M` chunks do not cover the bubble warm-up, drop pipeline parallelism entirely. Use FSDP / ZeRO-3 with a larger data-parallel dimension. Pipeline shines only when:

- The model does not fit on one node, and
- You have at least `2 · PP` micro-batches per step.

Below those thresholds every pipeline schedule in this advisor is worse than plain sharded data parallelism.

## Reference formulas

```
bubble_1f1b(PP, M, F, B)      = (PP - 1) * (F + B) / (M * (F + B))
bubble_zb_h2(PP, M, F, B)     = 0                          if M >= 2*PP
bubble_zb_h1(PP, M, F, B)     = (PP - 1) * (F + B) / (3 * M * (F + B))
bubble_dualpipe(PP, M, F, B, W) = max(0, (PP/2 - 1) * (max(F, B) + B - 3*W))
                                / (M * (F + B))
```

Treat these as first-order estimates. Always validate with a profiler run on a subset of the target cluster before committing to the schedule.
