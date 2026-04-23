"""DualPipe vs 1F1B pipeline parallelism simulator.

Stdlib-only event simulator. Each chunk splits into four components from the
DeepSeek-V3 report: attention, all-to-all dispatch, MLP, all-to-all combine.
Compare 1F1B (PipeDream / Megatron-LM baseline) against DualPipe, which
overlaps compute (attn + mlp) with comm (dispatch + combine) on each rank.
Units are ticks, so ratios matter, not absolute times.
"""

from __future__ import annotations

COSTS = {
    "attn": 4,
    "dispatch": 3,
    "mlp": 5,
    "combine": 3,
}

CHUNK_TIME = sum(COSTS.values())
COMPUTE_TIME = COSTS["attn"] + COSTS["mlp"]
COMM_TIME = COSTS["dispatch"] + COSTS["combine"]


def sequential_chunk(start, lane_busy):
    """Run attn -> dispatch -> mlp -> combine strictly in order. No overlap."""
    t = max(start, lane_busy["compute"], lane_busy["comm"])
    for name in ("attn", "dispatch", "mlp", "combine"):
        t += COSTS[name]
    lane_busy["compute"] = t
    lane_busy["comm"] = t
    return t, lane_busy


def overlapped_chunk(start, lane_busy):
    """DualPipe-style overlap: compute and comm lanes advance independently."""
    compute_start = max(start, lane_busy["compute"])
    compute_end = compute_start + COMPUTE_TIME
    comm_start = max(compute_start, lane_busy["comm"])
    comm_end = comm_start + COMM_TIME
    lane_busy["compute"] = compute_end
    lane_busy["comm"] = comm_end
    return max(compute_end, comm_end), lane_busy


def simulate_1f1b(num_stages, num_microbatches):
    """Canonical 1F1B: warm-up fills the pipe, steady state alternates F/B."""
    lanes = [{"compute": 0, "comm": 0} for _ in range(num_stages)]
    fwd_end = [[0] * num_microbatches for _ in range(num_stages)]
    bwd_end = [[0] * num_microbatches for _ in range(num_stages)]

    for mb in range(num_microbatches):
        for s in range(num_stages):
            start = fwd_end[s - 1][mb] if s > 0 else 0
            fwd_end[s][mb], lanes[s] = sequential_chunk(start, lanes[s])

    for mb in range(num_microbatches):
        for s in reversed(range(num_stages)):
            prev_stage = bwd_end[s + 1][mb] if s + 1 < num_stages else fwd_end[-1][mb]
            start = max(prev_stage, fwd_end[s][mb])
            bwd_end[s][mb], lanes[s] = sequential_chunk(start, lanes[s])

    total = max(bwd_end[s][-1] for s in range(num_stages))
    busy_per_rank = num_microbatches * CHUNK_TIME * 2
    bubble = 1.0 - busy_per_rank / total
    return total, bubble


def simulate_dualpipe(num_stages, num_microbatches):
    """DualPipe: bidirectional, lane-level overlap between chunks."""
    lanes = [{"compute": 0, "comm": 0} for _ in range(num_stages)]
    fwd_end = [[0] * num_microbatches for _ in range(num_stages)]
    bwd_end = [[0] * num_microbatches for _ in range(num_stages)]

    half = num_microbatches // 2

    for mb in range(half):
        for s in range(num_stages):
            start = fwd_end[s - 1][mb] if s > 0 else 0
            fwd_end[s][mb], lanes[s] = overlapped_chunk(start, lanes[s])

    for mb in range(half):
        for s in reversed(range(num_stages)):
            prev_stage = bwd_end[s + 1][mb] if s + 1 < num_stages else fwd_end[-1][mb]
            start = max(prev_stage, fwd_end[s][mb])
            bwd_end[s][mb], lanes[s] = overlapped_chunk(start, lanes[s])

    for mb in range(half, num_microbatches):
        for s in range(num_stages):
            start = fwd_end[s - 1][mb] if s > 0 else 0
            fwd_end[s][mb], lanes[s] = overlapped_chunk(start, lanes[s])

    for mb in range(half, num_microbatches):
        for s in reversed(range(num_stages)):
            prev_stage = bwd_end[s + 1][mb] if s + 1 < num_stages else fwd_end[-1][mb]
            start = max(prev_stage, fwd_end[s][mb])
            bwd_end[s][mb], lanes[s] = overlapped_chunk(start, lanes[s])

    total = max(bwd_end[s][-1] for s in range(num_stages))
    busy_per_rank = num_microbatches * 2 * max(COMPUTE_TIME, COMM_TIME)
    bubble = 1.0 - busy_per_rank / total
    return total, bubble


def rank_idle_fraction(num_stages, num_microbatches):
    """Rank idle fraction against a single shared ideal (slowest lane)."""
    t1, _ = simulate_1f1b(num_stages, num_microbatches)
    t2, _ = simulate_dualpipe(num_stages, num_microbatches)
    chunks_per_rank = num_microbatches * 2
    ideal = chunks_per_rank * max(COMPUTE_TIME, COMM_TIME)
    idle_1f1b = 1.0 - ideal / t1 if t1 else 0.0
    idle_dp = 1.0 - ideal / t2 if t2 else 0.0
    return idle_1f1b, idle_dp


def overlap_ratio():
    """Ratio of saved time inside a single chunk by running lanes in parallel."""
    sequential = CHUNK_TIME
    overlapped = max(COMPUTE_TIME, COMM_TIME)
    return sequential, overlapped, 1.0 - overlapped / sequential


def bubble_formula(num_stages, num_microbatches):
    """Paper formulas. 1F1B: (PP-1)(F+B). DualPipe: (PP/2-1)(F&B+B-3W)."""
    f = CHUNK_TIME
    b = CHUNK_TIME
    w = 0.3 * b

    onefob = (num_stages - 1) * (f + b)
    onefob_total = num_microbatches * (f + b)
    onefob_bubble = onefob / (onefob + onefob_total)

    dp = (num_stages / 2 - 1) * ((f + b) + b - 3 * w)
    dp_total = num_microbatches * (f + b)
    dp_bubble = max(0.0, dp / (dp + dp_total))

    return onefob_bubble, dp_bubble


def report():
    print("=" * 70)
    print("DualPipe: DeepSeek-V3 bidirectional pipeline parallelism")
    print("=" * 70)
    print(f"  Chunk components: attn={COSTS['attn']}, dispatch={COSTS['dispatch']}, "
          f"mlp={COSTS['mlp']}, combine={COSTS['combine']}")
    print(f"  Sequential chunk time : {CHUNK_TIME}")
    print(f"  Compute lane time     : {COMPUTE_TIME} (attn + mlp)")
    print(f"  Comm lane time        : {COMM_TIME} (dispatch + combine)")

    seq, over, saved = overlap_ratio()
    print(f"  Per-chunk overlap     : {seq} -> {over} ({saved:.0%} saved)")
    print()

    print("=" * 70)
    print("Schedule comparison (simulator)")
    print("=" * 70)
    print(f"  {'PP':>3} {'MB':>4} {'1F1B time':>10} {'DualPipe':>10} "
          f"{'speedup':>9} {'1F1B bub':>9} {'DP bub':>8}")
    print("  " + "-" * 56)
    for num_stages in (4, 8, 16):
        for num_microbatches in (8, 16, 32):
            t1, b1 = simulate_1f1b(num_stages, num_microbatches)
            t2, b2 = simulate_dualpipe(num_stages, num_microbatches)
            speedup = t1 / t2 if t2 else 0.0
            print(f"  {num_stages:>3} {num_microbatches:>4} {t1:>10d} {t2:>10d} "
                  f"{speedup:>8.2f}x {b1:>8.1%} {b2:>7.1%}")

    print()
    print("=" * 70)
    print("Apples-to-apples rank idle fraction (same ideal baseline)")
    print("=" * 70)
    print(f"  {'PP':>3} {'MB':>4} {'1F1B idle':>10} {'DualPipe idle':>14}")
    print("  " + "-" * 36)
    for num_stages in (4, 8, 16):
        for num_microbatches in (8, 16, 32):
            i1, i2 = rank_idle_fraction(num_stages, num_microbatches)
            print(f"  {num_stages:>3} {num_microbatches:>4} {i1:>9.1%} {i2:>13.1%}")

    print()
    print("=" * 70)
    print("Paper-formula bubble estimates (not simulator)")
    print("=" * 70)
    print(f"  {'PP':>3} {'MB':>4} {'1F1B bubble':>12} {'DualPipe bubble':>16}")
    print("  " + "-" * 40)
    for num_stages in (4, 8, 16):
        for num_microbatches in (8, 16, 32):
            b1, b2 = bubble_formula(num_stages, num_microbatches)
            print(f"  {num_stages:>3} {num_microbatches:>4} {b1:>11.1%} {b2:>15.1%}")

    print()
    print("=" * 70)
    print("Memory overhead (paper claims)")
    print("=" * 70)
    print("  1F1B     : 1x parameter copy, ~PP activation micro-batches in flight")
    print("  DualPipe : 2x parameter copies, ~PP+1 activation micro-batches")
    print("  DualPipeV: 1x parameter copy on PP/2 devices (V-shape variant)")


if __name__ == "__main__":
    report()
