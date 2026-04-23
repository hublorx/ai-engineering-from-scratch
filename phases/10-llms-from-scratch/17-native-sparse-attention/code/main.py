"""Native Sparse Attention from scratch.

Three parallel branches over the same Q, K, V:

    compression: coarse block summaries (global reach)
    selection:   top-n full-resolution blocks (precision where it matters)
    sliding:     last w tokens (local fluency)

A learned gate fuses the three outputs. One query at a time so the math stays
visible. Pure stdlib.

Reference: Yuan et al., "Native Sparse Attention: Hardware-Aligned and
Natively Trainable Sparse Attention", ACL 2025, arXiv:2502.11089.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class NSAConfig:
    d_head: int = 16
    block_l: int = 8
    stride_d: int = 4
    sel_block: int = 8
    top_n: int = 4
    window_w: int = 16
    num_init_blocks: int = 1
    num_local_blocks: int = 2


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


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


def compress_block(tokens, pos_bias, proj):
    d = len(tokens[0])
    pooled = [sum(tokens[i][j] + pos_bias[i][j] for i in range(len(tokens))) / len(tokens) for j in range(d)]
    return [sum(row[j] * pooled[j] for j in range(d)) for row in proj]


def build_compressed(keys, values, cfg, params):
    ck, cv, spans = [], [], []
    i = 0
    while i + cfg.block_l <= len(keys):
        ck.append(compress_block(keys[i:i + cfg.block_l], params["k_pos"], params["k_proj"]))
        cv.append(compress_block(values[i:i + cfg.block_l], params["v_pos"], params["v_proj"]))
        spans.append((i, i + cfg.block_l))
        i += cfg.stride_d
    return ck, cv, spans


def aggregate_scores(comp_probs, spans, num_sel, sel_block):
    scores = [0.0] * num_sel
    for p, (start, end) in zip(comp_probs, spans):
        for pos in range(start, end):
            sb = pos // sel_block
            if sb < num_sel:
                scores[sb] += p / (end - start)
    return scores


def pick_top(block_scores, cfg, t_query):
    total = len(block_scores)
    forced = set(range(min(cfg.num_init_blocks, total)))
    local_end = t_query // cfg.sel_block
    for i in range(max(0, local_end - cfg.num_local_blocks + 1), min(local_end + 1, total)):
        forced.add(i)
    ranked = sorted(range(total), key=lambda i: block_scores[i], reverse=True)
    chosen = list(forced)
    for idx in ranked:
        if len(chosen) >= cfg.top_n:
            break
        if idx not in forced:
            chosen.append(idx)
    return sorted(chosen)


def gather_blocks(keys, values, block_ids, sel_block, t_query):
    sk, sv = [], []
    for b in block_ids:
        start, end = b * sel_block, min((b + 1) * sel_block, t_query)
        sk.extend(keys[start:end])
        sv.extend(values[start:end])
    return sk, sv


def nsa_forward(q, keys, values, t, cfg, params):
    scale = 1.0 / math.sqrt(cfg.d_head)

    ck, cv, spans = build_compressed(keys[:t], values[:t], cfg, params)
    cmp_out = attention(q, ck, cv, scale)
    cmp_probs = softmax([dot(q, k) * scale for k in ck]) if ck else []

    num_sel = max(1, math.ceil(t / cfg.sel_block))
    sel_scores = aggregate_scores(cmp_probs, spans, num_sel, cfg.sel_block)
    sel_blocks = pick_top(sel_scores, cfg, t)
    sk, sv = gather_blocks(keys, values, sel_blocks, cfg.sel_block, t)
    slc_out = attention(q, sk, sv, scale)

    start = max(0, t - cfg.window_w)
    wk, wv = keys[start:t], values[start:t]
    win_out = attention(q, wk, wv, scale)

    gates = [sigmoid(dot(row, q)) for row in params["gate_w"]]
    fused = [gates[0] * cmp_out[j] + gates[1] * slc_out[j] + gates[2] * win_out[j] for j in range(cfg.d_head)]

    return {
        "output": fused,
        "gates": gates,
        "selected_blocks": sel_blocks,
        "compressed_count": len(ck),
        "sliding_count": len(wk),
        "full_tokens_read": len(ck) + len(sk) + len(wk),
    }


def full_attention(q, keys, values, t, d):
    return attention(q, keys[:t], values[:t], 1.0 / math.sqrt(d))


def nsa_cost(seq_len, cfg):
    compressed = max(0, (seq_len - cfg.block_l) // cfg.stride_d + 1)
    selected = cfg.top_n * cfg.sel_block
    sliding = min(cfg.window_w, seq_len)
    return compressed + selected + sliding


def random_matrix(rows, cols, rng):
    return [[rng.gauss(0.0, 0.1) for _ in range(cols)] for _ in range(rows)]


def demo():
    rng = random.Random(42)
    cfg = NSAConfig()
    seq_len = 128

    keys = [[rng.gauss(0, 1) for _ in range(cfg.d_head)] for _ in range(seq_len)]
    values = [[rng.gauss(0, 1) for _ in range(cfg.d_head)] for _ in range(seq_len)]
    query = [rng.gauss(0, 1) for _ in range(cfg.d_head)]

    params = {
        "k_pos": random_matrix(cfg.block_l, cfg.d_head, rng),
        "v_pos": random_matrix(cfg.block_l, cfg.d_head, rng),
        "k_proj": random_matrix(cfg.d_head, cfg.d_head, rng),
        "v_proj": random_matrix(cfg.d_head, cfg.d_head, rng),
        "gate_w": random_matrix(3, cfg.d_head, rng),
    }

    for t in (16, 64, 128):
        r = nsa_forward(query, keys, values, t, cfg, params)
        full_out = full_attention(query, keys, values, t, cfg.d_head)
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(r["output"], full_out)))
        print(f"t={t}")
        print(f"  compressed blocks: {r['compressed_count']}  selected: {r['selected_blocks']}  window: {r['sliding_count']}")
        print(f"  keys attended:     {r['full_tokens_read']} of {t}")
        print(f"  gate (cmp slc win): [{r['gates'][0]:.3f} {r['gates'][1]:.3f} {r['gates'][2]:.3f}]")
        print(f"  ||nsa - full||_2:  {diff:.4f}\n")

    print("asymptotic reads (DeepSeek defaults l=32 d=16 l'=64 n=16 w=512):")
    prod = NSAConfig(d_head=128, block_l=32, stride_d=16, sel_block=64, top_n=16, window_w=512)
    for seq in (8192, 32768, 65536):
        r = nsa_cost(seq, prod)
        print(f"  {seq:>6}: {r} keys  ({100 * r / seq:.2f}% of full)")


if __name__ == "__main__":
    demo()
