"""DeepSeek-V3 parameter calculator.

Walks the 671B / 37B architecture component by component using hyperparameters
from the DeepSeek-V3 Technical Report (arXiv:2412.19437) and the HuggingFace
config.json. Stdlib only. Verifies totals land near 671B / 37B.
"""

from __future__ import annotations


V3 = {
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 61,
    "first_dense_layers": 3,
    "num_attention_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "experts_per_token": 8,
    "vocab_size": 129280,
    "max_position_embeddings": 163840,
    "mtp_depth": 1,
    "tied_embeddings": False,
}


def mla_attention_params(c: dict) -> int:
    h, q, kv = c["hidden_size"], c["q_lora_rank"], c["kv_lora_rank"]
    heads, nope, rope, v = (c["num_attention_heads"], c["qk_nope_head_dim"],
                            c["qk_rope_head_dim"], c["v_head_dim"])
    return (h * q + q * heads * (nope + rope) + h * (kv + rope)
            + kv * heads * nope + kv * heads * v + heads * v * h + q + kv)


def swiglu_mlp_params(h: int, ff: int) -> int:
    return 3 * h * ff


def dense_block_params(c: dict) -> int:
    return (mla_attention_params(c)
            + swiglu_mlp_params(c["hidden_size"], c["intermediate_size"])
            + 2 * c["hidden_size"])


def moe_block_total_params(c: dict) -> int:
    h = c["hidden_size"]
    expert = swiglu_mlp_params(h, c["moe_intermediate_size"])
    experts = (c["n_routed_experts"] + c["n_shared_experts"]) * expert
    router = h * c["n_routed_experts"] + c["n_routed_experts"]
    return mla_attention_params(c) + experts + router + 2 * h


def moe_block_active_params(c: dict) -> int:
    h = c["hidden_size"]
    expert = swiglu_mlp_params(h, c["moe_intermediate_size"])
    active = (c["experts_per_token"] + c["n_shared_experts"]) * expert
    router = h * c["n_routed_experts"] + c["n_routed_experts"]
    return mla_attention_params(c) + active + router + 2 * h


def mtp_module_params(c: dict) -> int:
    h = c["hidden_size"]
    return 2 * h * h + 2 * h + moe_block_total_params(c)


def breakdown(c: dict) -> list[tuple[str, int, int]]:
    h, vocab = c["hidden_size"], c["vocab_size"]
    n_dense = c["first_dense_layers"]
    n_moe = c["num_hidden_layers"] - n_dense
    emb = vocab * h
    head = 0 if c["tied_embeddings"] else vocab * h
    dense = n_dense * dense_block_params(c)
    moe_tot = n_moe * moe_block_total_params(c)
    moe_act = n_moe * moe_block_active_params(c)
    mtp = c["mtp_depth"] * mtp_module_params(c)
    return [
        ("embedding", emb, emb),
        (f"dense blocks x{n_dense}", dense, dense),
        (f"moe blocks x{n_moe} (total)", moe_tot, 0),
        (f"moe blocks x{n_moe} (active)", 0, moe_act),
        ("final rmsnorm", h, h),
        ("lm head", head, head),
        (f"mtp modules x{c['mtp_depth']}", mtp, mtp),
    ]


def fmt(x: int) -> str:
    if x >= 1_000_000_000:
        return f"{x / 1e9:7.2f}B"
    if x >= 1_000_000:
        return f"{x / 1e6:7.2f}M"
    return f"{x:>10,}"


def kv_cache_bytes(c: dict, seq_len: int, dtype_bytes: int = 2) -> int:
    latent = c["kv_lora_rank"] + c["qk_rope_head_dim"]
    return 2 * c["num_hidden_layers"] * latent * seq_len * dtype_bytes


def mha_equivalent_kv_cache(c: dict, seq_len: int, dtype_bytes: int = 2) -> int:
    head_dim = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
    heads = c["num_attention_heads"]
    return 2 * c["num_hidden_layers"] * heads * head_dim * seq_len * dtype_bytes


def fmt_bytes(b: int) -> str:
    x = float(b)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * 72)


def main() -> None:
    c = V3
    print("=" * 72)
    print("DEEPSEEK-V3 PARAMETER CALCULATOR")
    print("=" * 72)
    print(f"  layers {c['num_hidden_layers']}  "
          f"({c['first_dense_layers']} dense + "
          f"{c['num_hidden_layers'] - c['first_dense_layers']} moe)  "
          f"hidden {c['hidden_size']}  vocab {c['vocab_size']}")
    print(f"  attn MLA, {c['num_attention_heads']} heads, "
          f"kv_lora={c['kv_lora_rank']}, q_lora={c['q_lora_rank']}")
    print(f"  moe {c['n_routed_experts']} routed + {c['n_shared_experts']} shared, "
          f"top-{c['experts_per_token']}, moe_ff={c['moe_intermediate_size']}")
    print(f"  dense ff (layers 1-3) {c['intermediate_size']}  "
          f"context {c['max_position_embeddings']:,}  mtp depth {c['mtp_depth']}")

    section("per-component breakdown")
    total = active = 0
    for name, p, a in breakdown(c):
        print(f"  {name:35s} total={fmt(p)}  active={fmt(a)}")
        total += p
        active += a
    print("-" * 72)
    mtp_t = c["mtp_depth"] * mtp_module_params(c)
    main_total, main_active = total - mtp_t, active - mtp_t
    print(f"  {'TOTAL (with MTP)':35s} total={fmt(total)}  active={fmt(active)}")
    print(f"  {'MAIN MODEL (excl. MTP)':35s} total={fmt(main_total)}  "
          f"active={fmt(main_active)}")

    section("MLA per-layer detail")
    h, q, kv = c["hidden_size"], c["q_lora_rank"], c["kv_lora_rank"]
    heads, nope, rope, v = (c["num_attention_heads"], c["qk_nope_head_dim"],
                            c["qk_rope_head_dim"], c["v_head_dim"])
    for name, val in [
        (f"W_DQ   ({h} x {q})", h * q),
        (f"W_UQ   ({q} x {heads} x {nope + rope})", q * heads * (nope + rope)),
        (f"W_DKV  ({h} x {kv + rope})", h * (kv + rope)),
        (f"W_UK   ({kv} x {heads} x {nope})", kv * heads * nope),
        (f"W_UV   ({kv} x {heads} x {v})", kv * heads * v),
        (f"W_O    ({heads} x {v} x {h})", heads * v * h),
    ]:
        print(f"  {name:40s} = {fmt(val)}")
    print(f"  {'total per-layer MLA':40s} = {fmt(mla_attention_params(c))}")

    section("MoE layer detail")
    expert = swiglu_mlp_params(h, c["moe_intermediate_size"])
    n_tot = c["n_routed_experts"] + c["n_shared_experts"]
    n_act = c["experts_per_token"] + c["n_shared_experts"]
    print(f"  one expert SwiGLU                    = {fmt(expert)}")
    print(f"  all {n_tot} experts stored               = {fmt(n_tot * expert)}")
    print(f"  {n_act} experts active per token          = {fmt(n_act * expert)}")
    print(f"  router ({h} x {c['n_routed_experts']})                   = "
          f"{fmt(h * c['n_routed_experts'])}")

    section("kv cache per sequence, bf16")
    for seq in [4096, 32768, 131072]:
        mla_kv = kv_cache_bytes(c, seq)
        mha_kv = mha_equivalent_kv_cache(c, seq)
        print(f"  {seq:>7,} tok  MLA={fmt_bytes(mla_kv):>10s}  "
              f"MHA-equiv={fmt_bytes(mha_kv):>10s}  "
              f"compression={mha_kv / mla_kv:.1f}x")

    section("headline check (main model, MTP reported separately)")
    print("  reported (DeepSeek-V3 tech report): 671B total / 37B active")
    print(f"  calculated                        : "
          f"{main_total / 1e9:.1f}B total / {main_active / 1e9:.1f}B active")
    print(f"  delta                             : "
          f"{abs(main_total - 671e9) / 671e9 * 100:.2f}% total, "
          f"{abs(main_active - 37e9) / 37e9 * 100:.2f}% active")
    print(f"  mtp module params (reported as 14B): {fmt(mtp_t)}")


if __name__ == "__main__":
    main()
