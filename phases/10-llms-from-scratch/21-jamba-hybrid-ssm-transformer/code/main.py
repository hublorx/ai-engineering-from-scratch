"""Jamba hybrid SSM-Transformer calculator.

Simulate the Jamba block schedule (attention / Mamba / MoE interleave) and
print parameter counts, KV cache, SSM state memory, and the ratio versus an
equivalent pure-Transformer stack. Stdlib only. No tensors, no training.

Reference config: Jamba-v0.1 (Lieber et al. 2024) -- 52B total, 12B active,
32 layers, 1 attention per 7 Mamba, MoE every 2 layers, 16 experts top-2.
"""

from __future__ import annotations

from dataclasses import dataclass


SHARED = {
    "vocab_size": 65536,
    "max_position_embeddings": 131072,
    "mamba_state_dim": 16,
    "mamba_conv_kernel": 4,
    "mamba_expand": 2,
    "attn_to_mamba_ratio": (1, 7),
    "moe_every": 2,
    "num_experts": 16,
    "experts_per_token": 2,
}


def make_config(**overrides) -> dict:
    return {**SHARED, **overrides}


CONFIGS = {
    "jamba-v0.1": make_config(
        hidden_size=4096, intermediate_size=14336,
        num_layers=32, num_attention_heads=32, num_key_value_heads=8,
    ),
    "jamba-1.5-mini": make_config(
        hidden_size=4096, intermediate_size=14336,
        num_layers=32, num_attention_heads=32, num_key_value_heads=8,
    ),
    "jamba-1.5-large": make_config(
        hidden_size=8192, intermediate_size=28672,
        num_layers=72, num_attention_heads=64, num_key_value_heads=8,
    ),
    "pure-mamba-7b": make_config(
        hidden_size=4096, intermediate_size=14336,
        num_layers=32, num_attention_heads=32, num_key_value_heads=8,
        attn_to_mamba_ratio=(0, 1), moe_every=0,
        num_experts=1, experts_per_token=1,
    ),
}


@dataclass
class Breakdown:
    name: str
    schedule: list[tuple[str, str]]
    total_params: int
    active_params: int
    attn_layers: int
    mamba_layers: int
    moe_layers: int
    kv_cache_bytes: int
    ssm_state_bytes: int
    pure_transformer_kv_bytes: int


def build_schedule(cfg: dict) -> list[tuple[str, str]]:
    a, m = cfg["attn_to_mamba_ratio"]
    period = a + m if (a + m) > 0 else 1
    moe_every = cfg["moe_every"]
    schedule: list[tuple[str, str]] = []
    for i in range(cfg["num_layers"]):
        slot = i % period
        attn_type = "attn" if slot < a else "mamba"
        if moe_every and ((i + 1) % moe_every == 0):
            mlp_type = "moe"
        else:
            mlp_type = "mlp"
        schedule.append((attn_type, mlp_type))
    return schedule


def attention_params(cfg: dict) -> int:
    h = cfg["hidden_size"]
    q_heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = h // q_heads
    q = h * h
    kv = 2 * h * (kv_heads * head_dim)
    o = h * h
    return q + kv + o


def mamba_params(cfg: dict) -> int:
    h = cfg["hidden_size"]
    expand = cfg["mamba_expand"]
    n = cfg["mamba_state_dim"]
    d_inner = expand * h
    k = cfg["mamba_conv_kernel"]
    in_proj = h * (2 * d_inner)
    conv = d_inner * k + d_inner
    x_proj = d_inner * (2 * n + 1)
    dt_proj = d_inner
    a_log = d_inner * n
    d_param = d_inner
    out_proj = d_inner * h
    return in_proj + conv + x_proj + dt_proj + a_log + d_param + out_proj


def swiglu_params(h: int, ff: int) -> int:
    return 2 * h * ff + ff * h


def rmsnorm_params(h: int) -> int:
    return h


def analyze(name: str, cfg: dict) -> Breakdown:
    h = cfg["hidden_size"]
    ff = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    n_layers = cfg["num_layers"]
    k_experts = cfg["experts_per_token"]
    num_experts = cfg["num_experts"]

    schedule = build_schedule(cfg)
    attn_count = sum(1 for a, _ in schedule if a == "attn")
    mamba_count = sum(1 for a, _ in schedule if a == "mamba")
    moe_count = sum(1 for _, m in schedule if m == "moe")

    emb = vocab * h
    attn_p = attention_params(cfg)
    mamba_p = mamba_params(cfg)
    mlp_p = swiglu_params(h, ff)
    norms_per_layer = 2 * rmsnorm_params(h)
    final_norm = rmsnorm_params(h)
    router_p = h * num_experts if num_experts > 1 else 0

    total = emb + final_norm
    active = emb + final_norm
    for attn_type, mlp_type in schedule:
        mixer = attn_p if attn_type == "attn" else mamba_p
        if mlp_type == "moe":
            layer_total = mixer + mlp_p * num_experts + router_p + norms_per_layer
            layer_active = mixer + mlp_p * k_experts + router_p + norms_per_layer
        else:
            layer_total = mixer + mlp_p + norms_per_layer
            layer_active = layer_total
        total += layer_total
        active += layer_active

    head_dim = h // cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    seq = cfg["max_position_embeddings"]
    kv_cache_bytes = 2 * attn_count * kv_heads * head_dim * seq * 2

    d_inner = cfg["mamba_expand"] * h
    ssm_state_bytes = mamba_count * d_inner * cfg["mamba_state_dim"] * 2

    pure_kv = 2 * n_layers * kv_heads * head_dim * seq * 2

    return Breakdown(
        name=name,
        schedule=schedule,
        total_params=total,
        active_params=active,
        attn_layers=attn_count,
        mamba_layers=mamba_count,
        moe_layers=moe_count,
        kv_cache_bytes=kv_cache_bytes,
        ssm_state_bytes=ssm_state_bytes,
        pure_transformer_kv_bytes=pure_kv,
    )


def fmt_params(x: int) -> str:
    if x >= 1_000_000_000:
        return f"{x / 1e9:.1f}B"
    if x >= 1_000_000:
        return f"{x / 1e6:.1f}M"
    return f"{x:,}"


def fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def render_schedule(schedule: list[tuple[str, str]]) -> str:
    cells = []
    for attn_type, mlp_type in schedule:
        a = "A" if attn_type == "attn" else "M"
        m = "e" if mlp_type == "moe" else "d"
        cells.append(f"{a}{m}")
    return " ".join(cells)


def print_breakdown(b: Breakdown, cfg: dict) -> None:
    a, m = cfg["attn_to_mamba_ratio"]
    pure = b.pure_transformer_kv_bytes
    ratio = f"{pure / b.kv_cache_bytes:.1f}x" if b.kv_cache_bytes else "inf"
    active_frac = b.active_params / max(b.total_params, 1)
    print(f"\n{b.name}")
    print("-" * 72)
    print(f"  layers        : {cfg['num_layers']} "
          f"(attn={b.attn_layers}, mamba={b.mamba_layers}, moe={b.moe_layers})")
    print(f"  interleave    : {a}:{m}  "
          f"MoE every {cfg['moe_every']}, {cfg['num_experts']} experts "
          f"top-{cfg['experts_per_token']}")
    print(f"  total / active: {fmt_params(b.total_params)} / "
          f"{fmt_params(b.active_params)}  ({active_frac:.1%})")
    print(f"  context       : {cfg['max_position_embeddings']:,} tokens")
    print(f"  KV cache      : {fmt_bytes(b.kv_cache_bytes)} "
          f"(vs {fmt_bytes(pure)} pure transformer -> {ratio} reduction)")
    print(f"  SSM state     : {fmt_bytes(b.ssm_state_bytes)} "
          f"(constant vs seq_len)")
    print(f"  schedule (Ad=attn+dense, Me=mamba+moe, Md=mamba+dense):")
    sched = render_schedule(b.schedule)
    for i in range(0, len(sched), 48):
        print(f"    {sched[i:i + 48]}")


def print_summary(results: list[Breakdown]) -> None:
    print()
    print("=" * 72)
    print("HEADLINE: SSM vs TRANSFORMER AT 128K")
    print("=" * 72)
    print(f"  {'model':18s}  {'total':>8s}  {'active':>8s}  "
          f"{'KV@ctx':>10s}  {'pure KV':>10s}  {'ratio':>6s}")
    for b in results:
        ratio = (f"{b.pure_transformer_kv_bytes / b.kv_cache_bytes:5.1f}x"
                 if b.kv_cache_bytes else "   inf")
        print(f"  {b.name:18s}  "
              f"{fmt_params(b.total_params):>8s}  "
              f"{fmt_params(b.active_params):>8s}  "
              f"{fmt_bytes(b.kv_cache_bytes):>10s}  "
              f"{fmt_bytes(b.pure_transformer_kv_bytes):>10s}  "
              f"{ratio:>6s}")


def main() -> None:
    print("=" * 72)
    print("JAMBA HYBRID SSM-TRANSFORMER CALCULATOR")
    print("=" * 72)
    results = []
    for name, cfg in CONFIGS.items():
        b = analyze(name, cfg)
        print_breakdown(b, cfg)
        results.append(b)
    print_summary(results)


if __name__ == "__main__":
    main()
