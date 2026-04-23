import math
import random
import sys

VOCAB = 16
PREFIX_LEN = 6
SEED = 2026


def make_distribution(rng, concentration=1.5):
    weights = [rng.random() ** concentration for _ in range(VOCAB)]
    z = sum(weights)
    return [w / z for w in weights]


def make_target(rng):
    return {tuple(rng.choices(range(VOCAB), k=PREFIX_LEN)): make_distribution(rng) for _ in range(32)}


def default_dist(rng):
    return make_distribution(rng)


def target_prob(target_dists, default, prefix):
    key = tuple(prefix[-PREFIX_LEN:]) if len(prefix) >= PREFIX_LEN else tuple(prefix + [0] * (PREFIX_LEN - len(prefix)))
    return target_dists.get(key, default)


def perturb(dist, noise, rng):
    n = len(dist)
    perturbed = [max(1e-9, d + noise * (rng.random() - 0.5)) for d in dist]
    z = sum(perturbed)
    return [p / z for p in perturbed]


def draft_prob(target_dists, draft_noise, default_draft, prefix, rng_stream):
    p = target_prob(target_dists, default_draft, prefix)
    return perturb(p, draft_noise, rng_stream)


def sample_from(dist, rng):
    r = rng.random()
    acc = 0.0
    for i, w in enumerate(dist):
        acc += w
        if r < acc:
            return i
    return len(dist) - 1


def plain_target_step(target_dists, default, prefix, rng):
    p = target_prob(target_dists, default, prefix)
    return sample_from(p, rng)


def speculative_step(target_dists, default_target, default_draft, prefix, K, draft_noise, rng, draft_rng):
    drafted = []
    q_at = []
    current = list(prefix)
    for _ in range(K):
        q = draft_prob(target_dists, draft_noise, default_draft, current, draft_rng)
        tok = sample_from(q, draft_rng)
        drafted.append(tok)
        q_at.append(q)
        current.append(tok)

    emitted = []
    accepts = 0
    for k, tok in enumerate(drafted):
        p_k = target_prob(target_dists, default_target, prefix + emitted)
        r = rng.random()
        ratio = p_k[tok] / max(q_at[k][tok], 1e-12)
        if r < min(1.0, ratio):
            emitted.append(tok)
            accepts += 1
            continue
        residual = [max(p_k[i] - q_at[k][i], 0.0) for i in range(VOCAB)]
        total = sum(residual)
        if total <= 0:
            emitted.append(sample_from(p_k, rng))
        else:
            emitted.append(sample_from([r_ / total for r_ in residual], rng))
        return emitted, accepts, k

    p_bonus = target_prob(target_dists, default_target, prefix + emitted)
    emitted.append(sample_from(p_bonus, rng))
    return emitted, accepts, K


def total_variation(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def empirical_distribution(samples):
    counts = [0] * VOCAB
    for s in samples:
        counts[s] += 1
    n = sum(counts)
    return [c / n for c in counts]


def expected_tokens_per_verify(alpha, K):
    if abs(alpha - 1.0) < 1e-9:
        return K + 1
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)


def run_exactness_test(n_trials, target_dists, default_target, default_draft, prefix, K, draft_noise):
    rng_spec = random.Random(SEED + 1)
    rng_draft = random.Random(SEED + 2)
    rng_plain = random.Random(SEED + 3)

    plain_samples = [plain_target_step(target_dists, default_target, prefix, rng_plain) for _ in range(n_trials)]

    spec_samples = []
    total_accepts = 0
    total_drafted = 0
    for _ in range(n_trials):
        emitted, accepts, stopped_at = speculative_step(
            target_dists, default_target, default_draft, prefix, K, draft_noise, rng_spec, rng_draft
        )
        spec_samples.append(emitted[0])
        total_accepts += accepts
        total_drafted += max(stopped_at, 1)

    p_plain = empirical_distribution(plain_samples)
    p_spec = empirical_distribution(spec_samples)
    alpha = total_accepts / total_drafted if total_drafted else 0.0
    return total_variation(p_plain, p_spec), alpha, p_plain, p_spec


def print_header(text):
    print("=" * 60)
    print(text)
    print("=" * 60)


def print_speedup_surface():
    print_header("Expected tokens per verify: (1 - alpha^(K+1)) / (1 - alpha)")
    print(f"{'alpha':>6}  " + "  ".join(f"K={K:<2}" for K in range(1, 9)))
    for alpha in (0.5, 0.6, 0.7, 0.8, 0.9):
        row = f"{alpha:>6.2f}  " + "  ".join(f"{expected_tokens_per_verify(alpha, K):>4.2f}" for K in range(1, 9))
        print(row)


def print_distribution_pair(label, p_plain, p_spec):
    print(f"{label}")
    print(f"{'token':>5}  {'plain':>8}  {'spec':>8}  {'|diff|':>8}")
    for i in range(VOCAB):
        print(f"{i:>5}  {p_plain[i]:>8.4f}  {p_spec[i]:>8.4f}  {abs(p_plain[i] - p_spec[i]):>8.4f}")


def main():
    rng = random.Random(SEED)
    target_dists = make_target(rng)
    default_target = default_dist(rng)
    default_draft = default_dist(rng)

    prefix = [rng.randrange(VOCAB) for _ in range(PREFIX_LEN)]
    K = 4
    draft_noise = 0.15
    n_trials = 200_000

    print_header("Leviathan 2023 speculative decoding: toy exactness check")
    print(f"vocab size         : {VOCAB}")
    print(f"prefix             : {prefix}")
    print(f"draft length K     : {K}")
    print(f"draft noise (|p-q|): {draft_noise}")
    print(f"trials             : {n_trials}")

    tv, alpha, p_plain, p_spec = run_exactness_test(
        n_trials, target_dists, default_target, default_draft, prefix, K, draft_noise
    )

    print_distribution_pair("\nEmpirical marginal at the first emitted position:", p_plain, p_spec)
    print(f"\ntotal variation (plain vs speculative): TV = {tv:.4f}")
    print(f"empirical acceptance rate alpha         : {alpha:.4f}")
    print(f"predicted tokens per verify @ K={K}       : {expected_tokens_per_verify(alpha, K):.4f}")

    print()
    print_speedup_surface()

    tv_tol = 0.01
    if tv < tv_tol:
        print(f"\nPASS  TV {tv:.4f} < {tv_tol} — speculative output matches target distribution within tolerance.")
        return 0
    print(f"\nFAIL  TV {tv:.4f} >= {tv_tol}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
