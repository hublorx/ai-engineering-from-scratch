# Mixture of Experts (MoE)

> Dense 70B transformer aktywuje każdy parametr dla każdego tokenu. 671B MoE aktywuje tylko 37B na token i bije go na każdym benchmarku. Sparsity to najważniejszy pomysł scalingowy dekady.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 05 (Full Transformer), Faza 7 · 07 (GPT)
**Czas:** ~45 minut

## Problem

Dense transformer's FLOPs przy inferencji equal jego parameter count (razy 2 dla forward pass). Skaluj dense model i każdy token płaci pełną cenę. Do 2024 frontier uderzał w compute wall: żeby być meaningful smarter, potrzebowałeś exponentially więcej FLOPs na token.

Mixture of Experts łamie to połączenie. Zastąp każdy FFN `E` niezależnymi ekspertami + router który wybiera `k` ekspertów na token. Total parameters = `E × FFN_size`. Aktywne parametry na token = `k × FFN_size`. Typowa konfiguracja 2026: `E=256`, `k=8`. Storage skaluje się z `E`, compute skaluje się z `k`.

Frontier 2026 to prawie całkowicie MoE: DeepSeek-V3 (671B total / 37B active), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. Na Artificial Analysis's independent leaderboard, top 10 open-source models to wszystkie MoE.

## Koncepcja

![Warstwa MoE: router wybiera k z E ekspertów na token](../assets/moe.svg)

### Zamiana FFN

Dense transformer block:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Każdy ekspert to niezależny FFN (typowo SwiGLU). Router to pojedyncza linear warstwa. Każdy token wybiera swoje `k` ekspertów i dostaje gated mixture ich outputs.

### Problem load-balancing

Jesli router włoży 90% tokenów przez eksperta 3, inni eksperci głodują. Trzy fixy były tried:

1. **Auxiliary load-balancing loss** (Switch Transformer, Mixtral). Dodaj karę proporcjonalną do wariancji w użyciu ekspertów. Działa, ale dodaje hyperparameter i drugi sygnał gradientowy.
2. **Expert capacity + token dropping** (early Switch). Każdy ekspert przetwarza max `C × N/E` tokenów; overflow tokens skipują warstwę. Rani jakość.
3. **Auxiliary-loss-free balancing** (DeepSeek-V3). Dodaj learned per-expert bias który przesuwa router's top-k selection. Bias jest updated outside training loss. Bez kary na głównym celu. 2024's big unlock.

Podejście DeepSeek-V3: po każdym kroku treningowym, dla każdego eksperta, sprawdź czy jego usage jest powyżej lub poniżej target. Nudge bias o `±γ`. Selection używa `scores + bias`. Expert probabilities used for gating to surowe `scores` unchanged. Decouples routing od expression.

### Shared eksperci

DeepSeek-V2/V3 też split ekspertów na *shared* i *routed*. Każdy token przechodzi przez wszystkich shared ekspertów. Routed eksperci są wybierani via top-k. Shared eksperci capture common knowledge; routed eksperci się specjalizują. V3 runs 1 shared expert plus top-8 z 256 routed.

### Fine-grained eksperci

Klasyczny MoE (GShard, Switch): każdy ekspert jest as wide jak full FFN. `E` jest małe (8–64), `k` jest małe (1–2).

Nowoczesny fine-grained MoE (DeepSeek-V3, Qwen-MoE): każdy ekspert jest węższy (1/8 FFN size). `E` jest duże (256+), `k` jest większe (8+). Te same total parameters, ale combinations scale much faster. `C(256, 8) = 400 trillion` możliwych "ekspertów" na token. Jakość rośnie, latency zostaje płaska.

### Profil kosztu

Na token, na warstwę:

| Konfiguracja | Aktywne params / token | Total params |
|--------------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 bije Llama 3 70B (dense) na prawie każdym benchmarku while doing **fewer active FLOPs per token**. Więcej parametrów = więcej wiedzy. Więcej active FLOPs = więcej compute na token. MoE je decouples.

### Podstęp: pamięć

Wszyscy eksperci mieszkają na GPU niezależnie od tego, który się zapala. 671B model potrzebuje ~1.3 TB VRAM dla fp16 weights. Frontier MoE deployment wymaga expert parallelism — shard ekspertów across GPUs, route tokens across the network. Latency jest zdominowana przez all-to-all communication, nie matmul.

## Zbuduj to

Zobacz `code/main.py`. Kompaktowa warstwa MoE w pure stdlib z:

- `n_experts=8` SwiGLU-ish ekspertów (one linear each, for illustration)
- top-k=2 routing
- softmax-normalized gating weights
- auxiliary-loss-free balancing via per-expert bias

### Krok 1: router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias wpływa na selection, nie na gate weight. To jest sztuczka DeepSeek-V3 — bias koryguje load imbalance bez steerowania predictions modelu.

### Krok 2: puść 100 tokenów przez router

Śledź który ekspert ile razy się zapala. Bez bias, usage jest skewed. Z pętlą aktualizacji bias (`-γ` dla over-used ekspertów, `+γ` dla under-used), usage converge do uniform distribution over a few iterations.

### Krok 3: porównanie param count

Wydrukuj "dense equivalent" konfiguracji MoE. DeepSeek-V3-shaped: 256 routed + 1 shared, 8 active, d_model=7168. Total parameter count jest eye-watering. Active count to jedna siódma dense Llama 3 70B.

## Użyj tego

HuggingFace loading:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 production inference: vLLM wspiera MoE routing natively. SGLang ma fastest expert-parallel path. Oba automatycznie obsługują top-k selection i expert parallelism.

**Kiedy wybrać MoE:**
- Chcesz frontier quality przy lower inference cost per token.
- Masz VRAM / expert-parallel infrastructure.
- Twoje workload jest token-heavy (chat, code) nie context-heavy (long docs).

**Kiedy NIE wybierać MoE:**
- Edge deployment — płacisz full storage za any active FLOP.
- Latency-critical single-user serving — expert routing dodaje overhead.
- Małe modele (<7B) — MoE's quality advantage pojawia się tylko powyżej compute threshold (~6B active params).

## Wyślij to

Zobacz `outputs/skill-moe-configurator.md`. Skill wybiera E, k i shared-expert layout dla nowego MoE przy given parameter budget, training tokens i deployment target.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Obserwuj jak auxiliary-loss-free bias update wyrównuje usage ekspertów over 50 iterations.
2. **Średnie.** Zastąp learned router hash-based router (deterministic, no learning). Porównaj quality i balance. Dlaczego learned router jest lepszy?
3. **Trudne.** Zaimplementuj GRPO-style "rollout-matched routing" (DeepSeek-V3.2 trick): log który ekspert się zapala podczas inference, wymuś to samo routing podczas gradient computation. Zmierz efekt na toy policy-gradient setup.

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|--------------------------|
| Expert | "Jeden FFN wśród wielu" | Niezależna feed-forward network; parametry dedykowane do sparse slice FFN computation. |
| Router | "Brama" | Tiny linear warstwa która ocenia każdy token przeciwko każdemu ekspertowi; top-k selection. |
| Top-k routing | "k aktywnych ekspertów na token" | Każdy token's FFN computation idzie przez dokładnie k ekspertów, ważonych przez gate. |
| Auxiliary loss | "Kara za load-balance" | Dodatkowy wyraz loss który karze skewed expert usage. |
| Auxiliary-loss-free | "Sztuczka DeepSeek-V3" | Balance przez per-expert bias na router's selection tylko; bez extra gradientu. |
| Shared expert | "Zawsze włączony" | Dodatkowy ekspert przez który przechodzi każdy token; capture common knowledge. |
| Expert parallelism | "Shard by expert" | Rozdziel różnych ekspertów do różnych GPU; route tokens across the network. |
| Sparsity | "Aktywne params < total params" | Ratio `k × expert_size / (E × expert_size)`; 37/671 ≈ 5.5% dla DeepSeek-V3. |

## Dalsze Czytanie

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — the idea.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, klasyczny MoE.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — paper o bias-based balancing.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — fine-grained + shared-expert split którego ten lesson's router używa.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — oryginalny shared-expert paper.