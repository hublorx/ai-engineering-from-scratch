# KV Cache, Flash Attention i Optymalizacja Inferencji

> Szkolenie jest równoległe i ograniczone przez FLOP-y. Inferencja jest szeregowa i ograniczona przez pamięć. Inny wąski gardło, inne triki.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 02 (Self-Attention), Faza 7 · 05 (Full Transformer), Faza 7 · 07 (GPT)
**Czas:** ~75 minut

## Problem

Naiwny dekoder autoregresyjny wykonuje `O(N²)` pracy, aby wygenerować `N` tokenów: na każdym kroku przelicza uwagę nad pełnym prefiksem. Dla odpowiedzi 4K tokenów to 16M operacji uwagi, z których większość jest redundantna. Każdy ukryty stan tokenu prefiksu jest deterministyczny po obliczeniu — musisz tylko uruchomić zapytanie nowego tokenu względem zbuforowanych kluczy i wartości wszystkiego przed nim.

Do tego sama uwaga przenosi dużo danych. Standardowa uwaga materializuje macierz N×N wyników, N×d wyjście softmax, N×d końcowe wyjście — zbyt wiele odczytów i zapisów do HBM. Dla N≥2K uwaga staje się ograniczona przez pamięć zanim stanie się ograniczona przez FLOP-y. Klasyczne jądra uwagi wykorzystują nowoczesne GPU o 4–10× za mało.

Dwie optymalizacje, obie od Dao et al., przesunęły inferencję frontu z "wolnej" na "szybką":

1. **KV cache.** Przechowuj wektory K i V każdego tokenu prefiksu. Uwaga każdego nowego tokenu to jedno zapytanie do zbuforowanych kluczy. Inferencja redukuje się z `O(N²)` do `O(N)` na krok generacji.
2. **Flash Attention.** Kafelkuje obliczenie uwagi tak, że pełna macierz N×N nigdy nie trafia do HBM. Całość softmax + matmul dzieje się w SRAM. Przyspieszenie 2–4× na A100; 5–10× na H100 z FP8.

Do 2026 oba są uniwersalne. Każdy stack inferencji produkcyjnej (vLLM, TensorRT-LLM, SGLang, llama.cpp) je zakłada. Każdy model frontu wysyła z Flash Attention włączonym.

## Koncepcja

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### Matematyka KV cache

Na warstwę dekodera, na token, na głowę:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

Dla modelu 7B z 32 warstwami, 32 głowami, d_head=128, fp16:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Dla Llama 3 70B (80 warstw, d_head=128, GQA z 8 głowami KV):

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

Te 10 GB to dlaczego Llama 3 70B przy 128K kontekście potrzebuje większości 40 GB A100 tylko dla KV cache przy batch size 1.

**GQA to wygrana dla KV-cache.** MHA z 64 głowami byłoby 32 GB. MLA kompresuje jeszcze bardziej.

### Flash Attention — trik z kafelkowaniem

Standardowa uwaga:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

Trzy rundy HBM. Na H100 przepustowość HBM to 3 TB/s; SRAM to 30 TB/s. Każda podróż do HBM to czynnik 10× spowolnienia względem trzymania wszystkiego na chipie.

Flash Attention:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

Jeden trip HBM na kafelek. Całkowity ślad pamięci spada z `O(N²)` do `O(N)`. Backward pass przelicza niektóre wartości z forward pass zamiast je przechowywać — kolejna wygrana pamięciowa.

**Triki numeryczne.** Running softmax utrzymuje `(max, sum)` across tiles więc końcowa normalizacja jest dokładna. Nie aproksymacja — Flash Attention oblicza bit-identyczne wyjście do standardowej uwagi (modulo fp16 non-associativity).

**Ewolucja wersji:**

| Wersja | Rok | Kluczowa zmiana | Przyspieszenie na referencyjnym sprzęcie |
|--------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | 2× na A100 |
| Flash 2 | 2023 | Better parallelism, causal-first ordering | 3× na A100 |
| Flash 3 | 2024 | Hopper asynchrony, FP8 | 1.5–2× na H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5-stage pipeline, software exp2 | Inference-first (początkowo tylko forward) |

Flash 4 jest tylko forward-pass przy starcie. Szkolenie nadal używa Flash 3. GQA i wsparcie varlen dla Flash 4 jest w toku (połowa 2026).

### Speculative decoding — druga wygrana na latency

Tani model proponuje N tokenów. Duży model weryfikuje wszystkie N równolegle. Jeśli weryfikacja zaakceptuje k tokenów, płacisz 1 forward pass dużego modelu za k generacji. Typowe k=3–5 na kodzie i prozie.

2026 domyślnie:
- **EAGLE 2 / Medusa.** Zintegrowane głowy draft które dzielą ukryte stany weryfikatora. Przyspieszenie 2–3× bez utraty jakości.
- **Speculative decoding z draft modelem.** Przyspieszenie 2–4× na sprzęcie konsumenckim.
- **Lookahead decoding.** Iteracja Jacobiego; nie potrzeba draft modelu. Niszowe ale darmowe.

### Continuous batching

Klasyczna wsadowa inferencja: czekaj aż najwolniejsza sekwencja skończy, potem startuj nowy batch. Marnuje GPU gdy krótkie odpowiedzi kończą się wcześniej.

Continuous batching (pierwszy wdrożony w Orca, teraz w vLLM, TensorRT-LLM, SGLang): wsuwaj nowe requesty do batcha jak tylko stare się skończą. Zysk przepustowości 5–10× dla typowych obciążeń czatowych.

### PagedAttention — KV cache jako pamięć wirtualna

Flagowa funkcja vLLM. KV cache jest alokowany w blokach 16 tokenów; tablica stron mapuje pozycje logiczne do bloków fizycznych. Pozwala dzielić KV między równoległe próbki (beam search, parallel sampling), hot-swap prefixów dla prompt caching, i defragmentować pamięć. Poprawa przepustowości 4× względem naiwnej ciągłej alokacji.

## Zbuduj To

Zobacz `code/main.py`. Implementujemy:

1. Naiwny dekoder `O(N²)` inkrementalny.
2. Dekoder z KV cache `O(N)`.
3. Tiled softmax który symuluje running-max algorytm Flash Attention.

### Krok 1: KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Proste: utrzymuj rosnące per-token K, V wektory w per-warstwa, per-głowa listach.

### Krok 2: tiled softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Bit-identical output to `softmax(qK) V` w jednym strzale, ale w dowolnym momencie working set to blok `tile × d_head`, nie pełne `N × d_head`.

### Krok 3: porównaj naiwny vs cached decoding na generacji 100 tokenów

Policz operacje uwagi. Naiwny: `O(N²)` = 5050. Z cache: `O(N)` = 100. Kod wypisuje oba.

## Użyj To

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM produkcyjnie:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Prefix caching między requestami to duża wygrana 2026 — ten sam system prompt, few-shot examples, lub długi dokument kontekstowy reuse KV across calls. Dla agent workload z powtarzającymi się tool prompts, prefix caching jest rutynowo 5× gain na przepustowości.

## Wyślij To

Zobacz `outputs/skill-inference-optimizer.md`. Skill wybiera implementację uwagi, strategię KV cache, kwantyzację, i speculative decoding dla nowego deploymentu inferencji.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Potwierdź że naiwny i cached dekodery produkują to samo wyjście; zanotuj różnicę w liczbie operacji.
2. **Średnie.** Zaimplementuj prefix caching: dany prompt P i kilka completions, uruchom jeden forward pass nad P żeby wypełnić KV cache, potem rozgałęź per-completion. Zmierz przyspieszenie vs re-encoding P dla każdego.
3. **Trudne.** Zaimplementuj toy PagedAttention: KV cache w ustalonych blokach 16 tokenów z free-listą. Gdy sekwencja się skończy, zwróć jej bloki do puli. Symuluj 1,000 czat completions z różnymi długościami. Porównaj fragmentację pamięci vs ciągłą alokację.

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| KV cache | "Triks który sprawia że dekodowanie jest szybkie" | Przechowywane K i V z każdego tokenu prefiksu; nowe zapytania uczestniczą w nich zamiast przeliczać. |
| HBM | "GPU main memory" | High Bandwidth Memory; 80 GB na H100, 192 GB na B200. ~3 TB/s bandwidth. |
| SRAM | "On-chip memory" | Per-SM fast memory, ~256 KB per SM na H100. ~30 TB/s bandwidth. |
| Flash Attention | "Tiled attention kernel" | Oblicza uwagę bez materializacji N×N w HBM. |
| Continuous batching | "No-wait batching" | Zamieniaj skończone sekwencje, nowe wchodzą, bez drainowania batcha. |
| PagedAttention | "vLLM's headline" | KV cache alokowany w ustalonych blokach z tablicą stron; eliminuje fragmentację. |
| Prefix caching | "Reuse long prompts" | Cache KV dla współdzielonego prefiksu między requestami; główna redukcja kosztów dla agentów. |
| Speculative decoding | "Draft + verify" | Tani draft model proponuje tokeny; duży model weryfikuje k w jednym passie. |

## Dalsze Czytanie

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5-stage pipeline and the software-exp2 trick; read the repo README for the forward-only launch caveats this lesson mentions.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM paper.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — spec decoding.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 paper for the integrated-draft approach the lesson cites.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — the Medusa approach referenced alongside EAGLE.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — the canonical deep dive on the 16-token block and page-table design.