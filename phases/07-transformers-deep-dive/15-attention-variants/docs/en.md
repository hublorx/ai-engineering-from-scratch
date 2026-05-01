# Warianty Uwagi — Sliding Window, Sparse, Różnicowa

> Pełna uwaga to koło. Każdy token widzi każdy token, a pamięć płaci cenę. Cztery warianty zmieniają kształt koła i odzyskują połowę kosztu.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 02 (Self-Attention), Faza 7 · 03 (Multi-Head), Faza 7 · 12 (KV Cache / Flash Attention)
**Czas:** ~60 minut

## Problem

Pełna uwaga kosztuje `O(N²)` pamięci i `O(N²)` compute w długości sekwencji. Dla 128K-context Llama 3 70B to 16 miliardów wpisów uwagi na warstwę, razy 80 warstw. Flash Attention (Lekcja 12) ukrywa `O(N²)` pamięć aktywacji ale nie zmienia kosztu arytmetycznego — każdy token nadal uczestniczy w każdym innym tokenie.

Trzy klasy wariantów zmieniają topologię samej macierzy uwagi:

1. **Sliding window attention (SWA).** Każdy token uczestniczy w ustalonym oknie sąsiadów, nie w pełnym prefiksie. Pamięć i compute spadają do `O(N · W)` gdzie `W` to okno. Gemma 2/3, pierwsze warstwy Mistral 7B, Phi-3-Long.
2. **Sparse / block attention.** Tylko wybrane pary `(i, j)` są scoringowane; reszta jest wymuszona do zero weight. Longformer, BigBird, OpenAI sparse transformer.
3. **Differential attention.** Oblicz dwie mapy uwagi z separate Q/K projections, odejmij jedną od drugiej. Zabija "attention sink" który wykrwawia wagę na pierwszych tokenach. Microsoft's DIFF Transformer (2024).

Te współistnieją. Model frontu 2026 często je miesza: większość warstw to SWA-1024, co piąta to global full attention, i garstka to differential heads które sprzątają retrieval. Gemma 3's 5:1 SWA-to-global ratio to obecny podręcznik default.

## Koncepcja

### Sliding Window Attention (SWA)

Każde zapytanie na pozycji `i` uczestniczy tylko w pozycjach `[i - W, i]` (causal SWA) lub `[i - W/2, i + W/2]` (bidirectional). Tokeny poza oknem dostają `-inf` w macierzy wyników.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

Dla `N = 8192` i `W = 1024`, macierz wyników ma 1024 × 8192 non-zero rows in expectation — 8× redukcja.

**KV cache shrinks with SWA.** Tylko ostatnie `W` tokenów K i V muszą być trzymane per layer. Dla konfiguracji podobnej do Gemma-3 (1024 window, 128K context), KV cache spada 128×.

**Koszt jakości.** SWA-only transformery mają problem z dalekosiężnym retrieval. Fix: przeplataj SWA layers z full-attention layers. Gemma 3 używa 5:1 SWA:global. Mistral 7B używał causal-SWA stack gdzie information "flows forward" przez overlapping windows — każda warstwa rozszerza effective receptive field o `W`, i po `L` warstwach model może uczestniczyć `L × W` tokens back.

### Sparse / Block Attention

Pick an `N × N` sparsity pattern ahead of time. Trzy kanoniczne kształty:

- **Local + strided (OpenAI sparse transformer).** Uczestnicz w ostatnich `W` tokenach plus every `stride`-th token przed nim. Przechwytuje zarówno local i long-range at `O(N · sqrt(N))` compute.
- **Longformer / BigBird.** Local window + mały zbiór global tokens (np. `[CLS]`) które uczestniczą w każdym i są uczestniczone przez każdego + random-sparse links. Empirical 2× context at matched quality.
- **Native Sparse Attention (DeepSeek, 2025).** Naucz się które bloki `(Q, K)` są ważne; pomiń zero blocks na poziomie kernela. FlashAttention-compatible.

Sparse attention to kernel-engineering story. Matematyka jest prosta (mask the score matrix); wygrana comes from never loading the zero entries into SRAM. FlashAttention-3 i 2026 FlexAttention API sprawiają że custom sparse patterns są first-class w PyTorch.

### Differential Attention (DIFF Transformer, 2024)

Regularna uwaga ma problem "attention sink": softmax wymusza że każdy wiersz sumuje się do 1, więc tokeny które nie chcą uczestniczyć w niczym szczególnym zrzucają wagę na pierwszy token (lub pierwsze kilka). To kradnie pojemność która powinna iść do prawdziwej treści.

Differential attention to naprawia przez obliczenie **dwóch** map uwagi i odjęcie:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

gdzie `λ` to learned scalar (typowo 0.5–0.8). A1 przechwytuje real content weights; A2 przechwytuje sink. Odejmowanie anuluje sink, przenosi wagę na relevant tokens.

Reported results (Microsoft 2024): 5–10% lower perplexity, 1.5–2× dłuższy effective context at same trained length, ostrzejsze needle-in-haystack retrieval.

### Variant Comparison

| Wariant | Compute | KV cache | Jakość vs full | Użycie produkcyjne |
|---------|---------|----------|-----------------|----------------|
| Pełna uwaga | O(N²) | O(N) per layer | baseline | domyślna warstwa każdego modelu |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl, good with global layers | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | mixed | similar to SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) approx | mixed | matches full at 2× context | early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models |

## Zbuduj To

Zobacz `code/main.py`. Implementujemy comparator масокuwagi który pokazuje full, SWA, local+strided, i differential attention side by side na toy sequence.

### Krok 1: pełna causal mask (baseline)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Baseline z Lekcji 07. Lower triangular; zero weight above the diagonal.

### Krok 2: sliding window causal mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Jeden parametr — `window`. Dla `window >= n`, odzyskujesz full causal attention. Dla `window = 1`, każdy token uczestniczy tylko w sobie.

### Krok 3: local + strided sparse mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Dense local window plus every `stride`-th token back to the start of the sequence. Receptive field rośnie w log steps z dodatkowymi warstwami.

### Krok 4: differential attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Dwa attention passes, subtract with a learned mixing coefficient. W kodzie porównujemy attention-sink heatmap single vs differential i obserwujemy że sink znika.

### Krok 5: KV cache sizes

Wydrukuj rozmiar cache per layer przy `N = 131072` dla każdego wariantu. SWA i sparse variants spadają o 10–100×. Differential się podwaja. Płać rachunek pamięci świadomie.

## Użyj To

 wzorce produkcyjne 2026:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 miesza SWA (window=1024) i global layers w 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention w PyTorch 2.5+ akceptuje maskę jako funkcję:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

To kompiluje się do custom Triton kernel. Within 10% of FlashAttention-3 speed for common patterns, a mask function to Python callable.

**Kiedy wybrać każdy:**

- **Pure full attention** — każda warstwa do ~16K context, lub gdy jakość retrieval jest paramount.
- **SWA + global mix** — long context (>32K), training i inference memory-bound. Default 2026 powyżej 32K.
- **Sparse block attention** — custom kernel, custom pattern. Zarezerwowane dla specjalizowanych workloadów (retrieval, audio).
- **Differential attention** — każdy workload gdzie attention-sink contamination boli (long-context RAG, needle-in-haystack).

## Wyślij To

Zobacz `outputs/skill-attention-variant-picker.md`. Skill wybiera topologię uwagi dla nowego modelu przy danym target context length, retrieval demands, i training/inference compute profile.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Zweryfikuj że SWA przy `window=4` zeruje wszystko poza ostatnimi 4 tokenami per wiersz. Zweryfikuj że `window=n` reprodukuje pełną causal attention bit-identically.
2. **Średnie.** Zaimplementuj causal SWA z `window=1024` na szczycie Lekcji 07 capstone. Trenuj przez 1,000 steps na tinyshakespeare. Jak bardzo val loss się cofa vs full attention? Jak bardzo spada peak memory?
3. **Trudne.** Zaimplementuj Gemma-3-style 5:1 layer mix (5 SWA, 1 global) w capstone model. Porównaj loss, memory, i jakość generacji przeciwko pure-SWA i pure-global baselines przy matched parameters.
4. **Trudne.** Zaimplementuj differential attention z learned `λ` per head. Trenuj na synthetic retrieval task (one needle, 2,000 distractors). Zmierz retrieval accuracy vs single-attention baseline przy matched parameters.

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Local attention" | Każde zapytanie uczestniczy w ostatnich `W` tokenach; KV cache maleje do `O(W)`. |
| Effective receptive field | "How far back the model sees" | W `L`-warstwowym SWA stack z oknem `W`, do `L × W` tokenów. |
| Longformer / BigBird | "Local + global + random" | Sparse patterns z kilkoma zawsze-uczestniczącymi global tokens; wczesne long-context podejście. |
| Native Sparse Attention | "DeepSeek's kernel trick" | Naucz się block-level sparsity; pomijaj zero blocks na poziomie kernela przy zachowaniu jakości. |
| Differential attention | "Two maps, one subtracts" | DIFF Transformer: odejmij learned `λ` razy drugą mapę uwagi od pierwszej żeby anulować attention sinks. |
| Attention sink | "Weight bleeds to token 0" | Softmax normalization wymusza że wiersze sumują się do 1; nieinformatywne zapytania zrzucają wagę na pozycję 0. |
| FlexAttention | "Mask-as-Python" | PyTorch 2.5+ API które kompiluje arbitrary mask functions into FlashAttention-shape kernels. |
| Layer type mix | "5:1 SWA-to-global" | Przeplataj sparse i full attention layers w stack żeby utrzymać jakość przy mniejszej pamięci. |

## Dalsze Czytanie

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — kanoniczny sliding-window + global-token paper.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — local+strided pattern OpenAI.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:global mix.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 mix z window=1024 który jest teraz podręcznik default.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer paper.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2's learned-sparsity attention.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — API reference dla mask-as-callable pattern w Use It.