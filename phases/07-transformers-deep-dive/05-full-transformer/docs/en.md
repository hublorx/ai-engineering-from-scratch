# Pełny Transformer — Encoder + Decoder

> Attention jest gwiazdą. Wszystko inne — residuals, normalization, feed-forward, cross-attention — to rusztowanie, które pozwala stackować to głęboko.

**Typ:** Buduj
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 02 (Self-Attention), Faza 7 · 03 (Multi-Head Attention), Faza 7 · 04 (Pozycyjne Kodowanie)
**Czas:** ~75 minut

## Problem

Pojedyncza warstwa attention to feature extractor, nie model. Jeden matmul per layer to nie wystarczająca pojemność dla języka. Potrzebujesz głębi — a głębia się łamie bez odpowiedniej hydrauliki.

Artykuł Vaswani z 2017 spakował sześć decyzji projektowych, które zamieniły jedną warstwę attention w stackowalny block. Każdy transformer od tego czasu — encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) — dziedziczy ten sam szkielet. W 2026 bloki zostały dopracowane (RMSNorm, SwiGLU, pre-norm, RoPE) ale szkielet jest identyczny.

Ta lekcja to szkielet. Następne lekcje go specjalizują — 06 dla encoderów, 07 dla decoderów, 08 dla encoder-decoder.

## Koncepcja

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### Sześć kawałków

1. **Embedding + positional signal.** Tokens → vectors. Pozycja wstrzykiwana przez RoPE (nowoczesne) lub sinusoidal (klasyczne).
2. **Self-attention.** Każda pozycja uczestniczy w każdej innej. Zamaskowane w decoderach.
3. **Feed-forward network (FFN).** Position-wise two-layer MLP: `W_2 · activation(W_1 · x)`. Expansion ratio 4× domyślnie.
4. **Residual connection.** `x + sublayer(x)`. Bez tego, gradienty znikają po ~6 warstwach.
5. **Layer normalization.** `LayerNorm` lub `RMSNorm` (nowoczesne). Stabilizuje residual stream.
6. **Cross-attention (tylko decoder).** Queries pochodzą z decodera, keys i values z encoder output.

### Encoder block (używany przez BERT, T5 encoder)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder jest bidirectional. Bez maskowania. Wszystkie pozycje widzą wszystkie pozycje.

### Decoder block (używany przez GPT, T5 decoder)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder ma trzy sublayers per block. Ten środkowy — cross-attention — to jedyne miejsce, gdzie informacja płynie z encoder do decoder. W czystej architekturze decoder-only (GPT), cross-attention jest pomijane i masz po prostu masked self-attention + FFN.

### Pre-norm vs post-norm

Oryginalny artykuł: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm stracił favour około 2019 — trudniej trenować głęboko bez careful warmup. Pre-norm (`LN` *before* sublayer) to 2026 default: Llama, Qwen, GPT-3+, Mistral wszyscy go używają.

### Modernized block w 2026

Vaswani 2017 wysyłał LayerNorm + ReLU. Nowoczesne stacki zastąpiły oba. Jak wyglądają production blocks w praktyce:

| Komponent | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU używa trzech macierzy, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (lub MLA) |
| Bias terms | Yes | No |

RMSNorm upuszcza mean-centering LayerNorm (jedno mniej subtraction), co oszczędza compute i empirycznie jest co najmniej tak stabilne. SwiGLU (`Swish(W1 x) ⊙ W3 x`) konsekwentnie bije ReLU/GELU FFN o ~0.5 punkt ppl w artykułach Llama, PaLM i Qwen.

### Liczba parametrów

Dla jednego blocka z `d_model = d` i FFN expansion `r`:

- MHA: `4 · d²` (Q, K, V, O projections)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norms: negligible

Przy `d = 4096, r = 2.6, layers = 32` (w przybliżeniu Llama 3 8B), total: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parametrów per layer × 32 ≈ 7B` (plus embeddings i head). Zgadza się z opublikowanymi liczbami.

## Zbuduj to

### Krok 1: budujące bloki

Używając tiny `Matrix` class z Lekcji 03 (skopiowane do tego pliku dla niezależności):

- `layer_norm(x, eps=1e-5)` — odejmij mean, podziel przez std.
- `rms_norm(x, eps=1e-6)` — podziel przez RMS. Bez mean subtraction.
- `gelu(x)` i `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` i `decoder_block(x, enc_out, params)`.

Zobacz `code/main.py` dla pełnego okablowania.

### Krok 2: podłącz 2-warstwowy encoder i 2-warstwowy decoder

Stackuj je. Przepuść encoder output do każdego decoder cross-attention. Dodaj final LN przed output projection.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Krok 3: uruchom forward na przykładzie zabawce

Przepuść 6-tokenowe źródło i 5-tokenowy target przez. Zweryfikuj, że output shape to `(5, vocab)`. Bez treningu — ta lekcja jest o architekturze, nie o loss.

### Krok 4: zamień na RMSNorm + SwiGLU

Zamień LayerNorm i ReLU-FFN na RMSNorm i SwiGLU. Potwierdź, że shapes nadal się zgadzają. To jest modernizacja 2026 z jedną substytucją funkcji.

## Użyj tego

PyTorch/TF reference implementations: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. Ale większość 2026 production kodu pisze własny block, bo:

- Flash Attention jest wywoływany inside attention, nie przez `nn.MultiheadAttention`.
- GQA / MLA nie są w stdlib reference.
- RoPE, RMSNorm, SwiGLU nie są domyślne w PyTorch.

HF `transformers` ma czyste reference blocks, które powinieneś przeczytać: `modeling_llama.py` to kanoniczny 2026 decoder-only block. To ~500 linii i warto przez to przejść raz.

**Encoder vs decoder vs encoder-decoder — kiedy wybrać:**

| Potrzeba | Wybierz | Przykład |
|------|------|---------|
| Classification, embeddings, QA nad tekstem | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Decoder-only wygrał język, bo najczystszy skalowanie i obsługuje zarówno comprehension i generation. Encoder-decoder wciąż najlepszy, gdy input ma wyraźną "source sequence" identity (translation, speech recognition, structured tasks).

## Wyślij to

Zobacz `outputs/skill-transformer-block-reviewer.md`. Skill przegląda nową implementację transformer block przeciwko 2026 defaults i flaguje missing pieces (pre-norm, RoPE, RMSNorm, GQA, FFN expansion ratio).

## Ćwiczenia

1. **Łatwe.** Policz parametry w swoim encoder_block przy `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Validate przez implementację blocka i użycie `sum(p.numel() for p in block.parameters())`.
2. **Średnie.** Przełącz z post-norm na pre-norm. Zainicjalizuj oba i zmierz activation norm po 12 stacked layers na losowym input. Post-norm activations powinny eksplodować; pre-norm powinny pozostać bounded.
3. **Trudne.** Zaimplementuj 4-warstwowy encoder-decoder na zabawce copy task (copy `x` reversed). Trenuj 100 steps. Raportuj loss. Zamień na RMSNorm + SwiGLU + RoPE — czy loss spada?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack norm + attention + norm + FFN, opakowany w residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; umożliwia przepływ gradientów przez głębokie stacki. |
| Pre-norm | "Normalize before, not after" | Nowoczesne: `x + sublayer(LN(x))`. Trenuje głębiej bez warmup gimnastyki. |
| RMSNorm | "LayerNorm without the mean" | Podziel przez RMS; jedna mniej op, ta sama empiryczna stabilność. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Bije ReLU/GELU na LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA z Q z decodera, K/V z encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio hidden-size do d_model, zwykle 4 (LayerNorm) lub 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Nowoczesne stacki pomijają biases w linear layers; lekkie ppl improvement, mniejszy model. |

## Dalsze Czytanie

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — oryginalna specyfikacja blocka.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — dlaczego pre-norm bije post-norm głęboko.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — artykuł o SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — kanoniczny 2026 decoder-only block.