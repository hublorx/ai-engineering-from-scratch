# Pozycyjne Kodowanie — Sinusoidal, RoPE, ALiBi

> Attention jest permutation-invariant. "The cat sat on the mat" i "mat the on sat cat the" produkują ten sam output bez sygnału pozycyjnego. Trzy algorytmy to naprawiają — każdy z innym zakładem na to, co "pozycja" oznacza.

**Typ:** Buduj
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 02 (Self-Attention), Faza 7 · 03 (Multi-Head Attention)
**Czas:** ~45 minut

## Problem

Scaled dot-product attention jest order-blind. Macierz attention `softmax(Q K^T / √d) V` jest obliczana z pairwise similarities. Przetasuj wiersze `X`, dostaniesz wiersze output shuffled tak samo. Nic inside attention nie dba o pozycję.

To nie jest bug w bag-of-words model. Dla języka, kodu, audio, video — cokolwiek gdzie kolejność niesie znaczenie — jest to fatalne.

Poprawka to wstrzyknąć pozycję w embeddings jakoś. Trzy ery odpowiedzi:

1. **Absolute sinusoidal** (Vaswani 2017). Dodaj `sin/cos` pozycji do embedding. Proste, bez learningu, słabo extrapoluje poza trenowane długości.
2. **RoPE — Rotary Position Embeddings** (Su 2021). Obracaj wektory Q i K o kąt proporcjonalny do pozycji. Koduje *względną* pozycję bezpośrednio w iloczynie skalarnym. Dominujące w 2026.
3. **ALiBi — Attention with Linear Biases** (Press 2022). Pomiń embeddings całkowicie; dodaj per-head linear penalty do attention scores na podstawie dystansu. Świetna długość extrapolation.

Od 2026, zasadniczo każdy frontier open model używa RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Garstka long-context models używa ALiBi lub jego nowoczesnych wariantów. Absolute sinusoidal jest historyczne.

## Koncepcja

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Absolute sinusoidal

Pre-oblicz stałą macierz `PE` o kształcie `(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Potem `X' = X + PE[:N]` przed attention. Każdy wymiar to sinusoid przy różnej częstotliwości. Model uczy się czytać pozycję z pattern fazy. Failuje poza `max_len`: nic nie powiedziało modelowi, co się dzieje na pozycji 2048, kiedy widział tylko pozycje 0–2047.

### RoPE

Obracaj wektory Q i K (nie embeddings). Dla pary wymiarów `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Zastosuj tę samą rotację do keys z pozycją `pos_k`. Iloczyn skalarny `q'_m · k'_n` staje się funkcją tylko `(m - n)`. To jest: **attention score zależy tylko od względnej odległości**, nawet jeśli rotacja była kluczowa off absolute positions. Piękna sztuczka.

Rozszerzanie RoPE: `base` może być skalowane (NTK-aware, YaRN, LongRoPE) żeby extrapolować do dłuższych kontekstów bez retrainingu. Llama 3 rozszerzyła z 8K do 128K kontekstu w ten sposób.

### ALiBi

Pomiń trick z embedding. Biasuj attention scores bezpośrednio:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Gdzie `m_h` to head-specific slope (np. `1 / 2^(8·h/H)`). Bliższe tokeny są wzmacniane; dalekie tokeny są karane. Zero kosztu treningowego. Artykuł pokazuje, że length extrapolation bije sinusoidal i matches RoPE na jego oryginalnej trenowanej długości.

### Co wybrać w 2026

| Wariant | Extrapolacja | Koszt treningu | Używany przez |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | słaba | darmowe | original transformer, early BERT |
| Learned absolute | żadna | malutki | GPT-2, GPT-3 |
| RoPE | dobra ze skalowaniem | darmowe | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | świetna | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | świetna | darmowe | BLOOM, MPT, Baichuan |

RoPE wygrało, bo wslotuje się w attention bez zmiany architektury, koduje relative position, a jego hyperparameter `base` daje czysty knob dla long-context fine-tuningu.

## Zbuduj to

### Krok 1: sinusoidal encoding

Zobacz `code/main.py`. 4-liniowa kalkulacja:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Dodaj to do macierzy embedding przed pierwszą warstwą attention.

### Krok 2: RoPE applied to Q, K

RoPE działa in-place na Q i K. Dla każdej pary wymiarów:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Kluczowe: zastosuj tę samą funkcję do Q na pozycji `m` i K na pozycji `n`. Ich iloczyn skalarny picks up `cos((m-n)·θ_i)` factor na każdej parze współrzędnych. Attention uczy się relative position za darmo.

### Krok 3: ALiBi slopes and bias

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Dodaj `bias[h]` do macierzy attention scores `(seq_len, seq_len)` głowy `h`, potem softmax.

### Krok 4: zweryfikuj właściwość względnego-dystansu RoPE

Wybierz dwa losowe wektory `a, b`. Obróć przez `(pos_a, pos_b)`. Potem przez `(pos_a + k, pos_b + k)`. Oba iloczyny skalarne muszą się zgadzać w granicach błędu zmiennoprzecinkowego. Ta właściwość jest całym punktem RoPE — jest niezmienniczy do absolute offset, tylko relative gap ma znaczenie.

## Użyj tego

PyTorch 2.5+ dostarcza RoPE utilities w `torch.nn.functional`. Większość production kodu używa `flash_attn` lub `xformers` gdzie RoPE jest aplikowane inside attention kernel.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Tricki long-context w 2026:**

- **NTK-aware interpolation.** Przeskaluj `base` do `base * (scale_factor)^(d/(d-2))` przy rozszerzaniu z 4K do 16K+.
- **YaRN.** Mądrzejsza interpolacja, która preservuje attention entropy na długich kontekstach. Llama 3.1 128K jej używa.
- **LongRoPE.** Metoda Microsoft z 2024, która używa evolutionary search do picking per-dimension scale factors. Phi-3-Long jej używa.
- **Position interpolation + fine-tuning.** Po prostu zmniejsz pozycje o współczynnik rozszerzenia i fine-tune na 1–5B tokenów. Zaskakująco efektywne.

## Wyślij to

Zobacz `outputs/skill-positional-encoding-picker.md`. Skill wybiera strategię kodowania dla nowego modelu przy danej docelowej długości kontekstu, potrzebach extrapolacji i budżecie treningowym.

## Ćwiczenia

1. **Łatwe.** Wykreśl sinusoidal `PE` matrix jako heatmap dla `max_len=512, d=128`. Potwierdź pattern "paski stają się szersze, gdy rośnie index wymiaru."
2. **Średnie.** Zaimplementuj NTK-aware RoPE scaling. Trenuj tiny LM na sekwencjach długości 256, potem testuj na długości 1024 z i bez skalowania. Zmierz perplexity.
3. **Trudne.** Zaimplementuj ALiBi i RoPE w tym samym module attention. Trenuj 4-warstwowy transformer na zadaniu copy z sekwencjami długości 512. Extrapoluj do 2048 w czasie testu. Porównaj degradację.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Jakikolwiek sygnał dodany do embeddings lub attention, który koduje pozycję. |
| Sinusoidal | "The original one" | `sin/cos` przy geometrycznych częstotliwościach dodane do embeddings; nie extrapoluje. |
| RoPE | "Rotary embeddings" | Obracaj Q, K przez position-dependent angle; iloczyn skalarny koduje relative distance. |
| ALiBi | "Linear bias trick" | Dodaj `-m·|i-j|` do attention scores; żaden embedding nie potrzebny, świetna extrapolacja. |
| base | "RoPE's knob" | Frequency scaler w RoPE; zwiększ, żeby rozszerzyć kontekst przy inferencji. |
| NTK-aware | "A RoPE scaling trick" | Przeskaluj `base` żeby wysokie częstotliwości dims nie były ściskane gdy kontekst się rozszerza. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation, która preservuje attention entropy. |
| Extrapolation | "Works beyond trained length" | Czy scheme pozycji może serwować correct output past `max_len` widziany w treningu? |

## Dalsze Czytanie

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — oryginalne sinusoidal.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — artykuł o RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — state of the art RoPE scaling.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta's Llama 2 long-context paper.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — metoda Microsoft używana przez Phi-3-Long i cytowana w sekcji Użyj tego.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — production-grade implementations każdego scheme skalowania RoPE (default, linear, dynamic, YaRN, LongRoPE, Llama-3).