# Self-Attention od Zera

> Attention to tabela lookup, gdzie każde słowo pyta "kto jest dla mnie ważny?" - i uczy się odpowiedzi.

**Typ:** Buduj
**Języki:** Python
**Wymagania wstępne:** Faza 3 (Deep Learning Core), Faza 5 Lekcja 10 (Sequence-to-Sequence)
**Czas:** ~90 minut

## Cele uczenia się

- Implementuj scaled dot-product self-attention od zera używając tylko NumPy, włącznie z projekcjami query/key/value i sumą ważoną softmax
- Zbuduj warstwę multi-head attention, która dzieli heads, oblicza równoległą attention i konkatenuje wyniki
- Śledź, jak macierz attention przechwytuje relacje między tokenami i wyjaśnij, dlaczego skalowanie przez sqrt(dk) zapobiega nasyceniu softmax
- Zastosuj causal masking, żeby przekonwertować bidirectional attention na autoregressive (decoder-style) attention

## Problem

RNN przetwarzają sekwencje jeden token na raz. Kiedy docierasz do tokena 50, informacja z tokena 1 została już ściśnięta przez 50 kroków kompresji. Zależności długiego zasięgu są miażdżone do ukrytego stanu o stałym rozmiarze - bottleneck, którego żadna ilość LSTM gating nie rozwiązuje całkowicie.

Artykuł Bahdanau attention z 2014 pokazał poprawkę: pozwól decoderowi patrzeć wstecz na każdą pozycję encoderu i zdecydować, które są ważne dla obecnego kroku. Ale nadal było to doczepione do RNN. Artykuł "Attention Is All You Need" z 2017 zadał ostrzejsze pytanie: co gdy attention jest *jedynym* mechanizmem? Bez rekurencji. Bez konwolucji. Tylko attention.

Self-attention pozwala każdej pozycji w sekwencji uczestniczyć w każdej innej pozycji w jednym równoległym kroku. To jest to, co czyni transformers szybkimi, skalowalnymi i dominującymi.

## Koncepcja

### Analogie bazy danych

Pomyśl o attention jako miękkim lookupie bazy danych:

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

Każdy token generuje trzy wektory:
- **Query (Q)**: "Czego szukam?"
- **Key (K)**: "Co zawieram?"
- **Value (V)**: "Jaką informację dostarczam, jeśli zostanę wybrany?"

Iloczyn skalarny między query a wszystkimi keys produkuje attention scores. Wysoki score oznacza "ten key pasuje do mojego query." Te scores ważą values. Output to ważona suma values.

### Obliczanie Q, K, V

Każdy token embedding jest projektowany przez trzy nauczone macierze wag:

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

Wizualnie, dla jednego tokena:

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### Macierz Attention

Kiedy masz już Q, K, V dla wszystkich tokenów, attention scores tworzą macierz:

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

### Dlaczego skalować?

Iloczyny skalarne rosną z wymiarem dk. Jeśli dk = 64, iloczyny skalarne mogą być w zakresie dziesiątek, pchając softmax w regiony, gdzie gradienty znikają. Poprawka: dziel przez sqrt(dk).

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

To utrzymuje wartości w zakresie, gdzie softmax produkuje użyteczne gradienty.

### Softmax zamienia scores w wagi

Softmax konwertuje surowe logity na rozkład prawdopodobieństwa w każdym wierszu:

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

Teraz każdy token ma zbiór wag mówiących, ile uczestniczyć w każdym innym tokenze.

### Ważona suma wartości

Końcowy output dla każdego tokena to ważona suma wszystkich wektorów value:

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### Pełny Pipeline

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

Formuła w jednej linii:

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## Zbuduj to

### Krok 1: Softmax od zera

Softmax konwertuje surowe logity na prawdopodobieństwa. Odejmij maximum dla numerycznej stabilności.

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### Krok 2: Scaled dot-product attention

Główna funkcja. Przyjmuje macierze Q, K, V i zwraca attention output plus macierz wag.

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### Krok 3: Klasa Self-attention z nauczonymi projekcjami

Pełny moduł self-attention z macierzami wag Wq, Wk, Wv zainicjalizowanymi z Xavier-like scaling.

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### Krok 4: Uruchom to na zdaniu

Stwórz fałszywe embeddings dla zdania i obserwuj attention weights.

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### Krok 5: Wizualizuj attention z ASCII heatmap

Mapuj attention weights na znaki dla szybkiej wizualizacji.

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## Użyj tego

PyTorch `nn.MultiheadAttention` robi dokładnie to, co zbudowaliśmy, plus multi-head splitting i output projection:

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

Kluczowa różnica: multi-head attention uruchamia wiele funkcji attention równolegle, każda z własnymi projekcjami Q, K, V o rozmiarze dk = d_model / n_heads, a potem konkatenuje wyniki. To pozwala modelowi uczestniczyć w różnych typach relacji jednocześnie.

## Wyślij to

Ta lekcja produkuje:
- `outputs/prompt-attention-explainer.md` - prompt do wyjaśniania attention przez analogię bazy danych

## Ćwiczenia

1. Zmodyfikuj `scaled_dot_product_attention`, żeby przyjmowała opcjonalną macierz mask, która ustawia pewne pozycje na ujemna nieskończoność przed softmax (to jest jak działa causal/decoder masking)
2. Zaimplementuj multi-head attention od zera: podziel Q, K, V na `n_heads` kawałków, uruchom attention na każdym, konkatenuj i projektuj przez końcową macierz wag Wo
3. Weź dwa różne zdania tej samej długości, przepuść je przez tę samą instancję SelfAttention i porównaj ich attention patterns. Co się zmienia? Co pozostaje takie same?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|----------------|----------------------|
| Query (Q) | "The question vector" | Nauczona projekcja wejścia reprezentująca, jakich informacji ten token szuka |
| Key (K) | "The label vector" | Nauczona projekcja reprezentująca, jakie informacje ten token zawiera, dopasowywana do queries |
| Value (V) | "The content vector" | Nauczona projekcja niosąca faktyczną informację, która jest agregowana na podstawie attention scores |
| Scaled dot-product attention | "The attention formula" | softmax(QK^T / sqrt(dk)) @ V - skalowanie zapobiega nasyceniu softmax w wysokich wymiarach |
| Self-attention | "The token looks at itself and others" | Attention gdzie Q, K, V wszystkie pochodzą z tej samej sekwencji, pozwalając każdej pozycji uczestniczyć w każdej innej pozycji |
| Attention weights | "How much focus" | Rozkład prawdopodobieństwa nad pozycjami, produkowany przez softmax nad skalowanymi iloczynami skalarnymi |
| Multi-head attention | "Parallel attention" | Uruchamianie wielu funkcji attention z różnymi projekcjami, potem konkatenowanie wyników dla bogatszych reprezentacji |

## Dalsze Czytanie

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - oryginalny artykuł o transformer
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - najlepszy wizualny przewodnik po pełnej architekturze
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - implementacja PyTorch linia po linii z wyjaśnieniami