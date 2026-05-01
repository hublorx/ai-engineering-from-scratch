# Multi-Head Attention

> Jedna głowa attention uczy się jednej relacji na raz. Osiem głów uczy się ośmiu. Głowy są darmowe. Weź ich więcej.

**Typ:** Buduj
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 02 (Self-Attention od Zera)
**Czas:** ~75 minut

## Problem

Pojedyncza głowa self-attention oblicza jedną macierz attention. Ta macierz przechwytuje jeden rodzaj relacji — zwykle ten, który minimalizuje loss na czymkolwiek, co jest sygnałem treningowym. Jeśli twoje dane mają zgodność podmiotu i orzeczenia, ko-referencję, długodystansowy dyskurs i syntactic chunking wszystkie poplątane razem, pojedyncza głowa rozmasowuje je w jeden miękki rozkład softmax i traci połowę sygnału.

Poprawka z artykułu Vaswani z 2017: uruchom kilka funkcji attention równolegle, każda z własnymi projekcjami Q, K, V i konkatenuj wyniki. Każda głowa operuje w mniejszej podprzestrzeni o wymiarze `d_model / n_heads`. Całkowita liczba parametrów zostaje taka sama. Siła ekspresyjna rośnie.

Multi-head attention to domyślne, z czym każdy transformer w 2026 jest wydawany. Jedynym argumentem jest *ile* głów i czy keys i values dzielą projekcje (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## Koncepcja

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Podziel.** Weź `X` o kształcie `(N, d_model)`. Projektuj do Q, K, V każdy o kształcie `(N, d_model)`. Reshape do `(N, n_heads, d_head)` gdzie `d_head = d_model / n_heads`. Transponuj do `(n_heads, N, d_head)`.

**Uczestnicz równolegle.** Uruchom scaled dot-product attention w każdej głowie. Każda głowa produkuje `(N, d_head)`. Głowy operują na różnych podprzestrzeniach embeddingu i nigdy nie rozmawiają podczas samego obliczania attention.

**Konkatenuj i projektuj.** Stack głowy z powrotem do `(N, d_model)` i pomnóż przez nauczoną macierz wyjściową `W_o` o kształcie `(d_model, d_model)`. `W_o` to miejsce, gdzie głowy się mieszają.

**Dlaczego to działa.** Każda głowa może się specjalizować bez konkurowania z innymi o budżet reprezentacyjny. Studia probing z 2019–2024 pokazują różne role głów: positional heads, głowy które uczestniczą w poprzednim tokenie, copy heads, named-entity heads, induction heads (które stanowią podstawę in-context learning).

**Linia wariantów w 2026:**

| Wariant | Q heads | K/V heads | Używany przez |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (np. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | skompresowane do low-rank | DeepSeek-V2, V3 |

GQA jest nowoczesnym domyślnym, bo tnie pamięć KV-cache o współczynnik `N/G` przy zachowaniu prawie pełnej jakości. MLA idzie dalej, kompresując K/V do przestrzeni latent, potem projektując z powrotem w czasie obliczeń — kosztuje FLOPs, oszczędza dużo więcej pamięci.

## Zbuduj to

### Krok 1: podziel głowy z single-head attention, którą już mamy

Weź `SelfAttention` z Lekcji 02 i owiń ją z pair split/concat. Zobacz `code/main.py` dla implementacji numpy; logika to:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Jeden reshape i jedna transpozycja. Bez pętli. To jest dokładnie to, co PyTorch robi pod `nn.MultiheadAttention`.

### Krok 2: uruchom scaled-dot-product attention per head

Każda głowa dostaje własny slice Q, K, V. Attention staje się wsadowym matmulem:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Na prawdziwym sprzęcie `Qh @ Kh.transpose(...)` to jeden `bmm`. GPU widzi jedno wsadowe mnożenie macierzy o kształcie `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Dodawanie głów jest darmowe.

### Krok 3: wariant Grouped-Query Attention

Tylko projekcje key i value się zmieniają. Q dostaje `n_heads` grup; K i V dostają `n_kv_heads < n_heads` grup i są powtarzane, żeby się dopasowały:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Podczas inferencji to oszczędza pamięć, bo tylko `n_kv_heads` kopii żyje w KV cache, nie `n_heads`. Llama 3 70B używa 64 query heads z 8 KV heads — 8× cache shrink.

### Krok 4: zbadaj, czego każda głowa się nauczyła

Uruchom MHA na krótkim zdaniu z 4 głowami. Dla każdej głowy, wydrukuj macierz attention `(N, N)`. Zobaczysz różne głowy wybierające różną strukturę nawet z losową inicjalizacją — to częściowo sygnał, częściowo rotational symmetry w podprzestrzeniach.

## Użyj tego

W PyTorch, wersja jednolinijkowa:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA od PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Ile głów?** Zasady kciuka z production models w 2026:

| Rozmiar modelu | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` prawie zawsze ląduje na 64 lub 128. To jest jednostka tego, ile jedna głowa może "zobaczyć." Zejdź poniżej 32 i głowy zaczynają walczyć ze skalą `sqrt(d_head)`; idź powyżej 256 i tracisz korzyść "wielu małych specjalistów".

## Wyślij to

Zobacz `outputs/skill-mha-configurator.md`. Skill poleca liczbę głów, liczbę kv-head i strategię projekcji dla nowego transformera przy danym budżecie parametrów, długości sekwencji i celu deployment.

## Ćwiczenia

1. **Łatwe.** Weź MHA z `code/main.py` i zmień `n_heads` z 1 na 16 przy `d_model=64` ustawionym. Wykreśl loss małego jednowarstwowego modelu na syntetycznym zadaniu copy. Czy więcej głów pomaga, plateau, czy szkodzi?
2. **Średnie.** Zaimplementuj MQA (jeden KV head współdzielony przez wszystkie query heads). Zmierz, jak bardzo spada liczba parametrów vs pełne MHA. Oblicz, jak bardzo KV-cache size maleje podczas inferencji dla N=2048.
3. **Trudne.** Zaimplementuj tiny version Multi-head Latent Attention: kompresuj K,V do rangi `r` latent, przechowuj latent w KV cache, dekompresuj w czasie attention. Przy jakim `r` pamięć cache przekracza poniżej 1/8 pełnego MHA, a jakość zostaje w 1 bicie validation ppl?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Head | "A single attention circuit" | Jedna projekcja Q/K/V o wymiarze `d_head = d_model / n_heads` z własną macierzą attention. |
| d_head | "Head dimension" | Per-head hidden width; prawie zawsze 64 lub 128 w production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose wokół attention. |
| W_o | "Output projection" | Macierz `(d_model, d_model)` aplikowana po konkatenacji głów; gdzie głowy się mieszają. |
| MQA | "One KV head" | Multi-Query Attention: jeden współdzielony projekcja K/V. Najmniejszy KV cache, trochę utraty jakości. |
| GQA | "The default since Llama 2" | Grouped-Query Attention z `n_kv_heads < n_heads`; powtarza, żeby dopasować Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V skompresowane do low-rank latent, dekompresowane w czasie attend. |
| Induction head | "The circuit behind in-context learning" | Para głów, która wykrywa poprzednie wystąpienia i kopiuje to, co po nich followowało. |

## Dalsze Czytanie

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — oryginalna specyfikacja multi-head.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — artykuł o MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — jak konwertować MHA do GQA po treningu.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA i dlaczego bije MHA/GQA na pamięci cache.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — mechanisticzny look na to, co głowy faktycznie robią.