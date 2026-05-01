# Operacje na tensorach

> Tensory są wspólnym językiem między danymi a deep learning. Każdy obraz, każde zdanie, każdy gradient przepływa przez nie.

**Typ:** Zbuduj
**Jezyk:** Python
**Wymagania wstępne:** Phase 1, Lesson 01 (Intuicja algebry liniowej), 02 (Wektory, Macierze i Operacje)
**Czas:** ~90 minut

## Cele uczenia się

- Zaimplementuj klasę tensor z kształtem, skokami, reshape, transpose i operacjami elementowymi od zera
- Zastosuj reguły broadcastingu do operowania na tensorach różnych kształtów bez kopiowania danych
- Napisz wyrażenia einsum dla iloczynów skalarnych, mnożenia macierzy, produktów zewnętrznych i operacji wsadowych
- Śledź dokładne kształty tensorów przez każdy krok multi-head attention

## Problem

Budujesz transformer. Forward pass wygląda czysto. Uruchamiasz i dostajesz: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`. Wpatrujesz się w kształty. Próbujesz transpose. Teraz mówi `Expected 4D input (got 3D input)`. Dodajesz unsqueeze. Coś innego się psuje.

Błędy kształtów to najczęstsze bugi w kodzie deep learning. Nie są trudne koncepcyjnie -- każda operacja ma kontrakt kształtu -- ale mnożą się szybko. Transformer ma dziesiątki reshape, transpose i broadcastów związanych ze sobą. Jeden zły axis i błąd się kaskaduje. Co gorsza, niektóre błędy kształtów w ogóle nie rzucają błędów. Cicho produkują śmieci przez broadcasting po złym wymiarze lub sumowanie po złym axis.

Macierze obsługują pairwise relationships między dwoma zbiorami rzeczy. Rzeczywiste dane nie mieszczą się w dwóch wymiarach. Batch 32 obrazów RGB w 224x224 to tensor 4D: `(32, 3, 224, 224)`. Self-attention z 12 heads to też 4D: `(batch, heads, seq_len, head_dim)`. Potrzebujesz struktury danych, która uogólnia się na dowolną liczbę wymiarów, z operacjami, które składają się czysto przez wszystkie. Ta struktura to tensor. Zapanuj nad jego operacjami i błędy kształtów stają się trywialnie debugowalne.

## Koncepcja

### Czym jest tensor

Tensor to wielowymiarowa tablica liczb o jednolitym typie danych. Liczba wymiarów to **rank** (lub **order**). Każdy wymiar to **axis**. **Shape** to krotka określająca rozmiar wzdłuż każdego axis.

```mermaid
graph LR
    S["Scalar<br/>rank 0<br/>shape: ()"] --> V["Vector<br/>rank 1<br/>shape: (3,)"]
    V --> M["Matrix<br/>rank 2<br/>shape: (2,3)"]
    M --> T3["3D Tensor<br/>rank 3<br/>shape: (2,2,2)"]
    T3 --> T4["4D Tensor<br/>rank 4<br/>shape: (B,C,H,W)"]
```

Całkowita liczba elementów = iloczyn wszystkich rozmiarów. Shape `(2, 3, 4)` trzyma `2 * 3 * 4 = 24` elementów.

### Ksztalty tensorow w deep learning

Rozne typy danych mapuja do konkretnych ksztaltow tensorow konwencjonalnie.

```mermaid
graph TD
    subgraph Vision
        V1["(B, C, H, W)<br/>32, 3, 224, 224"]
    end
    subgraph NLP
        N1["(B, T, D)<br/>16, 128, 768"]
    end
    subgraph Attention
        A1["(B, H, T, D)<br/>16, 12, 128, 64"]
    end
    subgraph Weights
        W1["Linear: (out, in)<br/>Conv2D: (out_c, in_c, kH, kW)<br/>Embedding: (vocab, dim)"]
    end
```

PyTorch używa NCHW (channels-first). TensorFlow domyślnie NHWC (channels-last). Niezgodne layouty powodują ciche spowolnienia lub błędy.

### Jak działa pamięć layout

Tablica 2D w pamięci to 1D sekwencja bajtów. **Strides** mówią ile elementów przeskoczyć, żeby przesunąć się o jeden krok wzdłuż każdego axis.

```mermaid
graph LR
    subgraph "Row-major (C order)"
        R["a b c d e f<br/>strides: (3, 1)"]
    end
    subgraph "Column-major (F order)"
        C["a d b e c f<br/>strides: (1, 2)"]
    end
```

Transpose nie przesuwa danych. Zamienia strides, co czyni tensor **non-contiguous** -- elementy wiersza nie są już przyległe w pamięci.

### Reguly broadcastingu

Broadcasting pozwala operować na tensorach różnych kształtów bez kopiowania danych. Wyrównaj kształty od prawej. Dwa wymiary są kompatybilne gdy są równe lub jeden to 1. Mniej wymiarów dostaje padding z 1 po lewej.

```
Tensor A:     (8, 1, 6, 1)
Tensor B:        (7, 1, 5)
Padded B:     (1, 7, 1, 5)
Result:       (8, 7, 6, 5)
```

### Einsum: uniwersalna operacja tensorowa

Einstein summation etykietuje każdy axis literą. Axes w input ale nie w output są sumowane. Axes w obu są zachowane.

```mermaid
graph LR
    subgraph "matmul: ik,kj -> ij"
        A["A(I,K)"] --> |"sum over k"| C["C(I,J)"]
        B["B(K,J)"] --> |"sum over k"| C
    end
```

Kluczowe wzorce: `i,i->` (iloczyn skalarny), `i,j->ij` (produkt zewnetrzny), `ii->` (trace), `ij->ji` (transpose), `bij,bjk->bik` (batch matmul), `bhtd,bhsd->bhts` (attention scores).

## Zbuduj to

Kod zyje w `code/tensors.py`. Kazdy krok odwoluje sie do implementacji tam.

### Krok 1: Tensor storage i strides

Tensor przechowuje plaska liste liczb plus metadata ksztaltu. Strides mowia logice indeksowania jak mapowac wielowymiarowe indeksy na plaskie pozycje.

```python
class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
```

Dla shape `(3, 4)`, strides to `(4, 1)` -- przeskocz 4 elementy, zeby przesunac sie o jeden wiersz, przeskocz 1 element, zeby przesunac sie o jedna kolumne.

### Krok 2: Reshape, squeeze, unsqueeze

Reshape zmienia shape bez zmiany kolejności elementów. Całkowita liczba elementów musi pozostać ta sama. Użyj `-1` dla jednego wymiaru, żeby wywnioskować jego rozmiar.

```python
t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))
```

Squeeze usuwa axes rozmiaru 1. Unsqueeze wstawia jeden. Unsqueezing jest krytyczny dla broadcastingu -- wektor bias `(D,)` dodany do batch `(B, T, D)` potrzebuje unsqueeze do `(1, 1, D)`.

```python
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)
```

### Krok 3: Transpose i permute

Transpose zamienia dwa axes. Permute zmienia kolejność wszystkich axes. To jak konwertujesz między NCHW a NHWC.

```python
mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.permute((0, 2, 3, 1))
```

Po transpose lub permute tensor jest non-contiguous w pamięci. W PyTorch `view` failuje na non-contiguous tensorach -- użyj `reshape` lub wywołaj `.contiguous()` najpierw.

### Krok 4: Operacje elementowe i redukcje

Operacje elementowe (add, multiply, subtract) stosuja sie niezaleznie do kazdego elementu i zachowuja shape. Redukcje (sum, mean, max) kolapsuja jeden lub wiecej axes.

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])
c = a + b
d = a * 2
s = a.sum(axis=0)
```

Global average pooling w CNN: `(B, C, H, W).mean(axis=[2, 3])` produkuje `(B, C)`. Sequence mean pooling w NLP: `(B, T, D).mean(axis=1)` produkuje `(B, D)`.

### Krok 5: Broadcasting z NumPy

Funkcja `demo_broadcasting_numpy()` w `tensors.py` pokazuje core patterns.

```python
activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
result = activations + bias

images = np.random.randn(2, 3, 4, 4)
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
result = images * scale

a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
outer = a * b
```

Pairwise distance przez broadcasting: reshape `(M, 2)` do `(M, 1, 2)` i `(N, 2)` do `(1, N, 2)`, odejmij, podnies do kwadratu, sum po ostatnim axis, wez pierwiastek kwadratowy. Result: `(M, N)`.

### Krok 6: Operacje einsum

Funkcje `demo_einsum()` i `demo_einsum_gallery()` przechodzą przez każdy common pattern.

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dot = np.einsum("i,i->", a, b)

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
matmul = np.einsum("ik,kj->ij", A, B)

batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)
batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)
```

Koszt obliczeniowy kontracji to iloczyn wszystkich rozmiarów indeksów (kept i summed). Dla `bij,bjk->bik` z B=32, I=128, J=64, K=128: `32 * 128 * 64 * 128 = 33,554,432` mnożeń-dodawań.

### Krok 7: Mechanizm attention przez einsum

Funkcja `demo_attention_einsum()` implementuje multi-head attention end to end.

```python
B, H, T, D = 2, 4, 8, 16
E = H * D

X = np.random.randn(B, T, E)
W_q = np.random.randn(E, E) * 0.02

Q = np.einsum("bte,ek->btk", X, W_q)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)

scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
weights = softmax(scores, axis=-1)
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
output = np.einsum("bte,ek->btk", concat, W_o)
```

Każdy krok to operacja tensorowa: projekcja (matmul przez einsum), head splitting (reshape + transpose), attention scores (batch matmul przez einsum), weighted sum (batch matmul przez einsum), head merging (transpose + reshape), output projekcja (matmul przez einsum).

## Uzyj tego

### Scratch vs NumPy

| Operacja | Scratch (klasa Tensor) | NumPy |
|---|---|---|
| Create | `Tensor([[1,2],[3,4]])` | `np.array([[1,2],[3,4]])` |
| Reshape | `t.reshape((3,4))` | `a.reshape(3,4)` |
| Transpose | `t.transpose(0,1)` | `a.T` lub `a.transpose(0,1)` |
| Squeeze | `t.squeeze(0)` | `np.squeeze(a, 0)` |
| Sum | `t.sum(axis=0)` | `a.sum(axis=0)` |
| Einsum | N/A | `np.einsum("ij,jk->ik", a, b)` |

### Scratch vs PyTorch

```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
t.shape
t.stride()
t.is_contiguous()

t.reshape(3, 2)
t.unsqueeze(0)
t.transpose(0, 1)
t.transpose(0, 1).contiguous()

torch.einsum("ik,kj->ij", A, B)
```

PyTorch dodaje autograd, GPU support i zoptymalizowane jądra BLAS. Semantyka kształtów jest identyczna. Jeśli rozumiesz wersje scratch, błędy kształtów PyTorch stają się czytelne.

### Każda warstwa sieci neuronowej jako operacja tensorowa

| Operacja | Forma Tensor | Einsum |
|---|---|---|
| Warstwa liniowa | `Y = X @ W.T + b` | `"bd,od->bo"` + bias |
| Attention QKV | `Q = X @ W_q` | `"btd,dh->bth"` |
| Attention scores | `Q @ K.T / sqrt(d)` | `"bhtd,bhsd->bhts"` |
| Attention output | `softmax(scores) @ V` | `"bhts,bhsd->bhtd"` |
| Batch norm | `(X - mu) / sigma * gamma` | element-wise + broadcast |
| Softmax | `exp(x) / sum(exp(x))` | element-wise + reduction |

## Dostarcz to

Ta lekcja produkuje dwa reusable prompts:

1. **`outputs/prompt-tensor-shapes.md`** -- Systematyczny prompt do debugowania niezgodności kształtów tensorów. Zawiera tabele decyzyjne dla każdej common operacji (matmul, broadcast, cat, Linear, Conv2d, BatchNorm, softmax) i tabele lookup poprawek.

2. **`outputs/prompt-tensor-debugger.md`** -- Krok po kroku debugging prompt, który wklejasz do dowolnego AI assistant gdy błąd kształtu cię blokuje. Wrzuć komunikat błędu i kształty tensorów, dostan z powrotem dokładną poprawkę.

## Ćwiczenia

1. **Łatwe -- Reshape round-trip.** Weź tensor shape `(2, 3, 4)`. Zreshapeuj do `(6, 4)`, potem do `(24,)`, potem z powrotem do `(2, 3, 4)`. Zweryfikuj że kolejność elementów jest zachowana w każdym kroku przez wydrukowanie płaskich danych.

2. **Średnie -- Zaimplementuj broadcasting.** Rozszerz klasę `Tensor` o metodę `broadcast_to(shape)` która rozszerza wymiary rozmiaru 1 żeby dopasować docelowy shape. Potem zmodyfikuj `_elementwise_op` żeby automatic broadcast przed operacją. Testuj z kształtami `(3, 1)` i `(1, 4)` produkując `(3, 4)`.

3. **Trudne -- Zbuduj einsum od zera.** Zaimplementuj podstawową funkcję `einsum(subscripts, *tensors)` która obsługuje co najmniej: iloczyn skalarny (`i,i->`), mnożenie macierzy (`ij,jk->ik`), produkt zewnętrzny (`i,j->ij`) i transpose (`ij->ji`). Parsuj string subscript, zidentyfikuj contracted indices i pętlaj przez wszystkie kombinacje indeksów. Porównaj wyniki z `np.einsum`.

4. **Trudne -- Attention shape tracker.** Napisz funkcję która przyjmuje `batch_size`, `seq_len`, `embed_dim` i `num_heads` jako inputs i drukuje dokładny shape w każdym kroku multi-head attention: input, Q/K/V projekcja, head split, attention scores, softmax weights, weighted sum, head merge, output projekcja. Zweryfikuj przeciwko output `demo_attention_einsum()`.

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|---|---|---|
| Tensor | "Macierz ale więcej wymiarów" | Wielowymiarowa tablica z jednolitym typem i zdefiniowanym shape, strides i operacjami |
| Rank | "Liczba wymiarów" | Liczba axes. Macierz ma rank 2, nie rank równy jej matrix rank |
| Shape | "Rozmiar tensora" | Krotka listująca rozmiar wzdłuż każdego axis. `(2, 3)` oznacza 2 wiersze, 3 kolumny |
| Stride | "Jak pamięć jest ułożona" | Liczba elementów do przeskoczenia żeby przesunąć się o jedną pozycję wzdłuż każdego axis |
| Broadcasting | "To po prostu działa gdy kształty się różnią" | Ścisły zbiór reguł: wyrównaj od prawej, wymiary muszą być równe lub jeden musi być 1 |
| Contiguous | "Tensor jest normalny" | Elementy przechowywane sekwencyjnie w pamięci bez przerw lub zmiany kolejności z logical layout |
| Einsum | "Fajny sposób na napisanie matmul" | Ogólna notacja która wyraża dowolną kontrakcję tensora, produkt zewnętrzny, trace lub transpose w jednej linii |
| View | "To samo co reshape" | Tensor dzielący ten sam bufor pamięci ale z inną shape/stride metadata. Failuje na non-contiguous danych |
| Contraction | "Sumowanie po indeksie" | Ogólna operacja gdzie wspólny indeks między tensorami jest mnożony i sumowany, produkując wynik niższego ranku |
| NCHW / NHWC | "Format PyTorch vs TensorFlow" | Konwencje layout pamięci dla obrazów tensorów. NCHW wkłada kanały przed spatial dims, NHWC po |

## Dalsza lektura

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) -- Kanoniczne reguły z wizualnymi przykładami
- [PyTorch Tensor Views](https://pytorch.org/docs/stable/tensor_view.html) -- Kiedy views działają a kiedy kopiuja
- [einops](https://github.com/arogozhnikov/einops) -- Biblioteka która czyni tensor reshaping czytelnym i bezpiecznym
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- Wizualizuje kształty tensorów przepływające przez attention
- [Einstein Summation in NumPy](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) -- Pełna dokumentacja einsum z przykładami