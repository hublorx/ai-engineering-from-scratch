# Mechanizm Attention — Przełom

> Dekoder przestaje mrużyć oczy na skompresowane podsumowanie i zaczyna patrzeć na całe źródło. Wszystko po tym to attention plus inżynieria.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 09 (Modele Sequence-to-Sequence)
**Czas:** ~45 minut

## Problem

Lekcja 09 zakończyła się mierzonym niepowodzeniem. GRU encoder-decoder wytrenowany na zadaniu kopiowania przechodzi z 89% dokładnością przy długości 5 do bliskiego losowemu przy długości 80. Powód jest strukturalny, nie błąd treningu: każdy bit informacji, który zebrał enkoder, musi się zmieścić w jednym wektorze o ustalonej wielkości, a dekoder nigdy nie widzi niczego innego.

Bahdanau, Cho i Bengio opublikowali trójlinijkową poprawkę w 2014 roku. Zamiast dawać dekoderowi tylko końcowy stan enkoderowy, zachowaj każdy stan enkoderowy. W każdym kroku dekodera oblicz średnią ważoną stanów enkoderowych, gdzie wagi mówią „ile dekoder musi spojrzeć na pozycję enkodera `i` właśnie teraz?". Ta średnia ważona to kontekst, i zmienia się w każdym kroku dekodera.

To jest cała idea. Transformery ją rozszerzyły. Self-attention zastosowało ją do pojedynczej sekwencji. Multi-head attention uruchomiło ją równolegle. Ale wersja z 2014 roku już przełamała wąskie gardło, i gdy już ją masz, przejście do transformerów to inżynieria, nie koncepcja.

## Koncepcja

![Bahdanau attention: dekoder odpytuje wszystkie stany enkoderowe](../assets/attention.svg)

W każdym kroku dekodera `t`:

1. Użyj poprzedniego stanu ukrytego dekodera `s_{t-1}` jako **query**.
2. Porównaj go z każdym stanem ukrytym enkodera `h_1, ..., h_T`. Jeden skalarny wynik na pozycję enkodera.
3. Przepuść wyniki przez softmax, aby uzyskać wagi attention `α_{t,1}, ..., α_{t,T}`, które sumują się do 1.
4. Wektor kontekstowy `c_t = Σ α_{t,i} * h_i`. Średnia ważona stanów enkoderowych.
5. Dekoder bierze `c_t` plus poprzedni token wyjściowy, generuje następny token.

Średnia ważona to sedno. Gdy dekoder potrzebuje przetłumaczyć „Je" na „I", wysoko waży stan enkoderowy nad „Je" i nisko pozostałe. Gdy potrzebuje „not", wysoko waży „pas". Wektor kontekstowy zmienia kształt przy każdym kroku.

## Kształty (to, co każdego psuje na początku)

To jest miejsce, gdzie każda implementacja attention idzie źle za pierwszym razem. Czytaj powoli.

| Element | Kształt | Uwagi |
|---------|---------|-------|
| Stany ukryte enkodera `H` | `(T_enc, d_h)` | Jeśli BiLSTM, `d_h = 2 * d_hidden` |
| Stan ukryty dekodera `s_{t-1}` | `(d_s,)` | Jeden wektor |
| Wynik attention `e_{t,i}` | skalarny | Jeden na pozycję enkodera |
| Waga attention `α_{t,i}` | skalarny | Po softmaxie względem wszystkich `i` |
| Wektor kontekstowy `c_t` | `(d_h,)` | Taki sam kształt jak stan enkoderowy |

**Wynik Bahdanau (addytywny).** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` ma kształt `(d_s,)`, `h_i` ma kształt `(d_h,)`.
- `W_a` ma kształt `(d_attn, d_s)`. `U_a` ma kształt `(d_attn, d_h)`.
- Ich suma wewnątrz tanh ma kształt `(d_attn,)`.
- `v_α` ma kształt `(d_attn,)`. Iloczyn skalarny z `v_α` redukuje do skalara. **To jest to, co robi `v_α`.** To nie jest magia. To jest projekcja, która zamienia wektor o wymiarze attention na skalarny wynik.

**Wynik Luong (multiplikatywny).** Trzy warianty:

- `dot`: `e_{t,i} = s_t^T * h_i`. Wymaga `d_s == d_h`. Trudne ograniczenie. Pomijaj, jeśli enkoder jest dwukierunkowy.
- `general`: `e_{t,i} = s_t^T * W * h_i` gdzie `W` ma kształt `(d_s, d_h)`. Usuwa ograniczenie równych wymiarów.
- `concat`: zasadniczo forma Bahdanau. Rzadko używana, bo dwa pierwsze są tańsze.

**Jedna pułapka Bahdanau/Luong, którą warto nazwać.** Bahdanau używa `s_{t-1}` (stan dekodera *przed* wygenerowaniem bieżącego słowa). Luong używa `s_t` (stan *po*). Pomylenie ich powoduje subtelnie błędne gradienty, które są niezwykle trudne do debugowania. Wybierz jedną pracę i trzymaj się jej konwencji.

## Zbuduj to

### Krok 1: addytywne (Bahdanau) attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Sprawdź swoje kształty względem tabeli powyżej. `encoder_states` ma kształt `(T_enc, d_h)`. `projected_enc` ma kształt `(T_enc, d_attn)`. `projected_dec` ma kształt `(d_attn,)` i się broadcastuje. `combined` ma kształt `(T_enc, d_attn)`. `scores` ma kształt `(T_enc,)`. `weights` ma kształt `(T_enc,)`. `context` ma kształt `(d_h,)`. Wysyłaj.

### Krok 2: Luong dot i general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Trzy linie każda. Dlatego praca Luonga tak dobrze wylądowała. Ta sama dokładność na większości zadań, dużo mniej kodu.

### Krok 3: przeprowadzony przykład numeryczny

Mając trzy stany enkoderowe (mniej więcej „cat", „sat", „mat") i stan dekodera, który najbardziej wyrównuje się z pierwszym, rozkład attention koncentruje się na pozycji 0. Gdy stan dekodera przesunie się bliżej trzeciego stanu enkoderowego, attention przesunie się do pozycji 2. Wektor kontekstowy to śledzi.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

Pierwszy wiersz wygrywa. Potem przesuń stan dekodera bliżej trzeciego stanu enkoderowego i obserwuj przesunięcie wag. To jest to. Attention to jawny alignment.

### Krok 4: dlaczego to jest most do transformerów

Przetłumacz język powyżej na Q/K/V:

- **Query** = stan dekodera `s_{t-1}`
- **Key** = stany enkoderowe (to, z czym porównujemy)
- **Value** = stany enkoderowe (to, co wagujemy i sumujemy)

W klasycznym attention keys i values są tym samym. Self-attention je rozdziela: możesz odpytywać sekwencję względem siebie, z różnymi nauczonymi projekcjami dla K i V. Multi-head attention uruchamia to równolegle z różnymi nauczonymi projekcjami. Transformery stackują cały etap wiele razy i porzucają RNN-y.

Matematyka jest ta sama. Kształty są takie same. Pedagogiczny skok od Bahdanau attention do scaled dot-product attention to głównie notacja.

## Użyj tego

PyTorch i TensorFlow mają attention wbudowane bezpośrednio.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

To jest warstwa attention transformera. Batch query 5 pozycji, klucz/wartość batch 10 pozycji, 128 wymiarów każdy, 8 heads. `output` to nowe query wzbogacone o kontekst. `weights` to macierz alignment 5x10, którą można zwizualizować.

### Kiedy klasyczne attention wciąż ma znaczenie

- Pedagogika. Wersja single-head, single-layer, oparta na RNN sprawia, że każda koncepcja jest widoczna.
- Zadania sekwencyjne na urządzeniach, gdzie transformery się nie mieszczą.
- Każda praca z lat 2014-2017. Bez znajomości konwencji Bahdanau ją źle odczytasz.
- Szczegółowa analiza alignment w MT. Surowe wagi attention to narzędzie interpretowalności nawet na modelach transformerowych, a ich odczytywanie wymaga wiedzy, czym są.

### Pułapka wag attention jako wyjaśnienia

Wagi attention wyglądają interpretowalnie. To wagi, które sumują się do jedności wzdłuż pozycji; można je wykreślić; wysokie znaczy „patrzyłem na to." Recenzenci je uwielbiają.

Nie są tak interpretowalne, jak wyglądają. Jain i Wallace (2019) wykazali, że rozkłady attention można permutować i zastępować dowolnymi alternatywami bez zmiany predykcji modelu dla niektórych zadań. Nigdy nie podawaj wag attention jako dowodu rozumowania bez ablacji lub sprawdzenia kontrfaktycznego.

## Wyślij to

Zapisz jako `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debugowanie błędów kształtów w implementacjach attention.
phase: 5
lesson: 10
---

Mając zepsutą implementację attention, identyfikujesz niezgodność kształtów. Wyjście:

1. Która macierz ma zły kształt. Nazwij tensor.
2. Jaki powinien być jej kształt, wyprowadzony z (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. Jednolinijkowa poprawka. Transponuj, zmień kształt lub zaprojektuj.
4. Test wychwytujący regresje. Zwykle: assert `output.shape == (batch, T_dec, d_h)` oraz `weights.shape == (batch, T_dec, T_enc)` oraz `weights.sum(dim=-1)` bliskie 1.

Odmawiaj polecania poprawek, które cicho broadcastują. Błędy ukryte przez broadcast ujawniają się później jako ciche pogorszenie dokładności, najgorszy rodzaj błędu attention.

Przy niejasnościach Bahdanau nalegaj, że wejście dekodera to `s_{t-1}` (stan przed krokiem). Przy Luong, `s_t` (stan po kroku). Przy dot-product, wskaż niezgodność wymiarów między query a key jako najczęstszy błąd przy pierwszym podejściu.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj maskowanie softmax, aby tokeny paddingu w enkoderze miały wagę attention zero. Testuj na batchu z sekwencjami o różnej długości.
2. **Średnie.** Dodaj multi-head attention do formy general Luong. Podziel `d_h` na `n_heads` grup, uruchom attention per head, konkatenuj. Zweryfikuj, że przypadek single-head odpowiada twojej wcześniejszej implementacji.
3. **Trudne.** Wytrenuj GRU encoder-decoder z Bahdanau attention na zadaniu kopiowania z lekcji 09. Wykreśl dokładność vs długość sekwencji. Porównaj z baseline bez attention. Powinieneś zobaczyć, że luka rośnie wraz z długością, potwierdzając że attention usuwa wąskie gardło.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Attention | Patrzenie na rzeczy | Średnia ważona sekwencji wartości, wagi obliczone z podobieństwa query-key. |
| Query, Key, Value | QKV | Trzy projekcje: Q pyta, K jest tym, co dopasować, V jest tym, co zwrócić. |
| Additive attention | Bahdanau | Wynik via feed-forward: `v^T tanh(W q + U k)`. |
| Multiplicative attention | Luong dot / general | Wynik to `q^T k` lub `q^T W k`. Tańsze, ta sama dokładność na większości zadań. |
| Alignment matrix | Ten ładny obrazek | Wagi attention jako siatka `(T_dec, T_enc)`. Czytaj ją, żeby zobaczyć, na co model patrzył. |

## Dalsze Czytanie

- Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate — oryginalna praca.
- Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation — trzy warianty wyników i ich porównanie.
- Jain and Wallace (2019). Attention is not Explanation — zastrzeżenie interpretowalności.
- Dive into Deep Learning — Bahdanau Attention — uruchamialny przewodnik z PyTorch.