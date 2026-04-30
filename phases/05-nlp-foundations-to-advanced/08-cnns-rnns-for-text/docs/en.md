# CNNy i RNN-y dla tekstu

> Konwolucje uczą się n-gramów. Rekurencje pamiętają. Oba zostały zastąpione przez atencję. Oba nadal mają znaczenie na ograniczonym sprzęcie.

**Typ:** Buduj
**Języki:** Python
**Wymagania wstępne:** Faza 3 · 11 (Wprowadzenie do PyTorch), Faza 5 · 03 (Osadzenia słów), Faza 4 · 02 (Konwolucje od zera)
**Czas:** około 75 minut

## Problem

TF-IDF i Word2Vec produkowały płaskie wektory, które ignorowały kolejność słów. Klasyfikator zbudowany na nich nie potrafił rozróżnić `dog bites man` od `man bites dog`. Kolejność słów czasami niesie sygnał.

Dwie rodziny architektur wypełniły tę lukę, zanim pojawiły się transformery.

**Splotowe sieci dla tekstu (TextCNN).** Stosuj sploty 1D nad sekwencjami osadzeń słów. Filtr o szerokości 3 to wyuczalny detektor trigramów: obejmuje trzy słowa i emituje wynik. Ułóż różne szerokości (2, 3, 4, 5), aby wykrywać wzorce na wielu skalach. Max-pool do reprezentacji o stałym rozmiarze. Płaskie, równoległe, szybkie.

**Rekurencyjne sieci (RNN, LSTM, GRU).** Przetwarzają tokeny jeden po drugim, utrzymując stan ukryty, który niesie informację do przodu. Sekwencyjne, z pamięcią, elastyczne długości wejściowe. Dominowały modelowanie sekwencji od 2014 do 2017, potem pojawiła się atencja.

Ta lekcja buduje obie, potem nazywa porażkę, która zmotywowala atencję.

## Koncepcja

![Filtry TextCNN vs. rozwinięcie stanu ukrytego RNN](./assets/cnn-rnn.svg)

**TextCNN** (Kim, 2014). Tokeny są osadzane. Splot 1D o szerokości `k` przesuwa filtr nad kolejnymi k-gramami osadzeń, produkując mapę cech. Global max-pooling nad tą mapą wybiera najsilniejszą aktywację. Połącz wyjścia max-pool z kilku szerokości filtrów. Przekaż do głowy klasyfikatora.

Dlaczego to działa. Filtr to wyuczalny n-gram. Max-pooling jest niezmienniczy pozycyjnie, więc "not good" wyzwala tę samą cechę na początku lub w środku recenzji. Trzy szerokości filtrów ze 100 filtrami każdy daje 300 wyuczonych detektorów n-gramów. Uczenie jest równoległe; brak zależności sekwencyjnej.

**RNN.** W każdym kroku czasowym `t`, stan ukryty `h_t = f(W * x_t + U * h_{t-1} + b)`. Współdziel `W`, `U`, `b` w czasie. Stan ukryty w czasie `T` to podsumowanie całego prefiksu. Dla klasyfikacji, agreguj跨越 `h_1 ... h_T` (max, mean, lub last).

Plain RNN-y cierpią z powodu zanikających gradientów. **LSTM** dodaje bramki, które decydują co zapomnieć, co przechować i co wyemitować, stabilizując gradienty przez długie sekwencje. **GRU** upraszcza LSTM do dwóch bramek; osiąga podobne wyniki z mniejszą liczbą parametrów.

**Dwukierunkowe RNN-y** uruchamiają jeden RNN do przodu i drugi do tyłu, łącząc stany ukryte. Reprezentacja każdego tokena widzi zarówno lewy, jak i prawy kontekst. Niezbędne dla zadań tagowania.

## Zbuduj to

### Krok 1: TextCNN w PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

Transpozycja `transpose(1, 2)` przekształca `[batch, seq_len, embed_dim]` w `[batch, embed_dim, seq_len]`, bo `nn.Conv1d` traktuje środkową oś jako kanały. Wynik pooled ma stały rozmiar niezależnie od długości wejścia.

### Krok 2: Klasyfikator LSTM

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool nad sekwencją, nie ostatni stan. Dla klasyfikacji, max-pooling zwykle bije branie ostatniego stanu ukrytego, bo informacja na końcu długiej sekwencji dominuje ostatni stan.

### Krok 3: demo zanikającego gradientu (intuicja)

Plain RNN bez bramek nie może nauczyć się dalekosiężnych zależności. Rozważ zadanie: przewiduj, czy token `A` pojawił się gdziekolwiek w sekwencji. Jeśli `A` jest na pozycji 1, a sekwencja ma 100 tokenów, gradient z loss musi przepłynąć wstecz przez 99 mnożeń rekurencyjnej wagi. Jeśli waga jest mniejsza niż 1, gradient znika. Jeśli większa niż 1, eksploduje.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# Przy wadze=0.9 przez 100 kroków:
#   0.9 ^ 100 ≈ 2.7e-5
# Gradient od kroku 100 do kroku 1 jest efektywnie zero.
```

LSTM-y to naprawiają za pomocą **stanu komórki**, który biegnie przez sieć tylko z interakcjami addytywnymi (bramka zapomnienia skaluje go mnożeniowo, ale gradienty nadal płyną przez "autostradę"). GRU robią coś podobnego z mniejszą liczbą parametrów. Oba dają stabilne uczenie przez sekwencje 100+ kroków.

### Krok 4: dlaczego to nadal nie wystarczyło

Trzy problemy persistowały nawet z LSTM.

1. **Wąskie gardło sekwencyjne.** Uczenie RNN na sekwencji długości 1000 wymaga 1000 seryjnych kroków forward/backward. Nie można zrównoleglić w czasie.
2. **Stały wektor kontekstowy w ustawieniach encoder-decoder.** Decoder widzi tylko finalny stan ukryty encodera, skompresowany nad całym wejściem. Długie wejścia tracą detale. Lekcja 09 omawia to bezpośrednio.
3. **Sufit dokładności dla odległych zależności.** LSTM-y przewyższają plain RNN-y, ale nadal walczą z propagacją specyficznych informacji przez 200+ kroków.

Atencja rozwiązała wszystkie trzy. Transformery porzuciły rekurencję całkowicie. Lekcja 10 to punkt zwrotny.

## Użyj tego

`nn.LSTM`, `nn.GRU` i `nn.Conv1d` z PyTorch są gotowe do produkcji. Kod uczenia jest standardowy.

Hugging Face dostarcza pretrained embeddings, które podłączasz jako warstwę wejściową:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Lista kontrolna: użyj, gdy pasuje do ograniczeń.

- **Edge / inference na urządzeniu.** TextCNN z osadzeniami GloVe jest 10-100x mniejszy niż transformer. Jeśli cel wdrożenia to telefon, to jest ten stack.
- **Strumieniowanie / klasyfikacja online.** RNN przetwarza jeden token na raz; transformery potrzebują pełnej sekwencji. Dla tekstu przychodzącego w czasie rzeczywistym, LSTM-y nadal wygrywają.
- **Male modele dla baseline'ów.** Szybka iteracja na nowym zadaniu. Naucz TextCNN w 5 minut na CPU.
- **Tagowanie sekwencji z ograniczonymi danymi.** BiLSTM-CRF (lekcja 06) to nadal produkcyjna architektura NER dla 1k-10k oznakowanych zdań.

Wszystko inne idzie do transformera.

## Wyślij to

Zapisz jako `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Ćwiczenia

1. **Łatwe.** Naucz TextCNN na 3-klasowym zbiorze (wymyśl dane). Sprawdź, czy filtry o szerokościach (2, 3, 4) przewyższają pojedynczą szerokość (3) w średnim F1.
2. **Średnie.** Zaimplementuj max-pool, mean-pool i last-state pooling dla klasyfikatora LSTM. Porównaj na małym zbiorze; udokumentuj, który pooling wygrywa i zaproponuj hipotezę dlaczego.
3. **Trudne.** Zbuduj tagger NER BiLSTM-CRF (połącz lekcję 06 i tę). Naucz na CoNLL-2003. Porównaj do baseline'u CRF z lekcji 06 i do BERT fine-tune. Raportuj czas uczenia, pamięć i F1.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| TextCNN | CNN for text | Stos splotów 1D na osadzeniach słów z globalnym max-poolingiem. Kim (2014). |
| RNN | Recurrent net | Stan ukryty aktualizowany w każdym kroku czasowym: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | Gated RNN | Dodaje bramki wejścia / zapomnienia / wyjścia + stan komórki. Uczy się stabilnie przez długie sekwencje. |
| GRU | Simpler LSTM | Dwie bramki zamiast trzech. Podobna dokładność, mniej parametrów. |
| Bidirectional | Oba kierunki | Przód + tył RNN połączone. Każdy token widzi obie strony kontekstu. |
| Vanishing gradient | Sygnał uczenia się umiera | Powtarzane mnożenie przez wagi <1 w plain RNN sprawia, że gradienty wczesnych kroków są efektywnie zero. |