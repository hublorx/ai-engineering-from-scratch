# Modele Sequence-to-Sequence

> Dwa RNN-y udające tłumacza. Wąskie gardło, które napotykają, to powód istnienia attention.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 5 · 08 (CNN + RNN dla tekstu), Phase 3 · 11 (Wprowadzenie do PyTorch)
**Szacowany czas:** ~75 minut

## Problem

Klasyfikacja mapuje sekwencję o zmiennej długości do pojedynczej etykiety. Tłumaczenie mapuje sekwencję o zmiennej długości do innej sekwencji o zmiennej długości. Wejście i wyjście należą do różnych słowników, możliwie różnych języków, bez gwarancji równej długości.

Architektura seq2seq (Sutskever, Vinyals, Le, 2014) rozwiązała ten problem celowo prostym przepisem. Dwa RNN-y. Jeden czyta zdanie źródłowe i produkuje wektor kontekstowy o stałym rozmiarze. Drugi czyta ten wektor i generuje zdanie docelowe token po tokenie. Ten sam kod, który napisałeś w lesson 08, sklejony inaczej.

Warto to studiować z dwóch powodów. Po pierwsze, wąskie gardło wektora kontekstowego to najbardziej pedagogicznie użyteczna porażka w NLP. Motywuje wszystko, czym attention i transformery są dobre. Po drugie, przepis treningowy (teacher forcing, scheduled sampling, beam search podczas wnioskowania) nadal obowiązuje w każdym nowoczesnym systemie generacji, w tym w LLM-ach.

## Koncepcja

![Koder-dekoder z wąskim gardłem wektora kontekstowego](./assets/seq2seq.svg)

**Koder.** RNN, który czyta zdanie źródłowe. Jego ostatni ukryty stan to **wektor kontekstowy** — stały podsumowanie całego wejścia. Nie tracisz niczego, poza źródłem, jakoby.

**Dekoder.** Kolejny RNN zainicjowany z wektora kontekstowego. W każdym kroku bierze wcześniej wygenerowany token jako wejście i produkuje rozkład nad słownikiem docelowym. Próbkuj lub argmax, żeby wybrać następny token. Wsuń go z powrotem. Powtarzaj, aż pojawi się token `<EOS>` lub zostanie osiągnięta maksymalna długość.

**Trening:** Cross-entropy loss w każdym kroku dekodera, zsumowane przez całą sekwencję. Standardowa backpropagation through time przez obie sieci.

**Teacher forcing.** Podczas treningu, wejście dekodera w kroku `t` to *prawdziwy* token na pozycji `t-1`, nie poprzednia predykcja dekodera. To stabilizuje trening; bez tego wczesne błędy kaskadują i model nigdy się nie uczy. Podczas wnioskowania musisz używać własnych predykcji modelu, więc zawsze istnieje różnica rozkładów train/inference. Ta różnica nazywa się **exposure bias**.

**Wąskie gardło.** Wszystko, czego koder się nauczył o źródle, musi być wyciśnięte w jeden wektor kontekstowy. Długie zdania tracą szczegóły. Rzadkie słowy się rozmywają. Zmiana kolejności (chat noir vs. black cat) musi być zapamiętana, nie obliczona.

Attention (lesson 10) to naprawia, pozwalając dekoderowi patrzeć na *każdy* ukryty stan kodera, nie tylko na ostatni. To cała obietnica.

## Zbuduj to

### Krok 1: koder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` ma kształt `[batch, seq_len, hidden_dim]` — jeden ukryty stan na pozycję wejściową. `hidden` ma kształt `[1, batch, hidden_dim]` — ostatni krok. Lesson 08 powiedziała "pool over outputs for classification". Tutaj trzymamy ostatni ukryty stan jako wektor kontekstowy i ignorujemy per-step outputs.

### Krok 2: dekoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Dekoder jest wywoływany jeden krok na raz. Wejście: batch pojedynczych tokenów i aktualny ukryty stan. Wyjście: logity słownikowe dla następnego tokena i zaktualizowany ukryty stan.

### Krok 3: pętla treningowa z teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Dwa pokrętła warte nazwania. `ignore_index=0` pomija loss na tokenach paddingu. `teacher_forcing_ratio` to prawdopodobieństwo użycia prawdziwego tokena vs. predykcji modelu w każdym kroku. Zaczynaj od 1.0 (pełny teacher forcing) i zmniejszaj stopniowo do ~0.5 przez trening, żeby zamknąć lukę exposure-bias.

### Krok 4: pętla wnioskowania (greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding wybiera token o najwyższym prawdopodobieństwie w każdym kroku. Może zbłądzić: raz gdy zobowiążesz się do tokenu, nie możesz go cofnąć. **Beam search** trzyma przy życiu top-`k` częściowych sekwencji i wybiera najwyżej ocenioną kompletną na końcu. Szerokość wiązki 3-5 to standard.

### Krok 5: wąskie gardło, zademonstrowane

Trenuj model na zadaniu kopiowania zabawki: źródło `[a, b, c, d, e]`, cel `[a, b, c, d, e]`. Zwiększaj długość sekwencji. Obserwuj dokładność.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Pojedynczy ukryty stan GRU nie może bezstratnie zapamiętać 40-tokenowego wejścia. Informacja jest tam przy każdym kroku kodera, ale dekoder widzi tylko ostatni stan. Attention to naprawia bezpośrednio.

## Użyj tego

PyTorch ma `nn.Transformer` i szablony seq2seq oparte na `nn.LSTM`. Biblioteka transformers Hugging Face dostarcza pełne modele encoder-decoder (BART, T5, mBART, NLLB) trenowane na miliardach tokenów.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Nowoczesne encoder-decodery porzuciły RNN-y na rzecz transformerów. Wysokopoziomowy kształt (encoder, decoder, generuj-token-po-tokenu) jest identyczny jak w pracy seq2seq z 2014 roku. Mechanizm wewnątrz każdego bloku jest inny.

### Kiedy nadal sięgać po RNN-based seq2seq

Prawie nigdy, dla nowych projektów. Konkretne wyjątki:

- Strumieniowe tłumaczenie, gdzie konsumujesz wejście jeden token na raz z ograniczoną pamięcią.
- Generacja tekstu na urządzeniu, gdzie koszt pamięci transformera jest prohibitive.
- Pedagogika. Zrozumienie wąskiego gardła encoder-decoder to najszybsza ścieżka do zrozumienia, dlaczego transformery wygrały.

### Exposure bias i jego łagodzenia

- **Scheduled sampling.** Zmniejszaj ratio teacher forcing podczas treningu, żeby model uczył się recoverować ze swoich własnych błędów.
- **Minimum risk training.** Trenuj na sentence-level BLEU score zamiast token-level cross-entropy. Bliżej tego, co faktycznie chcesz.
- **Reinforcement learning fine-tuning.** Nagrody dla generatora sekwencji z metryką. Używane we współczesnym LLM RLHF.

Wszystkie trzy nadal obowiązują w generation opartym na transformerach.

## Wyślij to

Zapisz jako `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Zaprojektuj pipeline sequence-to-sequence dla danego zadania.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj zadanie kopiowania zabawki. Trenuj GRU seq2seq na parach wejście-wyjście, gdzie cel równa się źródłu. Zmierz dokładność przy długościach 5, 10, 20. Odtwórz wąskie gardło.
2. **Średnie.** Dodaj beam search decoding z szerokością wiązki 3. Zmierz BLEU na małym korpusie równoległym wobec greedy. Udokumentuj, gdzie beam search wygrywa (zwykle ostatnie tokeny) i gdzie nie robi różnicy.
3. **Trudne.** Dostrój `facebook/bart-base` na datasetcie 10k par paraphrase. Porównaj wyjście modelu dostrojonego z beam-4 do wyjścia modelu bazowego na held-out inputs. Raportuj BLEU i wybierz 10 jakościowych przykładów.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Encoder | Input RNN | Czyta źródło. Produkuje per-step hidden states i final context vector. |
| Decoder | Output RNN | Zainicjowany z context vector. Generuje target tokens jeden na raz. |
| Context vector | Podsumowanie | Final encoder hidden state. Stały rozmiar. Wąskie gardło, które attention rozwiązuje. |
| Teacher forcing | Używaj prawdziwych tokenów | Podawaj ground-truth poprzedni token w czasie treningu. Stabilizuje uczenie. |
| Exposure bias | Luka train/test | Model treningowany na prawdziwych tokenach nigdy nie ćwiczył recoverowania ze swoich własnych błędów. |
| Beam search | Lepsze dekodowanie | Trzymaj top-k częściowych sekwencji przy życiu w każdym kroku zamiast commitować zachłannie. |

## Dalsze czytanie

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — oryginalna praca seq2seq. Cztery strony.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — wprowadziło GRU i framing encoder-decoder.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — praca o attention. Przeczytaj natychmiast po tej lekcji.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — budowalny kod seq2seq + attention.