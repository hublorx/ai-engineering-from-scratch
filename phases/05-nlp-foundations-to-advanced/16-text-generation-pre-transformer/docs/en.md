# Generowanie tekstu przed transformerami — modele językowe N-gram

> Jeśli słowo jest zaskakujące, model jest zły. Perplexity zamienia zaskoczenie w liczbę. Wygładzanie utrzymuje ją w określonych granicach.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 5 · 01 (Przetwarzanie tekstu), Phase 2 · 14 (Naiwny Bayes)
**Szacowany czas:** ~45 minut

## Problem

Zanim pojawiły się transformery, zanim pojawiły się RNN-y, zanim pojawiły się osadzania słów, model językowy przewidywał następne słowo poprzez zliczanie, jak często występowało po poprzednich `n-1` słowach. Zlicz "the cat" → "sat" 47 razy, "the cat" → "jumped" 12 razy, "the cat" → "refrigerator" 0 razy. Normalizuj, aby uzyskać rozkład prawdopodobieństwa.

To jest model językowy n-gram. Działał w każdym rozpoznawaczu mowy, każdym sprawdzarce pisowni i każdym systemie tłumaczenia maszynowego opartym na frazach od 1980 do 2015 roku. Wciąż działa, gdy potrzebujesz taniego modelowania językowego na urządzeniu.

Interesujący problem dotyczy tego, co zrobić z niespotykanymi n-gramami. Surowy model oparty na zliczaniu przypisuje zerowe prawdopodobieństwo всему, czego nie widział, co jest katastrofalne, bo zdania są długie i prawie każde długie zdanie zawiera przynajmniej jeden niespotkany ciąg. Pięćdziesiąt lat badań nad wygładzaniem to naprawiło. Kneser-Ney smoothing jest tego wynikiem, a nowoczesne głębokie uczenie odziedziczyło po nim tradycję empiryczną.

## Koncepcja

![Model n-gram: zlicz, wygładzaj, generuj](../assets/ngram.svg)

**Prawdopodobieństwo n-gram:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Ustalamy `n` (zazwyczaj 3 dla trigramów, 4 dla 4-gramów). Obliczamy ze zliczeń:

```text
P(w | kontekst) = count(kontekst, w) / count(kontekst)
```

**Problem zerowych zliczeń.** Każdy n-gram niespotkany w treningu otrzymuje zerowe prawdopodobieństwo. Badanie z 2007 roku na korpusie Browna wykazało, że nawet model 4-gramowy miał 30% 4-gramów ze zbioru testowego niespotkanych w treningu. Nie można ocenić na żadnym rzeczywistym tekście bez wygładzania.

**Podejścia do wygładzania, w kolejności złożoności:**

1. **Laplace (add-one).** Dodaj 1 do każdego zliczenia. Proste, fatalne dla rzadkich zdarzeń.
2. **Good-Turing.** Przydziel ponownie masę prawdopodobieństwa od zdarzeń o wyższej częstotliwości do niespotykanych na podstawie częstotliwości częstotliwości.
3. **Interpolacja.** Połącz oszacowania n-gram, (n-1)-gram itd. z regulowanymi wagami.
4. **Backoff.** Jeśli n-gram ma zero zliczeń, wróć do (n-1)-gramu. Katz backoff to normalizuje.
5. **Absolute discounting.** Odejmij stałą wartość dyskontową `D` od wszystkich zliczeń, rozdystrybuuj do niespotkanych.
6. **Kneser-Ney.** Absolute discounting plus przemyślany wybór dla modelu niższego rzędu: użyj *continuation probability* (w ilu kontekstach słowo się pojawia) zamiast surowej częstotliwości.

Odkrycie Kneser-Ney jest głębokie. "San Francisco" to częsty bigram. Unigram "Francisco" pojawia się głównie po "San." Naiwne absolute discounting nadaje "Francisco" wysokie prawdopodobieństwo unigramowe (bo zliczenie jest wysokie). Kneser-Ney zauważa, że "Francisco" pojawia się tylko w jednym kontekście i obniża jego continuation probability odpowiednio. Rezultat: nowy bigram kończący się na "Francisco" otrzymuje odpowiednio niskie prawdopodobieństwo.

**Ocena: perplexity.** Wykładnik średniej ujemnej log-wiarygodności na słowo na hold-out test set. Niższy jest lepszy. Perplexity 100 oznacza, że model jest tak zdezorientowany, jakby wybierał równomiernie spośród 100 słów.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | kontekst_i))
```

## Zbuduj to

### Krok 1: zliczenia trigramów

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Dane wejściowe to lista tokenizowanych zdań. Wynik to zliczenia n-gramów i kontekstów. `<s>` i `</s>` to granice zdań.

### Krok 2: Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Dodaj 1 do każdego zliczenia. Wygładza, ale przydziela za dużo masy niespotkanym zdarzeniom, szkodząc też rzadkim znanym zdarzeniom.

### Krok 3: Kneser-Ney (bigram, interpolated)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Trzy działające części. `continuation_prob` przechwytuje "w ilu różnych kontekstach pojawia się to słowo?" (innowacja Kneser-Ney). `lambda_prev` to masa uwolniona przez discount, używana do ważenia backoffu. Końcowe prawdopodobieństwo to zdyskontowany główny składnik plus ważony składnik continuation.

### Krok 4: generowanie tekstu z próbkowaniem

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Próbkowanie proporcjonalne do prawdopodobieństwa. Zawsze daje inny wynik na seed. Dla outputu podobnego do beam-search, wybierz argmax w każdym kroku (greedy) i dodaj mały pokrętło losowości (temperature).

### Krok 5: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Niższy jest lepszy. Dla korpusu Browna, dobrze dostrojony model 4-gram KN osiąga perplexity około 140. Transformer LM osiąga 15-30 na tym samym zestawie testowym. Przepaść to około 10x. Ta przepaść jest powodem, dla którego dziedzina poszła dalej.

## Użyj tego

- **Klasyczne nauczanie NLP.** Najjaśniejsza ekspozycja na wygładzanie, MLE i perplexity, jaką możesz uzyskać.
- **KenLM.** Produkcyjna biblioteka n-gram. Używana jako rescorer w systemach mowy i MT, gdzie liczy się niskie opóźnienie.
- **Autocomplete na urządzeniu.** Trigram models w klawiaturach. Wciąż.
- **Baseline'y.** Zawsze obliczaj perplexity n-gram LM przed ogłoszeniem, że twój neural LM jest dobry. Jeśli twój transformer nie bije KN o szeroki margines, coś jest nie tak.

## Wyślij to

Zapisz jako `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Ćwiczenia

1. **Łatwe.** Wytrenuj trigram LM na korpusie Shakespeare'a z 1 000 zdań. Wygeneruj 20 zdań. Będą lokalnie prawdopodobne, ale globalnie niespójne. To jest kanoniczna demo.
2. **Średnie.** Zaimplementuj perplexity dla swojego modelu KN na hold-out podzbiorze Shakespeare'a. Porównaj z Laplace. Powinieneś zobaczyć KN obniżający perplexity o 30-50%.
3. **Trudne.** Zbuduj trigram spell corrector: przy danym źle napisanym słowie i jego kontekście, generuj poprawki i rankuj je przez prawdopodobieństwo kontekstowe pod LM. Ewaluuj na korpusie pisowni Birkbeck (public).

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| N-gram | Ciąg słów | Sekwencja `n` kolejnych tokenów. |
| Smoothing | Unikanie zer | Ponowne przydzielanie masy prawdopodobieństwa, aby niespotkane zdarzenia otrzymały niezerowe prawdopodobieństwo. |
| Perplexity | Metryka jakości LM | `exp(-średnia log-prob)` na danych hold-out. Niższy jest lepszy. |
| Backoff | Wróć do krótszego kontekstu | Jeśli trigram count to zero, użyj bigramu. Katz backoff to formalizuje. |
| Kneser-Ney | Najlepsze wygładzanie dla n-gramów | Absolute discounting + continuation probability dla modelu niższego rzędu. |
| Continuation probability | Specyficzne dla KN | `P(w)` ważone liczbą kontekstów, w których `w` się pojawia, nie surowym zliczeniem. |

## Dalsze czytanie

- [Jurafsky and Martin — Speech and Language Processing, Rozdział 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — kanoniczne omówienie LM n-gram i wygładzania.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — artykuł, który ustalił Kneser-Ney jako najlepszy smoother n-gram.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — oryginalny artykuł KN.
- [KenLM](https://kheafield.com/code/kenlm/) — szybki produkcyjny n-gram LM, wciąż używany w 2026 dla aplikacji wrażliwych na opóźnienia.