# Tagowanie POS i analizowanie składniowe

> Gramatyka była niemodna przez jakiś czas. Potem każdy pipeline LLM potrzebował walidacji strukturalnej ekstrakcji i wróciła.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Szacowany czas:** ~45 minut

## Problem

Lekcja 01 obiecywała, że lematyzacja potrzebuje tagu części mowy. Bez wiedzy, że `running` jest czasownikiem, lematyzator nie może zredukować go do `run`. Bez wiedzy, że `better` jest przymiotnikiem, nie może zredukować do `good`.

To obietnica ukrywała całą subdyscyplinę. Tagowanie części mowy (POS tagging) przypisuje kategorie gramatyczne. Analiza składniowa (syntactic parsing) odzyskuje drzewiastą strukturę zdania: które słowo modyfikuje które, który czasownik rządzi którymi argumentami. Klasyczne NLP spędziło dwadzieścia lat nad doskonaleniem obu. Potem deep learning zredukował je do zadania klasyfikacji tokenów na szczycie pretrained transformera, a społeczność badawcza poszła dalej.

Nie społeczność stosowana. Każdy pipeline strukturalnej ekstrakcji nadal używa tagów POS i drzew zależności pod maską. LLM-generowany JSON jest walidowany względem ograniczeń gramatycznych. Systemy odpowiadania na pytania rozkładają zapytania używając analiz zależności. Oceniające jakość tłumaczenia maszynowego sprawdzają wyrównanie drzew analizy.

Warto wiedzieć. Ta lekcja wprowadza tagiesty, baseline'y i punkt, w którym przestajesz implementować od zera i wywołujesz spaCy.

## Koncepcja

![Przykład tagu POS i analizy zależności](./assets/pos-parse.svg)

**Tagowanie POS** etykietuje każdy token kategorią gramatyczną. Tagesta **Penn Treebank (PTB)** to domyślna angielska. 36 tagów z rozróżnieniami, które laik uznaje za pedantyczne: `NN` rzeczownik singular, `NNS` rzeczownik plural, `NNP` rzeczownik własny singular, `VBD` czasownik czas przeszły, `VBZ` czasownik 3. osoba singular prezent, i tak dalej. Tagesta **Universal Dependencies (UD)** jest grubsza (17 tagów) i niezależna od języka; stała się domyślna dla pracy wielojęzycznej.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Analiza składniowa** produkuje drzewo. Dwa główne style:

- **Analiza constituencujna.** Frazy rzeczownikowe, frazy czasownikowe, frazy przyimkowe zagnieżdżają się w sobie. Wynik to drzewo kategorii nie-terminalnych (NP, VP, PP) ze słowami jako liśćmi.
- **Analiza zależnościowa.** Każde słowo ma jedno słowo-nadrzędne, od którego zależy, oznaczone relacją gramatyczną. Wynik to drzewo, gdzie każda krawędź jest trójką (head, dependent, relation).

Analiza zależnościowa wygrała w latach 2010, ponieważ generalizuje się czysto między językami, szczególnie tymi z dowolnym porządkiem słów.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Zbuduj to

### Krok 1: baseline najczęstszego taga

Najgłupszy tagger POS, który działa. Dla każdego słowa przewiduj tag, który miało najczęściej podczas treningu.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Na korpusie Brown ten baseline osiąga ~85% dokładności. Nie dobrze, ale podłoga, poniżej której żaden poważny model nie powinien spaść.

### Krok 2: bigramowy tagger HMM

Modeluj prawdopodobieństwo łączne sekwencji:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Dwie tablice: prawdopodobieństwa przejścia (tag przy danym poprzednim tagu), prawdopodobieństwa emisji (słowo przy danym tagu). Estymuj oba ze zliczeń z wygładzaniem Laplace'a. Dekoduj z Viterbi (programowanie dynamiczne nad kratą tagów).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM na Brown osiąga ~93% dokładności. Skok z 85% na 93% to głównie prawdopodobieństwa przejścia — model uczy się, że `DET NOUN` jest częste, a `NOUN DET` rzadkie.

### Krok 3: dlaczego nowoczesne taggery to biją

Prawdopodobieństwa przejścia i emisji są lokalne. Nie mogą uchwycić, że `saw` jest rzeczownikiem w "I bought a saw", ale czasownikiem w "I saw the movie." CRF z dowolnymi cechami (suffix, kształt słowa, słowo przed i po, samo słowo) osiąga ~97%. BiLSTM-CRF lub transformer osiąga ~98%+.

Pułap tego zadania jest ustawiony przez niezgodę annotatorów. Ludzcy annotatorzy zgadzają się w ~97% przypadków na Penn Treebank. Modele powyżej 98% prawdopodobnie przeuczają się na zbiorze testowym.

### Krok 4: zarys analizy zależności

Pełna analiza zależności od zera jest poza zakresem; kanoniczna behandling w podręczniku jest w Jurafsky i Martin. Dwie klasyczne rodziny, które warto znać:

- **Parsery oparte na przejściach** (arc-eager, arc-standard) działają jak parser shift-reduce: czytają tokeny, przesuwają je na stos i stosują akcje redukcji, które tworzą łuki. Zachłanne dekodowanie jest szybkie. Klasyczna implementacja to MaltParser. Nowoczesna wersja neuronowa: parser oparty na przejściach Chen i Manning.
- **Parsery oparte na grafach** (algorytm Eisnera, biaffine Dozat-Manning) oceniają każdą możliwą krawędź head-dependent i wybierają maximum spanning tree. Wolniejsze, ale dokładniejsze.

Dla większości pracy stosowanej, wywołaj spaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Czytaj kolumnę `dep` od dołu do góry, a struktura gramatyczna zdania się uwypukli.

## Użyj tego

Każda produkcyjna biblioteka NLP wysyła tagery POS i parsery zależności jako część standardowego pipeline.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Szybkie, dokładne, zintegrowane z tokenizacją + NER + lematyzacją. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (relacja zależności).
- **Stanford NLP (stanza)**. Następca Stanford CoreNLP. Stan techniki na 60+ językach.
- **trankit**. Oparte na transformerze, dobra dokładność UD.
- **NLTK**. `pos_tag`. Używalne, wolne, starsze. Dobre do nauczania.

### Gdzie to nadal ma znaczenie w 2026

- **Lematyzacja.** Lekcja 01 potrzebuje POS do poprawnej lematyzacji. Zawsze.
- **Strukturalna ekstrakcja z wyjść LLM.** Waliduj, że wygenerowane zdanie respektuje ograniczenia gramatyczne (np. zgodność podmiotu z orzeczeniem, wymagane modyfikatory).
- **Sentiment oparty na aspektach.** Analizy zależności mówią ci, który przymiotnik modyfikuje który rzeczownik.
- **Rozumienie zapytań.** "movies directed by Wes Anderson starring Bill Murray" rozkłada się na strukturalne ograniczenia przez parse.
- **Transfer wielojęzyczny.** Tagi UD i relacje zależności są niezależne od języka, umożliwiając zero-shot strukturalną analizę nowych języków.
- **Pipeline niskich obliczeń.** Jeśli nie możesz dostarczyć transformera, POS + parse zależności + gazetteer daje ci zaskakująco daleko.

## Wyślij to

Zapisz jako `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Zaprojektuj klasyczny pipeline POS + zależności dla downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Ćwiczenia

1. **Łatwe.** Używając baseline najczęstszego taga na małym korpusie z tagami (np. podzbiór Browna z NLTK), zmierz dokładność na zdaniach hold-out. Zweryfikuj wynik ~85%.
2. **Średnie.** Wytrenuj powyższy bigramowy HMM i raportuj precision/recall per tag. Które tagi HMM myli najbardziej?
3. **Trudne.** Użyj analizy zależności spaCy do ekstrakcji trójek subject-verb-object ze 1000 zdaniowej próbki. Ewaluuj na 50 ręcznie oznakowanych trójkach. Udokumentuj, gdzie ekstrakcja zawodzi (często passiwa, koordynacje i eliptyczne podmioty).

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| POS tag | Typ słowa | Kategoria gramatyczna. PTB ma 36; UD ma 17. |
| Penn Treebank | Standardowy tagset | Specyficzny dla angielskiego. Drobnostkowe czasy czasowników i liczba rzeczowników. |
| Universal Dependencies | Wielojęzyczny tagset | Grubszy niż PTB; neutralny językowo; domyślny dla pracy międzyjęzykowej. |
| Dependency parse | Drzewo zdania | Każde słowo ma jedno head; każda krawędź ma relację gramatyczną. |
| Viterbi | Programowanie dynamiczne | Znajduje sekwencję tagów o najwyższym prawdopodobieństwie przy danych emisjach i przejściach. |

## Dalsze czytanie

- [Jurafsky i Martin — Speech and Language Processing, rozdziały 8 i 18](https://web.stanford.edu/~jurafsky/slp3/) — kanoniczna behandling podręcznikowa POS i parsowania.
- [Universal Dependencies project](https://universaldependencies.org/) — międzyjęzykowy tagset i kolekcja treebank używane przez każdy wielojęzyczny parser.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — praktyczne odniesienie dla każdego atrybutu eksponowanego na `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — paper który wprowadził neural parsery do mainstreamu.