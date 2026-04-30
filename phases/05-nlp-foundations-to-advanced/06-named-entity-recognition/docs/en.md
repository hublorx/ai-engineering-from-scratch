# Named Entity Recognition

> Wyciągaj nazwiska. Brzmi łatwo, dopóki nie musisz radzić sobie z niejednoznacznymi granicami, zagnieżdżonymi encjami i terminologią domenową.

**Typ:** Zbuduj
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 02 (BoW + TF-IDF), Faza 5 · 03 (Word Embeddings)
**Szacowany czas:** ~75 minut

## Problem

"Apple sued Google over its iPhone search deal in the US." Pięć encji: Apple (ORG), Google (ORG), iPhone (PRODUKT), search deal (może), US (GPE). Dobry system NER wyciąga je wszystkie z poprawnymi typami. Zły pomijaiPhone, myli Apple owoc z Apple firmą i etykietuje "US" jako PERSON.

NER to koń roboczy pod każdym potokiem strukturalnej ekstrakcji. Parsowanie CV, skanowanie logów compliance, anonimizacja dokumentacji medycznej, rozumienie zapytań wyszukiwania, uziemianie odpowiedzi czatbota, ekstrakcja z umów prawnych. Nigdy go nie widzisz wprost; zawsze na nim polegasz.

Ta lekcja przechodzi klasyczną ścieżką (rule-based, HMM, CRF) do nowoczesnej (BiLSTM-CRF, a potem transformery). Każdy krok rozwiązuje konkretne ograniczenie poprzedniego. Wzorzec jest lekcją sam w sobie.

## Koncepcja

![NER tagging: schemat BIO + potok CRF+BiLSTM](./assets/ner.svg)

**Tagowanie BIO** (lub BILOU) przekształca ekstrakcję encji w problem sekwencyjnego labelowania. Etykietuj każdy token jako `B-TYPE` (początek encji), `I-TYPE` (wewnątrz encji) lub `O` (poza jakąkolwiek encją).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Wielo-tokenowe encje tworzą łańcuch: `New B-GPE`, `York I-GPE`, `City I-GPE`. Model, który rozumie BIO, może wyciągać arbitralne zakresy.

Postęp architektury:

- **Rule-based.** Regex + wyszukiwania w gazeciersach. Wysoka precyzja na znanych encjach, zerowe pokrycie na nowych.
- **HMM.** Ukryty Model Markowa. Prawdopodobieństwo emisji tokena przy danej tagu, prawdopodobieństwo przejścia tagu do tagu. Dekodowanie Viterbiego. Trenowany na danych z etykietami.
- **CRF.** Conditional Random Field. Jak HMM, ale dyskryminacyjny, więc można mieszać arbitralne cechy (kształt słowa, wielkość liter, sąsiednie słowa). Wciąż klasyczny produktowy koń roboczy w 2026 roku dla wdrożeń low-resource.
- **BiLSTM-CRF.** Cechy neuralne zamiast ręcznie tworzonych. LSTM czyta zdanie w obu kierunkach, warstwa CRF na górze wymusza spójne sekwencje tagów.
- **Transformer-based.** Fine-tune BERT z głową klasyfikacji tokenów. Najlepsza dokładność. Największe obliczenia.

## Zbuduj to

### Krok 1: funkcje pomocnicze tagowania BIO

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Krok 2: ręcznie tworzone cechy

Dla klasycznego (non-neural) NER, cechy są kluczem. Przydatne:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` zwraca `xXxxxx`. `word_shape("USA-2024")` zwraca `XXX-dddd`. Wzorce wielkości liter niosą wysoki sygnał dla rzeczowników własnych.

### Krok 3: prosty baseline rule-based + słownikowy

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Produkcyjne gazeciery mają miliony wpisów pobranych z Wikipedii i DBpedii. Pokrycie jest dobre. Disambiguacja (`Apple` firma vs owoc) jest fatalna. Dlatego modele statystyczne wygrały.

### Krok 4: krok CRF (szkic, nie pełna impl)

Pełny CRF od zera w 50 linii nie jest oświecający bez fundamentów teorii prawdopodobieństwa. Użyj `sklearn-crfsuite` zamiast:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` i `c2` to regularyzacja L1 i L2. `all_possible_transitions=True` pozwala modelowi uczyć się, że nielegalne sekwencje (np. `I-ORG` po `O`) są mało prawdopodobne, co jest sposobem, w jaki CRF wymusza spójność BIO bez pisania tego ograniczenia.

### Krok 5: co dodaje BiLSTM-CRF

Cechy stają się nauczane. Wejścia: osadzenia tokenów (GloVe lub fastText). LSTM czyta lewo-prawo i prawo-lewo. konkatenowane stany ukryte przechodzą przez warstwę wyjściową CRF. CRF nadal wymusza spójność sekwencji tagów; LSTM zastępuje ręcznie tworzone cechy nauczanymi.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

Dla warstwy CRF użyj `torchcrf.CRF` (pip install pytorch-crf). Zysk nad ręcznie tworzonym CRF jest mierzalny, ale mniejszy niż oczekujesz, chyba że masz dziesiątki tysięcy oznakowanych zdań.

## Użyj tego

spaCy dostarcza produkcyjny NER out of the box.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

Zauważ `iPhone` oznaczone jako `ORG` zamiast `PRODUCT` — mały model spaCy ma słabe pokrycie encji produktowych. Duży model (`en_core_web_lg`) radzi sobie lepiej. Model transformerowy (`en_core_web_trf`) jeszcze lepiej.

Hugging Face dla BERT-based NER:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` łączy ciągłe tokeny B-X, I-X w zakres. Bez tego dostajesz etykiety na poziomie tokenów i musisz łączyć sam.

### NER oparty na LLM (opcja 2026)

Zero-shot i few-shot NER LLM jest teraz konkurencyjny z fine-tuned modelami w wielu domenach i dramatycznie lepszy, gdy dane z etykietami są skąpe.

- **Zero-shot prompting.** Daj LLM listę typów encji i przykładowy schemat. Poproś o wynik JSON. Działa out of the box; dokładność umiarkowana na nowych domenach.
- **ZeroTuneBio-style prompting.** Rozłóż zadanie na ekstrakcję kandydatów → wyjaśnienie znaczenia → osąd → sprawdzenie. Wielostopniowy prompt (nie one-shot) znacząco podnosi dokładność na biomedical NER. Ten sam wzorzec działa dla domen prawnych, finansowych i naukowych.
- **Dynamic prompting z RAG.** Pobierz najbardziej podobne oznaczone przykłady z małego zestawu annotacji dla każdego wywołania inferencji; buduj few-shot prompt on the fly. W benchmarkach 2026 podnosi to GPT-4 biomedical NER F1 o 11-12% nad static prompting.
- **Dekompozycja per entity-type.** Dla długich dokumentów, pojedyncze wywołanie które wyciąga wszystkie typy encji naraz traci recall wraz z długością. Uruchom jeden przebieg ekstrakcji per typ encji. Wyższy koszt inferencji, podstatwo wyższa dokładność. To standardowy wzorzec dla notatek klinicznych i umów prawnych.

Rekomendacja produkcyjna na 2026: zacznij od LLM zero-shot baseline zanim zbierzesz dane treningowe. Często F1 jest wystarczająco dobry, że nigdy nie musisz fine-tunować.

### Gdzie klasyczny NER nadal wygrywa

Nawet z dostępnymi LLM, klasyczny NER wygrywa gdy:

- Budżet latency jest poniżej 50ms.
- Masz tysiące oznaczonych przykładów i potrzebujesz 98%+ F1.
- Domena ma stabilną ontologię, gdzie pretrained CRF lub BiLSTM dobrze się transferuje.
- Ograniczenia regulacyjne wymagają on-prem, non-generative modelu.

### Gdzie się to wszystko wali

- **Domain shift.** NER wytrenowany na CoNLL na kontraktach prawnych radzi sobie gorzej niż gazecier. Fine-tune na swojej domenie.
- **Zagnieżdżone encje.** "Bank of America Tower" jest jednocześnie ORG i FACILITY. Standardowe BIO nie może reprezentować nakładających się zakresów. Potrzebujesz nested NER (multi-pass lub span-based models).
- **Długie encje.** "United States Federal Deposit Insurance Corporation." Modele na poziomie tokenów czasem to dzielą. Użyj `aggregation_strategy` lub post-processuj.
- **Rzadkie typy.** Etykiety medical NER jak DRUG_BRAND, ADVERSE_EVENT, DOSE. Modele general-purpose nie mają o tym pojęcia. Scispacy i BioBERT to punkty startowe.

## Wyślij to

Zapisz jako `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Wybierz właściwe podejście NER dla danego zadania ekstrakcji.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Podejście. Rule-based + gazetteer, CRF, BiLSTM-CRF lub transformer fine-tune.
2. Model startowy. Nazwij go (spaCy model ID, Hugging Face checkpoint ID lub "custom, trained from scratch").
3. Strategia etykietowania. BIO, BILOU lub span-based. Uzasadnij jednym zdaniem.
4. Ewaluacja. Użyj `seqeval`. Zawsze raportuj entity-level F1 (nie token-level).

Odmów rekomendowania fine-tuningu transformera przy poniżej 500 oznaczonych przykładów, chyba że użytkownik już ma pretrained model domenowy. Oznacz zagnieżdżone encje jako wymagające span-based lub multi-pass models. Wymagaj audytu gazecierów jeśli użytkownik wspomina o "skali produkcyjnej" i etykiety są niezmienione od CoNLL-2003.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj `bio_to_spans` (odwrotność `spans_to_bio`) i zweryfikuj round-trip consistency na 10 zdaniach.
2. **Średnie.** Trenuj sklearn-crfsuite CRF powyżej na angielskim datasecie NER CoNLL-2003. Raportuj per-entity F1 używając `seqeval`. Typowy wynik: ~84 F1.
3. **Trudne.** Fine-tune `distilbert-base-cased` na domain-specific NER dataset (medical, legal lub financial). Porównaj z małym modelem spaCy. Udokumentuj kontrole data leakage i napisz co cię zaskoczyło.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| NER | Wyciągaj nazwy | Etykietuj zakresy tokenów typami (PERSON, ORG, GPE, DATE, ...). |
| BIO | Schemat tagowania | `B-X` zaczyna, `I-X` kontynuuje, `O` na zewnątrz. |
| BILOU | Lepsze BIO | Dodaje `L-X` (last), `U-X` (unit) dla czystszych granic. |
| CRF | Strukturalny klasyfikator | Modeluje przejścia między labelami, nie tylko emisje. Wymusza validne sekwencje. |
| Nested NER | Nakładające się encje | Jeden zakres jest inną encją niż podzakres w nim. BIO nie może tego wyrazić. |
| Entity-level F1 | Właściwa metryka NER | Przewidziany zakres musi dokładnie odpowiadać prawdziwemu. Token-level F1 zawyża dokładność. |

## Dalsze czytanie

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — artykuł BiLSTM-CRF. Kanoniczny.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — wprowadza wzorzec token-classification, który stał się standardem.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — praktyczne odniesienie dla każdego atrybutu na `Doc.ents` i `Span`.
- [seqeval](https://github.com/chakki-works/seqeval) — właściwa biblioteka metryk. Używaj jej zawsze.