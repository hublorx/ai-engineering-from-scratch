```markdown
# Przetwarzanie tekstu — Tokenizacja, Stemming, Lematyzacja

> Język jest ciągły. Modele są dyskretne. Przetwarzanie wstępne jest mostem.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 2 · 14 (Naive Bayes)
**Szacowany czas:** ~45 minut

## Problem

Model nie może przeczytać "The cats were running." Przeczytuje liczby całkowite.

Każdy system NLP zaczyna się od trzech pytań. Gdzie słowo się zaczyna. Jaki jest korzeń słowa. Jak traktować "run", "running", "ran" jako to samo, gdy to pomaga, i jako różne rzeczy, gdy to nie pomaga.

Jeśli tokenizer potraktuje `don't` jako jeden token, ale `do n't` jako dwa, dystrybucja treningowa się rozszczepia. Jeśli stemmer zredukuje `organization` i `organ` do tego samego stemu, topic modeling umiera. Jeśli lemmatizer potrzebuje kontekstu części mowy, ale nie otrzymuje go, czasowniki są traktowane jako rzeczowniki.

Ta lekcja buduje trzy prymitywy przetwarzania wstępnego od zera, następnie pokazuje jak NLTK i spaCy wykonują tę samą pracę, żebyś mógł zobaczyć kompromisy.

## Koncepcja

Trzy operacje. Każda ma swoje zadanie i tryb awarii.

![Potok przetwarzania wstępnego: surowy tekst → tokeny → stemy lub lemmy → model](./assets/pipeline.svg)

**Tokenizacja** dzieli string na tokeny. "Token" jest celowo niejasne, bo właściwa granularność zależy od zadania. Poziom słów dla klasycznego NLP, podsłowo dla transformerów, znak dla języków bez spacji.

**Stemming** obcina sufiksy za pomocą reguł. Szybki, agresywny, głupi — `running -> run`, `organization -> organ`. Drugi przypadek — to tryb awarii.

**Lematyzacja** redukuje słowo do jego formy słownikowej używając wiedzy gramatycznej. Wolniejsza, dokładna, potrzebuje tabeli odnośników lub analizatora morfologicznego. `ran -> run` (musi wiedzieć, że "ran" to czas przeszły "run"). `better -> good` (musi znać formy stopniowania).

Zasada kciuka. Steming gdy liczy się szybkość i możesz tolerować szum (indeksowanie wyszukiwania, zgrubna klasyfikacja). Lematyzuj gdy liczy się znaczenie (odpowiadanie na pytania, wyszukiwanie semantyczne, wszystko co użytkownik będzie czytać).

## Zbuduj to

### Krok 1: regex word tokenizer

Najprostszy użyteczny tokenizer dzieli na podstawie znaków niealfanumerycznych, zachowując interpunkcję jako własne tokeny. Nie idealny, nie ostateczny ale działa w jednej linii.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Trzy wzorce w kolejności priorytetu. Słowa z opcjonalnym wewnętrznym apostrofem (`don't`, `it's`). Same liczby. Każdy pojedynczy znak niebędący białym znakiem, i niealfanumeryczny jako samodzielny token (interpunkcja).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Tryby awarii, na które warto zwrócić uwagę. `3pm` dzieli się na `['3', 'pm']` bo przełączamy się między ciągami liter a ciągami cyfr. Wystarczająco dobre dla większości zadań. URL, e-maile, hashtagi wszystkie się psują. Dla produkcji, dodaj wzorce przed ogólnymi.

### Krok 2: Porter stemmer (tylko krok 1a)

Pełny algorytm Portera ma pięć faz reguł. Sam krok 1a obejmuje najczęstsze angielskie sufiksy i uczy wzorca.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Czytaj reguły od góry na dół. Reguła `ies -> i` wyjaśnia dlaczego `ponies -> poni`, a nie `pony`. Prawdziwy Porter ma krok 1b, który by to naprawił. Reguły konkurują. Wcześniejsze reguły wygrywają. Kolejność ma większe znaczenie niż jakakolwiek pojedyncza reguła.

### Krok 3: lookup-based lemmatizer

Właściwa lematyzacja wymaga morfologii. Nadająca się do nauczania wersja używa małej tabeli lemmatów i fallbacka.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

Ostatni przypadek to kluczowy moment nauczania. `watched` nie jest w naszej tabeli, a nasz fallback obsługuje tylko `ing`. Prawdziwa lematyzacja obejmuje `ed`, czasowniki nieregularne, przymiotniki w stopniu wyższym, liczbę mnogą ze zmianami dźwiękowymi (`children -> child`). Dlatego produkcyjne systemy używają WordNet, morfologizera spaCy, lub pełnego analizatora morfologicznego.

### Krok 4: łącz je w potok

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

Brakującym elementem jest POS tagger. Phase 5 · 07 (POS Tagging) buduje jeden. Na razie domyślnie wszystko ustawiaj na `NOUN` i przyznaj ograniczenie.

## Użyj tego

NLTK i spaCy dostarczają wersje produkcyjne. Po kilka linii każda.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` obsługuje kontrakcje, Unicode, przypadki brzegowe, które Twój regex pomija. `PorterStemmer` uruchamia wszystkie pięć faz. `WordNetLemmatizer` potrzebuje taga POS przetłumaczonego ze schematu Penn Treebank NLTK na skrótowy zestaw WordNet. Powyższe okablowanie translacji to bit, który większość tutoriali pomija.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy ukrywa cały potok za `nlp(text)`. Tokenizacja, POS tagging i lematyzacja wszystkie działają. Szybsze niż NLTK na skali. Dokładniejsze bez dodatkowej konfiguracji. Kompromis polega na tym, że nie możesz łatwo wymieniać pojedynczych komponentów.

### Kiedy wybrać co

| Sytuacja | Wybierz |
|----------|---------|
| Nauczanie, badania, wymiana komponentów | NLTK |
| Produkcja, wielojęzyczność, szybkość ma znaczenie | spaCy |
| Transformer pipeline (i tak tokenizujesz tokenizerem modelu) | Użyj `tokenizers` / `transformers`, i pomiń klasyczny preprocessing |

### Dwa tryby awarii, o których nikt nie ostrzega

Większość tutoriali uczy algorytmów i kończy. Dwie rzeczy ugryzą prawdziwy potok przetwarzania wstępnego, a prawie nigdy nie są omawiane.

**Reproducibility drift.** NLTK i spaCy zmieniają zachowanie tokenizacji i lemmatyzera między wersjami. To, co produkowało `['do', "n't"]` w spaCy 2.x, może produkować `["don't"]` w 3.x. Twój model był trenowany na jednej dystrybucji. Inferencja teraz działa na innej, dokładność cicho się pogarsza i nikt nie wie, dlaczego. Przypnij wersje bibliotek w `requirements.txt`. Napisz regresyjny test preprocessingu, który zamraża oczekiwaną tokenizację 20 przykładowych zdań. Uruchamiaj go przy każdej aktualizacji.

**Training / inference mismatch.** Trenuj z agresywnym przetwarzaniem wstępnym (małe litery, usuwanie słów stop, stemming), wdrażaj na surowym inputcie użytkownika, patrz, jak performance spada. To jest najczęstszy produkcyjny błąd NLP. Jeśli przetwarzasz wstępnie podczas treningu, musisz uruchomić identyczną funkcję podczas inferencji. Wysyłaj preprocessing jako funkcję wewnątrz pakietu modelu, nie jako komórkę notebooka, którą zespół servingowy przepisze.

## Wyślij to

Wielokrotnie użyteczny prompt, który pomaga inżynierom wybrać strategię przetwarzania wstępnego bez czytania trzech podręczników.

Zapisz jako `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Ćwiczenia

1. **Łatwe.** Rozszerz `tokenize` żeby zachować URL jako pojedynczy token. Test: `tokenize("Visit https://example.com today.")` powinno produkować jeden token URL.
2. **Średnie.** Zaimplementuj Porter step 1b. Jeśli słowo zawiera samogłoskę i kończy się na `ed` lub `ing`, usuń to. Obsłuż regułę podwójnej spółgłoski (`hopping -> hop`, nie `hopp`).
3. **Trudne.** Zbuduj lemmatizer, który używa WordNet jako tabeli odnośników, ale wraca do Twojego Portera gdy WordNet nie ma wpisu. Zmierz dokładność na otagowanym korpusie wobec czystego WordNet i czystego Portera.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Token | Słowo | Cokolwiek jednostka którą model konsumuje. Może być słowem, podłowem, znakiem lub bajtem. |
| Stem | Korzeń słowa | Wynik regułowego obcinania sufiksu. Nie zawsze prawdziwe słowo. |
| Lemma | Forma słownikowa | Forma, której byś użył do wyszukiwania. Wymaga kontekstu gramatycznego do poprawnego obliczenia. Nie zawsze prawdziwe słowo. |
| POS tag | Część mowy | Kategoria jak NOUN, VERB, ADJ. Potrzebna do dokładnej lematyzacji. |
| Morfologia | Zasady kształtu słowa | Jak słowo zmienia formę na podstawie czasu, liczby, przypadku. Lematyzacja od tego zależy. |

## Dalsze czytanie

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — oryginalny artykuł, pięć stron, wciąż najjaśniejsze wyjaśnienie.
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — jak prawdziwy potok jest okablowany.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — przypadki brzegowe tokenizacji o których jeszcze nie pomyślałeś.
```