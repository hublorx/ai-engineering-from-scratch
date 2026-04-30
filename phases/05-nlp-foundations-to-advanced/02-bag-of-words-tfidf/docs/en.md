# Bag of Words, TF-IDF i reprezentacja tekstu

> Licz najpierw, myśl później. TF-IDF wciąż pokonuje embeddings w dobrze zdefiniowanych zadaniach w 2026 roku.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 01 (Przetwarzanie tekstu), Faza 2 · 02 (Regresja liniowa od zera)
**Szacowany czas:** ~75 minut

## Problem

Model potrzebuje liczb. Ty masz stringi.

Każdy pipeline NLP musi odpowiedzieć na to samo pytanie. Jak zamienić strumień tokenów o zmiennej długości na wektor o stałym rozmiarze, który klasyfikator będzie mógł przetworzyć. Pierwsza odpowiedź, na jaką wpadła dziedzina, była najgłupsza, która działa. Policz słowa. Zrób wektor.

Ten wektor wsparł więcej produkcyjnego NLP niż jakikolwiek model embeddingowy. Filtry spamu, klasyfikatory tematów, wykrywanie anomalii w logach, ranking wyszukiwania (przed BM25), pierwsza fala analizy sentymentu, pierwsza dekada akademickich benchmarków NLP. Praktycy w 2026 nadal sięgają po niego w pierwszej kolejności przy wąskich zadaniach klasyfikacji. Jest szybki, interpretowalny i często nie do odróżnienia od modelu embeddingowego z 400M parametrów w zadaniach, gdzie to obecność słów ma znaczenie.

Ta lekcja buduje bag of words, a następnie TF-IDF od zera. Potem pokazuje, jak scikit-learn robi to samo w trzech liniach. Potem wymienia tryb awarii, który sprawia, że sięgasz po embeddings.

## Koncepcja

![BoW vs TF-IDF representation flow](./assets/bow-tfidf.svg)

**Bag of Words (BoW)** wyrzuca kolejność. Dla każdego dokumentu zlicza, ile razy pojawia się każde słowo ze słownika. Długość wektora to rozmiar słownika. Pozycja `i` to zliczenie słowa `i`.

**TF-IDF** przeważa BoW. Słowo, które pojawia się w każdym dokumencie, jest nieinformacyjne, więc skalujemy je w dół. Słowo rzadkie w całym korpusie, ale częste w jednym dokumencie, to sygnał, więc skalujemy je w górę.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Gdzie `TF` to term frequency w dokumencie, `df` to document frequency (ile dokumentów zawiera słowo), `N` to całkowita liczba dokumentów. `log` utrzymuje wagę w ryzach dla wszechobecnych słów.

Kluczowa właściwość: oba produkują sparse vectors z interpretowalnymi osiami. Możesz spojrzeć na wagi wytrenowanego klasyfikatora i odczytać, które słowa przesuwają dokument w kierunku każdej klasy. Nie możesz tego zrobić z 768-wymiarowym embeddingiem BERT.

## Zbuduj to

### Krok 1: zbuduj słownik

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Input: lista tokenizowanych dokumentów (dowolny tokenizer na poziomie słów; `code/main.py` w tej lekcji używa uproszczonej wersji lowercase). Output: słownik `{word: index}`. Stabilna kolejność wstawiania oznacza, że słowo o indeksie 0 to pierwsze słowo widziane w pierwszym dokumencie. Konwencja się różni; scikit-learn sortuje alfabetycznie.

### Krok 2: bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Wiersze to dokumenty. Kolumny to indeksy słownika. Element `[i][j]` to "ile razy słowo `j` pojawia się w dokumencie `i`." Dokument 1 ma `cat` dwa razy, bo tyle ma. Dokument 0 ma `ran` zero razy, bo tyle ma.

### Krok 3: term frequency i document frequency

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Dwie sztuczki smoothing warte wymienienia. `(n+1)/(d+1)` unika `log(x/0)`. Końcowe `+1` zapewnia, że słowo w każdym dokumencie nadal ma IDF 1 (nie 0), co odpowiada domyślnemu zachowaniu scikit-learn. Inne implementacje używają surowego `log(N/df)`. Obie działają; wygładzona wersja jest przyjaźniejsza.

### Krok 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Trzy dokumenty, pięć słów w słowniku (`the`, `cat`, `sat`, `dog`, `ran`). `the` pojawia się we wszystkich trzech, więc jego IDF jest niskie. `dog` pojawia się w jednym, więc jego IDF jest wysokie. Wektory są sparse (większość elementów to małe wartości), a dyskryminacyjne słowa się wyróżniają.

### Krok 5: L2-normalizacja wierszy

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Bez normalizacji, dłuższy dokument dostaje większy wektor i dominuje w wynikach podobieństwa. L2 normalizacja umieszcza każdy dokument na unit hypersphere. Cosine similarity między wierszami to teraz po prostu iloczyn skalarny.

## Użyj tego

scikit-learn dostarcza wersję produkcyjną.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` robi tokenizację, słownik i BoW jednym wywołaniem. `TfidfVectorizer` dodaje ważenie IDF i L2 normalizację. Oba zwracają sparse matrices. Dla 100k dokumentów, wersja dense nie zmieści się w pamięci; pozostań sparse, dopóki klasyfikator nie zażąda dense.

Pokrętła, które zmieniają wszystko:

| Argument | Efekt |
|----------|-------|
| `ngram_range=(1, 2)` | Uwzględnij bigramy. Zwykle poprawia klasyfikację. |
| `min_df=2` | Usuń słowa występujące w mniej niż 2 dokumentach. Przycina słownik przy zaszumionych danych. |
| `max_df=0.95` | Usuń słowa występujące w więcej niż 95% dokumentów. Przybliża usuwanie stopwords bez hardcoded listy. |
| `stop_words="english"` | Wbudowana lista stopwords scikit-learn. Zależy od zadania — analiza sentymentu NIE powinna usuwać negacji. |
| `sublinear_tf=True` | Użyj `1 + log(tf)` zamiast surowego `tf`. Pomaga, gdy termin powtarza się wiele razy w jednym dokumencie. |

### Kiedy TF-IDF nadal wygrywa (stan na 2026)

- Wykrywanie spamu, etykietowanie tematów, flagowanie anomalii w logach. Obecność słów ma znaczenie; niuanse semantyczne nie.
- Tryby z małą ilością danych (setki oznakowanych przykładów). TF-IDF plus regresja logistyczna nie ma kosztu pretrainingu.
- Wszędzie tam, gdzie liczy się latency. TF-IDF plus model liniowy odpowiada w mikrosekundach. Embedding dokumentu przez transformer zajmuje 10-100ms.
- Systemy, które muszą wyjaśniać swoje predykcje. Sprawdź współczynniki klasyfikatora. Najwyższe pozytywne słowa to powód.

### Kiedy TF-IDF zawodzi

Tryb awarii: ślepota semantyczna. Rozważ te dwa dokumenty:

- "Film był w ogóle nie dobry."
- "Film był doskonały."

Jeden to negatywna recenzja. Jeden to pozytywna. Ich nakładanie TF-IDF to dokładnie `{the, movie, was}`. Klasyfikator bag-of-words musi zapamiętać, że słowo `not` blisko `good` odwraca etykietę. Może się tego nauczyć przy wystarczającej ilości danych, ale nigdy tak elegancko jak model, który rozumie składnię.

Inna awaria: słowa out-of-vocabulary podczas inferencji. Model BoW trenowany na recenzjach IMDb nie wie, co zrobić z `Zoomer-approved`, jeśli ten token nigdy nie pojawił się w treningu. Subword embeddings (lekcja 04) to obsługują. TF-IDF nie.

### Hybryda: TF-IDF weighted embeddings

Pragmatyczny domyślny wybór na 2026 dla klasyfikacji przy średniej ilości danych: użyj wag TF-IDF jako attention nad word embeddings.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Dostajesz pojemność semantyczną z embeddings i nacisk na rzadkie słowa z TF-IDF. Klasyfikator trenowany na pooled vector. To przewyższa oba podejścia osobno przy klasyfikacji sentymentu, tematu i intencji poniżej około 50k oznakowanych przykładów.

## Wyślij to

Zapisz jako `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj `cosine_similarity(doc_vec_a, doc_vec_b)` na L2-znormalizowanym wyjściu TF-IDF. Zweryfikuj, że identyczne dokumenty oceniają się na 1.0, a dokumenty z rozłącznym słownikiem oceniają się na 0.0.
2. **Średnie.** Dodaj obsługę `n-gram` do `bag_of_words`. Parametr `n` produkuje zliczenia dla `n`-gramów. Przetestuj, że `n=2` na `["the", "cat", "sat"]` produkuje bigram counts dla `["the cat", "cat sat"]`.
3. **Trudne.** Zbuduj hybrydę TF-IDF-weighted-embedding powyżej używając wektorów GloVe 100d (pobierz raz, cacheuj). Porównaj dokładność klasyfikacji względem plain TF-IDF i plain mean-pooled embeddings na zbiorze danych 20 Newsgroups. Raportuj, co wygrywa gdzie.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| BoW | Wektor częstotliwości słów | Zliczenia słów ze słownika w jednym dokumencie. Wyrzuca kolejność. |
| TF | Term frequency | Zliczenie słowa w dokumencie, opcjonalnie znormalizowane przez długość dokumentu. |
| DF | Document frequency | Zliczenie dokumentów zawierających słowo co najmniej raz. |
| IDF | Inverse document frequency | `log(N / df)` wygładzone. Zaniża słowa, które pojawiają się wszędzie. |
| Sparse vector | Głównie zera | Słownik ma zwykle 10k-100k słów; większość nie występuje w żadnym konkretnym dokumencie. |
| Cosine similarity | Kąt wektora | Iloczyn skalarny L2-znormalizowanych wektorów. 1 to identyczne, 0 to ortogonalne. |

## Dalsze czytanie

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — kanoniczny reference API, plus notatki o każdym pokrętle.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — artykuł, który uczynił TF-IDF domyślnym wyborem na dekadę.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — perspektywa na 2026, kiedy stara metoda wygrywa i dlaczego.