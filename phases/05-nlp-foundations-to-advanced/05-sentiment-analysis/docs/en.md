# Analiza sentymentu

> Kanoniczne zadanie NLP. Większość tego, co musisz wiedzieć o klasycznej klasyfikacji tekstu, pojawia się właśnie tutaj.

**Typ:** Zbuduj
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 02 (BoW + TF-IDF), Faza 2 · 14 (Naive Bayes)
**Szacowany czas:** ~75 minut

## Problem

" Jedzenie nie było dobre." Pozytywne czy negatywne?

Sentyment brzmi prosto. Recenzent powiedział, że coś mu się podobało lub nie. Etykietuj zdanie. Powód, dla którego stał się kanonicznym zadaniem NLP, polega na tym, że każdy wyglądający łatwo przypadek skrywa trudny. Negacja odwraca znaczenie. Sarkazm odwraca je ponownie. "Wcale nie było źle" jest pozytywne, mimo dwóch negatywnie kodowanych słów. Emoji niosą więcej sygnału niż otaczający tekst. Słownictwo dziedzinowe ma znaczenie (`tight` w recenzji muzycznej versus `tight` w recenzji mody).

Sentyment jest working lab dla klasycznego NLP. Jeśli rozumiesz, dlaczego każdy naiwny baseline ma określony tryb awarii, rozumiesz, dlaczego każdy bogatszy model został wynaleziony. Ta lekcja buduje baseline Naive Bayes od zera, dodaje regresję logistyczną i nazywa pułapki, które czynią produkcyjny sentyment problemem klasy compliance.

## Koncepcja

![Potok sentymentu: tokeny → cechy → klasyfikator → etykieta](./assets/sentiment.svg)

Klasyczny sentyment to przepis dwuetapowy.

1. **Reprezentacja.** Zamień tekst w wektor cech. BoW, TF-IDF lub n-gramy.
2. **Klasyfikacja.** Dopasuj model liniowy (Naive Bayes, regresja logistyczna, SVM) do oznaczonych przykładów.

Naive Bayes to najgłupszy model, który działa. Załóż, że każda cecha jest niezależna od etykiety. Oszacuj `P(word | positive)` i `P(word | negative)` ze zliczeń. Podczas wnioskowania pomnóż prawdopodobieństwa. "Naiwne" założenie o niezależności jest śmiesznie błędne, a wyniki są zaskakująco silne. Powód: przy rzadkich cechach tekstowych i umiarkowanych danych klasyfikatorowi zależy bardziej na tym, po której stronie każde słowo się przechyla, niż ile waży.

Regresja logistyczna naprawia założenie o niezależności. Uczy się wagi dla każdej cechy, w tym ujemnych wag. `not good` jako cecha bigramowa dostaje ujemną wagę. Naive Bayes nie może tego zrobić dla bigramów, których nigdy nie etykietował.

## Zbuduj to

### Krok 1: prawdziwy mini-dataset

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Małe celowo. Prawdziwa praca używa dziesiątek tysięcy przykładów (IMDb, SST-2, Yelp polarity). Matematyka jest identyczna.

### Krok 2: wielomianowy Naive Bayes od zera

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing (alpha=1.0) to Laplace smoothing. Bez niego słowo niewidziane w klasie ma prawdopodobieństwo zero i log wybucha. `alpha=0.01` jest powszechne w praktyce. `alpha=1.0` to domyślna wartość do nauczania.

### Krok 3: regresja logistyczna od zera

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization ma tu znaczenie. Cechy tekstowe są rzadkie, bez L2 model zapamiętuje przykłady treningowe. Zacznij od `0.01` i dostrój.

### Krok 4: obsługa negacji (tryb awarii)

Rozważ "not good" i "not bad". Klasyfikator BoW widzi `{not, good}` i `{not, bad}` i uczy się od tego, które pojawiło się częściej podczas treningu. Klasyfikator bigramowy widzi `not_good` i `not_bad` i uczy się ich jako odrębnych cech. To zazwyczaj wystarczy.

Gorsza poprawka, która działa gdy nie masz bigramów: **negation scoping**. Prefiksuje tokeny następujące po słowie negacji prefiksem `NOT_` aż do następnego interpunkcyjnego.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Teraz `good` i `NOT_good` to różne cechy. Klasyfikator może im przypisać przeciwne wagi. Trzy linie preprocessingu, mierzalny skok dokładności na benchmarkach sentymentu.

### Krok 5: metryki ewaluacyjne, które mają znaczenie

Sama dokładność jest myląca, jeśli klasy są niezrównoważone. Rzeczywiste korpusy sentymentu są zazwyczaj w 70-80% pozytywne lub w 70-80% negatywne, stały klasyfikator większościowy uzyskuje 80% dokładności i jest bezużyteczny. Raportuj każdą z poniższych:

- **Precision i recall per klasa.** Jedna para na klasę. Uśrednij je makro, żeby uzyskać jedną liczbę, która szanuje równowagę klas.
- **Macro-F1 (główna metryka dla niezrównoważonych danych).** Średnia per-klasowych F1, ważona równo. Użyj tego zamiast dokładności, gdy klasy są niezrównoważone.
- **Weighted-F1 (alternatywa).** To samo co macro, ale ważone częstością klas. Raportuj obok macro-F1, gdy niezrównoważenie ma znaczenie biznesowe.
- **Confusion matrix.** Surowe zliczenia. Zawsze sprawdzaj przed zaufaniem dowolnej skalarnej metryce, ujawnia która para klas myli model.
- **Przykłady błędów per klasa.** Wyciągnij 5 błędnych predykcji na klasę. Przeczytaj je. Nic nie zastąpi czytania rzeczywistych błędów.

Przy mocno niezrównoważonych danych (> 95-5 ratio), raportuj **AUROC** i **AUPRC** zamiast dokładności. AUPRC jest bardziej czuły na klasę mniejszościową, która jest tym, co zwykle cię interesuje (spam, fraud, rzadki sentyment).

**Częsty bug do unikania.** Raportowanie micro-F1 zamiast macro-F1 na niezrównoważonych danych daje liczbę, która wygląda wysoko, bo jest zdominowana przez klasę większościową. Macro-F1 zmusza cię do zobaczenia wydajności klasy mniejszościowej.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Użyj tego

scikit-learn robi to w sześciu linijkach, poprawnie.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Trzy rzeczy do zauważenia. `stop_words=None` zachowuje negacje. `ngram_range=(1, 2)` dodaje bigramy, więc `not_good` staje się cechą. `sublinear_tf=True` tłumi powtórzone słowa. Te trzy flagi to różnica między baseline na 75% dokładności a baseline na 85% dokładności na SST-2.

### Kiedy sięgać po transformer

- Wykrywanie sarkazmu. Klasyczne modele tutaj zawodzą. Koniec.
- Długie recenzje, gdzie sentyment zmienia się w połowie dokumentu.
- Sentyment oparty na aspektach. "Aparat był świetny, ale bateria była okropna." Musisz przypisać sentyment do aspektów. Tylko transformery lub modele ze strukturalnym outputem.
- Języki nieangielskie, low-resource. Multilingual BERT daje ci zero-shot baseline za darmo.

Jeśli potrzebujesz czegokolwiek z powyższych, przejdź do fazy 7 (głębokie zanurzenie w transformery). W przeciwnym razie, Naive Bayes lub regresja logistyczna na TF-IDF plus bigramy plus obsługa negacji to twój produkcyjny baseline na 2026 rok.

### Pułapka reprodukowalności (znowu)

Przeuczanie modeli sentymentu jest rutynowe. Ponowna ewaluacja nie jest. Liczby dokładności raportowane w artykułach używają specyficznych podziałów, specyficznego preprocessingu, specyficznych tokenizerów. Jeśli porównujesz swój nowy model do baseline bez użycia identycznego potoku, uzyskasz mylące delty. Zawsze regeneruj baseline na swoim potoku, nie na liczbie z artykułu.

## Wyślij to

Zapisz jako `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Ćwiczenia

1. **Łatwe.** Dodaj `apply_negation` jako krok preprocessingu w potoku scikit-learn i zmierz deltę F1 na małym datasetcie sentymentu.
2. **Średnie.** Zaimplementuj klasyfikator regresji logistycznej z wagami klas (przekaż `class_weight="balanced"` do scikit-learn lub wyprowadź gradient samodzielnie). Zmierz efekt na syntetycznym niezrównoważeniu klas 90-10.
3. **Trudne.** Zbuduj detektor sarkazmu trenując drugi klasyfikator na residuach modelu sentymentu. Udokumentuj swój eksperymentalny setup. Ostrzeż czytelnika, gdy twoja dokładność jest poniżej szansy (szansa na 2-klasowym sarkazmie to ~50%, a większość pierwszych prób tam ląduje).

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Polarity | Pozytywny lub negatywny | Binarna etykieta; czasem rozszerzona do neutralnego lub fine-grained (5 gwiazdek). |
| Aspect-based sentiment | Per-aspect polarity | Przypisz sentyment do specyficznych encji lub atrybutów wspomnianych w tekście. |
| Negation scoping | Odwracanie pobliskich tokenów | Prefiksuje tokeny po "not" z `NOT_` aż do interpunkcji. |
| Laplace smoothing | Dodawanie 1 do zliczeń | Zapobiega cechom z prawdopodobieństwem zero w Naive Bayes. |
| L2 regularization | Kurczenie wag | Dodaje `lambda * sum(w^2)` do loss. Niezbędne dla rzadkich cech tekstowych. |

## Dalsza lektura

- Pang and Lee (2008). Opinion Mining and Sentiment Analysis — foundational survey. Long, ale pierwsze cztery sekcje pokrywają wszystko klasyczne.
- Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification — paper który pokazał że bigramy + Naive Bayes jest trudny do pokonania na short text.
- scikit-learn text feature extraction docs — reference dla `CountVectorizer`, `TfidfVectorizer` i każdego pokrętła które dostroisz.