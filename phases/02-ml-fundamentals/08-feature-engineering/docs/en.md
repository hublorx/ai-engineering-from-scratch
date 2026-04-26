# Inżynieria cech i ich selekcja

> Dobra cecha jest warta tysiąc punktów danych.

**Typ:** Zbuduj
**Języki:** Python
**Wymagania wstępne:** Phase 1 (Statystyka dla ML, Algebra liniowa), Phase 2 Lekcje 1-7
**Szacowany czas:** ~90 minut

## Cele uczenia się

- Implementuj transformacje numeryczne (standaryzacja, skalowanie min-max, transformacja logarytmiczna, binning) i wytłumacz, kiedy każda z nich jest odpowiednia
- Buduj encoding typu one-hot, label i target dla cech kategorycznych oraz zidentyfikuj ryzyko data leakage w target encoding
- Zbuduj wektoryzator TF-IDF od zera i wytłumacz, dlaczego przewyższa surowe zliczenia słów w klasyfikacji tekstu
- Zastosuj selekcję cech opartą na filtrach (próg wariancji, korelacja, mutual information) aby zmniejszyć wymiarowość

## Problem

Masz dataset. Wybierasz algorytm. Uczysz go. Wyniki są przeciętne. Próbujesz bardziej wyrafinowanego algorytmu. Wciąż przeciętne. Spędzasz tydzień na strojeniu hiperparametrów. Minimalna poprawa.

Potem ktoś transformuje surowe dane na lepsze cechy i prosty regresja logistyczna pokonuje twój dostrojony ensemble oparty na gradient boosting.

To dzieje się ciągle. W klasycznym ML, reprezentacja danych ma większe znaczenie niż wybór algorytmu. Model cen domów z "powierzchnią w stopach kwadratowych" i "liczbą sypialni" pokona model z "adresem jako surowym ciągiem znaków" bez względu na to, jak wyrafinowany jest learner. Algorytm może pracować tylko z tym, co mu dasz.

Inżynieria cech to proces transformacji surowych danych w reprezentacje, które ułatwiają modelom znajdowanie wzorców. Selekcja cech to proces odrzucania cech, które dodają szum bez dodawania sygnału. Razem stanowią one aktywność o najwyższej dźwigni w klasycznym ML.

## Koncepcja

### Pipeline cech

```mermaid
flowchart LR
    A[Raw Data] --> B[Handle Missing Values]
    B --> C[Numerical Transforms]
    B --> D[Categorical Encoding]
    B --> E[Text Features]
    C --> F[Feature Interactions]
    D --> F
    E --> F
    F --> G[Feature Selection]
    G --> H[Model-Ready Data]
```

### Cechy numeryczne

Surowe liczby rzadko są gotowe dla modelu. Typowe transformacje:

**Skalowanie:** Przekształcaj cechy na ten sam zakres, aby algorytmy oparte na odległości (K-Means, KNN, SVM) traktowały wszystkie cechy jednakowo. Skalowanie min-max mapuje do [0, 1]. Standaryzacja (z-score) mapuje do mean=0, std=1.

**Transformacja logarytmiczna:** Kompresuje rozkłady skośne w prawo (dochód, populacja, zliczenia słów). Zamienia relacje multiplikatywne na addytywne.

**Binning:** Konwertuje wartości ciągłe na kategorie. Przydatne, gdy relacja między cechą a target jest nieliniowa, ale krokowa (np. grupy wiekowe).

**Cechy wielomianowe:** Tworzy wyrazy x^2, x^3, x1*x2. Pozwala liniowym modelom uchwycić nieliniowe relacje za cenę większej liczby cech.

### Cechy kategoryczne

Modele potrzebują liczb. Kategorie potrzebują encodingu.

**One-hot encoding:** Tworzy kolumnę binarną dla każdej kategorii. "color = red/blue/green" staje się trzema kolumnami: is_red, is_blue, is_green. Działa dobrze dla niskiej kardynalności, ale wybucha przy wielu kategoriach.

**Label encoding:** Mapuje każdą kategorię na liczbę całkowitą: red=0, blue=1, green=2. Wprowadza fałszywe uporządkowanie (model może myśleć, że green > blue > red). Odpowiednie tylko dla modeli drzewiastych, które dzielą na pojedyncze wartości.

**Target encoding:** Zastępuje każdą kategorię średnią zmiennej target dla tej kategorii. Potężne, ale niebezpieczne: wysokie ryzyko data leakage. Musi być obliczane tylko na danych treningowych i stosowane na danych testowych.

### Cechy tekstowe

**Count vectorizer:** Zlicza, ile razy każde słowo pojawia się w dokumencie. "the cat sat on the mat" staje się {the: 2, cat: 1, sat: 1, on: 1, mat: 1}.

**TF-IDF:** Term Frequency-Inverse Document Frequency. Waży słowa przez ich unikalność w dokumentach. Częste słowa jak "the" otrzymują niską wagę. Rzadkie, charakterystyczne słowa otrzymują wysoką wagę.

```
TF(word, doc) = count(word in doc) / total words in doc
IDF(word) = log(total docs / docs containing word)
TF-IDF = TF * IDF
```

### Brakujące wartości

Rzeczywiste dane mają dziury. Strategie:

- **Usuwanie wierszy:** Tylko gdy brakujące dane są rzadkie i losowe
- **Imputacja średnią/medianą:** Prosta, zachowuje kształt rozkładu (mediana jest bardziej odporna na outliers)
- **Imputacja modą:** Dla cech kategorycznych
- **Kolumna wskaźnikowa:** Dodaj kolumnę binarną "was_this_missing" przed imputacją. Fakt, że dane są brakujące, może sam w sobie być informacyjny
- **Forward/backward fill:** Dla danych szeregów czasowych

### Interakcje cech

Czasami relacja tkwi w kombinacji. "Wzrost" i "waga" same w sobie są mniej predykcyjne niż "BMI = waga / wzrost^2". Interakcje cech mnożą przestrzeń cech, więc używaj wiedzy domenowej, aby wybrać właściwe.

### Selekcja cech

Więcej cech nie zawsze oznacza lepsze wyniki. Nieistotne cechy dodają szum, zwiększają czas treningu i mogą powodować overfitting.

**Metody filtrujące (przed modelem):**
- Korelacja: usuń cechy silnie skorelowane ze sobą (redundancja)
- Mutual information: mierzy, ile wiedzy o cesze redukuje niepewność o target
- Próg wariancji: usuń cechy, które prawie nie różnią się

**Metody wrapper (oparte na modelu):**
- Regularyzacja L1 (Lasso): zeruje wagi nieistotnych cech
- Rekursywna eliminacja cech: trenuj, usuń najmniej ważną cechę, powtórz

**Dlaczego selekcja ma znaczenie:** Model z 10 dobrymi cechami zwykle przewyższy model z 10 dobrymi i 90 szumowymi cechami. Szumowe cechy dają modelowi możliwości do overfittingu na wzorce treningowe, które się nie uogólniają.

## Zbuduj to

### Krok 1: Transformacje numeryczne od zera

```python
import math


def min_max_scale(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    return [(v - mean) / std for v in values]


def log_transform(values):
    return [math.log(v + 1) for v in values]


def bin_values(values, n_bins=5):
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / n_bins
    if bin_width == 0:
        return [0] * len(values)
    result = []
    for v in values:
        bin_idx = int((v - min_val) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)
        result.append(bin_idx)
    return result


def polynomial_features(row, degree=2):
    n = len(row)
    result = list(row)
    if degree >= 2:
        for i in range(n):
            result.append(row[i] ** 2)
        for i in range(n):
            for j in range(i + 1, n):
                result.append(row[i] * row[j])
    return result
```

### Krok 2: Encoding kategoryczny od zera

```python
def one_hot_encode(values):
    categories = sorted(set(values))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    n_cats = len(categories)

    encoded = []
    for v in values:
        row = [0] * n_cats
        row[cat_to_idx[v]] = 1
        encoded.append(row)

    return encoded, categories


def label_encode(values):
    categories = sorted(set(values))
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    return [cat_to_int[v] for v in values], cat_to_int


def target_encode(feature_values, target_values, smoothing=10):
    global_mean = sum(target_values) / len(target_values)

    category_stats = {}
    for feat, target in zip(feature_values, target_values):
        if feat not in category_stats:
            category_stats[feat] = {"sum": 0.0, "count": 0}
        category_stats[feat]["sum"] += target
        category_stats[feat]["count"] += 1

    encoding = {}
    for cat, stats in category_stats.items():
        cat_mean = stats["sum"] / stats["count"]
        weight = stats["count"] / (stats["count"] + smoothing)
        encoding[cat] = weight * cat_mean + (1 - weight) * global_mean

    return [encoding[v] for v in feature_values], encoding
```

### Krok 3: Cechy tekstowe od zera

```python
def count_vectorize(documents):
    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    vectors = []
    for doc in documents:
        vec = [0] * len(vocab)
        for word in doc.lower().split():
            vec[vocab[word]] += 1
        vectors.append(vec)

    return vectors, vocab


def tfidf(documents):
    n_docs = len(documents)

    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    doc_freq = {}
    for doc in documents:
        seen = set()
        for word in doc.lower().split():
            if word not in seen:
                doc_freq[word] = doc_freq.get(word, 0) + 1
                seen.add(word)

    vectors = []
    for doc in documents:
        words = doc.lower().split()
        word_count = len(words)
        tf_map = {}
        for word in words:
            tf_map[word] = tf_map.get(word, 0) + 1

        vec = [0.0] * len(vocab)
        for word, count in tf_map.items():
            tf = count / word_count
            idf = math.log(n_docs / doc_freq[word])
            vec[vocab[word]] = tf * idf
        vectors.append(vec)

    return vectors, vocab
```

### Krok 4: Imputacja brakujących wartości od zera

```python
def impute_mean(values):
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values), 0.0
    mean = sum(present) / len(present)
    return [v if v is not None else mean for v in values], mean


def impute_median(values):
    present = sorted(v for v in values if v is not None)
    if not present:
        return [0.0] * len(values), 0.0
    n = len(present)
    if n % 2 == 0:
        median = (present[n // 2 - 1] + present[n // 2]) / 2
    else:
        median = present[n // 2]
    return [v if v is not None else median for v in values], median


def impute_mode(values):
    present = [v for v in values if v is not None]
    if not present:
        return values, None
    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    mode = max(counts, key=counts.get)
    return [v if v is not None else mode for v in values], mode


def add_missing_indicator(values):
    return [0 if v is not None else 1 for v in values]
```

### Krok 5: Selekcja cech od zera

```python
def correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def mutual_information(feature, target, n_bins=10):
    feat_min = min(feature)
    feat_max = max(feature)
    bin_width = (feat_max - feat_min) / n_bins if feat_max != feat_min else 1.0
    feat_binned = [
        min(int((f - feat_min) / bin_width), n_bins - 1) for f in feature
    ]

    n = len(feature)
    target_classes = sorted(set(target))

    feat_bins = sorted(set(feat_binned))
    p_feat = {}
    for b in feat_bins:
        p_feat[b] = feat_binned.count(b) / n

    p_target = {}
    for t in target_classes:
        p_target[t] = target.count(t) / n

    mi = 0.0
    for b in feat_bins:
        for t in target_classes:
            joint_count = sum(
                1 for fb, tv in zip(feat_binned, target) if fb == b and tv == t
            )
            p_joint = joint_count / n
            if p_joint > 0:
                mi += p_joint * math.log(p_joint / (p_feat[b] * p_target[t]))

    return mi


def variance_threshold(features, threshold=0.01):
    n_features = len(features[0])
    n_samples = len(features)
    selected = []

    for j in range(n_features):
        col = [features[i][j] for i in range(n_samples)]
        mean = sum(col) / n_samples
        var = sum((v - mean) ** 2 for v in col) / n_samples
        if var >= threshold:
            selected.append(j)

    return selected


def remove_correlated(features, threshold=0.9):
    n_features = len(features[0])
    n_samples = len(features)

    to_remove = set()
    for i in range(n_features):
        if i in to_remove:
            continue
        col_i = [features[r][i] for r in range(n_samples)]
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            col_j = [features[r][j] for r in range(n_samples)]
            corr = abs(correlation(col_i, col_j))
            if corr >= threshold:
                to_remove.add(j)

    return [i for i in range(n_features) if i not in to_remove]
```

### Krok 6: Pełny pipeline i demo

```python
import random


def make_housing_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        sqft = random.uniform(500, 5000)
        bedrooms = random.choice([1, 2, 3, 4, 5])
        age = random.uniform(0, 50)
        neighborhood = random.choice(["downtown", "suburbs", "rural"])
        has_pool = random.choice([True, False])

        sqft_with_missing = sqft if random.random() > 0.05 else None
        age_with_missing = age if random.random() > 0.08 else None

        price = (
            50 * sqft
            + 20000 * bedrooms
            - 1000 * age
            + (50000 if neighborhood == "downtown" else 10000 if neighborhood == "suburbs" else 0)
            + (15000 if has_pool else 0)
            + random.gauss(0, 20000)
        )

        data.append({
            "sqft": sqft_with_missing,
            "bedrooms": bedrooms,
            "age": age_with_missing,
            "neighborhood": neighborhood,
            "has_pool": has_pool,
            "price": price,
        })
    return data


if __name__ == "__main__":
    data = make_housing_data(200)

    print("=== Raw Data Sample ===")
    for row in data[:3]:
        print(f"  {row}")

    sqft_raw = [d["sqft"] for d in data]
    age_raw = [d["age"] for d in data]
    prices = [d["price"] for d in data]

    print("\n=== Missing Value Handling ===")
    sqft_missing = sum(1 for v in sqft_raw if v is None)
    age_missing = sum(1 for v in age_raw if v is None)
    print(f"  sqft missing: {sqft_missing}/{len(sqft_raw)}")
    print(f"  age missing: {age_missing}/{len(age_raw)}")

    sqft_indicator = add_missing_indicator(sqft_raw)
    age_indicator = add_missing_indicator(age_raw)
    sqft_imputed, sqft_fill = impute_median(sqft_raw)
    age_imputed, age_fill = impute_mean(age_raw)
    print(f"  sqft filled with median: {sqft_fill:.0f}")
    print(f"  age filled with mean: {age_fill:.1f}")

    print("\n=== Numerical Transforms ===")
    sqft_scaled = standardize(sqft_imputed)
    age_scaled = min_max_scale(age_imputed)
    sqft_log = log_transform(sqft_imputed)
    age_binned = bin_values(age_imputed, n_bins=5)
    print(f"  sqft standardized: mean={sum(sqft_scaled)/len(sqft_scaled):.4f}, std={math.sqrt(sum(v**2 for v in sqft_scaled)/len(sqft_scaled)):.4f}")
    print(f"  age min-max: [{min(age_scaled):.2f}, {max(age_scaled):.2f}]")
    print(f"  age bins: {sorted(set(age_binned))}")

    print("\n=== Categorical Encoding ===")
    neighborhoods = [d["neighborhood"] for d in data]

    ohe, ohe_cats = one_hot_encode(neighborhoods)
    print(f"  One-hot categories: {ohe_cats}")
    print(f"  Sample encoding: {neighborhoods[0]} -> {ohe[0]}")

    le, le_map = label_encode(neighborhoods)
    print(f"  Label encoding map: {le_map}")

    te, te_map = target_encode(neighborhoods, prices, smoothing=10)
    print(f"  Target encoding: {({k: round(v) for k, v in te_map.items()})}")

    print("\n=== Text Features ===")
    descriptions = [
        "large modern house with pool",
        "small cozy cottage near downtown",
        "spacious family home with large yard",
        "modern apartment downtown with view",
        "rustic cabin in rural area",
    ]
    cv, cv_vocab = count_vectorize(descriptions)
    print(f"  Vocabulary size: {len(cv_vocab)}")
    print(f"  Doc 0 non-zero features: {sum(1 for v in cv[0] if v > 0)}")

    tf, tf_vocab = tfidf(descriptions)
    print(f"  TF-IDF vocabulary size: {len(tf_vocab)}")
    top_words = sorted(tf_vocab.keys(), key=lambda w: tf[0][tf_vocab[w]], reverse=True)[:3]
    print(f"  Doc 0 top TF-IDF words: {top_words}")

    print("\n=== Polynomial Features ===")
    sample_row = [sqft_scaled[0], age_scaled[0]]
    poly = polynomial_features(sample_row, degree=2)
    print(f"  Input: {[round(v, 4) for v in sample_row]}")
    print(f"  Polynomial: {[round(v, 4) for v in poly]}")
    print(f"  Features: [x1, x2, x1^2, x2^2, x1*x2]")

    print("\n=== Feature Selection ===")
    feature_matrix = [
        [sqft_scaled[i], age_scaled[i], float(sqft_indicator[i]), float(age_indicator[i])]
        + ohe[i]
        for i in range(len(data))
    ]

    print(f"  Total features: {len(feature_matrix[0])}")

    surviving_var = variance_threshold(feature_matrix, threshold=0.01)
    print(f"  After variance threshold (0.01): {len(surviving_var)} features kept")

    surviving_corr = remove_correlated(feature_matrix, threshold=0.9)
    print(f"  After correlation filter (0.9): {len(surviving_corr)} features kept")

    binary_prices = [1 if p > sum(prices) / len(prices) else 0 for p in prices]
    print("\n  Mutual information with target:")
    feature_names = ["sqft", "age", "sqft_missing", "age_missing"] + [f"neigh_{c}" for c in ohe_cats]
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        mi = mutual_information(col, binary_prices, n_bins=10)
        print(f"    {feature_names[j]}: MI={mi:.4f}")

    print("\n  Correlation with price:")
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        corr = correlation(col, prices)
        print(f"    {feature_names[j]}: r={corr:.4f}")
```

## Użyj tego

Z scikit-learn, te transformacje są kompozycyjnymi pipelineami:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["sqft", "age"]),
    ("cat", categorical_pipe, ["neighborhood"]),
])
```

Wersje od zera pokazują dokładnie, co dzieje się w środku każdej transformacji. Wersje biblioteczne dodają obsługę przypadków brzegowych, wsparcie dla sparse matrices i kompozycję pipeline, ale matematyka jest ta sama.

## Wyślij to

Ta lekcja wytwarza:
- `outputs/prompt-feature-engineer.md` - prompt do systematycznego inżynierowania cech z surowych danych

## Ćwiczenia

1. Dodaj robust scaling (używając mediany i rozstępu międzykwartylowego zamiast średniej i odchylenia standardowego) do transformacji numerycznych. Porównaj go ze standardowym skalowaniem na danych z ekstremalnymi outliers.
2. Implementuj leave-one-out target encoding: dla każdego wiersza oblicz średnią target wykluczając własną wartość target tego wiersza. Pokaż, jak to zmniejsza overfitting w porównaniu z naiwnym target encoding.
3. Zbuduj automatyczny pipeline selekcji cech, który łączy próg wariancji, filtrowanie korelacji i ranking mutual information. Zastosuj go do housing dataset i porównaj wydajność modelu (użyj prostej regresji liniowej) ze wszystkimi cechami vs wybranymi cechami.

## Kluczowe pojęcia

| Pojęcie | Co ludzie mówią | Co to faktycznie oznacza |
|---------|----------------|----------------------|
| Feature engineering | "Tworzenie nowych kolumn" | Transformacja surowych danych w reprezentacje, które ujawniają wzorce modelowi |
| Standardization | "Normowanie" | Odejmowanie średniej i dzielenie przez odchylenie standardowe, aby cecha miała mean=0 i std=1 |
| One-hot encoding | "Tworzenie zmiennych dummy" | Tworzenie jednej kolumny binarnej na kategorię, gdzie dokładnie jedna kolumna jest 1 dla każdego wiersza |
| Target encoding | "Używanie odpowiedzi do kodowania" | Zastępowanie każdej kategorii średnią wartością target dla tej kategorii, z smoothingiem aby zapobiec overfitting |
| TF-IDF | "Wyrafinowane zliczenia słów" | Term Frequency razy Inverse Document Frequency: słowa ważone przez ich charakterystyczność w korpusie |
| Imputation | "Wypełnianie pustych miejsc" | Zastępowanie brakujących wartości szacowanymi wartościami (średnia, mediana, moda lub przewidywane przez model) |
| Feature selection | "Wyrzucanie złych kolumn" | Usuwanie cech, które dodają szum lub redundancję, zatrzymując tylko te z sygnałem o target |
| Mutual information | "Ile jedna rzecz mówi ci o drugiej" | Miara redukcji niepewności o zmiennej Y uzyskanej przez obserwację zmiennej X |
| Data leakage | "Przypadkowe oszustwo" | Używanie informacji podczas treningu, która nie byłaby dostępna w czasie predykcji, dające fałszywie optymistyczne wyniki |

## Dalsze czytanie

- [Feature Engineering and Selection (Max Kuhn & Kjell Johnson)](http://www.feat.engineering/) - darmowa książka online pokrywająca pełne spektrum inżynierii cech
- [scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html) - praktyczne odniesienie dla wszystkich standardowych transformacji
- [Target Encoding Done Right (Micci-Barreca, 2001)](https://dl.acm.org/doi/10.1145/507533.507538) - oryginalny artykuł o target encoding z smoothingiem