# Selekcja cech

> Więcej cech nie znaczy lepiej. Właściwe cechy znaczy lepiej.

**Typ:** Zbuduj
**Język:** Python
**Wymagania wstępne:** Faza 2, Lekcje 01-09, 08 (inżynieria cech)
**Czas:** ~75 minut

## Cele uczenia się

- Zaimplementuj metody filtrujące (próg wariancji, informacja wzajemna, chi-kwadrat) i metody opakowaniowe (RFE, selekcja postępowa) od zera
- Wyjaśnij, dlaczego informacja wzajemna przechwytuje nieliniowe relacje cecha-cechadocelowa, których korelacja nie wykrywa
- Porównaj regularyzację L1 (selekcja osadzona) z RFE (selekcja opakowaniowa) oraz oceń ich kompromisy obliczeniowe
- Zbuduj potok selekcji cech łączący wiele metod i wykaż poprawę uogólniania na danych wstrzymanych

## Problem

Masz 500 cech. Twój model uczy się powoli, przeucza się ciągle i nikt nie potrafi wyjaśnić, czego się nauczył. Dodajesz więcej cech, mając nadzieję na poprawę wyników. Robi się gorzej.

To jest przekleństwo wymiarowości w działaniu. Wraz ze wzrostem liczby cech, objętość przestrzeni cech eksploduje. Punkty danych stają się rozrzedzone. Odległości między punktami zbiegają do siebie. Model potrzebuje wykładniczo więcej danych, aby znaleźć prawdziwe wzorce. Cechy szumowe zagłuszają sygnałowe. Przeuczanie staje się domyślne.

Selekcja cech jest antidotum. Odrzuć szum. Usuń redundancję. Zachowaj cechy, które faktycznie niosą informację o celu. Rezultat: szybsze uczenie, lepsze uogólnianie i modele, które naprawdę możesz wyjaśnić.

Celem nie jest użycie wszystkich dostępnych informacji. Chodzi o użycie właściwych informacji.

## Koncepcja

### Trzy kategorie selekcji cech

Każda metoda selekcji cech należy do jednej z trzech kategorii:

**Metody filtrujące** oceniają każdą cechę niezależnie za pomocą miary statystycznej. Nie używają modelu. Szybkie, ale pomijają interakcje między cechami.

**Metody opakowaniowe** trenują model do oceny podzbiorów cech. Używają wydajności modelu jako wyniku. Lepsze rezultaty, ale kosztowne, ponieważ wymagają wielokrotnego ponownego uczenia modelu.

**Metody osadzone** wybierają cechy jako część uczenia modelu. Regularyzacja L1 redukuje wagi do zera. Drzewa decyzyjne dzielą na najbardziej użytecznych cechach. Selekcja odbywa się podczas dopasowywania, nie jako osobny krok.

### Próg wariancji

Najprostszy filtr. Jeśli cecha prawie nie zmienia się między próbkami, niesie prawie żadnej informacji.

Rozważ cechę, która wynosi 0.0 dla 999 z 1000 próbek. Jej wariancja jest bliska zeru. Żaden model nie może użyć jej do rozróżniania klas. Usuń ją.

Ustaw próg (np. 0.01). Upuść każdą cechę z wariancją poniżej progu. To usuwa stałe lub prawie stałe cechy bez patrzenia na zmienną docelową.

Kiedy używać: jako krok wstępnego przetwarzania przed innymi metodami. Wyłapuje oczywiście bezużyteczne cechy za niemal zerowy koszt.

Ograniczenie: cecha może mieć wysoką wariancję i wciąż być czystym szumem. Próg wariancji jest konieczny, ale niewystarczający.

### Informacja wzajemna

Informacja wzajemna mierzy, ile wiedzy o wartości cechy X zmniejsza niepewność o celu Y.

Jeśli X i Y są niezależne, p(x, y) = p(x) * p(y), więc wyraz logarytmiczny wynosi zero i I(X; Y) = 0. Im więcej X mówi ci o Y, tym wyższa informacja wzajemna.

Kluczowa zaleta nad korelacją: informacja wzajemna przechwytuje nieliniowe relacje. Cechy mogą mieć zerową korelację z celem, ale wysoką informację wzajemną, ponieważ relacja jest kwadratowa lub okresowa.

Dla ciągłych cech, najpierw zdyskretyzuj do przedziałów (estymacja oparta na histogramie). Liczba przedziałów wpływa na oszacowanie -- zbyt mało przedziałów traci informację, zbyt wiele dodaje szum. Częsty wybór: sqrt(n) przedziałów lub reguła Sturgesa (1 + log2(n)).

### Rekursywna eliminacja cech (RFE)

RFE to metoda opakowaniowa. Używa ważności cech samego modelu do iteracyjnego przycinania:

1. Trenuj model ze wszystkimi cechami
2. Ranguj cechy według ważności (współczynniki dla modeli liniowych, redukcja zanieczyszczeń dla drzew)
3. Usuń najmniej ważną cechę(y)
4. Powtarzaj, aż pozostanie pożądana liczba cech

RFE uwzględnia interakcje między cechami, ponieważ model widzi wszystkie pozostałe cechy razem. Usunięcie jednej cechy zmienia ważność innych. To czyni ją bardziej dokładną niż metody filtrujące.

Koszt: trenujesz model N - razy. Z 500 cechami i celem 10, to 490 przebiegów treningowych. Dla kosztownych modeli, to wolne. Możesz przyspieszyć usuwając wiele cech na krok (np. usuń dolne 10% każdej rundy).

### Regularyzacja L1 (Lasso)

Regularyzacja L1 dodaje wartość bezwzględną wag do funkcji straty:

Parametr alpha kontroluje, jak agresywnie cechy są przycinane. Wyższy alpha oznacza więcej wag dokładnie na zero.

Dlaczego dokładnie zero? Kara L1 tworzy diamentowy region ograniczeń w przestrzeni wag. Optymalne rozwiązanie dąży do trafienia w róg diamentu, gdzie jedna lub więcej wag jest zero. Regularyzacja L2 (grzbiet) tworzy okrągły region ograniczeń, gdzie wagi się zmniejszają, ale rzadko osiągają dokładnie zero.

To jest osadzona selekcja cech: model uczy się podczas treningu, które cechy zignorować. Cechy z zerową wagą są efektywnie usunięte.

Zalety: pojedynczy przebieg treningowy, obsługuje skorelowane cechy (wybiera jedną i zeruje inne), wbudowane w większość implementacji modeli liniowych.

Ograniczenie: działa tylko dla modeli liniowych. Nie może przechwycić nieliniowej ważności cech.

### Ważność cech oparta na drzewach

Drzewa decyzyjne i ich zespoły (losowe lasy, gradient boosting) naturalnie rankują cechy. Każde rozgałęzienie redukuje zanieczyszczenie (Gini lub entropia dla klasyfikacji, wariancja dla regresji). Cechy produkujące większe redukcje zanieczyszczeń są ważniejsze.

To daje znormalizowany wynik ważności dla każdej cechy. Automatycznie obsługuje nieliniowe relacje i interakcje między cechami.

Ostrożność: ważność oparta na drzewach jest stronnicza w stronę cech o wielu unikalnych wartościach (wysoka kardynalność). Losowa kolumna ID będzie wyglądać na ważną, ponieważ idealnie dzieli każdą próbkę. Użyj ważności permutacyjnej jako sprawdzenia.

### Ważność permutacyjna

Metoda niezależna od modelu:

1. Trenuj model i zapisz bazową wydajność na danych walidacyjnych
2. Dla każdej cechy: losowo przetasuj jej wartości, zmierz spadek wydajności
3. Im większy spadek, tym ważniejsza cecha

Jeśli tasowanie cechy nie szkodzi wydajności, model od niej nie zależy. Jeśli wydajność się załamuje, ta cecha jest krytyczna.

Ważność permutacyjna unika stronniczości kardynalności ważności opartej na drzewach. Ale jest wolna: jedna pełna ewaluacja na cechę, powtórzona wiele razy dla stabilności.

### Tabela porównawcza

| Metoda | Typ | Szybkość | Nieliniowa | Interakcje cech |
|--------|------|---------|------------|-----------------|
| Próg wariancji | Filtr | Bardzo szybka | Nie | Nie |
| Informacja wzajemna | Filtr | Szybka | Tak | Nie |
| Filtr korelacji | Filtr | Szybka | Nie | Nie |
| RFE | Opakowaniowa | Wolna | Zależy od modelu | Tak |
| L1 / Lasso | Osadzona | Szybka | Nie (liniowa) | Nie |
| Ważność drzew | Osadzona | Średnia | Tak | Tak |
| Ważność permutacyjna | Niezależna od modelu | Wolna | Tak | Tak |

### Schemat decyzyjny

## Zbuduj to

### Krok 1: Generuj syntetyczne dane ze znaną strukturą cech

```python
import numpy as np


def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)

    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)

    informative = np.column_stack([x1, x2, x3, x4, x5])

    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])

    noise = rng.randn(n_samples, 10) * 0.5

    X = np.hstack([informative, correlated, noise])
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)

    feature_names = (
        [f"info_{i}" for i in range(5)]
        + [f"corr_{i}" for i in range(5)]
        + [f"noise_{i}" for i in range(10)]
    )

    return X, y, feature_names
```

Znamy prawdę podstawową: cechy 0-4 są informatywne (plus 3 i 4 są skorelowanymi kopiami 0 i 1), cechy 5-9 są skorelowane z cechami informatywnymi, cechy 10-19 to czysty szum. Dobra metoda selekcji powinna rangować 0-4 najwyżej i 10-19 najniżej.

### Krok 2: Próg wariancji

```python
def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances
```

### Krok 3: Informacja wzajemna (dyskretna)

```python
def discretize(x, n_bins=10):
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned


def mutual_information(X, y, n_bins=10):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples

    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))

        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi

    return mi_scores
```

### Krok 4: Rekursywna eliminacja cech

```python
def simple_logistic_importance(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        w -= lr * (X.T @ error) / n_samples
        b -= lr * np.mean(error)

    return w, b


def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total

    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)

        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)

    for idx in remaining:
        rankings[idx] = 1

    selected_mask = rankings == 1
    return selected_mask, rankings
```

### Krok 5: Selekcja cech L1

```python
def soft_threshold(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y

        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)

        w -= lr * gradient_w
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b

    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w
```

### Krok 6: Ważność oparta na drzewach (proste drzewo decyzyjne)

```python
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_idx):
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0

    best_threshold = None
    best_gain = -1.0
    parent_gini = gini_impurity(y)
    n = len(y)

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            continue

        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[right_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def tree_importance(X, y, n_trees=50, max_depth=5, seed=42):
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    for _ in range(n_trees):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        feature_subset = rng.choice(n_features, size=max(1, int(np.sqrt(n_features))), replace=False)

        X_boot = X[sample_idx]
        y_boot = y[sample_idx]

        tree_imp = _build_tree_importance(X_boot, y_boot, feature_subset, max_depth)
        importances += tree_imp

    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


def _build_tree_importance(X, y, feature_subset, max_depth, depth=0):
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    if depth >= max_depth or len(np.unique(y)) <= 1 or len(y) < 4:
        return importances

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    for f in feature_subset:
        threshold, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_threshold = threshold

    if best_feature is None or best_gain <= 0:
        return importances

    importances[best_feature] += best_gain * len(y)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    importances += _build_tree_importance(X[left_mask], y[left_mask], feature_subset, max_depth, depth + 1)
    importances += _build_tree_importance(X[right_mask], y[right_mask], feature_subset, max_depth, depth + 1)

    return importances
```

### Krok 7: Uruchom wszystkie metody i porównaj

Plik z kodem uruchamia wszystkie pięć metod na tym samym syntetycznym zbiorze danych i drukuje tabelę porównawczą pokazującą, które cechy każda metoda wybiera.

## Użyj tego

Z scikit-learn, selekcja cech jest wbudowana w potok:

```python
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)

mi_scores = mutual_info_classif(X, y)
top_k = np.argsort(mi_scores)[-10:]

rfe_selector = RFE(LogisticRegression(), n_features_to_select=10)
rfe_selector.fit(X, y)
X_rfe = rfe_selector.transform(X)

lasso_selector = SelectFromModel(Lasso(alpha=0.01))
lasso_selector.fit(X, y)
X_lasso = lasso_selector.transform(X)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

Implementacje od zera pokazują dokładnie, co dzieje się wewnątrz każdej metody. Próg wariancji to just computing `var(X, axis=0)` and applying a mask. Informacja wzajemna to counting joint and marginal frequencies in a contingency table. RFE to a loop that trains, ranks, and prunes. L1 to gradient descent with a soft-thresholding step. Ważność drzew accumulates impurity reductions across splits. No magic -- just statistics and loops.

Wersje sklearn dodają odporność (np. mutual_info_classif używa estymacji gęstości k-NN zamiast binning), szybkość (implementacje C) i integrację potokową.

## Wyślij to

Ta lekcja produkuje:
- `outputs/skill-feature-selector.md` -- szybka referencja schematu decyzyjnego dla wyboru właściwej metody selekcji cech

## Ćwiczenia

1. **Selekcja postępowa**: zaimplementuj przeciwieństwo RFE. Zacznij od zero cech. W każdym kroku dodaj cechę, która najbardziej poprawia wydajność modelu. Zatrzymaj się, gdy dodawanie cech już nie pomaga. Porównaj wybrane cechy z wynikami RFE. Która jest szybsza? Która daje lepsze rezultaty?

2. **Selekcja stabilnościowa**: uruchom selekcję cech L1 50 razy, każdorazowo na losowym podzbiorze 80% danych, z nieznacznie różnymi wartościami alpha. Policz, jak często każda cecha jest wybierana. Cechy wybrane w > 80% przebiegów są "stabilne." Porównaj stabilne cechy z selekcją L1 jednorazową. Która jest bardziej niezawodna?

3. **Wykrywanie wielokolinearności**: oblicz macierz korelacji dla wszystkich cech. Zaimplementuj funkcję, która przy danym progu korelacji (np. 0.9), usuwa jedną cechę z każdej silnie skorelowanej pary (zachowując tę z wyższą informacją wzajemną z celem). Przetestuj na syntetycznym zbiorze danych i sprawdź, czy usuwa redundantne skorelowane cechy.

4. **Potok selekcji cech**: połącz próg wariancji, filtr informacji wzajemnej i RFE w jeden potok. Najpierw usuń cechy o wariancji bliskiej zeru, następnie zachowaj górne 50% według informacji wzajemnej, potem uruchom RFE na pozostałych. Porównaj ten potok z samym RFE na wszystkich cechach. Czy potok jest szybszy? Czy jest równie dokładny?

5. **Ważność permutacyjna od zera**: zaimplementuj ważność permutacyjną. Dla każdej cechy przetasuj jej wartości 10 razy, zmierz średni spadek wyniku F1. Porównaj ranking z ważnością opartą na drzewach. Znajdź przypadki, gdzie się różnią i wyjaśnij dlaczego (wskazówka: skorelowane cechy).

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|----------------|----------------------|
| Metoda filtrująca | "Oceniaj cechy niezależnie" | Podejście do selekcji cech, które rankuje cechy używając miary statystycznej bez uczenia modelu, oceniając każdą cechę izolowanie |
| Metoda opakowaniowa | "Użyj modelu do wyboru cech" | Podejście do selekcji cech, które ocenia podzbiory cech trenując model i używając jego wydajności jako kryterium selekcji |
| Metoda osadzona | "Model wybiera cechy podczas uczenia" | Selekcja cech zachodząca jako część dopasowywania modelu, taka jak regularyzacja L1 redukująca wagi do zera |
| Informacja wzajemna | "Ile jedna zmienna mówi ci o drugiej" | Miara redukcji niepewności o Y przy znajomości X, przechwytująca zarówno liniowe jak i nieliniowe zależności |
| Rekursywna eliminacja cech | "Trenuj, ranguj, przycinaj, powtarzaj" | Iteracyjna metoda opakowaniowa, która trenuję model, usuwa najmniej ważną cechę(y) i powtarza aż do osiągnięcia docelowej liczby |
| Regularyzacja L1 / Lasso | "Kara, która zabija cechy" | Dodawanie sumy bezwzględnych wartości wag do funkcji straty, co redukuje wagi nieistotnych cech dokładnie do zera |
| Próg wariancji | "Usuń stałe cechy" | Upuszczanie cech, których wariancja między próbkami spada poniżej określonego progu, filtrując cechy niosące żadnej informacji |
| Ważność cech | "Które cechy mają największe znaczenie" | Wynik wskazujący, ile każda cecha przyczynia się do predykcji modelu, obliczany z zysków rozgałęzień (drzewa) lub wielkości współczynników (liniowe) |
| Ważność permutacyjna | "Tasuj i mierz szkody" | Ewaluowanie ważności cech przez losowe tasowanie wartości każdej cechy i mierzenie wynikowego spadku wydajności modelu |
| Przekleństwo wymiarowości | "Za dużo cech, za mało danych" | Zjawisko, gdzie dodawanie cech zwiększa objętość przestrzeni cech wykładniczo, czyniąc dane rozrzedzonymi i odległości bez znaczenia |

## Dalsza lektura

- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) -- foundational survey on feature selection methods, still widely referenced
- [scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html) -- practical reference for filter, wrapper, and embedded methods with code examples
- [Stability Selection (Meinshausen & Buhlmann, 2010)](https://arxiv.org/abs/0809.2932) -- combines subsampling with feature selection for robust, reproducible results
- [Beware Default Random Forest Importances (Strobl et al., 2007)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25) -- demonstrates the cardinality bias in tree-based importance and proposes conditional importance as an alternative