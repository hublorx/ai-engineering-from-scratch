# Regresja Logistyczna

> Regresja logistyczna zagina linię prostą w kształt litery S, żeby odpowiadać na pytania typu tak lub nie, używając prawdopodobieństw.

**Typ:** Zbuduj
**Języki:** Python
**Wymagania wstępne:** Faza 2 Lekcja 1-2 (Czym jest ML, Regresja Liniowa)
**Czas:** ~90 minut

## Cele uczenia się

- Zaimplementuj regresję logistyczną od zera, używając funkcji sigmoidalnej i straty entropii krzyżowej binarnej
- Obliczaj i interpretuj precision, recall, wynik F1 oraz macierz pomyłek dla klasyfikacji binarnej
- Wyjaśnij, dlaczego MSE nie nadaje się do klasyfikacji i dlaczego binarna entropia krzyżowa tworzy wypukłą powierzchnię kosztu
- Zbuduj model regresji softmax do klasyfikacji wieloklasowej i oceń kompromisy związane z dostrajaniem progu

## Problem

Chcesz przewidzieć, czy guz jest złośliwy czy łagodny na podstawie jego rozmiaru. Próbujesz regresji liniowej. Wydaje ona liczby takie jak 0,3 lub 1,7 lub -0,5. Co one oznaczają? Czy 1,7 oznacza „bardzo złośliwy"? Czy -0,5 oznacza „bardzo łagodny"? Regresja liniowa zwraca nieograniczone liczby. Klasyfikacja wymaga ograniczonych prawdopodobieństw między 0 a 1, a także, jasnej decyzji: tak lub nie.

Regresja logistyczna rozwiązuje ten problem. Pobiera tę samą kombinację liniową (wx + b) i przepuszcza ją przez funkcję sigmoidalną, która kompresuje dowolną liczbę do zakresu (0, 1). Wynik to prawdopodobieństwo. Ustawiasz próg (zwykle 0,5) i podejmujesz decyzję.

Jest to jeden z najczęściej używanych algorytmów w praktyce. Pomimo swojej nazwy, regresja logistyczna jest algorytmem klasyfikacji, nie regresji. Nazwa pochodzi od funkcji logistycznej (sigmoidalnej), której używa.

## Koncepcja

### Dlaczego regresja liniowa nie nadaje się do klasyfikacji

Wyobraź sobie przewidywanie zaliczenia/niezaliczenia (1/0) na podstawie godzin nauki. Regresja liniowa dopasowuje linię do danych:

```
hours:  1   2   3   4   5   6   7   8   9   10
actual: 0   0   0   0   1   1   1   1   1   1
```

Liniowe dopasowanie może generować predykcje takie jak -0,2 przy 1 godzinie i 1,3 przy 10 godzinach. Te wartości nie są prawdopodobieństwami. Schodzą poniżej 0 i powyżej 1. Co gorsza, pojedynczy element odstający, (ktoś, kto uczył się 50 godzin,) przeciągnąłoby całą linię, zmieniając predykcje dla wszystkich.

Klasyfikacja wymaga funkcji, która:
- Zwraca wartości między 0 a 1 (prawdopodobieństwa)
- Tworzy ostre przejście (granicę decyzji)
- Nie jest zniekształcana przez elementy odstające daleko od granicy

### Funkcja sigmoidalna

Funkcja sigmoidalna robi dokładnie to:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Właściwości:
- Gdy z jest duże i dodatnie, sigmoid(z) zbliża się do 1
- Gdy z jest duże i ujemne, sigmoid(z) zbliża się do 0
- Gdy z = 0, sigmoid(z) = 0,5
- Wynik jest zawsze między 0 a 1
- Funkcja jest gładka i różniczkowalna wszędzie

Pochodna ma wygodny kształt: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)). To sprawia, że obliczanie gradientu jest efektywne.

### Regresja logistyczna = model liniowy + sigmoid

Model oblicza z = wx + b (tak jak w regresji liniowej), a następnie stosuje sigmoid:

```mermaid
flowchart LR
    X[Input features x] --> L["Linear: z = wx + b"]
    L --> S["Sigmoid: p = 1/(1+e^-z)"]
    S --> D{"p >= 0.5?"}
    D -->|Yes| P[Predict 1]
    D -->|No| N[Predict 0]
```

Wynik p jest interpretowany jako P(y=1 | x), czyli prawdopodobieństwo, że wejście należy do klasy 1. Granica decyzji to miejsce, gdzie wx + b = 0, co sprawia, że sigmoid zwraca dokładnie 0,5.

### Strata entropii krzyżowej binarnej

Nie możesz użyć MSE dla regresji logistycznej. MSE z sigmoidalną funkcją aktywacji tworzy niewypukłą powierzchnię kosztu z wieloma minimami lokalnymi. Zamiast tego użyj entropii krzyżowej binarnej (log loss):

```
Loss = -(1/n) * sum(y * log(p) + (1-y) * log(1-p))
```

Dlaczego to działa:
- Gdy y=1 i p jest bliskie 1: log(1) = 0, więc strata jest bliska 0 (poprawnie, niski koszt)
- Gdy y=1 i p jest bliskie 0: log(0) dąży do ujemnej nieskończoności, więc strata jest ogromna (błędnie, wysoki koszt)
- Gdy y=0 i p jest bliskie 0: log(1) = 0, więc strata jest bliska 0 (poprawnie, niski koszt)
- Gdy y=0 i p jest bliskie 1: log(0) dąży do ujemnej nieskończoności, więc strata jest ogromna (błędnie, wysoki koszt)

Ta funkcja straty jest wypukła dla regresji logistycznej, co gwarantuje istnienie pojedynczego minimum globalnego.

### Spadek gradientu dla regresji logistycznej

Gradienty dla entropii krzyżowej binarnej z sigmoidalną funkcją aktywacji mają czystą postać:

```
dL/dw = (1/n) * sum((p - y) * x)
dL/db = (1/n) * sum(p - y)
```

Wyglądają identycznie jak gradienty regresji liniowej. Różnica polega na tym, że p = sigmoid(wx + b) zamiast p = wx + b. Sigmoid wprowadza nieliniowość, ale reguła aktualizacji gradientu pozostaje taka sama.

```mermaid
flowchart TD
    A[Initialize w=0, b=0] --> B[Forward pass: z = wx+b, p = sigmoid z]
    B --> C[Compute loss: binary cross-entropy]
    C --> D["Compute gradients: dw = (1/n) * sum((p-y)*x)"]
    D --> E[Update: w = w - lr*dw, b = b - lr*db]
    E --> F{Converged?}
    F -->|No| B
    F -->|Yes| G[Model trained]
```

### Granica decyzji

Dla dwuwymiarowego wejścia (dwie cechy), granica decyzji to linia, gdzie:

```
w1*x1 + w2*x2 + b = 0
```

Punkty po jednej stronie są klasyfikowane jako 1, punkty po drugiej stronie jako 0. Regresja logistyczna zawsze tworzy liniową granicę decyzji. Jeśli potrzebujesz zakrzywionej granicy, możesz, albo dodać wielomianowe cechy, albo użyć nieliniowego modelu.

### Klasyfikacja wieloklasowa z softmax

Binarna regresja logistyczna obsługuje dwie klasy. Dla k klas użyj funkcji softmax:

```
softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)
```

Każda klasa ma swój własny wektor wag. Model oblicza wynik z_i dla każdej klasy, a następnie, softmax konwertuje wyniki na prawdopodobieństwa sumujące się do 1. Przewidywana klasa to ta z najwyższym prawdopodobieństwem.

Funkcja straty staje się entropią krzyżową kategoryczną:

```
Loss = -(1/n) * sum(sum(y_k * log(p_k)))
```

gdzie y_k to 1 dla prawdziwej klasy i 0 dla wszystkich pozostałych (kodowanie one-hot).

### Metryki oceny

Sama dokładność nie wystarczy. Dla zbioru danych z 95% negatywnych i 5% pozytywnych, model, który zawsze przewiduje negatywne, uzyskuje 95% dokładności, ale jest bezużyteczny.

**Macierz pomyłek**:

| | Przewidywany pozytywny | Przewidywany negatywny |
|---|---|---|
| Faktycznie pozytywny | True Positive (TP) | False Negative (FN) |
| Faktycznie negatywny | False Positive (FP) | True Negative (TN) |

**Precision**: spośród wszystkich przewidywanych pozytywnych, ile faktycznie jest pozytywnych?
```
Precision = TP / (TP + FP)
```

**Recall** (Czułość): spośród wszystkich faktycznie pozytywnych, ile złapaliśmy?
```
Recall = TP / (TP + FN)
```

**Wynik F1**: średnia harmoniczna precision i recall. Równoważy obie metryki.
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Kiedy priorytetyzować:
- **Precision**: gdy fałszywie pozytywne są kosztowne (filtr spamu, nie chcesz blokować legalnej poczty)
- **Recall**: gdy fałszywie negatywne są kosztowne (badania przesiewowe raka, nie chcesz przeoczyć guza)
- **F1**: gdy potrzebujesz pojedynczej zrównoważonej metryki

## Zbuduj to

### Krok 1: funkcja sigmoidalna i generowanie danych

```python
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")
```

### Krok 2: regresja logistyczna od zera

```python
class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("\n=== Training Logistic Regression ===")
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
print(f"Bias: {model.bias:.4f}")
```

### Krok 3: macierz pomyłek i metryki od zera

```python
class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


y_pred_test = [model.predict(x) for x in X_test]
print("\n=== Classification Report (Test Set) ===")
metrics = ClassificationMetrics(y_test, y_pred_test)
metrics.print_report()
```

### Krok 4: analiza granicy decyzji

```python
print("\n=== Decision Boundary ===")
w1, w2 = model.weights
b = model.bias
print(f"Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Solved for x2:     x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

print("\nSample predictions near the boundary:")
test_points = [
    [3.0, 3.0],
    [3.5, 3.5],
    [4.0, 4.0],
    [2.5, 2.5],
    [5.0, 5.0],
]
for point in test_points:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f"  [{point[0]}, {point[1]}] -> prob={prob:.4f}, class={pred}")
```

### Krok 5: wieloklasowa klasyfikacja z softmax

```python
class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0
            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


random.seed(42)
X_3class = []
y_3class = []

centers = [(1, 1), (5, 1), (3, 5)]
for label, (cx, cy) in enumerate(centers):
    for _ in range(50):
        X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
        y_3class.append(label)

combined = list(zip(X_3class, y_3class))
random.shuffle(combined)
X_3class, y_3class = zip(*combined)
X_3class = list(X_3class)
y_3class = list(y_3class)

split_3 = int(0.8 * len(X_3class))
X_train_3 = X_3class[:split_3]
y_train_3 = y_3class[:split_3]
X_test_3 = X_3class[split_3:]
y_test_3 = y_3class[split_3:]

print("\n=== Multi-class Softmax Regression (3 classes) ===")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

print("\nSample predictions:")
for i in range(5):
    probs = softmax_model.predict_proba(X_test_3[i])
    pred = softmax_model.predict(X_test_3[i])
    print(f"  True: {y_test_3[i]}, Predicted: {pred}, Probs: [{', '.join(f'{p:.3f}' for p in probs)}]")
```

### Krok 6: dostrajanie progu

```python
print("\n=== Threshold Tuning ===")
print("Default threshold: 0.5. Adjusting the threshold trades precision for recall.\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in thresholds:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")
```

## Użyj tego

Teraz to samo z scikit-learn.

```python
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_0 = np.random.randn(100, 2) + [2, 2]
X_1 = np.random.randn(100, 2) + [5, 5]
X_sk = np.vstack([X_0, X_1])
y_sk = np.array([0] * 100 + [1] * 100)

X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = SklearnLR()
lr.fit(X_tr_sc, y_tr)
y_pred = lr.predict(X_te_sc)

print("=== Scikit-learn Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_te, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")
```

Twoja implementacja od zera tworzy tę samą granicę decyzji i metryki. Scikit-learn dodaje opcje solverów (liblinear, lbfgs, saga), automatyczną regularyzację, strategie wieloklasowe (one-vs-rest, multinomial) i optymalizacje stabilności numerycznej.

## Wyślij to

Ta lekcja tworzy:
- `code/logistic_regression.py` - regresja logistyczna od zera z metrykami

## Ćwiczenia

1. Wygeneruj zbiór danych, który NIE jest liniowo separowalny (np. dwa współśrodkowe okręgi). Wytrenuj regresję logistyczną i obserwuj jej porażkę. Następnie dodaj wielomianowe cechy (x1^2, x2^2, x1*x2) i trenuj ponownie. Pokaż, że dokładność się poprawia.
2. Zaimplementuj wieloklasową macierz pomyłek dla modelu softmax z 3 klasami. Oblicz precision i recall dla każdej klasy. Która klasa jest najtrudniejsza do sklasyfikowania?
3. Zbuduj krzywą ROC od zera. Dla 100 wartości progu od 0 do 1 oblicz współczynnik prawdziwie pozytywnych i współczynnik fałszywie pozytywnych. Oblicz AUC (pole pod krzywą) używając reguły trapezoidalnej.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|----------------|----------------------|
| Logistic regression | „Regresja do klasyfikacji" | Model liniowy z sigmoidalną funkcją aktywacji, który zwraca prawdopodobieństwa klas |
| Sigmoid function | „Funkcja w kształcie S" | Funkcja 1/(1+e^(-z)), która mapuje dowolną liczbę rzeczywistą na zakres (0, 1) |
| Binary cross-entropy | „Log loss" | Funkcja straty -[y*log(p) + (1-y)*log(1-p)], która surowo karze pewne błędne predykcje |
| Decision boundary | „Linia podziału" | Powierzchnia, gdzie wynik modelu osiąga prawdopodobieństwo 0,5, oddzielając przewidywane klasy |
| Softmax | „Sigmoid dla wielu klas" | Funkcja, która konwertuje wektor wyników na prawdopodobieństwa sumujące się do 1 |
| Precision | „Ile wybranych jest istotnych" | TP / (TP + FP), frakcja pozytywnych predykcji, które faktycznie są pozytywne |
| Recall | „Ile istotnych jest wybranych" | TP / (TP + FN), frakcja faktycznie pozytywnych, które model poprawnie identyfikuje |
| F1 score | „Zrównoważona dokładność" | Średnia harmoniczna precision i recall: 2*P*R / (P+R) |
| Confusion matrix | „Rozbicie błędów" | Tabela pokazująca liczby TP, TN, FP, FN dla każdej pary klas |
| Threshold | „Wartość odcięcia" | Wartość prawdopodobieństwa, powyżej której model przewiduje klasę 1 (domyślnie 0,5, konfigurowalna) |
| One-hot encoding | „Binarne kolumny dla kategorii" | Reprezentacja klasy k jako wektora zer z 1 na pozycji k |
| Categorical cross-entropy | „Log loss dla wielu klas" | Rozszerzenie binarnej entropii krzyżowej na k klas używające etykiet kodowanych one-hot |