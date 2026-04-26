# Regresja liniowa

> Regresja liniowa rysuje najlepszą prostą przez twoje dane. To jest "hello world" uczenia maszynowego.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 1 (Algebra liniowa, Rachunek różniczkowy, Optymalizacja), Lekcja Fazy 2 nr 1
**Czas:** ~90 minut

## Cele uczenia się

- Odtwórz reguły aktualizacji spadku gradientu dla błędu średniokwadratowego i zaimplementuj regresję liniową od podstaw
- Porównaj spadek gradientu z równaniem normalnym pod względem złożoności obliczeniowej i określ, kiedy stosować każde z nich
- Zbuduj model wielokrotnej regresji liniowej ze standaryzacją cech i zinterpretuj nauczone wagi
- Wyjaśnij, jak regresja grzbietowa (regularyzacja L2) zapobiega nadmiernemu dopasowaniu poprzez karanie dużych wag

## Problem

Masz dane: wielkości domów i ich ceny sprzedaży. Chcesz przewidzieć cenę nowego domu na podstawie jego wielkości. Możesz to oszacować na oko na wykresie rozproszenia, ale potrzebujesz wzoru. Potrzebujesz linii, która najlepiej dopasuje się do danych, abyś mógł podstawić dowolną wielkość i uzyskać przewidywaną cenę.

Regresja liniowa daje ci tę linię. Co ważniejsze, wprowadza całą pętlę treningową ML: zdefiniuj model, zdefiniuj funkcję kosztu, optymalizuj parametry. Każdy algorytm ML podąża za tym samym wzorem. Opanuj to tutaj w najprostszym przypadku, a rozpoznasz to wszędzie.

To nie jest tylko do prostych problemów. Regresja liniowa jest używana w systemach produkcyjnych do prognozowania popytu, analizy testów A/B, modelowania finansowego i jako punkt wyjścia dla każdego zadania regresji.

## Koncepcja

### Model

Regresja liniowa zakłada liniową zależność między wejściem (x) a wyjściem (y):

```
y = wx + b
```

- `w` (waga/nachylenie): o ile y zmienia się, gdy x wzrasta o 1
- `b` (obciążenie/wyraz wolny): wartość y gdy x = 0

Dla wielu wejść (cech), rozszerza się to do:

```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

Albo w formie wektorowej: `y = w^T * x + b`

Cel: znajdź wartości w i b, które sprawiają, że przewidywane y jest jak najbliższe rzeczywistemu y we wszystkich przykładach treningowych.

### Funkcja kosztu (Błąd średniokwadratowy)

Jak mierzysz "jak najbliżej"? Potrzebujesz jednej liczby, która uchwyci, jak bardzo mylne są twoje przewidywania. Najczęstszym wyborem jest Błąd średniokwadratowy (MSE):

```
MSE = (1/n) * sum((y_predicted - y_actual)^2)
```

Dlaczego kwadrat? Dwa powody. Po pierwsze, kara duże błędy bardziej niż małe (błąd 10 jest 100 razy gorszy niż błąd 1, nie 10 razy). Po drugie, funkcja kwadratowa jest gładka i różniczkowalna wszędzie, co sprawia, że optymalizacja jest prosta.

Funkcja kosztu tworzy powierzchnię. Dla pojedynczej wagi w i obciążenia b, powierzchnia MSE wygląda jak miska (wypukły paraboloid). Dno miski to miejsce, gdzie MSE jest minimalizowane. Trening oznacza znalezienie tego dna.

### Spadek gradientu

Spadek gradientu znajduje dno miski, robiąc kroki w dół.

```mermaid
flowchart TD
    A[Initialize w and b randomly] --> B[Compute predictions: y_hat = wx + b]
    B --> C[Compute cost: MSE]
    C --> D[Compute gradients: dMSE/dw, dMSE/db]
    D --> E[Update parameters]
    E --> F{Cost low enough?}
    F -->|No| B
    F -->|Yes| G[Done: optimal w and b found]
```

Gradienty mówią ci dwie rzeczy: w którym kierunku przesunąć każdy parametr i o ile.

Dla MSE z y_hat = wx + b:

```
dMSE/dw = (2/n) * sum((y_hat - y) * x)
dMSE/db = (2/n) * sum(y_hat - y)
```

Reguła aktualizacji:

```
w = w - learning_rate * dMSE/dw
b = b - learning_rate * dMSE/db
```

Współczynnik uczenia się kontroluje rozmiar kroku. Zbyt duży: przekraczasz minimum i rozbiegasz się. Zbyt mały: trening trwa wiecznie. Typowe wartości początkowe: 0,01, 0,001 lub 0,0001.

### Równanie normalne (rozwiązanie w formie zamkniętej)

Dla regresji liniowej konkretnie, istnieje bezpośredni wzór, który daje optymalne wagi bez żadnej iteracji:

```
w = (X^T * X)^(-1) * X^T * y
```

To odwraca macierz, aby rozwiązać dla w w jednym kroku. Działa idealnie dla małych zbiorów danych. Dla dużych zbiorów danych (miliony wierszy lub tysiące cech), preferowany jest spadek gradientu, ponieważ odwracanie macierzy ma złożoność O(n^3) względem liczby cech.

### Wielokrotna regresja liniowa

Z wieloma cechami, model staje się:

```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

Wszystko działa tak samo: MSE jest funkcją kosztu, spadek gradientu aktualizuje wszystkie wagi jednocześnie. Jedyna różnica polega na tym, że dopasowujesz hiperpłaszczyznę zamiast linii.

Skalowanie cech ma tutaj znaczenie. Jeśli jedna cecha mieści się w zakresie od 0 do 1, a inna od 0 do 1 000 000, spadek gradientu będzie zmagać się, bo powierzchnia kosztu staje się wydłużona. Standaryzuj cechy (odejmij średnią, podziel przez odchylenie standardowe) przed treningiem.

### Regresja wielomianowa

Co jeśli zależność nie jest liniowa? Możesz nadal używać regresji liniowej, tworząc cechy wielomianowe:

```
y = w1*x + w2*x^2 + w3*x^3 + b
```

To nadal jest "liniowa" regresja, bo model jest liniowy względem wag (w1, w2, w3). Po prostu używasz nieliniowych cech x.

Wielomiany wyższego stopnia mogą dopasować bardziej złożone krzywe, ale ryzykują nadmierne dopasowanie. Wielomian stopnia 10 przejdzie przez każdy punkt w 10-punktowym zbiorze danych, ale będzie słabo przewidywać na nowych danych.

### Współczynnik R-kwadrat

MSE mówi ci, jak bardzo się mylisz, ale liczba zależy od skali y. R-kwadrat (R^2) daje miarę niezależną od skali:

```
R^2 = 1 - (sum of squared residuals) / (sum of squared deviations from mean)
    = 1 - SS_res / SS_tot
```

- R^2 = 1,0: doskonałe przewidywania
- R^2 = 0,0: model nie jest lepszy niż przewidywanie średniej za każdym razem
- R^2 < 0,0: model jest gorszy niż przewidywanie średniej

### Podgląd regularyzacji (Regresja grzbietowa)

Gdy masz wiele cech, model może nadmiernie dopasować się, przypisując duże wagi. Regresja grzbietowa (regularyzacja L2) dodaje karę:

```
Cost = MSE + lambda * sum(w_i^2)
```

Wyraz kary zniechęca do dużych wag. Hiperparametr lambda kontroluje kompromis: wyższa lambda oznacza mniejsze wagi i więcej regularyzacji. To jest omówione szczegółowo w późniejszej lekcji. Na razie wiedz, że istnieje i dlaczego pomaga.

## Zbuduj to

### Krok 1: Generuj przykładowe dane

```python
import random
import math

random.seed(42)

TRUE_W = 3.0
TRUE_B = 7.0
N_SAMPLES = 100

X = [random.uniform(0, 10) for _ in range(N_SAMPLES)]
y = [TRUE_W * x + TRUE_B + random.gauss(0, 2.0) for x in X]

print(f"Generated {N_SAMPLES} samples")
print(f"True relationship: y = {TRUE_W}x + {TRUE_B} (+ noise)")
print(f"First 5 points: {[(round(X[i], 2), round(y[i], 2)) for i in range(5)]}")
```

### Krok 2: Regresja liniowa od podstaw ze spadkiem gradientu

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n
        return cost

    def compute_gradients(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        dw = (2 / n) * sum((pred - actual) * x for pred, actual, x in zip(predictions, y, X))
        db = (2 / n) * sum(pred - actual for pred, actual in zip(predictions, y))
        return dw, db

    def fit(self, X, y, epochs=1000, print_every=200):
        for epoch in range(epochs):
            dw, db = self.compute_gradients(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


print("=== Training Linear Regression (Gradient Descent) ===")
model = LinearRegression(learning_rate=0.005)
model.fit(X, y, epochs=1000, print_every=200)
print(f"\nLearned: y = {model.w:.4f}x + {model.b:.4f}")
print(f"True:    y = {TRUE_W}x + {TRUE_B}")
print(f"R-squared: {model.r_squared(X, y):.4f}")
```

### Krok 3: Równanie normalne (rozwiązanie w formie zamkniętej)

```python
class LinearRegressionNormal:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        n = len(X)
        x_mean = sum(X) / n
        y_mean = sum(y) / n
        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((X[i] - x_mean) ** 2 for i in range(n))
        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean
        return self

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


print("\n=== Normal Equation (Closed-Form) ===")
model_normal = LinearRegressionNormal()
model_normal.fit(X, y)
print(f"Learned: y = {model_normal.w:.4f}x + {model_normal.b:.4f}")
print(f"R-squared: {model_normal.r_squared(X, y):.4f}")
```

### Krok 4: Wielokrotna regresja liniowa

```python
class MultipleLinearRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        return sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


random.seed(42)
N = 100
X_multi = []
y_multi = []
for _ in range(N):
    size = random.uniform(500, 3000)
    bedrooms = random.randint(1, 5)
    age = random.uniform(0, 50)
    price = 50 * size + 10000 * bedrooms - 1000 * age + 50000 + random.gauss(0, 20000)
    X_multi.append([size, bedrooms, age])
    y_multi.append(price)


def standardize(X):
    n_features = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(n_features)]
    stds = []
    for j in range(n_features):
        variance = sum((X[i][j] - means[j]) ** 2 for i in range(len(X))) / len(X)
        stds.append(variance ** 0.5)
    X_scaled = []
    for i in range(len(X)):
        row = [(X[i][j] - means[j]) / stds[j] if stds[j] > 0 else 0 for j in range(n_features)]
        X_scaled.append(row)
    return X_scaled, means, stds


y_mean_val = sum(y_multi) / len(y_multi)
y_std_val = (sum((yi - y_mean_val) ** 2 for yi in y_multi) / len(y_multi)) ** 0.5
y_scaled = [(yi - y_mean_val) / y_std_val for yi in y_multi]

X_scaled, x_means, x_stds = standardize(X_multi)

print("\n=== Multiple Linear Regression (3 features) ===")
print("Features: house size, bedrooms, age")
multi_model = MultipleLinearRegression(n_features=3, learning_rate=0.01)
multi_model.fit(X_scaled, y_scaled, epochs=1000, print_every=200)

print(f"\nWeights (standardized): {[round(w, 4) for w in multi_model.weights]}")
print(f"Bias (standardized): {multi_model.bias:.4f}")
print(f"R-squared: {multi_model.r_squared(X_scaled, y_scaled):.4f}")
```

### Krok 5: Regresja wielomianowa

```python
class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.01):
        self.degree = degree
        self.weights = [0.0] * degree
        self.bias = 0.0
        self.lr = learning_rate

    def make_features(self, X):
        return [[x ** (d + 1) for d in range(self.degree)] for x in X]

    def predict(self, X):
        features = self.make_features(X)
        return [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]

    def fit(self, X, y, epochs=1000, print_every=200):
        features = self.make_features(X)
        n = len(y)
        for epoch in range(epochs):
            predictions = [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(self.degree):
                grad = (2 / n) * sum(errors[i] * features[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost = sum(e ** 2 for e in errors) / n
                print(f"  Epoch {epoch:4d} | Cost: {cost:.6f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


random.seed(42)
X_poly = [x / 10.0 for x in range(0, 50)]
y_poly = [0.5 * x ** 2 - 2 * x + 3 + random.gauss(0, 1.0) for x in X_poly]

x_max = max(abs(x) for x in X_poly)
X_poly_norm = [x / x_max for x in X_poly]
y_poly_mean = sum(y_poly) / len(y_poly)
y_poly_std = (sum((yi - y_poly_mean) ** 2 for yi in y_poly) / len(y_poly)) ** 0.5
y_poly_norm = [(yi - y_poly_mean) / y_poly_std for yi in y_poly]

print("\n=== Polynomial Regression (degree 2 vs degree 5) ===")
print("True relationship: y = 0.5x^2 - 2x + 3")

print("\nDegree 2:")
poly2 = PolynomialRegression(degree=2, learning_rate=0.1)
poly2.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly2.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 5:")
poly5 = PolynomialRegression(degree=5, learning_rate=0.1)
poly5.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly5.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 2 fits the true curve well. Degree 5 fits training data slightly better")
print("but risks overfitting on new data.")
```

### Krok 6: Regresja grzbietowa (regularyzacja L2)

```python
class RidgeRegression:
    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            mse = sum(e ** 2 for e in errors) / n
            reg_term = self.alpha * sum(w ** 2 for w in self.weights)
            cost = mse + reg_term
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                grad += 2 * self.alpha * self.weights[j]
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | L2 penalty: {reg_term:.4f}")
        return self


print("\n=== Ridge Regression (L2 Regularization) ===")
print("Same data as multiple regression, with alpha=0.1")
ridge = RidgeRegression(n_features=3, learning_rate=0.01, alpha=0.1)
ridge.fit(X_scaled, y_scaled, epochs=1000, print_every=200)
print(f"\nRidge weights: {[round(w, 4) for w in ridge.weights]}")
print(f"Plain weights: {[round(w, 4) for w in multi_model.weights]}")
print("Ridge weights are smaller (shrunk toward zero) due to the L2 penalty.")
```

## Użyj tego

Teraz to samo z scikit-learn, który jest tym, czego będziesz faktycznie używać w produkcji.

```python
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

np.random.seed(42)
X_sk = np.random.uniform(0, 10, (100, 1))
y_sk = 3.0 * X_sk.squeeze() + 7.0 + np.random.normal(0, 2.0, 100)

X_train, X_test, y_train, y_test = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

lr = SklearnLR()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("=== Scikit-learn Linear Regression ===")
print(f"Coefficient (w): {lr.coef_[0]:.4f}")
print(f"Intercept (b): {lr.intercept_:.4f}")
print(f"R-squared (test): {r2_score(y_test, y_pred):.4f}")
print(f"MSE (test): {mean_squared_error(y_test, y_pred):.4f}")

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_sk = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

lr_poly = SklearnLR()
lr_poly.fit(X_poly_sk, y_train)
print(f"\nPolynomial degree 2 R-squared: {r2_score(y_test, lr_poly.predict(X_poly_test)):.4f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print(f"Ridge R-squared: {r2_score(y_test, ridge.predict(X_test_scaled)):.4f}")
print(f"Ridge coefficient: {ridge.coef_[0]:.4f}")
```

Twoja implementacja od podstaw i scikit-learn dają te same wyniki. Różnica: scikit-learn obsługuje przypadki brzegowe, stabilność numeryczną i optymalizacje wydajności. Używaj biblioteki w produkcji. Używaj wersji od podstaw, aby zrozumieć, co się dzieje.

## Wyślij to

Ta lekcja tworzy:
- `outputs/skill-regression.md` - umiejętność wyboru odpowiedniego podejścia regresji na podstawie problemu

## Ćwiczenia

1. Zaimplementuj wsadowy spadek gradientu, stochastyczny spadek gradientu (SGD) i mini-batch spadek gradientu. Porównaj szybkość konwergencji na tym samym zbiorze danych. Który zbiega najszybciej? Który ma najgładszą krzywą kosztu?

2. Generuj dane z funkcji sześciennej (y = ax^3 + bx^2 + cx + d + szum). Dopasuj wielomiany stopnia 1, 3 i 10. Porównaj R^2 treningowy i testowy. Przy jakim stopniu nadmierne dopasowanie staje się oczywiste?

3. Zaimplementuj regresję Lasso (regularyzacja L1: kara = alpha * sum(|w_i|)). Trenuj na danych mieszkaniowych z wieloma cechami. Porównaj, które wagi dążą do zera w porównaniu z Ridge. Dlaczego L1 produkuje rzadkie rozwiązania, a L2 nie?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|----------------|----------------------|
| Regresja liniowa | "Narysuj linię przez dane" | Znajdź wagę w i obciążenie b, które minimalizują sumę kwadratów różnic między wx+b a rzeczywistymi wartościami y |
| Funkcja kosztu | "Jak zły jest model" | Funkcja, która odwzorowuje parametry modelu na pojedynczą liczbę mierzącą błąd przewidywania, którą optymalizacja minimalizuje |
| Błąd średniokwadratowy | "Średnia kwadratów błędów" | (1/n) * suma (przewidywane - rzeczywiste)^2, karająca duże błędy nieproporcjonalnie |
| Spadek gradientu | "Idź w dół" | Iteracyjnie dostosowuj parametry w kierunku, który zmniejsza funkcję kosztu, używając pochodnych cząstkowych |
| Współczynnik uczenia się | "Rozmiar kroku" | Skalar kontrolujący, ile parametry zmieniają się na każdym kroku spadku gradientu |
| Równanie normalne | "Rozwiąż to bezpośrednio" | Rozwiązanie w formie zamkniętej w = (X^T X)^-1 X^T y, które daje optymalne wagi bez iteracji |
| R-kwadrat | "Jak dobre jest dopasowanie" | Ułamek wariancji y wyjaśnionej przez model, od minus nieskończoności do 1,0 |
| Skalowanie cech | "Spraw, żeby cechy były porównywalne" | Przekształcanie cech do podobnych zakresów (np. zero średnia, wariancja jednostkowa), żeby spadek gradientu szybciej zbiegał |
| Regularyzacja | "Karaż złożoność" | Dodawanie wyrazu do funkcji kosztu, który zmniejsza wagi, zapobiegając nadmiernemu dopasowaniu |
| Regresja grzbietowa | "Regularyzacja L2" | Regresja liniowa z karą lambda * sum(w_i^2) dodaną do MSE |
| Regresja wielomianowa | "Dopasowywanie krzywych matematyką liniową" | Regresja liniowa na cechach wielomianowych (x, x^2, x^3, ...), nadal liniowa względem wag |
| Nadmierne dopasowanie | "Zapamiętywanie danych treningowych" | Używanie modelu tak złożonego, że dopasowuje szum w danych treningowych i zawodzi na nowych danych |

## Dalsze czytanie

- [An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/) -- darmowy PDF, rozdziały 3 i 6 obejmują regresję liniową i regularyzację z praktycznymi przykładami w R
- [The Elements of Statistical Learning (ESL)](https://hastie.su.domains/ElemStatLearn/) -- darmowy PDF, bardziej matematyczny towarzysz ISLR z głębszym omówieniem ridge i lasso
- [Stanford CS229 Lecture Notes on Linear Regression](https://cs229.stanford.edu/main_notes.pdf) -- notatki Andrew'ego Ng, które odtwarzają równanie normalne i spadek gradientu od podstaw
- [scikit-learn LinearRegression documentation](https://scikit-learn.org/stable/modules/linear_model.html) -- praktyczne odniesienie dla LinearRegression, Ridge, Lasso i ElasticNet z przykładami kodu