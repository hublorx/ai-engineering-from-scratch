# Wektory, Macierze i Operacje

> Każda sieć neuronowa to po prostu mnożenie macierzy z dodatkowymi krokami.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Intuicja algebry liniowej)
**Time:** ~60 minutes

## Cele uczenia się

- Zbuduj klasę Matrix z operacjami elementowymi, mnożeniem macierzy, transpozycją, wyznacznikiem i odwrotnością
- Rozróżniaj mnożenie elementowe od mnożenia macierzy i wyjaśniaj, kiedy każde się stosuje
- Zaimplementuj pojedynczą gęstą warstwę sieci neuronowej (`relu(W @ x + b)`) używając tylko klasy Matrix od zera
- Wyjaśnij reguły broadcasting i jak działa dodawanie biasu w frameworkach sieci neuronowych

## Problem

Chcesz zbudować sieć neuronową. Czytasz kod i widzisz:

```
output = activation(weights @ input + bias)
```

Ten `@` to mnożenie macierzy. `weights` to macierz. `input` to wektor. Jeśli nie wiesz, co te operacje robią, ta linia to magia. Jeśli wiesz, to cały forward pass warstwy w trzech operacjach.

Każdy obraz przetwarzany przez twój model to macierz wartości pikseli. Każdy word embedding to wektor. Każda warstwa każdej sieci neuronowej to transformacja macierzowa. Nie możesz budować systemów AI bez biegłości w operacjach macierzowych tak, jak nie możesz pisać kodu bez rozumienia zmiennych.

Ta lekcja buduje tę biegłość od zera.

## Koncepcja

### Wektory: uporządkowane listy liczb

Wektor to lista liczb z kierunkiem i magnitudą. W AI wektory reprezentują punkty danych, cechy lub parametry.

```
v = [3, 4]        -- wektor 2D
w = [1, 0, -2]    -- wektor 3D
```

Wektor 2D `[3, 4]` wskazuje na współrzędne (3, 4) na płaszczyźnie. Jego długość (magnituda) wynosi 5 (trójkąt 3-4-5).

### Macierze: siatki liczb

Macierz to 2D siatka. Wiersze i kolumny. Macierz m x n ma m wierszy i n kolumn.

```
A = | 1  2  3 |     -- macierz 2x3 (2 wiersze, 3 kolumny)
    | 4  5  6 |
```

W sieciach neuronowych macierze wag przekształcają wektory wejściowe w wektory wyjściowe. Warstwa z 784 wejściami i 128 wyjściami używa macierzy wag 128x784.

### Dlaczego kształty mają znaczenie

Mnożenie macierzy ma ścisłą regułę: `(m x n) @ (n x p) = (m x p)`. Wewnętrzne wymiary muszą się zgadzać.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Wewnętrzne wymiary: 784 = 784  -- prawidłowe
```

Jeśli dostaniesz błąd niezgodności kształtów w PyTorch, to dlatego.

### Mapa operacji

| Operacja | Co robi | Zastosowanie w sieci neuronowej |
|-----------|-------------|-------------------|
| Dodawanie | Łączenie elementowe | Dodawanie biasu do wyjścia |
| Mnożenie skalarne | Skalowanie każdego elementu | learning_rate * gradients |
| Mnożenie macierzy | Transformacja wektorów | Forward pass warstwy |
| Transpozycja | Odwracanie wierszy i kolumn | Backpropagation |
| Wyznacznik | Pojedyncza liczba podsumowująca | Sprawdzanie odwracalności |
| Odwrotność | Cofnięcie transformacji | Rozwiązywanie układów liniowych |
| Macierz tożsamości | Macierz nic-nie-robienia | Inicjalizacja, połączenia rezydualne |

### Elementowe vs mnożenie macierzy

Ta distincja sprawia, że początkujący często się mylą.

Elementowe: mnożenie pasujących pozycji. Obie macierze muszą mieć ten sam kształt.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Mnożenie macierzy: iloczyny skalarne wierszy i kolumn. Wewnętrzne wymiary muszą się zgadzać.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Różne operacje, różne wyniki, różne reguły.

### Broadcasting

Gdy dodajesz wektor bias do macierzy wyjść, kształty się nie zgadzają. Broadcasting rozciąga mniejszą tablicę, żeby pasowała.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting rozciąga wektor przez wiersze:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Każdy nowoczesny framework robi to automatycznie. Rozumienie tego zapobiega confuzji, gdy kształty wydają się nieprawidłowe, ale kod działa.

## Buduj to

### Krok 1: Klasa Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Krok 2: Klasa Matrix z podstawowymi operacjami

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Krok 3: Zobacz jak to działa

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Krok 4: Połącz z sieciami neuronowymi

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

To jest pojedyncza gęsta warstwa: `output = relu(W @ x + b)`. Każda gęsta warstwa w każdej sieci neuronowej robi dokładnie to.

## Użyj tego

NumPy robi wszystko powyższe w mniejszej liczbie linii i wielokrotnie szybciej.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Operator `@` w Pythonie wywołuje `__matmul__`. NumPy implementuje go z zoptymalizowanymi procedurami BLAS napisanymi w C i Fortran. Ta sama matematyka, 100x szybciej.

Broadcasting w NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy automatycznie broadcastuje 1D bias przez oba wiersze. Tak działa dodawanie biasu w każdym frameworku sieci neuronowych.

## Wyślij to

Ta lekcja tworzy prompt do nauczania operacji macierzowych przez geometryczną intuicję. Zobacz `outputs/prompt-matrix-operations.md`.

Klasa Matrix zbudowana tutaj to fundament mini frameworku sieci neuronowej, który budujemy w Fazie 3, Lekcji 10.

## Ćwiczenia

1. **Zweryfikuj odwrotność.** Pomnóż `A @ A.inverse_2x2()` i potwierdź, że dostajesz macierz tożsamości. Wypróbuj z trzema różnymi macierzami 2x2. Co się dzieje, gdy wyznacznik jest zero?

2. **Zaimplementuj odwrotność 3x3.** Rozszerz klasę Matrix, żeby obliczała odwrotności dla macierzy 3x3 używając metody adjungowanej. Przetestuj ją przeciwko `np.linalg.inv`.

3. **Zbuduj sieć dwuwarstwową.** Używając tylko swojej klasy Matrix (bez NumPy), stwórz sieć neuronową dwuwarstwową: input (3) -> hidden (4) -> output (2). Zainicjuj losowe wagi, uruchom forward pass i zweryfikuj, że wszystkie kształty są poprawne.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|----------------|----------------------|
| Wektor | "Strzałka" | Uporządkowana lista liczb. W AI: punkt w przestrzeni wysokowymiarowej. |
| Macierz | "Tabela liczb" | Transformacja liniowa. Mapuje wektory z jednej przestrzeni w drugą. |
| Mnożenie macierzy | "Po prostu pomnóż liczby" | Iloczyny skalarne każdego wiersza pierwszej macierzy i każdej kolumny drugiej. Kolejność ma znaczenie. |
| Transpozycja | "Odwróć to" | Zamienia wiersze i kolumny. Zamienia macierz m x n w n x m. Krytyczna w backpropagation. |
| Wyznacznik | "Jakaś liczba z macierzy" | Mierzy, jak bardzo macierz skaluje pole (2D) lub objętość (3D). Zero oznacza, że transformacja redukuje wymiar. |
| Odwrotność | "Cofnij macierz" | Macierz odwracająca transformację. Istnieje tylko, gdy wyznacznik nie jest zero. |
| Macierz tożsamości | "Nudna macierz" | Macierz odpowiadająca mnożeniu przez 1. Używana w połączeniach rezydualnych (ResNets). |
| Broadcasting | "Magiczne dopasowanie kształtów" | Rozciąganie mniejszej tablicy, żeby pasowała do większej przez powtarzanie w brakujących wymiarach. |
| Elementowe | "Regularne mnożenie" | Mnożenie pasujących pozycji. Obie tablice muszą mieć ten sam kształt (lub być broadcastowalne). |

## Dalsze czytanie

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - wizualna intuicja dla każdej operacji tutaj omówionej
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - dokładne reguły, których NumPy przestrzega
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - zwięzłe odniesienie do algebry liniowej specyficznej dla ML