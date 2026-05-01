# Normy i odległości

> Twoja funkcja odległości określa, co oznacza "podobny". Jeśli wybierzesz źle, wszystko downstream się popsuje.

**Typ:** Zbuduj to
**Język:** Python
**Wymagania wstępne:** Faza 1, Lekcje 01 (Intuicja algebry liniowej), 02 (Wektory, Macierze i Operacje)
**Czas:** około 90 minut

## Cele uczenia się

- Zaimplementować odległości L1, L2, cosinusową, Mahalanobisa, Jaccarda i edycyjną od podstaw
- Wybrać odpowiednią miarę odległości dla danego zadania ML i wyjaśnić, dlaczego alternatywy zawodzą
- Połączyć normy L1 i L2 z regularyzacją LASSO i Ridge oraz ich geometrycznymi obszarami ograniczeń
- Zademonstrować, jak ten sam zbiór danych generuje różne najbliższe sąsiedztwa przy różnych metrykach

## Problem

Masz dwa wektory. Może to są osadzenia słów. Może to są profile użytkowników. Może to są tablice pikseli. Musisz wiedzieć: jak blisko siebie są?

Odpowiedź zależy całkowicie od tego, jaką funkcję odległości wybierzesz. Dwa punkty danych mogą być najbliższymi sąsiadami według jednej metryki, a odległe według innej. Twój klasyfikator KNN, twój silnik rekomendacji, twoja baza wektorowa, twój algorytm grupowania, twoja funkcja straty -- wszystkie zależą od tego wyboru. Jeśli się pomylisz, twój model optymalizuje niewłaściwą rzecz.

Nie istnieje uniwersalna najlepsza odległość. L2 sprawdza się dla danych przestrzennych. Podobieństwo cosinusowe dominuje w NLP. Jaccard obsługuje zbiory. Odległość edycyjna obsługuje ciągi znaków. Mahalanobis uwzględnia korelacje. Wasserstein przenosi masę prawdopodobieństwa. Każda z nich koduje inne założenie o tym, co oznacza "podobny".

Ta lekcja buduje od podstaw każdą główną funkcję odległości, pokazuje, kiedy każda z nich jest właściwym narzędziem, oraz demonstruje, jak te same dane generują całkowicie różne najbliższe sąsiedztwa w zależności od użytej metryki.

## Koncepcja

### Normy: mierzenie wielkości wektora

Norma mierzy "rozmiar" wektora. Każdą funkcję odległości między dwoma wektorami można zapisać jako normę ich różnicy: d(a, b) = ||a - b||. Zatem rozumienie norm oznacza rozumienie odległości.

### Norma L1 (odległość Manhattan)

Norma L1 sumuje wartości bezwzględne wszystkich składowych.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Nazywa się ją odległością Manhattan, ponieważ mierzy, jak daleko idziesz po siatce ulic, gdzie możesz poruszać się tylko wzdłuż osi. Bez przekątnych.

```
Punkt A = (1, 1)
Punkt B = (4, 5)

Odległość L1 = |4-1| + |5-1| = 3 + 4 = 7

Na siatce idziesz 3 przecznice na wschód i 4 na północ.
```

Kiedy używać L1:
- Dane rzadkie o wysokiej wymiarowości (cechy tekstowe, kodowania one-hot)
- Gdy chcesz odporności na elementy odstające (pojedyncza ogromna różnica nie dominuje)
- Problemy z selekcją cech (regularyzacja L1 promuje rzadkość)

Połączenie z regularyzacją L1 (Lasso): dodanie ||w||_1 do funkcji straty kara za sumę bezwzględnych wartości wag. To pcha małe wagi dokładnie do zera, wykonując automatyczną selekcję cech. Kara L1 tworzy rombowate obszary ograniczeń w przestrzeni wag, a rogi rombu leżą na osiach, gdzie niektóre wagi wynoszą zero.

Połączenie z funkcjami straty: Mean Absolute Error (MAE) to średnia odległość L1 między predykcjami a celami. Kara za wszystkie błędy jest liniowa, co czyni ją odporną na elementy odstające w porównaniu z MSE.

### Norma L2 (odległość euklidesowa)

Norma L2 to odległość w linii prostej. Pierwiastek kwadratowy z sumy kwadratów składowych.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

To jest odległość, której uczyłeś się na lekcjach geometrii. Pitagoras w n wymiarach.

```
Punkt A = (1, 1)
Punkt B = (4, 5)

Odległość L2 = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

Linia prosta, przecinająca ukośnie siatkę.
```

Kiedy używać L2:
- Dane ciągłe o niskiej do średniej wymiarowości
- Gdy skale cech są porównywalne
- Odległości fizyczne (dane przestrzenne, odczyty czujników)
- Podobieństwo obrazów na poziomie pikseli

Połączenie z regularyzacją L2 (Ridge): dodanie ||w||_2^2 do funkcji straty kara za duże wagi. W przeciwieństwie do L1, nie pcha wag do zera. Zmniejsza wszystkie wagi proporcjonalnie ku zero. Kara L2 tworzy koliste obszary ograniczeń, więc nie ma rogów na osiach. Wagi stają się małe, ale rzadko dokładnie zero.

Połączenie z funkcjami straty: Mean Squared Error (MSE) to średnia z kwadratów odległości L2. Podnoszenie do kwadratu kara za duże błędy znacznie bardziej niż za małe.

```
MAE (strata L1):  |y - y_hat|         Kara liniowa. Odporna na elementy odstające.
MSE (strata L2):  (y - y_hat)^2       Kara kwadratowa. Wrażliwa na elementy odstające.
```

### Normy Lp: ogólna rodzina

L1 i L2 to szczególne przypadki normy Lp:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Różne wartości p produkują różnie ukształtowane "kule jednostkowe" (zbiór wszystkich punktów w odległości 1 od początku):

```
p=1:    Kształt rombu      (rogi na osiach)
p=2:    Koło/kula          (zwykła okrągła kula)
p=3:    Superelipsa        (zaokrąglony kwadrat)
p=inf:  Kwadrat/hiperkostka (płaskie boki wzdłuż osi)
```

### Norma L-nieskończoność (odległość Czebyszewa)

Gdy p dąży do nieskończoności, norma Lp zbiega do maksymalnej wartości bezwzględnej składowej.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Odległość między dwoma punktami jest określana przez wymiar, w którym różnią się najbardziej. Wszystkie inne wymiary są ignorowane.

```
Punkt A = (1, 1)
Punkt B = (4, 5)

Odległość L-inf = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Kiedy używać L-nieskończoności:
- Gdy najgorszy przypadek odchylenia w dowolnym pojedynczym wymiarze ma znaczenie
- Plansze do gier (król w szachach porusza się w L-nieskończoności: jeden krok w dowolnym kierunku kosztuje 1)
- Tolerancje produkcyjne (każdy wymiar musi być w specyfikacji)

### Podobieństwo cosinusowe i odległość cosinusowa

Podobieństwo cosinusowe mierzy kąt między dwoma wektorami, ignorując ich wielkości.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Zakres od -1 (przeciwne kierunki) do +1 (ten sam kierunek). Prostopadłe wektory mają podobieństwo cosinusowe równe 0.

Odległość cosinusowa konwertuje to na odległość: cosine_distance = 1 - cosine_similarity. Zakres od 0 (identyczny kierunek) do 2 (przeciwne kierunki).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Dlaczego cosinus dominuje w NLP i osadzeniach: w tekście długość dokumentu nie powinna wpływać na podobieństwo. Dokument o kotach, który jest dwa razy dłuższy niż inny dokument o kotach, powinien nadal być "podobny". Podobieństwo cosinusowe ignoruje wielkość (długość) i dba tylko o kierunek. Dwa dokumenty z tą samą dystrybucją słów, ale różnymi długościami, wskazują w tym samym kierunku i otrzymują podobieństwo cosinusowe 1.0.

Kiedy używać podobieństwa cosinusowego:
- Podobieństwo tekstu (wektory TF-IDF, osadzenia słów, osadzenia zdań)
- Każda domena, gdzie wielkość jest szumem, a kierunek jest sygnałem
- Systemy rekomendacji (wektory preferencji użytkowników)
- Wyszukiwanie osadzeń (bazy wektorowe prawie zawsze używają cosinusa lub iloczynu skalarnego)

### Podobieństwo iloczynu skalarnego vs podobieństwo cosinusowe

Iloczyn skalarny dwóch wektorów to:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(kąt)
```

Podobieństwo cosinusowe to iloczyn skalarny znormalizowany przez obie wielkości. Gdy oba wektory są już znormalizowane jednostkowo (wielkość = 1), iloczyn skalarny i podobieństwo cosinusowe są identyczne.

```
Jeśli ||a|| = 1 i ||b|| = 1:
    a . b = cos(kąt między a i b)
```

Gdy się różnią: iloczyn skalarny zawiera informację o wielkości. Wektor o większej wielkości otrzymuje wyższy wynik iloczynu skalarnego. Ma to znaczenie w niektórych systemach wyszukiwania, gdzie chcesz, aby "popularne" elementy były wyżej rankowane. Wielkość działa jako niejawny sygnał jakości lub ważności.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

iloczyn(a, b) = 3     iloczyn(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Obie zgadzają się co do kierunku, ale iloczyn skalarny odzwierciedla też wielkość.
```

W praktyce:
- Używaj podobieństwa cosinusowego, gdy chcesz czystego podobieństwa kierunkowego
- Używaj iloczynu skalarnego, gdy wielkości niosą znaczące informacje
- Wiele baz wektorowych (Pinecone, Weaviate, Qdrant) pozwala wybierać między nimi
- Jeśli twoje osadzenia są znormalizowane L2, wybór nie ma znaczenia

### Odległość Mahalanobisa

Odległość euklidesowa traktuje wszystkie wymiary równo. Ale jeśli twoje cechy są skorelowane lub mają różne skale, L2 daje mylące wyniki.

Odległość Mahalanobisa uwzględnia strukturę kowariancji danych.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

gdzie S to macierz kowariancji danych.

Intuicyjnie: odległość Mahalanobisa najpierw dekoreluje i normalizuje dane (whitening), a następnie oblicza odległość L2 w tej przestrzeni transformowanej. Jeśli S jest macierzą jednostkową (neskorelowane cechy o jednostkowej wariancji), odległość Mahalanobisa redukuje się do odległości euklidesowej.

```
Przykład: wzrost i waga są skorelowane.
Ktoś 190 cm i 82 kg nie jest niezwykły.
Ktoś 152 cm i 82 kg jest niezwykły.

Odległość euklidesowa może powiedzieć, że są równie daleko od średniej.
Odległość Mahalanobisa poprawnie identyfikuje drugą osobę jako element odstający,
ponieważ uwzględnia korelację wzrost-waga.
```

Kiedy używać odległości Mahalanobisa:
- Wykrywanie elementów odstających (punkty o dużej odległości Mahalanobisa od średniej są elementami odstającymi)
- Klasyfikacja, gdy cechy mają różne skale i korelacje
- Gdy masz wystarczająco dużo danych, aby oszacować wiarygodną macierz kowariancji
- Kontrola jakości w produkcji (wielowymiarowe monitorowanie procesów)

### Podobieństwo Jaccarda (dla zbiorów)

Podobieństwo Jaccarda mierzy nakładanie się dwóch zbiorów.

```
J(A, B) = |A intersect B| / |A union B|
```

Zakres od 0 (brak nakładania) do 1 (identyczne zbiory). Odległość Jaccarda = 1 - podobieństwo Jaccarda.

```
A = {kot, pies, ryba}
B = {kot, ptak, ryba, wąż}

Część wspólna = {kot, ryba}         rozmiar = 2
Suma = {kot, pies, ryba, ptak, wąż}  rozmiar = 5

Podobieństwo Jaccarda = 2/5 = 0.4
Odległość Jaccarda = 0.6
```

Kiedy używać Jaccarda:
- Porównywanie zbiorów tagów, kategorii lub cech
- Podobieństwo dokumentów oparte na obecności słów (nie częstotliwości)
- Wykrywanie niemal-duplikatów (przybliżenie MinHash podobieństwa Jaccarda)
- Porównywanie binarnych wektorów cech (dane obecności/nieobecności)
- Ewaluacja modeli segmentacji (Intersection over Union = Jaccard)

### Odległość edycyjna (odległość Levenshteina)

Odległość edycyjna liczy minimalną liczbę operacji jednoznakowych potrzebnych do przekształcenia jednego ciągu w drugi. Operacje to: wstaw, usuń lub podstaw.

```
"kotek" -> "siedzenie"

kotek -> sietek  (podstaw k -> s)
sietek -> sidenek  (podstaw e -> i)
sidenek -> siedzenie (wstaw g)

Odległość edycyjna = 3
```

Obliczana za pomocą programowania dynamicznego. Wypełniasz macierz, gdzie wpis (i, j) to odległość edycyjna między pierwszymi i znakami ciągu A a pierwszymi j znakami ciągu B.

```
        ""  s  i  e  d  z  e  n  i  e
    ""   0  1  2  3  4  5  6  7  8  9
    k    1  1  2  3  4  5  6  7  8  9
    o    2  2  2  3  4  5  6  7  8  9
    t    3  3  3  3  4  5  6  7  8  9
    e    4  4  4  4  3  4  5  6  7  8
    k    5  5  5  5  4  4  5  6  7  8
```

Kiedy używać odległości edycyjnej:
- Sprawdzanie i poprawianie pisowni
- Wyrównywanie sekwencji DNA (z ważonymi operacjami)
- Rozmyte dopasowywanie ciągów
- Deduplikacja bałaganiarskich danych tekstowych

### Dywergencja KL (nie jest odległością, ale używana jak jedna)

Dywergencja KL mierzy, jak bardzo jedna dystrybucja prawdopodobieństwa różni się od drugiej. Omówiona w Lekcji 09, ale należy do tej dyskusji, ponieważ ludzie używają jej jako "odległości", mimo że nią nie jest.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Krytyczna właściwość: dywergencja KL NIE jest symetryczna.

```
D_KL(P || Q) != D_KL(Q || P)
```

Oznacza to, że nie spełnia podstawowego wymogu metryki odległości. Nie spełnia też nierówności trójkąta. To dywergencja, nie odległość.

Forward KL (D_KL(P || Q)) jest "szukająca średniej": Q stara się pokryć wszystkie mody P.
Reverse KL (D_KL(Q || P)) jest "szukająca mody": Q koncentruje się na pojedynczej modzie P.

Kiedy widzisz dywergencję KL:
- VAE (wyraz KL w ELBO popycha dystrybucję latentną w kierunku prioru)
- Destylacja wiedzy (student stara się dopasować dystrybucję nauczyciela)
- RLHF (kara KL utrzymuje dostrojony model blisko modelu bazowego)
- Metody policy gradient (ograniczanie aktualizacji polityki)

### Odległość Wassersteina (odległość Earth Mover's)

Odległość Wassersteina mierzy minimalną "pracę" potrzebną do przekształcenia jednej dystrybucji prawdopodobieństwa w drugą. Pomyśl o tym jak: jeśli jedna dystrybucja to sterta ziemi, a druga to dziura, ile ziemi musisz przenieść i jak daleko?

```
W(P, Q) = inf nad wszystkimi planami transportu gamma z E[d(x, y)]
```

Dla dystrybucji jednowymiarowych upraszcza się do całki z bezwzględnej różnicy dystrybuant:

```
W_1(P, Q) = całka |CDF_P(x) - CDF_Q(x)| dx
```

Dlaczego Wasserstein ma znaczenie:
- Jest prawdziwą metryką (symetryczna, spełnia nierówność trójkąta)
- Daje gradienty nawet gdy dystrybucje się nie nakładają (dywergencja KL dąży do nieskończoności)
- Ta właściwość uczyniła ją centralną w Wasserstein GAN (WGAN), które rozwiązały niestabilność treningową oryginalnych GAN

```
Dystrybucje bez nakładania się:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

Dywergencja KL: nieskończoność (log zera)
Wasserstein: 4 (przenieś całą masę o 4 przedziały)

Wasserstein daje znaczący gradient. KL nie.
```

Kiedy używać Wassersteina:
- Trening GAN (WGAN, WGAN-GP)
- Porównywanie dystrybucji, które mogą się nie nakładać
- Problemy optymalnego transportu
- Wyszukiwanie obrazów (porównywanie histogramów kolorów)

### Dlaczego różne zadania wymagają różnych odległości

| Zadanie | Najlepsza odległość | Dlaczego |
|--------|---------------------|----------|
| Podobieństwo tekstu | Cosinusowa | Wielkość jest szumem, kierunek jest znaczeniem |
| Porównanie pikseli obrazu | L2 | Relacje przestrzenne mają znaczenie, cechy są porównywalnej skali |
| Rzadkie cechy wysokowymiarowe | L1 | Odporna, nie wzmacnia rzadkich dużych różnic |
| Nakładanie się zbiorów (tagi, kategorie) | Jaccard | Dane są naturalnie wartościami zbiorowymi, nie wektorowymi |
| Dopasowywanie ciągów | Odległość edycyjna | Operacje odpowiadają ludzkiej intuicji edycji |
| Wykrywanie elementów odstających | Mahalanobis | Uwzględnia korelacje i skale cech |
| Porównywanie dystrybucji | Dywergencja KL | Mierzy informację straconą przez używanie Q zamiast P |
| Trening GAN | Wasserstein | Daje gradienty nawet gdy dystrybucje nie mają nakładania |
| Osadzenia (wektorowa baza danych) | Cosinusowa lub iloczyn skalarny | Osadzenia są trenowane, aby kodować znaczenie w kierunku |
| Rekomendacja | Iloczyn skalarny | Wielkość może kodować popularność lub pewność |
| Sekwencje DNA | Ważona odległość edycyjna | Koszty podstawień różnią się w zależności od pary nukleotydów |
| Kontrola jakości produkcji | L-nieskończoność | Najgorszy przypadek odchylenia w dowolnym wymiarze ma znaczenie |

### Połączenie z funkcjami straty

Funkcje straty to funkcje odległości zastosowane do predykcji vs celów.

```
Funkcja straty       Używana odległość       Zachowanie
MSE                 L2 squared             Kara za duże błędy mocno
MAE                 L1                     Kara za wszystkie błędy równo
Huber loss          L1 dla dużych błędów,  Najlepsze z obu: odporna na elementy
                    L2 dla małych błędów   odstające, gładki gradient blisko zera
Cross-entropy       Dywergencja KL         Mierzy niedopasowanie dystrybucji
Hinge loss          max(0, margin - d)     Kara tylko poniżej marginesu
Triplet loss        L2 (zazwyczaj)         Przyciąga pozytywy blisko, odpycha
                                           negatywy
Contrastive loss    L2                     Podobne pary blisko, niepodobne
                                           pary poza marginesem
```

### Połączenie z regularyzacją

Regularyzacja dodaje karę normy wag do funkcji straty.

```
Regularyzacja L1 (Lasso):   loss + lambda * ||w||_1
  -> Rzadkie wagi. Niektóre wagi stają się dokładnie zero.
  -> Automatyczna selekcja cech.
  -> Rozwiązanie ma rogi (nieróżniczkowalne w zero).

Regularyzacja L2 (Ridge):   loss + lambda * ||w||_2^2
  -> Małe wagi. Wszystkie wagi zmniejszają się ku zero.
  -> Brak selekcji cech (nic nie idzie dokładnie do zera).
  -> Gładkie rozwiązanie wszędzie.

Elastic Net:                loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Łączy rzadkość L1 ze stabilnością L2.
  -> Grupy skorelowanych cech są trzymane lub usuwane razem.
```

Dlaczego L1 produkuje rzadkość, a L2 nie: wyobraź sobie obszar ograniczeń w 2D przestrzeni wag. L1 to diament, L2 to koło. Kontury funkcji straty (elipsy) najprawdopodobniej dotykają diamentu w rogu, gdzie jedna waga wynosi zero. Dotykają koła w gładkim punkcie, gdzie obie wagi są niezerowe.

### Wyszukiwanie najbliższego sąsiada

Każda funkcja odległości implikuje problem wyszukiwania najbliższego sąsiada: mając punkt zapytania, znajdź najbliższe punkty w zbiorze danych.

Dokładne wyszukiwanie najbliższego sąsiada to O(n * d) na zapytanie w zbiorze danych n punktów z d wymiarami. Dla dużych zbiorów danych jest to zbyt wolne.

Algorytmy Approximate Nearest Neighbor (ANN) wymieniają niewielką dokładność na ogromne zyski prędkości:

```
Algorytm         Podejście                      Używany przez
KD-trees         Podział przestrzeni            scikit-learn (niska wymiarowość)
                 wyrównany do osi
Ball trees       Zagnieżdżone hipersfery        scikit-learn (średnia wymiarowość)
LSH               Losowe projekcje haszujące    Wykrywanie niemal-duplikatów
HNSW              Hierarchiczny nawigowany      FAISS, Qdrant, Weaviate
                  mały świat (graf)
IVF               Odwrócony indeks plików z     FAISS (miliardowa skala)
                  wyszukiwaniem opartym
                  na klastrach
Product quant.    Kompresja wektorów,           FAISS (pamięć ograniczona)
                  wyszukiwanie
                  w skompresowanej przestrzeni
```

HNSW (Hierarchical Navigable Small World) to dominujący algorytm w nowoczesnych wektorowych bazach danych. Buduje wielowarstwowy graf, gdzie każdy węzeł łączy się ze swoimi przybliżonymi najbliższymi sąsiadami. Wyszukiwanie zaczyna się od górnej warstwy (rzadka, długie skoki) i schodzi do dolnej warstwy (gęsta, krótkie skoki).

## Zbuduj to

### Krok 1: Wszystkie funkcje norm i odległości

Zobacz `code/distances.py` po pełną implementację. Każda funkcja jest zbudowana od podstaw używając tylko podstawowej matematyki Pythona.

### Krok 2: Te same dane, różne odległości, różni sąsiedzi

Demo w `distances.py` tworzy zbiór danych, wybiera punkt zapytania i pokazuje, jak najbliższy sąsiad zmienia się w zależności od metryki odległości. Punkt, który jest "najbliższy" według L1, może nie być najbliższy według L2 lub cosinusa.

### Krok 3: Wyszukiwanie podobieństwa osadzeń

Kod zawiera mock wyszukiwania podobieństwa osadzeń, które znajduje najbardziej "podobne" dokumenty do zapytania używając podobieństwa cosinusowego vs odległości L2, pokazując, że rankingi mogą się różnić.

## Uzyj tego

Najczęstsze praktyczne zastosowanie: znajdowanie podobnych elementów w wektorowej bazie danych.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Gdy wywołujesz `model.encode(text)` a następnie wyszukujesz w wektorowej bazie danych, to jest to, co dzieje się pod maską. Model osadzający mapuje tekst na wektory. Wektorowa baza danych oblicza podobieństwo cosinusowe (lub iloczyn skalarny) między twoim wektorem zapytania a każdym przechowywanym wektorem, używając algorytmów ANN, aby uniknąć sprawdzania wszystkich.

## Cwiczenia

1. Oblicz odległości L1, L2 i L-nieskończoność między (1, 2, 3) i (4, 0, 6). Zweryfikuj, że L-inf <= L2 <= L1 zawsze zachodzi dla dowolnej pary punktów. Udowodnij, dlaczego ta kolejność jest gwarantowana.

2. Stwórz dwa wektory, gdzie podobieństwo cosinusowe jest wysokie (> 0.9), ale odległość L2 jest duża (> 10). Wyjaśnij geometrycznie, co się dzieje. Następnie stwórz dwa wektory, gdzie podobieństwo cosinusowe jest niskie (< 0.3), ale odległość L2 jest mała (< 0.5).

3. Zaimplementuj funkcję, która przyjmuje zbiór danych i punkt zapytania i zwraca najbliższego sąsiada według odległości L1, L2, cosinusowej i Mahalanobisa. Znajdź zbiór danych, gdzie wszystkie cztery nie zgadzają się co do tego, który punkt jest najbliższy.

4. Oblicz odległość Wassersteina między [0.5, 0.5, 0, 0] a [0, 0, 0.5, 0.5] ręcznie używając metody CDF. Następnie oblicz ją między [0.25, 0.25, 0.25, 0.25] a [0, 0, 0.5, 0.5]. Która jest większa i dlaczego?

5. Zaimplementuj MinHash dla przybliżonego podobieństwa Jaccarda. Wygeneruj 100 losowych zbiorów, oblicz dokładne podobieństwo Jaccarda dla wszystkich par i porównaj z przybliżeniem MinHash używając 50, 100 i 200 funkcji haszujących. Wykreśl błąd przybliżenia.

## Kluczowe pojecia

| Pojęcie | Co ludzie mówią | Co to faktycznie oznacza |
|---------|-----------------|--------------------------|
| Norma | "Rozmiar wektora" | Funkcja, która mapuje wektor na nieujemny skalar, spełniająca nierówność trójkąta, jednorodność bezwzględną i zero tylko dla wektora zerowego |
| Norma L1 | "Odległość Manhattan" | Suma wartości bezwzględnych składowych. Produkuje rzadkość w optymalizacji. Odporna na elementy odstające |
| Norma L2 | "Odległość euklidesowa" | Pierwiastek kwadratowy z sumy kwadratów składowych. Odległość w linii prostej w przestrzeni euklidesowej |
| Norma Lp | "Uogólniona norma" | Pierwiastek p-tego stopnia z sumy p-tych potęg wartości bezwzględnych składowych. L1 i L2 to szczególne przypadki |
| Norma L-nieskończoność | "Norma max" lub "odległość Czebyszewa" | Maksymalna wartość bezwzględna składowej. Granica Lp gdy p dąży do nieskończoności |
| Podobieństwo cosinusowe | "Kąt między wektorami" | Iloczyn skalarny znormalizowany przez obie wielkości. Zakres od -1 do +1. Ignoruje długość wektora |
| Odległość cosinusowa | "1 minus podobieństwo cosinusowe" | Konwertuje podobieństwo cosinusowe na odległość. Zakres od 0 do 2 |
| Iloczyn skalarny | "Nieznormalizowany cosinus" | Suma iloczynów składowych. Równa się podobieństwu cosinusowemu razy obie wielkości |
| Odległość Mahalanobisa | "Odległość uwzględniająca korelacje" | Odległość L2 w przestrzeni, która została wybielona (dekorelowana i znormalizowana) używając macierzy kowariancji danych |
| Podobieństwo Jaccarda | "Nakładanie się zbiorów" | Rozmiar przecięcia podzielony przez rozmiar sumy. Dla zbiorów, nie wektorów |
| Odległość edycyjna | "Odległość Levenshteina" | Minimalne wstawienia, usunięcia i podstawienia do przekształcenia jednego ciągu w drugi |
| Dywergencja KL | "Odległość między dystrybucjami" | Nie jest prawdziwą odległością (nie jest symetryczna). Mierzy dodatkowe bity z używania Q do kodowania P |
| Odległość Wassersteina | "Odległość Earth mover's" | Minimalna praca do transportu masy z jednej dystrybucji do drugiej. Prawdziwa metryka |
| Przybliżony najbliższy sąsiad | "Wyszukiwanie ANN" | Algorytmy (HNSW, LSH, IVF), które znajdują przybliżone najbliższe punkty znacznie szybciej niż dokładne wyszukiwanie |
| HNSW | "Algorytm wektorowej bazy danych" | Hierarchiczny nawigowany mały świat. Wielowarstwowy graf dla szybkiego przybliżonego wyszukiwania najbliższego sąsiada |
| Regularyzacja L1 | "Lasso" | Dodawanie normy L1 wag do straty. Pcha wagi do zera (rzadkość) |
| Regularyzacja L2 | "Ridge" lub "weight decay" | Dodawanie kwadratu normy L2 wag do straty. Zmniejsza wagi ku zero bez rzadkości |
| Elastic Net | "L1 + L2" | Łączy regularyzację L1 i L2. Obsługuje grupy skorelowanych cech lepiej niż każda z osobna |

## Dalsza lektura

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Biblioteka Meta do wyszukiwania podobieństwa w skali miliardów
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - artykuł, który wprowadził odległość Earth Mover's do GAN
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - podstawowy algorytm ANN
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, gdzie podobieństwo cosinusowe stało się domyślne dla osadzeń
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - praktyczny przewodnik po metrykach odległości i algorytmach sąsiedztwa w scikit-learn