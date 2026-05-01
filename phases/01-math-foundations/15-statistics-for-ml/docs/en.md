# Statystyka dla uczenia maszynowego

> Statystyka to sposób, zeby dowiedziec sie, czy twoj model faktycznie dziala, czy po prostu mial szczescie.

**Typ:** Zbuduj
**Jezyk:** Python
**Wymagania wstepne:** Faza 1, Lekcje 06 (Prawdopodobienstwo i rozklady), 07 (Twierdzenie Bayesa)
**Czas:** ~120 minut

## Cele uczenia sie

- Obliczac statystyki opisowe, korelacje Pearsona/Spearmana i macierze kowariancji od zera
- Przeprowadzac testy hipotez (test t, chi-kwadrat) i poprawnie interpretowac wartosci p oraz przedzialy ufnosci
- Uzywac bootstrap resamplingu do konstruowania przedzialow ufnosci dla dowolnej metryki bez zalozen dystrybucyjnych
- Rozrozniac istotnosc statystyczna od istotnosci praktycznej za pomoca miar wielkosci efektu

## Problem

Wyszkoliles dwa modele. Model A uzyskuje 0.87 na twoim zbiorze testowym. Model B uzyskuje 0.89. Wdrazasz Model B. Trzy tygodnie pozniej metryki produkcyjne sa gorsze niz wczesniej. Co sie stalo?

Model B w rzeczywistosci nie okazal sie lepszy niz Model A. Roznica 0.02 to byl szum. Twoj zbior testowy byl zbyt maly, albo wariancja zbyt wysoka, albo jedno i drugie. Wdrozyles losowosc przebrana za poprawe.

To sie dzieje stale. Przetasowania na tablicach wynikow Kaggle. Artykuly, ktore nie udaje sie odtworzyc. Testy A/B, ktore oglaszaja zwyciezcow na podstawie kilkuset prob. Przyczyna jest zawsze ta sama: ktos pominąl statystyke.

Statystyka daje ci narzedzia do rozroznienia sygnalu od szumu. Mowi ci, kiedy roznica jest rzeczywista, jak pewien powinienes byc i ile danych potrzebujesz, zanim mozesz zaufac wynikowi. Kazdy potok ML, kazde porownanie modeli, kazdy eksperyment potrzebuje statystyki. Bez niej zgadujesz.

## Koncepcja

### Statystyki opisowe: podsumowanie danych

Zanim cokolwiek zamodelujesz, musisz wiedziec, jak twoje dane wygladaja. Statystyki opisowe kompresuja zbior danych do kilku liczb, ktore oddaja jego ksztalt.

**Miary tendencji centralnej** odpowiadaja na pytanie "gdzie jest srodek?"

```
Srednia:   suma wszystkich wartosci / liczba
           mu = (1/n) * sum(x_i)

Mediana:   srodkowa wartosc po posortowaniu
           Odporna na wartosci odstajace. Jesli masz [1, 2, 3, 4, 1000],
           srednia to 202, ale mediana to 3.

Moda:      najczestsza wartosc
           Przydatna dla danych kategorycznych. Dla danych ciaglych,
           rzadko jest informacyjna.
```

Srednia to punkt rownowagi. Mediana to polowa drogi. Kiedy sie roznia, twoj rozklad jest skosny. Rozkłady dochodow maja srednia >> mediana (prawe skoszenie od miliarderow). Rozkłady strat podczas treningu czesto maja srednia << mediana (lewe skoszenie od latwych probek).

**Miary rozproszenia** odpowiadaja na pytanie "jak bardzo dane sa rozproszone?"

```
Wariancja:   srednie kwadratowe odchylenie od sredniej
             sigma^2 = (1/n) * sum((x_i - mu)^2)

Odchylenie standardowe: pierwiastek kwadratowy z wariancji
                        sigma = sqrt(sigma^2)
                        W tych samych jednostkach co dane, wiec bardziej interpretowalne.

Zakres:      max - min
             Wrazliwy na wartosci odstajace. Prawie nigdy nie uzytczny sam.

IQR:         Q3 - Q1 (rozstep miedzykwartylowy)
             Zakres srodkowych 50% danych.
             Odporny na wartosci odstajace. Uzywany do wykresow pudelkowych i wykrywania odstajacych.
```

**Percentyle** dziela posortowane dane na 100 rownych czesci. 25. percentyl (Q1) oznacza, ze 25% wartosci znajduje sie ponizej tego punktu. 50. percentyl to mediana. 75. percentyl to Q3.

```
Dla monitorowania opoznien:
  P50 = mediana opoznienia        (typowe doswiadczenie uzytkownika)
  P95 = 95. percentyl             (zle, ale nie najgorszy przypadek)
  P99 = 99. percentyl             (opoznienie ogonowe, czesto 10x mediana)
```

W ML, interesuja cie percentyle dla opoznien wnioskowania, rozkladow pewosci predykcji i zrozumienia rozkladow bledow. Model z niskim srednim bledem, ale terriblem P99 moze byc bezużyteczny dla zastosowan krytycznych dla bezpieczenstwa.

**Statystyki probki a populacji.** Przy obliczaniu wariancji z probki, dziel przez (n-1) zamiast n. To jest korekta Bessela. Kompensuje fakt, ze srednia z probki nie jest prawdziwa srednia populacji. Z n w mianowniku systematycznie niedoszacowujesz prawdziwa wariancje. Z (n-1), oszacowanie jest nieobciazone.

```
Wariancja populacji: sigma^2 = (1/N) * sum((x_i - mu)^2)
Wariancja probki:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

W praktyce: jesli n jest duze (tysiecy probek), roznica jest znikoma. Jesli n jest male (dziesiatki probek), ma to znaczenie.

### Korelacja: jak zmienne poruszaja sie razem

Korelacja mierzy sile i kierunek liniowego zwiazku miedzy dwiema zmiennymi.

**Wspolczynnik korelacji Pearsona** mierzy asocjacje linowa:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  doskonale dodatni zwiazek linowy
r = -1:  doskonale ujemny zwiazek linowy
r =  0:  brak zwiazku linowego (ale moze byc nieliniowy!)

Zakres: [-1, 1]
```

Pearson zaklada, ze zwiazek jest linowy i obie zmienne sa w przyblizeniu normalnie rozlozone. Jest wrazliwy na wartosci odstajace. Jeden skrajny punkt moze przeciagnac r z 0.1 do 0.9.

**Korelacja rangowa Spearmana** mierzy asocjacje monotoniczna:

```
1. Zastap kazda wartosc jej ranga (1, 2, 3, ...)
2. Oblicz korelacje Pearsona na rangach

Spearman wychwytuje dowolny zwiazek monotoniczny, nie tylko linowy.
Jesli y = x^3, Pearson daje r < 1 ale Spearman daje rho = 1.
```

**Kiedy uzywac ktoregos:**

```
Pearson:    Obie zmienne sa ciagle i w przyblizeniu normalne.
            Zalezy ci na zwiazku linowym konkretnie.
            Brak skrajnych wartosci odstajacych.

Spearman:   Dane porzadkowe ( rankingi, oceny).
            Dane nie sa normalnie rozlozone.
            Podejrzewasz zwiazek monotoniczny, ale nie linowy.
            Obecne sa wartosci odstajace.
```

**Złota zasada:** korelacja nie oznacza przyczynowosci. Sprzedaz lodow i zgony przez utonięcia sa skorelowane, bo obie rosna latem. Dokladnosc twojego modelu i liczba parametrow sa skorelowane, ale dodawanie parametrow automatycznie nie poprawia dokladnosci (patrz: nadmierne dopasowanie).

### Macierz kowariancji

Kowariancja miedzy dwiema zmiennymi mierzy, jak razem sie zmieniaja:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X i Y maja tendencje do wzrostu razem
Cov(X, Y) < 0:  gdy X rośnie, Y ma tendencje do spadku
Cov(X, Y) = 0:  brak wspolruchu linowego
```

Dla d cech, macierz kowariancji C to macierz d x d, gdzie C[i][j] = Cov(feature_i, feature_j). Elementy diagonalne C[i][i] to wariancje kazdej cechy.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Wlasciwosci:
  - Symetryczna: C[i][j] = C[j][i]
  - Polododatnio okreslona: wszystkie wartosci wlasne >= 0
  - Przekatna = wariancje
  - Poza przekatna = kowariancje
```

**Polaczenie z PCA.** PCA rozkłada macierz kowariancji. Wektory wlasne to glowny składowe (kierunki maksymalnej wariancji). Wartosci wlasne mowia ci, ile wariancji kazdy składowa przechwytuje. To jest dokladnie to, co omowila Lekcja 10, ale teraz widzisz, dlaczego macierz kowariancji to wlasciwa rzecz do rozkładu: koduje wszystkie parami linowe zwiazki w twoich danych.

**Polaczenie z korelacja.** Macierz korelacji to macierz kowariancji standaryzowanych zmiennych (kazda podzielona przez jej odchylenie standardowe). Korelacja normalizuje kowariancje, wiec wszystkie wartosci mieszcza sie w [-1, 1].

### Testowanie hipotez

Testowanie hipotez to framework do podejmowania decyzji w warunkach niepewnosci. Zaczynasz od twierdzenia, zbierasz dane i okreslasz, czy dane sa zgodne z twierdzeniem.

**Ustalenia:**

```
Hipoteza zerowa (H0):        domyslne zalozenie, zazwyczaj "brak efektu"
Hipoteza alternatywna (H1):  to, co probujesz wykazac

Przyklad:
  H0: Model A i Model B maja ta sama dokladnosc
  H1: Model B ma wyzsza dokladnosc niz Model A
```

**Wartosc p** to prawdopodobienstwo zobaczenia danych tak ekstremalnych jak zaobserwowane, zakladajac, ze H0 jest prawdziwe. To NIE jest prawdopodobienstwo, ze H0 jest prawdziwe. To jest najczestsze nieporozumienie w statystyce.

```
p-value = P(data this extreme | H0 is true)

Jezeli p-value < alpha (zazwyczaj 0.05):
    Odrzuc H0. Wynik jest "istotny statystycznie."
Jezeli p-value >= alpha:
    Nie mozna odrzucic H0. Nie masz wystarczajaco duzo dowodow.
    To NIE oznacza, ze H0 jest prawdziwe.
```

**Przedzialy ufnosci** daja zakres plauzybilnych wartosci dla parametru:

```
95% przedzial ufnosci dla sredniej:
    x_bar +/- z * (s / sqrt(n))

gdzie z = 1.96 dla 95% ufnosci

Interpretacja: jesli powtorzylbys ten eksperyment wiele razy,
95% obliczonych przedzialow zawieraloby prawdziwa srednia.
To NIE oznacza, ze jest 95% prawdopodobienstwo, ze prawdziwa srednia
jest w tym konkretnym przedziale.
```

Szerokosc przedzialu ufnosci mowi ci o precyzji. Szerokie przedzialy oznaczaja wysoka niepewnosc. Waskie przedzialy oznaczaja, ze twoje oszacowanie jest precyzyjne (ale niekoniecznie dokladne, jesli twoje dane sa obciazone).

### Test t

Test t porownuje srednie. Jest kilka odmian.

**Test t dla jednej probki:** czy srednia populacji rozni sie od hipotezowanej wartosci?

```
t = (x_bar - mu_0) / (s / sqrt(n))

stopnie swobody = n - 1
```

**Test t dla dwoch probek (niezalezne):** czy srednie dwoch grup sie roznia?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

To jest test t Welcha, ktory nie zaklada rownych wariancji.
Zawsze uzywaj testu Welcha, chyba ze masz konkretny powod do rownych wariancji.
```

**Test t dla par:** gdy pomiary sa parami (ten sam model oceniany na tych samych podzialach danych):

```
Oblicz d_i = x_i - y_i dla kazdej pary
Nastepnie przeprowadz test t dla jednej probki na wartosciach d_i przeciwko mu_0 = 0
```

W ML, test t dla par jest powszechny: uruchamiasz oba modele na tych samych 10 podzialach walidacji krzyzowej i porownujesz ich wyniki parami.

### Test chi-kwadrat

Test chi-kwadrat sprawdza, czy zaobserwowane czestosci odpowiadaja oczekiwanym czestosciom. Uzyteczny dla danych kategorycznych.

```
chi^2 = sum((observed - expected)^2 / expected)

Przyklad: czy rozklad wyjsciowy modelu jezykowego odpowiada rozkladowi
treningowemu w kategoriach?

Kategoria    Zaobserwowane   Oczekiwane
Pozytywne        120            100
Negatywne         80            100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

Z 1 stopniem swobody, chi^2 = 8 daje p < 0.005.
Roznica jest istotna.
```

### Testowanie A/B dla modeli ML

Testowanie A/B w ML to nie to samo co webowe testowanie A/B. Porownanie modeli ma specyficzne wyzwania:

```
1. Ten sam zbior testowy:   Oba modele musza byc oceniane na identycznych danych.
                              Rózne zbiory testowe czynia porownanie bezuzytecznym.

2. Wiele metryk:            Sama dokladnosc to za malo. Potrzebujesz precyzji,
                              kompletnosci, F1, opoznienia i metryk sprawiedliwosci.

3. Wariancja:               Uzyj walidacji krzyzowej lub bootstrap,
                              aby oszacowac wariancje kazdej metryki, nie tylko oszacowania punktowe.

4. Wyciek danych:           Jesli zbior testowy byl uzywany podczas wyboru modelu,
                              twoje porownanie jest obciazone. Odłóż ostateczny zbior testowy.
```

**Procedura:**

```
1. Zdefiniuj swoja metryke i poziom istotnosci (alpha = 0.05)
2. Uruchom oba modele na tych samych podzialach walidacji krzyzowej
3. Zbierz parami wyniki: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Oblicz roznice: d_i = b_i - a_i
5. Przeprowadz test t dla par na roznicyach
6. Sprawdz: czy srednia roznica jest istotnie rozna od 0?
7. Oblicz przedzial ufnosci dla sredniej roznicy
8. Oblicz wielkosc efektu (Cohen's d), aby ocenic praktyczna istotnosc
```

### Istotnosc statystyczna a istotnosc praktyczna

Wynik moze byc istotny statystycznie, ale praktycznie bezuzyteczny. Przy wystarczajaco duzych danych, nawet trywialna roznica staje sie istotna statystycznie.

```
Przyklad:
  Dokladnosc Modelu A: 0.9234
  Dokladnosc Modelu B: 0.9237
  n = 1,000,000 probek testowych
  p-value = 0.001

Istotne statystycznie? Tak.
Istotne praktycznie? Poprawa o 0.03% nie jest warta
kosztow inżynieryjnych wdrozenia nowego modelu.
```

**Wielkosc efektu** kwantyfikuje, jak duza jest roznica, niezaleznie od wielkosci probki:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  maly efekt
d = 0.5:  sredni efekt
d = 0.8:  duzy efekt
```

Zawsze podawaj zarowno wartosc p, jak i wielkosc efektu. Wartosc p mowi ci, czy roznica jest rzeczywista. Wielkosc efektu mowi ci, czy ma znaczenie.

### Problem wielokrotnych porownan

Gdy testujesz wiele hipotez, niektore beda "istotne" przez przypadek. Jesli testujesz 20 rzeczy przy alpha = 0.05, spodziewasz sie 1 falszywie dodatniego nawet gdy nic nie jest rzeczywiste.

```
P(co najmniej jeden falszywie dodatni) = 1 - (1 - alpha)^m

m = 20 testow, alpha = 0.05:
P(falszywie dodatni) = 1 - 0.95^20 = 0.64

Masz 64% szans na co najmniej jeden falszywie dodatni wynik.
```

**Korekta Bonferroniego:** dziel alpha przez liczbe testow.

```
Skorygowane alpha = alpha / m = 0.05 / 20 = 0.0025

Odrzucaj H0 tylko jesli p-value < 0.0025.
Konserwatywne, ale proste. Dziala, gdy testy sa niezalezne.
```

W ML, to ma znaczenie, gdy porownujesz model w wielu metrykach, testujesz wiele konfiguracji hiperparametrow lub ewaluujesz na wielu zbiorach danych.

### Metody bootstrap

Bootstrapping szacuje rozklad probkowy statystyki poprzez resampling danych z zamiana. Zadne zalozenia o podstawowym rozkladzie nie sa wymagane.

**Algorytm:**

```
1. Masz n punktow danych
2. Losuj n probek Z ZAMIA (niektore punkty pojawiaja sie wielokrotnie,
   niektore w ogole nie)
3. Oblicz swoja statystyke na tej probce bootstrap
4. Powtorz B razy (zazwyczaj B = 1000 do 10000)
5. Rozklad statystyk bootstrap przybliza rozklad probkowy
```

**Przedzial ufnosci bootstrap (metoda percentylowa):**

```
Posortuj B statystyk bootstrap
95% CI = [2.5. percentyl, 97.5. percentyl]
```

**Dlaczego bootstrap ma znaczenie dla ML:**

```
- Dokladnosc zbioru testowego to oszacowanie punktowe. Bootstrap daje ci
  przedzialy ufnosci.
- Nie mozesz zakladac, ze rozklady metryk sa normalne (szczegolnie
  dla AUC, F1, precyzji przy k).
- Bootstrap dziala dla DOWOLNEJ statystyki: mediany, ilorazu dwoch srednich,
  roznicy w AUC miedzy dwoma modelami.
- Nie potrzebujesz wzoru w formie zamknietej.
```

**Bootstrap dla porownania modeli:**

```
1. Masz predykcje Modelu A i Modelu B na tym samym zbiorze testowym
2. Dla kazdej iteracji bootstrap:
   a. Resampluj indeksy testowe z zamiana
   b. Oblicz metric_A i metric_B na resamplowanym zbiorze
   c. Zapisz diff = metric_B - metric_A
3. 95% CI dla roznicy:
   [2.5. percentyl roznicy, 97.5. percentyl roznicy]
4. Jesli CI nie zawiera 0, roznica jest istotna
```

To jest bardziej odporne niz test t dla par, bo nie robi zadnych zalozen dystrybucyjnych.

### Testy parametryczne a nieparametryczne

**Testy parametryczne** zakladaja konkretny rozklad (zazwyczaj normalny):

```
test t:         zaklada normalnie rozlozone dane (lub duze n dzieki CLT)
ANOVA:          zaklada normalnosc i rownosc wariancji
Pearson r:      zaklada dwuzmiennowa normalnosc
```

**Testy nieparametryczne** nie robia zalozen dystrybucyjnych:

```
Mann-Whitney U:           porownuje dwie grupy (zastepuje niezalezny test t)
Wilcoxon signed-rank:     porownuje dane parami (zastepuje test t dla par)
Spearman rho:             korelacja na rangach (zastepuje Pearson)
Kruskal-Wallis:           porownuje wiele grup (zastepuje ANOVA)
```

**Kiedy uzywac nieparametrycznych:**

```
- Mala wielkosc probki (n < 30) i dane sa wyraznie nienormalne
- Dane porzadkowe (oceny, rankingi)
- Skrajne wartosci odstajace, ktorych nie mozesz usunac
- Skosne rozklady
```

**Kiedy uzywac parametrycznych:**

```
- Duza wielkosc probki (CLT czyni statystyke testowa w przyblizeniu normalna)
- Dane sa w przyblizeniu symetryczne bez skrajnych odstajacych
- Wieksza moc statystyczna (lepsza w wykrywaniu prawdziwych roznicy)
```

W eksperymentach ML zazwyczaj masz male n (5 lub 10 podzialow walidacji krzyzowej), wiec testy nieparametryczne jak Wilcoxon signed-rank sa czesto bardziej odpowiednie niz testy t.

### Centralne twierdzenie graniczne: praktyczne implikacje

CLT mowi, ze rozklad srednich probek dązy do rozkladu normalnego wraz ze wzrostem n, niezaleznie od podstawowego rozkladu populacji.

```
Jezeli X_1, X_2, ..., X_n sa iid ze srednia mu i wariancja sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Dziala dla n >= 30 w wiekszosci przypadkow.
Dla silnie skosnych rozkladow mozesz potrzebowac n >= 100.
```

**Dlaczego to ma znaczenie dla ML:**

```
1. Uzasadnia przedzialy ufnosci i testy t na zagregowanych metrykach
2. Wyjasnia, dlaczego uśrednianie przez podzialy walidacji krzyzowej daje
   stabilne oszacowania, nawet gdy poszczegolne podzialy bardzo sie roznia
3. Mini-batch gradient descent dziala, bo sredni gradient
   przez batch przybliza prawdziwy gradient (CLT w akcji)
4. Metody zespolowe: uśrednianie predykcji z wielu modeli daje
   bardziej stabilny wynik niz kazdy pojedynczy model
```

**Co CLT NIE robi:**

```
- NIE czyni twoich danych normalnymi. Czyni normalnym SREDNIA probek.
- NIE dziala dla rozkladow o grubych ogonach z nieskonczona wariancja
  (rozkład Cauchy'ego).
- NIE ma zastosowania do danych zaleznych (szeregi czasowe bez korekty).
```

### Czeste bledy statystyczne w artykulach ML

1. **Testowanie na zbiorze treningowym.** Gwarantuje nadmierne dopasowanie. Zawsze odłóż dane, ktorych model nigdy nie widzial podczas treningu.

2. **Brak przedzialow ufnosci.** Podawanie pojedynczej liczby dokladnosci bez niepewnosci czyni wyniki niepowtarzalnymi i nieweryfikowalnymi.

3. **Ignorowanie wielokrotnych porownan.** Testowanie 50 konfiguracji i podawanie najlepszej bez korekty zawyzca wspolczynnik falszywie dodatnich.

4. **Mylenie istotnosci statystycznej i praktycznej.** Wartosc p 0.001 przy poprawie dokladnosci o 0.01% nie jest znaczaca.

5. **Uzywanie dokladnosci na niezbalansowanych danych.** 99% dokladnosci na zbiorze z 99% klasy ujemnej oznacza, ze model niczego sie nie nauczyl. Uzywaj precyzji, kompletnosci, F1 lub AUC.

6. **Selektywne podawanie metryk.** Podawanie tylko metryki, w ktorej twój model wygrywa. Rzetelna ewaluacja podaje wszystkie istotne metryki.

7. **Wyciek informacji przez podzialy train/test.** Normalizacja przed podzialem lub uzywanie przyszlych danych do przewidywania przeszlosci.

8. **Male zbiory testowe bez oszacowan wariancji.** Ewaluacja na 100 probkach i twierdzenie o 2% poprawie to szum, nie sygnal.

9. **Zakladanie niezaleznosci, gdy dane nie sa niezalezne.** Obrazy medyczne od tego samego pacjenta, wiele zdan z tego samego dokumentu. Obserwacje w grupie sa skorelowane.

10. **P-hacking.** Probowanie roznych testow, podzbiorow lub kryteriow wykluczenia, az uzyskasz p < 0.05. Wynik to artifact poszukiwania.

## Zbuduj to

Zaimplementujesz:

1. **Statystyki opisowe od zera** (srednia, mediana, moda, odchylenie standardowe, percentyle, IQR)
2. **Funkcje korelacji** (Pearson i Spearman, z macierza kowariancji)
3. **Testy hipotez** (test t dla jednej probki, test t dla dwoch probek, test chi-kwadrat)
4. **Przedzialy ufnosci bootstrap** (dla dowolnej statystyki, bez zalozen)
5. **Symulator testu A/B** (generuj dane, testuj, sprawdzaj bledy typu I i II)
6. **Demo istotnosci statystycznej vs praktycznej** (pokazujace, ze duze n czyni wszystko "istotnym")

Wszystko od zera, uzywajac tylko `math` i `random`. Bez numpy, bez scipy.

## Kluczowe pojecia

| Termin | Definicja |
|--------|-----------|
| Srednia | Suma wartosci podzielona przez liczbe. Wrazliwa na wartosci odstajace. |
| Mediana | Srodkowa wartosc posortowanych danych. Odporna na wartosci odstajace. |
| Odchylenie standardowe | Pierwiastek kwadratowy z wariancji. Mierzy rozproszenie w oryginalnych jednostkach. |
| Percentyl | Wartosc, ponizej ktorej spada dany procent danych. |
| IQR | Rozstep miedzykwartylowy. Q3 minus Q1. Rozproszenie srodkowych 50%. |
| Korelacja Pearsona | Mierzy asocjacje linowa miedzy dwiema zmiennymi. Zakres [-1, 1]. |
| Korelacja Spearmana | Mierzy asocjacje monotoniczna za pomoca rang. |
| Macierz kowariancji | Macierz parami kowariancji miedzy wszystkimi cechami. |
| Hipoteza zerowa | Domyslne zalozenie braku efektu lub roznicy. |
| Wartosc p | Prawdopodobienstwo danych tak ekstremalnych przy prawdziwej hipotezie zerowej. |
| Przedzial ufnosci | Zakres plauzybilnych wartosci dla parametru przy danym poziomie ufnosci. |
| Test t | Sprawdza, czy srednie roznia sie istotnie. Uzywa rozkladu t. |
| Test chi-kwadrat | Sprawdza, czy zaobserwowane czestosci roznia sie od oczekiwanych. |
| Wielkosc efektu | Wielkosc roznicy, niezaleznie od wielkosci probki. Cohen's d jest powszechny. |
| Korekta Bonferroniego | Dzieli próg istotnosci przez liczbe testow, aby kontrolowac falszywie dodatnie. |
| Bootstrap | Resampling z zamiana do szacowania rozkladow probkowych. |
| Blad typu I | Falszywie dodatni. Odrzucenie H0, gdy jest prawdziwa. |
| Blad typu II | Falszywie ujemny. Niezdolnosc do odrzucenia H0, gdy jest falszywa. |
| Moc statystyczna | Prawdopodobienstwo poprawnego odrzucenia falszywego H0. Moc = 1 minus wspolczynnik bledu typu II. |
| Centralne twierdzenie graniczne | Srednie probek daza do rozkladu normalnego wraz ze wzrostem wielkosci probki. |
| Test parametryczny | Zaklada konkretny rozklad danych (zazwyczaj normalny). |
| Test nieparametryczny | Nie robi zalozen dystrybucyjnych. Dziala na rangach lub znakach. |