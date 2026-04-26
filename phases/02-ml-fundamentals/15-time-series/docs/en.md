<think>
The user wants me to apply the corrections to the document. Let me go through and make the three corrections needed:

1. Add comma before "jeśli" - "jest stacjonarny, jeśli"
2. Add comma after "stycznia" - "ze stycznia, nauczył się"
3. Add Polish explanation for "i.i.d." - "niezależne i jednakowo rozłożone (i.i.d.)"

Let me write out the corrected document.
</think>

# Podstawy Szeregów Czasowych

- **Typ:** Budowanie
- **Język:** Python
- **Wymagania wstępne:** Phase 2, Lekcje 01-09
- **Czas:** ~90 minut

## Cele uczenia się

Po tej lekcji będziesz potrafić:

1. Dekomponuj szereg czasowy na składowe: trend, sezonowość i resztę oraz testuj stacjonarność

2. Implementuj opóźnione cechy i statystyki ruchome, aby przekształcić szereg czasowy w problem nadzorowanego uczenia się

3. Zbuduj framework walidacji typu krok-do-przodu, który zapobiega wyciekowi przyszłych danych do treningu

4. Wyjaśnij, dlaczego losowe podziały train/test są nieprawidłowe dla szeregów czasowych i zademonstruj różnicę w wydajności w porównaniu z właściwymi podziałami czasowymi

## Problem

Prognozowanie szeregów czasowych jest trudne, ponieważ standardowy zestaw narzędzi ML zakłada, że możesz losowo podzielić dane na train/test i użyć walidacji krzyżowej. Szeregi czasowe łamią to założenie. Kiedy tasujesz szeregi czasowe, wyciek przyszłych informacji w przeszłość. To dlatego model z 95% dokładnością przy losowej walidacji krzyżowej może uzyskać 55% przy właściwej ocenie opartej na czasie.

## Koncepcja

### Co wyróżnia szeregi czasowe

Standardowe ML zakłada niezależne i jednakowo rozłożone (i.i.d.).

Każda próbka jest pobrana z tej samej dystrybucji i niezależnie od innych próbek.

Szeregi czasowe łamią oba założenia.

Próbki są wymienne. Tasowanie ich nic nie zmienia.

W szeregach czasowych kolejność jest wszystkim. Tasowanie niszczy sygnał.

### Składowe szeregu czasowego

Każdy szereg czasowy jest kombinacją trendu, sezonowości i reszty.

**Trend:** Kierunek długoterminowy - przychody rosnące 10% rocznie

**Sezonowość:** Powtarzające się wzorce w ustalonych interwałach - sprzedaż detaliczna gwałtownie rośnie w grudniu, użycie klimatyzacji osiąga szczyt w lipcu

**Reszta:** Losowa zmienność po usunięciu trendu i sezonowości

Jeśli reszta wygląda jak biały szum, oznacza to, że dekompozycja uchwyciła sygnał.

### Stacjonarność

Szereg czasowy jest stacjonarny, jeśli jego właściwości statystyczne (średnia, wariancja, autokorelacja) nie zmieniają się w czasie.

Większość metod prognozowania zakłada stacjonarność.

**Dlaczego to ważne:** Seria niestacjonarna ma średnią, która dryfuje. Model trenowany na danych ze stycznia, nauczył się innej średniej niż ta, którą pokaże luty. Będzie systematycznie błędny.

**Jak sprawdzić:** Oblicz średnią ruchomą i odchylenie standardowe ruchome przez okna. Jeśli dryfują, seria jest niestacjonarna.

**Jak naprawić:** Różnicowanie. Zamiast modelować surowe wartości, modeluj zmianę między kolejnymi wartościami. Jeśli jedna runda różnicowania nie czyni serii stacjonarną, zastosuj ją ponownie (różnicowanie drugiego rzędu). Większość rzeczywistych serii potrzebuje maksymalnie dwóch rund.

**Przykład:**

Oryginalna seria: [100, 102, 106, 112, 120]

Pierwsza różnica: [2, 4, 6, 8]

Druga różnica: [2, 2, 2]

Oryginalna seria miała trend kwadratowy. Pierwsze różnicowanie zamieniło go w trend liniowy. Drugie różnicowanie spłaszczyło. W praktyce rzadko potrzeba więcej niż dwie rundy.

**Test formalny:** Rozszerzony test Dickeya-Fullera (ADF). Hipoteza zerowa brzmi „seria jest niestacjonarna". Wartość p poniżej 0.05 oznacza, że możemy odrzucić hipotezę zerową i stwierdzić stacjonarność.