# Modelowanie Wizualne Autoregresyjne (VAR): Predykcja Następnej Skali

> Modele dyfuzyjne próbkują iteracyjnie w czasie (kroki odszumiania). VAR próbkuje iteracyjnie w skali — przewiduje token 1x1, następnie 2x2, następnie 4x4, aż do finalnej rozdzielczości, każda skala warunkowana przez poprzednią. Artykuł z 2024 roku wykazał, że VAR dorównuje prawom skalującym w stylu GPT dla generowania obrazów i bije DiT przy tym samym budżecie obliczeniowym. Ta lekcja buduje podstawowy mechanizm.

**Typ:** Zbuduj
**Języki:** Python (z PyTorch)
**Wymagania wstępne:** Lekcja 03 fazy 7 (Multi-Head Attention), Lekcja 06 fazy 8 (DDPM)
**Czas:** ~90 minut

## Problem

Generacja autoregresyjna zdominowała modelowanie języka, ponieważ skaluje się przewidywalnie: więcej obliczeń, więcej parametrów, niższa perplexity, lepsze wyniki. Generacja obrazów miała dwa główne podejścia AR przed 2024: PixelRNN/PixelCNN (piksel po pikselu) i DALL-E 1 / Parti / MuseGAN (token po tokenie na kodach VQ-VAE).

Oba cierpiały z powodu problemu kolejności generacji. Piksle i tokeny są ułożone w siatkę 2D, ale model AR musi je odwiedzać w kolejności rastrowej 1D, bo wczesny piksel w rogu nie ma pojęcia, czym obraz ostatecznie się stanie. Jakość generacji skalowała się gorzej niż GPT na tekście i nigdy nie osiągnęła jakości modeli dyfuzyjnych przy dopasowanym zestawie obliczeń.

VAR naprawia problem kolejności generacji, zmieniając to, co jest generowane. Zamiast przewidywać tokeny obrazowe jeden po drugim w przestrzeni, VAR przewiduje cały obraz w rosnących rozdzielczościach. Krok 1: przewiduj token 1x1 (podsumowanie całego obrazu). Krok 2: przewiduj siatkę tokenów 2x2 (grubsze cechy). Krok 3: przewiduj siatkę 4x4. Krok K: przewiduj finalną siatkę (H/p)x(W/p).

Każda skala uczestniczy we wszystkich poprzednich skalach (przyczynowo w "kolejności skalowej") i równolegle w ramach własnej skali. Problem kolejności znika: cały obraz w skali k jest produkowany w jednym przejściu transformatora.

## Koncepcja

### Wieloskalowy tokenizer VQ-VAE

VAR potrzebuje **wieloskalowego tokenizera dyskretnego**. Dla obrazu x produkuje on sekwencję progresywnie wyższych rozdzielczości siatek tokenów:

```
x -> encoder -> latent f
f -> tokenizacja przy 1x1: siatka tokenów z_1 of shape (1, 1)
f -> tokenizacja przy 2x2: siatka tokenów z_2 of shape (2, 2)
...
f -> tokenizacja przy (H/p)x(W/p): siatka tokenów z_K of shape (H/p, W/p)
```

Każde z_k używa tego samego słownika kodów (typowy rozmiar 4096-16384). Tokenizacja na każdej skali nie jest niezależna — jest trenowana tak, że sumowanie residuów na każdej skali rekonstruuje f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

To jest wariant **residualnego VQ**. Skala k przechwytuje to, co skale 1..k-1 przegapiły. Dekoder bierze sumę wszystkich osadzeń skalowych i produkuje obraz.

Wieloskalowy tokenizer VQ jest trenowany raz (jak VQGAN) i potem zamrożony. Cała praca generatywna jest wykonywana przez model autoregresyjny na wierzchu.

### Predykcja następnej skali

Model generatywny to transformator, który widzi tokeny ze wszystkich poprzednich skal i przewiduje tokeny na następnej skali.

Struktura sekwencji wejściowej:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Osadzenia pozycyjne kodują zarówno indeks skali, jak i pozycję przestrzenną w skali. Uwaga jest przyczynowa w kolejności skalowej: token na skali k, pozycja (i, j) może uczestniczyć we wszystkich tokenach na skalach 1..k i w tokenach na skali k samej, które przychodzą wcześniej w dowolnej wewnątrzskalowej kolejności (VAR używa stałej uwagi pozycyjnej bez wewnątrzskalowej przyczynowości — wszystkie pozycje w skali są przewidywane równolegle).

Funkcja straty treningowej: na każdej skali k, przewiduj tokeny z_k przy danych wszystkich poprzednich tokenów skalowych. Funkcja straty entropii krzyżowej na dyskretnych kodach VQ. Ta sama struktura jak GPT, z wyjątkiem że "sekwencja" jest teraz uporządkowana w skali.

### Generacja

Podczas wnioskowania:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

Dla K = 10 skal, generacja to 10 przejść transformatora. Każde przejście produkuje całą swoją skalę równolegle — brak autoregresji per-token w skali. Dla obrazu 256x256 to mniej więcej 10 przejść vs 28-50 DiT.

### Dlaczego następna skala wygrywa z następnym tokenem

Trzy strukturalne zwycięstwa:
1. **Grubsze do drobniejszego alignuje się ze statystykami naturalnych obrazów.** Ludzka percepcja wizualna i zestawy danych obrazów wykazują regularności zależne od skali: niskoczęstotliwościowa struktura jest stabilna i przewidywalna; wysokoczęstotliwościowy szczegół jest warunkowany przez zawartość niskich częstotliwości. Predykcja następnej skali to wykorzystuje.
2. **Równoległa generacja w skali.** W przeciwieństwie do tokenowej AR w stylu GPT, VAR produkuje wszystkie tokeny w skali w jednym kroku. Efektywna długość generacji jest logarytmiczna zamiast liniowa.
3. **Brak obciążenia kolejnością generacji.** Tokeny na skali k widzą całą skalę k-1; nie ma obciążenia "na lewo od" lub "nad" które zmusza wczesne tokeny do podejmowania decyzji przed dostępności późniejszego kontekstu.

### Prawo skalujące

Tian i in. wykazali, że VAR podąża za krzywą prawa potęgowego dla FID na ImageNet — dokładnie jak GPT robi dla perplexity. Podwajanie parametrów lub obliczeń wiarygodnie zmniejsza błąd o połowę. To był pierwszy generatywny model obrazowy, który wykazuje tego typu zachowanie skalowania tak czysto jak modele językowe. Wynik jest taki, że predykcje VAR-scale stają się przewidywalne z obliczeń, nie empiryczne zgadywania per architektura.

### Związek z dyfuzją

VAR i dyfuzja dzielą tę samą historię kompresji danych: oba rozbijają problem generacji na sekwencję łatwiejszych podproblemów.

- Dyfuzja: stopniowo dodaje szum, uczy się cofać jeden krok.
- VAR: stopniowo dodaje rozdzielczość, uczy się przewidywać następną skalę.

To są różne osie przez problem. Obie dają tractable rozkłady warunkowe. Empirycznie VAR jest szybszy przy wnioskowaniu (mniej przejść, wszystko równoległe w skali) i dorównuje lub przebija DiT na klasowo-kondycjonalnym ImageNet. Text-kondycjonalny VAR (VARclip, HART) to aktywny kierunek badań.

## Zbuduj to

W `code/main.py`:
1. Zbuduj miniaturowy **wieloskalowy tokenizer VQ** na syntetycznych danych "obrazowych" (2D pierścienie Gaussowskie).
2. Trenuj **transformator w stylu VAR** do predykcji następnej skali tokenów.
3. Próbkuj wywołując transformator 4 razy (4 skale) i dekodując.
4. Zweryfikuj, że trening uporządkowany w skali sprawia, że generacja jest równoległa w skali.

To jest implementacja pomocnicza. Chodzi o to, żeby zobaczyć maskę uwagi uporządkowaną w skali i równoległą generację w skali faktycznie działającą.

## Wyślij to

Ta lekcja produkuje `outputs/skill-var-tokenizer-designer.md` — umiejętność projektowania wieloskalowego tokenizera: liczba skal, proporcje skal, rozmiar słownika kodów, współdzielenie residuów, architektura dekodera.

## Ćwiczenia

1. **Ablacja liczby skal.** Trenuj VAR z 4, 6, 8, 10 skalami. Mierz jakość rekonstrukcji vs liczbę autoregresyjnych przejść. Więcej skal = drobniejsze residua = lepsza jakość, ale więcej przejść.

2. **Rozmiar słownika kodów.** Trenuj tokenizery z rozmiarami słownika kodów 512, 4096, 16384. Większe słowniki kodów dają lepszą rekonstrukcję, ale trudniejsze przewidywanie. Znajdź kolano.

3. **Sprawdzenie równoległości w skali.** Dla trenowanego VAR, zmierz wzorzec uwagi jawnie. W skali k, czy model uczestniczy w pozycjach międzyskalowych, ale nie wewnątrzskalowych? Zweryfikuj implementację maski.

4. **VAR vs DiT skalowanie.** Dla tego samego zadania klasowo-kondycjonalnego ImageNet, trenuj VAR i DiT przy dopasowanych budżetach parametrów (np. 33M, 130M, 458M). Wykreśl FID vs obliczenia. VAR powinien wyprzedzać DiT przy każdym rozmiarze — odtwórz wynik artykułu w małej skali.

5. **Kondycjonowanie tekstowe.** Rozszerz VAR o osadzenie tekstowe (CLIP pooled) jako dodatkowe wejście kondycjonalne przez adaLN. To jest przepis HART. O ile poprawia się FID na próbkowaniu wyrównanym z tekstem?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| VAR | "Visual AutoRegressive" | Generacja obrazów przez predykcję następnej skali nad piramidą siatek tokenów VQ |
| Predykcja następnej skali | "Przewiduj grubsze, potem drobniejsze" | Model przewiduje tokeny w rosnących rozdzielczościach, warunkując na wszystkich poprzednich skalach |
| Wieloskalowy tokenizer VQ | "Residual VQ" | VQ-VAE który produkuje K siatek tokenów rosnącej rozdzielczości, z deoderem sumującym wszystkie skale |
| Skala k | "Poziom piramidy k" | Jeden z K poziomów rozdzielczości, od 1x1 przy k=1 do (H/p)x(W/p) przy k=K |
| Równoległe w skali | "Jedno przejście na skalę" | Wszystkie tokeny w skali k są przewidywane w jednym przejściu transformatora, nie autoregresyjnie |
| Przyczynowe w skalach | "Uwaga uporządkowana w skali" | Token na skali k może uczestniczyć we wszystkim z skal 1..k, ale nie w skalach k+1..K |
| Residualne VQ | "Additive tokenization" | Tokeny każdej skali kodują residuum pozostawione przez niższe skale; dekoder sumuje wszystkie osadzenia skalowe |
| Prawo skalujące VAR | "Image GPT scaling" | FID podąża za przewidywalnym prawem potęgowym w obliczeniach, jak perplexity modeli językowych |
| HART | "Hybrid VAR + text" | Wariant VAR kondycjonalny tekstowo łączący dekodowanie iteracyjne w stylu MaskGIT ze strukturą skalową VAR |
| Osadzenie pozycji skali | "(scale, row, col) triple" | Kodowanie pozycyjne niesie zarówno indeks skali, jak i współrzędne przestrzenne w skali |

## Dalsze czytanie

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — artykuł VAR, kanoniczne odniesienie
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, linia bazowa porównania dyfuzji
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, rodzina tokenizera którą wieloskalowy tokenizer VAR rozszerza
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, fundament dyskretnej tokenizacji obrazów
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — VAR kondycjonalny tekstem