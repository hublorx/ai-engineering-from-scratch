# StyleGAN

> Większość generatorów miesza `z` w każdej warstwie naraz. StyleGAN rozłożył to na części: najpierw mapuje `z` na pośrednią zmienną `w`, a następnie *wstrzykuje* `w` na każdym poziomie rozdzielczości poprzez AdaIN. Ta jedna zmiana rozplątała przestrzeń latentną i na siedem lat uczyniła fotorealistyczne twarze rozwiązanym problemem.

**Typ:** Budowa
**Języki:** Python
**Wymagania wstępne:** Faza 8 · 03 (GANy), Faza 4 · 08 (Normalizacja), Faza 3 · 07 (CNN-y)
**Szacowany czas:** ~45 minut

## Problem

DCGAN mapuje `z` na obraz poprzez stos transponowanych splotów. Problem polega na tym, że `z` kontroluje wszystko — pozę, oświetlenie, tożsamość, tło — pomieszane ze sobą. Przesuń się wzdłuż jednej osi `z`, a zmienią się wszystkie cztery. Nie możesz poprosić modelu „ta sama osoba, inna poza", ponieważ reprezentacja nie rozkłada się w ten sposób.

Karras i współpracownicy (2019, NVIDIA) zaproponowali: przestań podawać `z` bezpośrednio do warstw splotowych. Podawaj stały tensor `4×4×512` jako wejście sieci. Naucz 8-warstwowy MLP, który mapuje `z ∈ Z → w ∈ W`. Wstrzykuj `w` na każdej rozdzielczości poprzez *adaptacyjną normalizację instancji* (AdaIN): normalizuj każdą mapę cech splotu, a następnie skaluj i przesuwaj przez afiniczne rzutowania `w`. Dodaj szum dla każdej warstwy dla stochastycznego detalu (pory skóry, pasma włosów).

Rezultat: `W` ma mniej więcej ortogonalne osie dla „stylu wysokiego poziomu" (poza, tożsamość) vs „stylu finezyjnego" (oświetlenie, kolor). Możesz zamieniać style między dwoma obrazami, używając `w` obrazu A dla niskich poziomów rozdzielczości i `w` obrazu B dla wysokich. To odblokowało edycję, stylizację między domenami i całą linię badań „inwersji StyleGAN".

## Koncepcja

![StyleGAN: sieć mapująca + AdaIN + szum dla każdej warstwy](../assets/stylegan.svg)

**Sieć mapująca.** `f: Z → W`, 8-warstwowy MLP. `Z = N(0, I)^512`. `W` nie jest wymuszone jako Gaussowskie — uczy się adaptowanego do danych kształtu.

**Sieć syntezy.** Zaczyna się od nauczonej stałej `4×4×512`. Każdy blok rozdzielczości: `upsample → conv → AdaIN(w_i) → szum → conv → AdaIN(w_i) → szum`. Rozdzielczości podwajają się: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

gdzie `y_scale` i `y_bias` pochodzą z afinicznych rzutowań `w`. Normalizuj per mapa cech, następnie restylizuj. „Styl" tutaj to statystyki pierwszego i drugiego rzędu mapy cech.

**Szum dla każdej warstwy.** Jednokanałowy szum Gaussowski dodawany do każdej mapy cech, skalowany przez nauczony współczynnik per-kanał. Kontroluje stochastyczny detal bez wpływu na globalną strukturę.

**Sztuczka obcięcia.** W czasie wnioskowania, próbkuj `z`, oblicz `w = mapping(z)`, a następnie `w' = ŵ + ψ·(w - ŵ)` gdzie `ŵ` to średnie `w` z wielu próbek. `ψ < 1` wymienia różnorodność na jakość. Prawie każda demonstracja StyleGAN używa `ψ ≈ 0.7`.

## StyleGAN 1 → 2 → 3

| Wersja | Rok | Innowacja |
|--------|------|------------|
| StyleGAN | 2019 | Sieć mapująca + AdaIN + szum + progresywne narastanie. |
| StyleGAN2 | 2020 | Demodulacja wag zastępuje AdaIN (naprawia artefakty kropelkowe); architektura skip/residual; regularyzacja długości ścieżki. |
| StyleGAN3 | 2021 | Splot wolny od aliasów + jądra równoważne; eliminuje przyklejanie tekstury do siatki pikseli. |
| StyleGAN-XL | 2022 | Warunkowe na klasę, 1024², ImageNet. |
| R3GAN | 2024 | Zmiana marki z mocniejszą regularyzacją; zmniejsza dystans do dyfuzji na FFHQ-1024 przy 20x mniejszej liczbie parametrów. |

W 2026 StyleGAN3 pozostaje domyślnym wyborem dla (a) wąskodomenowego fotorealizmu przy wysokim FPS, (b) adaptacji domeny przy kilku próbkach (trenuj na nowym zbiorze danych ze 100 obrazami, zamroź sieć mapującą), (c) edycji opartej na inwersji (znajdź `w`, które rekonstruuje prawdziwe zdjęcie, a następnie edytuj to `w`). Dla otwartodomenowego generowania tekst-na-obraz, to nie jest narzędzie — dyfuzja jest.

## Zbuduj to

`code/main.py` implementuje zabawkową „style-GAN lite" w 1-D: sieć mapującą MLP, funkcję syntezy, która przyjmuje nauczony wektor stały i moduluje go skalą/obciążeniem pochodzącym od `w`, oraz szum dla każdej warstwy. Pokazuje, że wstrzykiwanie `w` poprzez modulację afiniczną dorównuje lub przewyższa konkatenację `z` do wejścia generatora.

### Krok 1: sieć mapująca

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Krok 2: adaptacyjna normalizacja instancji

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Skala i obciążenie per mapa cech pochodzą od `w` poprzez projekcję liniową.

### Krok 3: szum dla każdej warstwy

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma per-kanał jest nauczalna.

## Pułapki

- **Artefakty kropelkowe.** StyleGAN 1 produkował kropelkowe plamy w mapach cech, ponieważ AdaIN zerował średnią. Demodulacja wag StyleGAN2 naprawia to poprzez skalowanie wag splotu zamiast aktywacji.
- **Przyklejanie tekstury.** StyleGAN 1 i 2 tekstury podążały za współrzędnymi pikseli, nie obiektu (widoczne przy interpolacji). Sploty wolne od aliasów StyleGAN3 naprawiają to poprzez filtry okienkowe sinc.
- **Pokrycie mody.** Obcięcie `ψ < 0.7` wygląda czysto, ale próbkuje z wąskiego stożka; użyj `ψ = 1.0` jeśli potrzebujesz różnorodności.
- **Inwersja jest stratna.** Inwersja prawdziwego zdjęcia w `W` jest zwykle wykonywana poprzez optymalizację lub enkoder (e4e, ReStyle, HyperStyle). Wyniki dryfują przez wiele iteracji.

## Zastosuj to

| Przypadek użycia | Podejście |
|----------|----------|
| Fotorealistyczne ludzkie twarze (anime, produkt, wąska domena) | StyleGAN3 FFHQ / niestandardowy fine-tune |
| Edycja twarzy ze zdjęcia | e4e inwersja + StyleSpace / kierunki InterFaceGAN |
| Zamiana twarzy / reanimacja | StyleGAN + enkoder + mieszanie |
| Potoki awatarów | StyleGAN3 w/ ADA dla fine-tune przy małych danych |
| Adaptacja domeny z kilku obrazów | Zamroź sieć mapującą, fine-tune syntezy |
| Wielomodowa lub tekstowo-warunkowa generacja | Nie — użyj dyfuzji |

Dla produktowych demonstracji, gdzie odpowiedź brzmi „zdjęcie twarzy osoby", StyleGAN bije dyfuzję pod względem kosztu wnioskowania (pojedynczy przebieg forward, <10ms na 4090) i ostrości przy tej samej granicy jakości.

## Wyślij to

Zapisz `outputs/skill-stylegan-inversion.md`. Umiejętność przyjmuje prawdziwe zdjęcie i zwraca: metodę inwersji (e4e / ReStyle / HyperStyle), oczekiwany latent loss, budżet edycji (jak daleko w `W` można się przesunąć zanim pojawią się artefakty) oraz listę sprawdzonych kierunków edycji (wiek, ekspresja, poza).

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` z `adain_on=True` i `adain_on=False`. Porównaj rozrzut wyników dla ustalonego latentu vs zaburzonego latentu.
2. **Średnie.** Zaimplementuj regularyzację mieszania: dla batcha treningowego oblicz `w_a`, `w_b` i zastosuj `w_a` dla pierwszej połowy syntezy i `w_b` dla drugiej połowy. Czy dekoder uczy się rozdzielonych stylów?
3. **Trudne.** Weź pretrained model StyleGAN3 FFHQ (ffhq-1024.pkl). Znajdź kierunek `w` kontrolujący „uśmiech" trenując SVM na oznaczonych próbkach; podaj jak daleko można pójść zanim tożsamość zacznie dryfować.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Sieć mapująca | „MLP" | `f: Z → W`, 8 warstw, rozdziela geometrię latentną od statystyk danych. |
| Przestrzeń W | „Przestrzeń stylu" | Wyjście sieci mapującej; mniej więcej rozdzielona. |
| AdaIN | „Adaptacyjna norma instancji" | Normalizuj mapę cech, następnie skaluj + przesuwaj przez projekcję `w`. |
| Sztuczka obcięcia | „Psi" | `w = średnia + ψ·(w - średnia)`, ψ<1 wymienia różnorodność na jakość. |
| Regularyzacja długości ścieżki | „PL reg" | Kara za duże zmiany obrazu na jednostkę zmiany `w`; wygładza `W`. |
| Demodulacja wag | „Poprawka StyleGAN2" | Normalizuj wagi splotu zamiast aktywacji; zabija artefakty kropelkowe. |
| Wolny od aliasów | „Sztuczka StyleGAN3" | Filtry okienkowe sinc; eliminuje przyklejanie tekstury do siatki pikseli. |
| Inwersja | „Znajdź w dla prawdziwego obrazu" | Optymalizuj lub koduj `x → w` tak, że `G(w) ≈ x`. |

## Uwaga produkcyjna: dlaczego StyleGAN nadal wysyłany jest w 2026

StyleGAN3 na 4090 generuje twarz FFHQ 1024² w mniej niż 10 ms — `num_steps = 1`, brak dekodu VAE, brak przebiegu cross-attention. W terminologii produkcyjnej to minimalne opóźnienie dla dowolnego generatora obrazów. Potok SDXL z 50 krokami + dekod VAE przy tej samej rozdzielczości to ~3 sekundy. To jest **300× różnica**, a dla wąskodomenowych produktów (usługi awatarów, potoki dokumentów ID, generowanie twarzy stockowych) wygrywa pod względem TCO.

Dwie operacyjne konsekwencje:

- **Brak schedulera, brak batchera.** Statyczny batch przy docelowej zajętości jest optymalny. Ciągły batching (niezbędny dla LLM-ów i dyfuzji) nie daje żadnej korzyści, bo każde żądanie zajmuje te same FLOPS-y.
- **`ψ` to bezpieczny pokrętło.** `ψ < 0.7` próbkuje z wąskiego stożka zakresu sieci mapującej. To jedyne narzędzie, jakie ma warstwa obsługująca na kontrolę wariancji próbek. Obniż `ψ` przy szczytowym obciążeniu, podnieś dla premium użytkowników.

## Dalsza lektura

- Karras i współpracownicy (2019). A Style-Based Generator Architecture for GANs — StyleGAN.
- Karras i współpracownicy (2020). Analyzing and Improving the Image Quality of StyleGAN — StyleGAN2.
- Karras i współpracownicy (2021). Alias-Free Generative Adversarial Networks — StyleGAN3.
- Tov i współpracownicy (2021). Designing an Encoder for StyleGAN Image Manipulation — e4e inversion.
- Sauer i współpracownicy (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets — StyleGAN-XL.
- Huang i współpracownicy (2024). R3GAN: The GAN is dead; long live the GAN! — nowoczesny minimalny przepis na GAN.