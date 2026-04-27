# Modele generatywne — systematyzacja i historia

> Każdy model obrazu, tekstu, wideo i model 3D mieści się w jednym z pięciu kubłów. Wybierz zły kubieł i będziesz walczyć z matematyką przez tygodnie. Wybierz właściwy, a dwanaście ostatnich lat postępu w tej dziedzinie ułoży się czysto w twojej głowie.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 2 (Podstawy ML), Faza 3 (Deep Learning — rdzeń), Faza 7 · 14 (Transformery)
**Szacowany czas:** ~45 minut

## Problem

Model generatywny wykonuje jedną pracę: mając próbki treningowe z nieznanej dystrybucji `p_data(x)`, wyprowadza nowe próbki, które wyglądają jakby pochodziły z tej samej dystrybucji. Twarze, zdania, pliki MIDI, struktury białek — ten sam problem, jeśli przymkniesz oko.

Sedno sprawy polega na tym, że `p_data` żyje w przestrzeni z milionami wymiarów (obraz 512x512 RGB to ~786k wymiarów), próbki siedzą na cienkiej rozmaitości wewnątrz tej przestrzeni, a ty masz może zaledwie 10M przykładów. Metoda siłowa w obliczaniu gęstości jest skazana na porażkę. Każdy model generatywny to kompromis, który wymienia jeden trudny problem na nieco łatwiejszy.

Pięć rodzin przetrwało ostatnie dwanaście lat. Wiedza o tym, jaki kompromis każda rodzina zawiera, mówi ci, dlaczego wygrywa w niektórych zadaniach, a w innych się załamuje.

## Koncepcja

![Pięć rodzin modeli generatywnych — systematyzacja według tego, co modelują](../assets/taxonomy.svg)

**1. Jawna gęstość, ostra (tractable).** Zapisz `log p(x)` jako sumę, którą możesz faktycznie obliczyć. Modele autoregresyjne (PixelCNN, WaveNet, GPT) rozkładają `p(x) = ∏ p(x_i | x_<i)`. Normalizujące przepływy (RealNVP, Glow) budują `p(x)` jako odwracalną transformację prostego rozkładu bazowego. Zaleta: dokładne prawdopodobieństwo (likelihood), czysta funkcja straty treningowej. Wada: wnioskowanie autoregresyjne jest sekwencyjne (wolne dla długich sekwencji), przepływy wymagają odwracalnych architektur (architektonicznie restrykcyjne).

**2. Jawna gęstość, przybliżona.** Ogranicz `log p(x)` od dołu (ELBO) i optymalizuj to ograniczenie. VAE (Kingma 2013) używają enkodera-dekodera z wariacyjnym posterierem. Modele dyfuzyjne (DDPM, Ho 2020) trenują denoiser, który niejawnie optymalizuje ważone ELBO. Dyfuzja jest dominującym backbone'em obrazu, wideo i 3D w 2026 roku.

**3. Niejawne gęstość.** Pomiń gęstość całkowicie; naucz generator `G(z)`, który produkuje próbki, i dyskryminator `D(x)`, który rozpoznaje prawdziwe od fałszywych. GANy (Goodfellow 2014). Szybkie przy wnioskowaniu (jeden przebieg do przodu), ale niesławnie niestabilne podczas treningu. StyleGAN 1/2/3 pozostają stanem techniki dla fotorealistycznych obrazów z wąskiej domeny (twarze, sypialnie) nawet w 2026 roku.

**4. Oparte na wyniku (score-based) / czasie ciągłym.** Naucz bezpośrednio gradient logarytmu gęstości `∇_x log p(x)` (wynik, score). Song i Ermon (2019) pokazali, że score matching uogólnia dyfuzję na SDE. Flow matching (Lipman 2023) to hit 2024-2026: trening bez symulacji, prostsze ścieżki, 4-10x szybsze próbkowanie niż DDPM. Stable Diffusion 3, Flux, AudioCraft 2 wszystkie używają flow matching.

**5. Oparte na tokenach, autoregresyjne na dyskretnych kodach.** Skompresuj dane wysokiego wymiaru za pomocą VQ-VAE lub kwantyzatora rezydualnego w krótką sekwencję dyskretnych tokenów, następnie użyj Transformera do modelowania sekwencji tokenów. Parti, MuseNet, AudioLM, VALL-E, tokenizer Sora wszystkie to używają. To kubła 1 plus nauczony tokenizer.

## Krótka historia

| Rok | Model | Dlaczego miał znaczenie |
|------|-------|--------------------------|
| 2013 | VAE (Kingma) | Pierwszy głęboki model generatywny z użyteczną funkcją straty treningowej. |
| 2014 | GAN (Goodfellow) | Niejawne gęstość, bez prawdopodobieństwa — zaskakująco ostre próbki. |
| 2015 | DRAW, PixelCNN | Sekwencyjna generacja obrazów. |
| 2017 | Glow, RealNVP | Odwracalne przepływy; dokładne prawdopodobieństwo z głębią. |
| 2017 | Progressive GAN | Pierwsze twarze w megapikselach. |
| 2019 | StyleGAN / StyleGAN2 | Fotorealistyczne twarze wciąż trudne do pokonania w tej jednej domenie. |
| 2020 | DDPM (Ho) | Dyfuzja staje się praktyczna. |
| 2021 | CLIP, DALL-E 1, VQGAN | Tekst-do-obrazu staje się głównym nurtem. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + warunkowanie tekstowe = commodity. |
| 2022 | ControlNet, LoRA | Precyzyjna kontrola nad pretrained dyfuzją. |
| 2023 | SDXL, Midjourney v5, Flow matching | Skalowanie + lepsza dynamika treningu. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Wideo dyfuzja; flow matching wygrywa. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Wideo klasy produkcyjnej. |
| 2026 | Consistency + Rectified Flow | Jednokrokowe próbkowanie z backbone'ów dyfuzyjnych. |

## Triangulacja pięciu pytań

Gdy pojawi się nowy artykuł o modelu generatywnym, odpowiedz na te pięć pytań zanim przeczytasz sekcję metod.

1. **Co jest modelowane?** Piksele, latenty, dyskretne tokeny, Gaussy 3D, siatki, fale?
2. **Czy gęstość jest jawna czy niejawna?** Czy zapisują `log p(x)`?
3. **Próbkowanie: jednokrokowe czy iteracyjne?** Iteracyjne oznacza wolniejsze wnioskowanie; jednokrokowe zwykle oznacza adversarial lub destylowane.
4. **Warunkowanie: bezwarunkowe, klasowe, tekstowe, obrazowe, pozy?** To determinuje funkcję straty i rusztowanie architektoniczne.
5. **Ewaluacja: FID, CLIP score, IS, preferencje ludzkie, dokładność zadania?** Każda ma znane tryby awarii (zobacz Lekcja 14).

Będziesz odpowiadać na te pięć pytań dla każdej lekcji w tej fazie. Pod koniec będzie to odruch.

## Zbuduj to

Kod tej lekcji to lekka wizualizacja: dopasuj 1-D mieszankę Gaussian do próbek używając trzech podejść (estymacja jądrowa gęstości, dyskretny histogram i najbliższy-sąsiad "GAN-owski" generator), żebyś mógł zobaczyć różnicę między jawną a niejawną gęstością na problemie, który możesz wydrukować na jednym ekranie.

Uruchom `code/main.py`. Rysuje 2000 próbek z dwumodalnej mieszanki Gaussian, potem drukuje:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Zauważ: pierwsze dwa pozwalają zapytać "jak prawdopodobny jest ten punkt?" Trzeci nie może. To jest rozróżnienie *jawne vs niejawne*, które będzie miało znaczenie dla każdej przyszłej lekcji.

## Użyj tego

Która rodzina, dla którego zadania, w 2026?

| Zadanie | Najlepsza rodzina | Dlaczego |
|---------|-------------------|----------|
| Fotorealistyczne twarze, wąska domena | StyleGAN 2/3 | Wciąż najostrzejsze, najszybsze wnioskowanie. |
| Ogólne tekst-do-obraz | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| Szybki tekst-do-obraz | Rectified flow + destylacja | SDXL-Turbo, SD3-Turbo, LCM. |
| Tekst-do-wideo | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| Mowa + muzyka | Token-based AR (AudioLM, VALL-E, MusicGen) lub flow matching (AudioCraft 2) | Dyskretne tokeny skalują tanio. |
| Sceny 3D | Gaussian Splatting fit, dyfuzja prior | 3D-GS do rekonstrukcji, dyfuzja do nowych widoków. |
| Estymacja gęstości (bez próbkowania) | Flows | Jedyna rodzina z dokładnym `log p(x)`. |
| Symulacja / fizyka | Flow matching, score SDE | Proste ścieżki, gładkie pola wektorowe. |

## Wyślij to

Zapisz jako `outputs/skill-model-chooser.md`.

Umiejętność bierze opis zadania i wyprowadza: (1) którą rodzinę użyć, (2) ranking trzech opcji open source i trzech hostowanych, (3) prawdopodobny tryb awarii, na który powinieneś uważać, i (4) budżet compute/czasu.

## Ćwiczenia

1. **Łatwe.** Dla każdego z tych pięciu produktów zidentyfikuj rodzinę i backbone: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Dowody powinny pochodzić z publicznych raportów technicznych.
2. **Średnie.** Artykuł, który będziesz czytać jutro, twierdzi 100x szybsze próbkowanie niż dyfuzja. Zapisz trzy pytania, żeby sprawdzić, czy przyspieszenie przetrwa warunkowanie i wysoką rozdzielczość.
3. **Trudne.** Weź jedną domenę, która cię interesuje (np. struktura białek, CAD, cząsteczki, trajektorie). Odpowiedz na pięciopytaniową triangulację dla obecnego SOTA w tej domenie i naszkicuj, co lepszy model by zmienił.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Model generatywny | "Tworzy nowe rzeczy" | Uczy samplera dla `p_data(x)`, opcjonalnie udostępnia `log p(x)`. |
| Jawna gęstość | "Możesz to obliczyć" | Model podaje zamkniętą formę lub ostra `log p(x)`. |
| Niejawne gęstość | "GAN-style" | Tylko sampler — nie ma sposobu na obliczenie `p(x)` dla danego punktu. |
| ELBO | "Evidence lower bound" | Ostra dolna granica na `log p(x)`; VAE i dyfuzja ją optymalizują. |
| Score | "Gradient log-gęstości" | `∇_x log p(x)`; modele dyfuzyjne i SDE uczą się tego pola. |
| Hipoteza o rozmaitości | "Dane żyją na powierzchni" | Dane wysokiego wymiaru koncentrują się na niskowymiarowej rozmaitości; dlatego redukcja wymiarów działa. |
| Autoregresyjny | "Przewiduj następny kawałek" | Rozkładaj łączny rozkład jako iloczyn warunkowych. |
| Latent | "Skompresowany kod" | Reprezentacja niskiego wymiaru, z której dekoder może zrekonstruować wejście. |

## Uwaga produkcyjna: pięć rodzin, pięć kształtów wnioskowania

Każda rodzina odpowiada innej krzywej kosztu serwera wnioskowania. Literatura produkcyjnego wnioskowania ramuje wnioskowanie LLM jako prefill + decode; to samo rozbicie tutaj się stosuje:

- **Autoregresyjny (kubła 1 i 5).** Sekwencyjny decode dominuje latency; KV-cache, continuous batching i speculative decoding wszystkie stosują się bezpośrednio.
- **VAE / dyfuzja / flow-matching (kubła 2 i 4).** Nie ma decode w sensie LLM. Koszt = `num_steps × step_cost`, a `step_cost` to przebieg Transformera lub U-Neta przy pełnej rozdzielczości latentnej. Pokrętła produkcyjne to liczba kroków (DDIM / DPM-Solver / destylacja), batch size i precyzja (bf16 / fp8 / int4).
- **GAN (kubła 3).** Jeden przebieg do przodu. Bez schedule, bez KV-cache. TTFT ≈ całkowite latency. Dlatego StyleGAN wciąż wygrywa w wąskiej domenie UX.

Gdy widzisz "szybsze niż dyfuzja" w abstrakcji artykułu, przetłumacz to na "mniej kroków × ten sam koszt kroku" lub "tyle samo kroków × tańszy koszt kroku". Wszystko inne to marketing.

## Dalsza lektura

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — artykuł o GAN.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — artykuł o VAE.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — artykuł o DDPM.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — dyfuzja jako SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — artykuł o flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.