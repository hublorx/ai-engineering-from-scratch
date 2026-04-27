# Inpainting, Outpainting i edycja obrazu

> Text-to-image tworzy nowe rzeczy. Inpainting naprawia stare. W produkcji, 70% płatnej pracy nad obrazem to edycja — zamiana tła, usunięcie logo, rozszerzenie płótna, regeneracja dłoni. Inpainting to miejsce, gdzie dyfuzja sprawdza się najlepiej.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Czas:** ~75 minut

## Problem

Klient przesyła idealne zdjęcie produktu z rozpraszającym znakiem w tle. Chcesz wymazać znak i pozostawić wszystko inne piksel-po-pikselu identyczne. Nie możesz uruchomić text-to-image od zera — wynik będzie miał inny kolor, inne oświetlenie, inny kąt produktu. Chcesz zregenerować *tylko* zamaskowany region, a regeneracja ma szanować otaczający kontekst.

To jest inpainting. Warianty:

- **Inpainting.** Regeneruj wewnątrz maski, zachowaj piksele na zewnątrz.
- **Outpainting.** Regeneruj na zewnątrz maski (lub poza płótnem), zachowaj wnętrze.
- **Edycja obrazu.** Regeneruj cały obraz, ale zachowaj wierność semantyczną lub strukturalną oryginałowi (SDEdit, InstructPix2Pix).

Każda dyfuzyjna pipeline w 2026 roku ma tryb inpainting. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. Działają na tej samej zasadzie.

## Koncepcja

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### Naiwne podejście (i dlaczego jest błędne)

Uruchom standardowy text-to-image z maską. W każdym kroku próbkowania zastąp zamaskowany region潛 latentnego szumu forward-diffused czystym obrazem. To działa... źle. Artefakty graniczne przesączają się, ponieważ model nie ma informacji o tym, co znajduje się w zamaskowanym regionie.

### Właściwy model inpainting

Trenuj zmodyfikowany U-Net, który przyjmuje 9 kanałów wejściowych zamiast 4:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

Dodatkowe kanały to kopia obrazu źródłowego zakodowanego przez VAE plus jednokanałowa maska. Podczas treningu losowo maskujesz regiony obrazu i trenujesz model do denoisingu tylko zamaskowanego regionu, podczas gdy zamaskowany region jest podawany jako czysty sygnał warunkujący. W czasie wnioskowania model może "widzieć", co otacza zamaskowany region i generuje spójne uzupełnienia.

SD-Inpaint, SDXL-Inpaint, Flux-Fill wszystkie używają tego 9-kanałowego (lub analogicznego) wejścia. Diffusers `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — darmowa edycja

Dodaj szum do obrazu źródłowego do pewnego pośredniego `t`, a następnie uruchom odwrotny łańcuch od `t` do 0 z nowym promptem. Bez ponownego treningu. Wybór początkowego `t` handluje wiernością za kreatywną swobodę:

- `t/T = 0.3` → prawie identyczny z źródłem, małe zmiany stylistyczne
- `t/T = 0.6` → umiarkowane edycje, zachowuje zgrubną strukturę
- `t/T = 0.9` → generowane z prawie szumu, minimalne zachowanie źródła

### InstructPix2Pix (Brooks et al., 2023)

Dostrój dyfuzyjny model na trójkach `(input_image, instruction, output_image)`. We wnioskowaniu warunkuj zarówno na obrazie wejściowym, jak i instrukcji tekstowej ("zmień na zachód słońca", "dodaj smoka"). Dwie skale CFG: skala obrazu i skala tekstu.

### RePaint (Lugmayr et al., 2022)

Zachowaj standardowy bezwarunkowy model dyfuzyjny. W każdym odwrotnym kroku ponownie próbkuj — skacz czasem do bardziej zaszumionego stanu i regeneruj. Unika artefaktów granicznych. Używany, gdy nie masz wytrenowanego modelu inpainting.

## Zbuduj to

`code/main.py` implementuje zabawkowy 1-W inpainting scheme na 5-wymiarowych danych. Trenujemy DDPM na 5-D danych mieszaninowych, gdzie każda próbka to 5 floatów z jednego z dwóch klastrów. We wnioskowaniu "maskujemy" 2 z 5 wymiarów, wstrzykujemy zaszumioną-przekierowaną wersję niezamaskowanych trzech w każdym kroku i regenerujemy tylko zamaskowane wymiary.

### Krok 1: dane DDPM 5-D

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Krok 2: trenuj denoiser na wszystkich 5 wymiarach

Standardowy DDPM. Sieć wyprowadza 5-D predykcję szumu dla 5-D zaszumionego wejścia.

### Krok 3: we wnioskowaniu, mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

To jest naiwne podejście i działa na zabawkowych 1-D danych. Prawdziwy inpainting obrazu używa 9-kanałowego wejścia, ponieważ spójność tekstury ma większe znaczenie.

### Krok 4: outpainting

Outpainting to inpainting z odwróconą maską: maskuj nowe (wcześniej nieistniejące) płótno, wypełnij resztę oryginałem. Identyczny cel treningowy.

## Pułapki

- **Szewy.** Naiwne podejście pozostawia widoczne granice, ponieważ informacje o gradientach nie przepływają przez maskę. Napraw: dylatuj maskę o 8-16 pikseli lub użyj właściwego modelu inpainting.
- **Przeciek maski.** Jeśli zamaskowany region obrazu warunkującego jest niskiej jakości lub zaszumiony, zanieczyszcza generację wewnątrz maski. Delikatnie denoisuj lub rozmyj.
- **CFG oddziałuje z rozmiarem maski.** Wysokie CFG na małej masce = nasyconaplatka. Zmniejsz CFG dla małych edycji.
- **Klif wierności SDEdit.** Przejście z `t/T = 0.5` na `t/T = 0.6` może utracić tożsamość obiektu. Przeszukuj i checkpointuj.
- **Niedopasowanie promptu.** Prompt powinien opisywać *cały* obraz, nie tylko nową treść. "Kot siedzący na krześle", a nie "kot".

## Użyj tego

| Zadanie | Pipeline |
|---------|----------|
| Usuń obiekt, mała maska | SD-Inpaint lub Flux-Fill, standardowy prompt |
| Zastąp niebo | SD-Inpaint + "niebieskie niebo o zachodzie słońca" |
| Rozszerz płótno | SDXL tryb outpaint (8px feather) lub Flux-Fill z maską outpaint |
| Zregeneruj dłoń/twarz | SD-Inpaint z promptem ponownie opisującym obiekt + ControlNet-Openpose |
| Zmień styl jednego regionu | SDEdit przy `t/T=0.5` na zamaskowanym regionie |
| "Zmień na zachód słońca" | InstructPix2Pix lub Flux-Kontext |
| Zastąpienie tła | Maska SAM → SD-Inpaint |
| Ultra-wysoka wierność | Flux-Fill lub GPT-Image (hostowane) dla najtrudniejszych przypadków |

SAM (Meta's Segment Anything, 2023) + dyfuzyjny inpaint to pipeline usuwania tła w 2026. SAM 2 (2024) działa na wideo.

## Wyślij to

Zapisz `outputs/skill-editing-pipeline.md`. Umiejętność przyjmuje oryginalny obraz + opis edycji + opcjonalną maskę (lub prompt SAM) i wyprowadza: podejście do generowania maski, model bazowy, skale CFG (obraz + tekst), SDEdit-t lub tryb inpainting oraz listę kontrolną QA.

## Ćwiczenia

1. **Łatwe.** W `code/main.py`, zmień frakcję wymiarów zamaskowanych od 0.2 do 0.8. Przy jakiej frakcji jakość inpaint (residual w zamaskowanych wymiarach) równa się generowaniu bezwarunkowemu?
2. **Średnie.** Zaimplementuj RePaint: w każdym 10. odwrotnym kroku, skacz 5 kroków wstecz (dodaj szum) i ponownie denoisuj. Zmierz, czy zmniejsza to residual graniczny na krawędzi maski.
3. **Trudne.** Użyj Hugging Face diffusers do porównania: SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill na 20 zadaniach regeneracji twarzy. Oceniaj przestrzeganie pozy i zachowanie tożsamości osobno.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Inpainting | "Wypełnij dziurę" | Regeneruj wewnątrz maski; zachowaj piksele na zewnątrz. |
| Outpainting | "Rozszerz płótno" | Regeneruj poza płótnem; zachowaj wnętrze. |
| 9-kanałowy U-Net | "Właściwy model inpainting" | U-Net z `noisy | encoded-source | mask` jako wejściem. |
| SDEdit | "Img2img z poziomem szumu" | Szum do czasu `t`, denoisuj z nowym promptem. |
| InstructPix2Pix | "Edycje tylko tekstowe" | Dostrojony dyfuzyjny na trójkach (obraz, instrukcja, wynik). |
| RePaint | "Bez ponownego treningu" | Ponownie szumuj okresowo podczas odwrotnego przejścia, aby zmniejszyć szwy. |
| SAM | "Segment Anything" | Generator masek przez kliknięcia lub ramki; paruje z inpaint. |
| Flux-Kontext | "Edycja z kontekstem" | Wariant Flux, który przyjmuje obraz referencyjny + instrukcję do edycji. |

## Uwaga produkcyjna: pipeline edycyjne są wrażliwe na opóźnienia

Użytkownicy edytujący obraz oczekują round-tripów poniżej 5 sekund. SDXL-Inpaint w 30 krokach przy 1024² to 3-4 s na L4, plus generowanie maski SAM (~200 ms) i VAE encode/decode (~500 ms łącznie). W kontekście produkcyjnym, to jest bound przez TTFT, a nie throughput — batch 1, niska współbieżność, minimalizuj każdy etap:

- **SAM-H jest tym wolnym.** SAM-H przy 1024² to ~200 ms; SAM-ViT-B to ~40 ms z niewielką utratą jakości. SAM 2 (wideo) dodaje narzut czasowy; nie używaj go do edycji pojedynczych obrazów.
- **Pomiń encode gdy możliwe.** `pipe.image_processor.preprocess(img)` enkoduje do latentów. Jeśli masz latenty z poprzedniej generacji (typowe w iteracyjnych interfejsach edycji), przekaż je bezpośrednio przez `latents=...`, aby pominąć jeden VAE encode.
- **Dylatacja maski ma znaczenie także dla throughput.** Mała maska oznacza, że większość forward pass U-Neta jest marnowana (zamaskowane piksele i tak są clampowane). `diffusers`' `StableDiffusionInpaintPipeline` uruchamia pełny U-Net niezależnie; tylko właściwe warianty inpainting z 9 kanałami wykorzystują masked compute.
- **Flux-Kontext to odpowiedź 2025.** Pojedynczy forward pass przez `(source_image, instruction)` — bez osobnej maski, bez przeszukiwania szumu SDEdit. Na H100 wysyła edycję w ~1.5 s. Architektoniczna lekcja: złóż etapy.

## Dalsza lektura

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — inpainting bez treningu.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — edycja instrukcjami tekstowymi.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, źródło masek.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — wideo SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — edycja na poziomie attention.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — narzędzia 2024.