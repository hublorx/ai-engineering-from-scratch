# Conditional GANs i Pix2Pix

> Pierwszym dużym przełomem lat 2014–2017 było kontrolowanie tego, co generuje GAN. Dołącz etykietę, obraz lub zdanie. Pix2Pix zrobił wersję obrazową i nadal bije każdy generyczny model text-to-image na wąskich zadaniach image-to-image.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Szacowany czas:** ~75 minut

## Problem

Bezwarunkowy GAN próbkuje arbitralne twarze. Przydatne na demo, bezużyteczne w produkcji. Potrzebujesz: *mapuj szkic na zdjęcie*, *mapuj mapę na zdjęcie lotnicze*, *mapuj scenę dzienną na nocną*, *koloryzuj obraz w skali szarości*. We wszystkich tych przypadkach dostajesz obraz wejściowy `x` i musisz wygenerować `y` z pewną semantyczną odpowiedniością. Jest wiele prawdopodobnych `y` dla danego `x`. Błąd średniokwadratowy spłaszcza je w papkę. Loss adwersarialny nie, bo "wygląda realistycznie" jest ostre.

Conditional GAN (Mirza & Osindero, 2014) dodaje warunek `c` jako wejście do `G` i `D`. Pix2Pix (Isola et al., 2017) to uspecjalizował: warunek to pełny obraz wejściowy, generator to U-Net, dyskryminator to *patch-based* klasyfikator (PatchGAN), a loss to adversarial + L1. Ten przepis przewyższa modele text-to-image od zera na wąskich domenach image-to-image nawet w 2026, bo jest trenowany na *paired data* — masz dokładnie ten sygnał, którego potrzebujesz.

## Koncepcja

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**G warunkowy.** `G(x, z) → y`. W Pix2Pix, `z` to dropout wewnątrz G (brak szumu wejściowego — Isola stwierdził, że jawny szum był ignorowany).

**D warunkowy.** `D(x, y) → [0, 1]`. Wejście to *para* (warunek, wyjście). To jest kluczowa różnica: D musi ocenić, czy `y` jest spójne z `x`, nie tylko czy `y` wygląda realistycznie.

**Generator U-Net.** Encoder-decoder z połączeniami skip przez bottleneck. Krytyczne dla zadań, gdzie input i output dzielą niskopoziomową strukturę (krawędzie, sylwetkę). Bez skipów, szczegóły wysokiej częstotliwości znikają.

**Dyskryminator PatchGAN.** Zamiast wyświetlać jeden wynik real/fake, D wyświetla siatkę `N×N`, gdzie każda komórka ocenia receptive field ~70×70 pikseli. Uśrednione. To jest założenie Markov random field: realizm jest lokalny. Znacznie szybsze trenowanie, mniej parametrów, ostrzejszy output.

**Loss.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

Składnik L1 stabilizuje trening i popycha G w kierunku znanego targetu. L1 daje ostrzejsze krawędzie niż L2 (mediany, nie średnie). `λ = 100` było domyślne w Pix2Pix.

## CycleGAN — gdy nie masz par

Pix2Pix potrzebuje paired `(x, y)` data. CycleGAN (Zhu et al., 2017) porzuca to wymaganie za cenę dodatkowego lossa: *cycle consistency loss*. Dwa generatory `G: X → Y` i `F: Y → X`. Trenuj je tak, żeby `F(G(x)) ≈ x` i `G(F(y)) ≈ y`. To pozwala tłumaczyć konie na zebry, lato na zimę, bez paired przykładów.

W 2026, unpaired image-to-image jest głównie robione przez diffusion (ControlNet, IP-Adapter) zamiast CycleGAN, ale idea cycle-consistency przetrwała w prawie każdym paperze o unpaired domain adaptation.

## Zbuduj to

`code/main.py` implementuje tiny conditional GAN na danych 1-D. Warunek `c` to etykieta klasy (0 lub 1). Zadanie: wygeneruj próbkę z rozkładu warunkowego dla danej klasy.

### Krok 1: dodaj warunek do wejść G i D

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot encoding to najprostszy sposób. Większe modele używają learned embeddings, modulacji FiLM lub cross-attention.

### Krok 2: trenuj warunkowo

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator musi matchować real distribution *dla danego warunku*, nie marginal.

### Krok 3: zweryfikuj per-class output

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Pułapki

- **Warunek ignorowany.** G uczy się marginalizować, D nigdy nie karze bo sygnał warunku jest słaby. Fix: warunkuj D bardziej agresywnie (early layer, nie tylko late), użyj projection discriminator (Miyato & Koyama 2018).
- **L1 weight za niski.** G dryfuje do arbitrary real-looking outputs, nie faithful ones. Startuj z λ≈100 dla zadań stylu Pix2Pix.
- **L1 weight za wysoki.** G produkuje rozmyte outputs bo L1 to nadal L_p norm. Anneal down raz trening się ustabilizuje.
- **Ground-truth leakage w D.** Konkatenuj `(x, y)` jako D input, nie tylko `y`. Bez tego D nie może sprawdzić spójności.
- **Mode collapse per class.** Każda klasa może collapse niezależnie. Uruchamiaj class-conditional diversity checks.

## Użyj to

Stan image-to-image tasks w 2026:

| Zadanie | Najlepsze podejście |
|---------|---------------------|
| Szkic → zdjęcie, ta sama domena, paired data | Pix2Pix / Pix2PixHD (nadal szybki, nadal ostry) |
| Szkic → zdjęcie, unpaired | ControlNet z Scribble conditioning model |
| Semantyczna seg → zdjęcie | SPADE / GauGAN2 lub SD + ControlNet-Seg |
| Style transfer | Diffusion z IP-Adapter lub LoRA; metody GAN to legacy |
| Depth → zdjęcie | ControlNet-Depth nad Stable Diffusion |
| Super-resolution | Real-ESRGAN (GAN), ESRGAN-Plus lub SD-Upscale (diffusion) |
| Koloryzacja | ColTran, colorizery bazujące na diffusion, lub Pix2Pix-color |
| Dzień → noc, pory roku, pogoda | CycleGAN lub ControlNet-based |

Pix2Pix pozostaje właściwym narzędziem gdy (a) masz tysiące paired examples, (b) zadanie jest wąskie i powtarzalne, i (c) potrzebujesz szybkiej inferencji. Na generic open-domain tasks, diffusion wygrywa.

## Wyślij to

Zapisz `outputs/skill-img2img-chooser.md`. Skill bierze opis zadania, dostępność danych (paired vs unpaired, N samples), i budżet latency/quality, potem wyświetla: podejście (Pix2Pix, CycleGAN, wariant ControlNet, SDXL + IP-Adapter), wymagania training data, koszt inferencji, i protokół eval (LPIPS, FID, task-specific).

## Ćwiczenia

1. **Łatwe.** Zmodyfikuj `code/main.py` żeby dodać trzecią klasę. Potwierdź, że G nadal mapuje szum każdej klasy na poprawny mode.
2. **Średnie.** Zastąp L1 perceptual-style loss w ustawieniu 1-D (np. mały zamrożony D działający jako feature extractor). Czy to zmienia ostrość rozkładu warunkowego?
3. **Trudne.** Naszkicuj CycleGAN w ustawieniu 1-D: dwa rozkłady, dwa generatory, cycle loss. Pokaż, że uczy się mapować między nimi bez paired data.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Conditional GAN | "GAN z etykietami" | G(z, c), D(x, c). Obie sieci widzą warunek. |
| Pix2Pix | "Image-to-image GAN" | Paired cGAN z U-Net G i PatchGAN D + L1 loss. |
| U-Net | "Encoder-decoder ze skipami" | Symetryczna sieć konw.; skipy zachowują high-freq. |
| PatchGAN | "Local-realism classifier" | D wyświetla per-patch score zamiast globalnego score. |
| CycleGAN | "Unpaired image translation" | Dwa G's + cycle-consistency loss; brak paired data. |
| SPADE | "GauGAN" | Normalizuje intermediate activations z mapą semantyczną; segmentation-to-image. |
| FiLM | "Feature-wise linear modulation" | Per-feature affine transform z warunku; cheap conditioning. |

## Uwaga produkcyjna: Pix2Pix jako baseline ograniczony latency

Gdy masz paired data i wąskie zadanie (szkic → render, mapa semantyczna → zdjęcie, dzień → noc), jednorazowa inferencja Pix2Pix bije diffusion o rząd wielkości na latency. Produkcyna porównanie to zwykle:

| Ścieżka | Kroków | Typowy latency przy 512² na pojedynczym L4 |
|---------|--------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint lub SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix wygrywa na throughput w statycznych batchach (każdy request to te same FLOPs). Diffusion wygrywa na jakości i generalizacji. Nowoczesna strategia to często wysłać model destylowany w stylu Pix2Pix dla wąskiego zadania i fallback diffusion dla tail inputs.

## Dalsze czytanie

- Mirza & Osindero (2014). *Conditional Generative Adversarial Nets* — paper o cGAN.
- Isola et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks* — Pix2Pix.
- Zhu et al. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* — CycleGAN.
- Wang et al. (2018). *High-Resolution Image Synthesis with Conditional GANs* — Pix2PixHD.
- Park et al. (2019). *Semantic Image Synthesis with Spatially-Adaptive Normalization* — SPADE / GauGAN.
- Miyato & Koyama (2018). *cGANs with Projection Discriminator* — projekcyjny D.