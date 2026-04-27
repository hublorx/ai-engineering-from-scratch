# GANy — Generator vs Dyskryminator

> Sztuczka Goodfellowa z 2014 roku polegała na całkowitym pominięciu gęstości. Dwie sieci. Jedna tworzy fałszywki. Jedna je łapie. Walczą, dopóki fałszywki nie staną się nieodróżnialne od prawdziwych. Nie powinno to działać. Często nie działa. Gdy działa, próbki są wciąż najostrzejsze w literaturze dla wąskich domen.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 3 · 02 (Backprop), Faza 3 · 08 (Optymizatory), Faza 8 · 02 (VAE)
**Szacowany czas:** ~75 minut

## Problem

VAE wytwarzają rozmazane próbki, ponieważ ich funkcja straty dekodera MSE jest optymalna w sensie Bayesa dla *średniego* obrazu — a średnia z wielu prawdopodobnych cyfr to rozmazana cyfra. Potrzebujesz funkcji straty, która nagradza *wiarygodność*, a nie pikselową zgodność z jakimkolwiek jednym celem. Nie ma zamkniętej formy na wiarygodność. Musisz się jej nauczyć.

Pomysł Goodfellowa: trenuj klasyfikator `D(x)`, aby rozróżniał prawdziwe obrazy od fałszywych. Trenuj generator `G(z)`, aby oszukiwał `D`. Sygnał straty dla `G` to to, co aktualnie `D` uważa za to, co sprawia, że coś wygląda realistycznie. Ten sygnał aktualizuje się wraz z poprawą `G`, goniąc za ruchomym celem. Jeśli obie sieci zbiegają się, `G` nauczyło się rozkładu danych bez nigdzie zapisywania `log p(x)`.

To jest trening adversarial. Matematyka to gra minimaksowa:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

W 2026 roku GANy nie są już najnowocześniejszym generatorem (diffusion i flow matching przejęły tę koronę). Ale StyleGAN 2/3 pozostają najostrzejszymi modelami twarzy jakie kiedykolwiek wydano, dyskryminatory GAN są używane jako *perceptual losses* w treningu diffusion, a trening adversarial zasila szybkie destylacje 1-krokowe (SDXL-Turbo, SD3-Turbo, LCM), które pozwalają na wdrożenie real-time diffusion.

## Koncepcja

![Trening GAN: generator i dyskryminator w minimax](../assets/gan.svg)

**Generator `G(z)`.** Odwzorowuje wektor szumu `z ~ N(0, I)` na próbkę `x̂`. Sieć w kształcie dekodera (dense lub transposed conv).

**Dyskryminator `D(x)`.** Odwzorowuje próbkę na skalarną probabilistykę (lub wynik). Prawdziwe → 1, fałszywe → 0.

**Funkcja straty.** Dwa naprzemienne aktualizacje:

- **Trenuj `D`:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. Binarna entropia krzyżowa dla real=1, fake=0.
- **Trenuj `G`:** `loss_G = -log D(G(z))`. To jest forma *non-saturating*, której użył Goodfellow (oryginalne `log(1 - D(G(z)))` nasyca się i zabija gradienty, gdy `D` jest pewne).

**Pętla treningowa.** Jeden krok `D`, jeden krok `G`. Powtarzaj.

**Dlaczego to działa.** Jeśli `G` idealnie dopasuje `p_data`, wtedy `D` nie może zrobić niczego lepszego niż przypadek i zwraca 0.5 wszędzie; `G` nie dostaje już gradientu. Równowaga.

**Dlaczego się psuje.** Mode collapse (`G` znajduje jeden tryb, którego `D` nie potrafi sklasyfikować i bije go wiecznie), zanikający gradient (`D` uczy się za szybko i `log D` nasyca się), niestabilność treningu (learning rates, batch sizes, cokolwiek).

## Warianty, które sprawiły, że GANy zadziałały

| Rok | Innowacja | Rozwiązanie |
|------|------------|-------------|
| 2015 | DCGAN | Konw/deconv, batch norm, LeakyReLU — pierwsza stabilna architektura. |
| 2017 | WGAN, WGAN-GP | Zastąp BCE odległością Wassersteina + gradient penalty. Naprawia zanikający gradient. |
| 2017 | Spectral normalization | Wiąże dyskryminator z Lipschitz. Wciąż używane w dyskryminatorach z 2026. |
| 2018 | Progressive GAN | Trenuj najpierw niskie rozdzielczości, dodawaj warstwy. Pierwsze wyniki megapikselowe. |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm. Stan techniki dla fotorealizmu w ustalonej domenie. |
| 2021 | StyleGAN3 | Alias-free, translation-equivariant — wciąż złoty standard twarzy w 2026. |
| 2022 | StyleGAN-XL | Warunkowy, class-aware, większa skala. |
| 2024 | R3GAN | Zmiana marki z silniejszą regularyzacją; działa na 1024² bez sztuczek. |

## Zbuduj to

`code/main.py` trenuje małego GANa na danych 1-D: mieszaninie dwóch Gaussians. Generator i dyskryminator to jedno-warstwowe ukryte MLP. Implementujemy forward, backward i pętlę minimaks ręcznie. Cel to zobaczenie dwóch kluczowych trybów awarii (mode collapse + zanikający gradient) w akcji.

### Krok 1: non-saturating loss

Oryginalna funkcja straty Goodfellowa `log(1 - D(G(z)))` dąży do 0, gdy D klasyfikuje fałszywki G jako fałszywe z wysoką pewnością. W tym punkcie gradient dla G jest praktycznie zerowy — G nie może się poprawić. Forma non-saturating `-log D(G(z))` ma przeciwny asymptomatyczny zachowanie: eksploduje, gdy D jest pewne, dając G silny sygnał.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Krok 2: jeden krok dyskryminatora na krok generatora

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

Świeże fałszywki dla G, w przeciwnym razie gradienty są stare.

### Krok 3: obserwuj mode collapse

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Kanoniczny symptom: jeden z dwóch prawdziwych trybów przestaje być generowany. Dyskryminator przestaje go korygować, bo nigdy nie jest widziany jako fałszywy.

## Pułapki

- **Dyskryminator za silny.** Zmniejsz learning rate D o 2-5x, lub dodaj instance/layer noise. Jeśli D osiąga >95% accuracy, G jest martwe.
- **Generator zapamiętuje tryb.** Dodaj szum do wejść D, użyj warstwy minibatch-discriminator, lub przełącz na WGAN-GP.
- **Batch norm przeciekający statystyki.** Real batch + fake batch przepływające przez tę samą warstwę BN mieszają ich statystyki. Użyj instance norm lub spectral norm zamiast tego.
- **Inception-score gaming.** FID i IS są zaszumione przy niskich liczbach próbek. Użyj ≥10k próbek przy ewaluacji.
- **One-shot sampling to kłamstwo dla warunkowych zadań.** Wciąż potrzebujesz skal CFG, truncation tricks i re-sampling, aby uzyskać użyteczne wyniki.

## Użyj tego

Stos GAN 2026:

| Sytuacja | Wybierz |
|-----------|---------|
| Fotorealistyczne ludzkie twarze, ustalona poza | StyleGAN3 (najostrzejszy, najmniejszy) |
| Anime / stylizowane twarze | StyleGAN-XL lub Stable Diffusion LoRA |
| Image-to-image translation | Pix2Pix / CycleGAN (Faza 8 · 04) lub ControlNet (Faza 8 · 08) |
| Szybki 1-krok text-to-image | Destylacja adversarial diffusion (SDXL-Turbo, SD3-Turbo) |
| Perceptual loss wewnątrz trenera diffusion | Mały dyskryminator GAN na cropach obrazu |
| Cokolwiek wielomodowe, otwarte | Nie — użyj diffusion lub flow matching |

GANy są ostre, ale wąskie. Gdy tylko twoja domena się otwiera — zdjęcia, arbitralne tekstowe prompty, wideo — przełącz się na diffusion. Sztuczka adversarial żyje dalej jako komponent (perceptual losses, distillation), nie jako samodzielny generator.

## Wydaj to

Zapisz `outputs/skill-gan-debugger.md`. Skill bierze nieudany przebieg GAN (krzywe strat, siatka próbek, rozmiar datasetu) i zwraca ranked listę prawdopodobnych przyczyn, jednoliniowe poprawki i protokół ponownego uruchomienia.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` z ustawieniami stock. Następnie ustaw `D_LR = 5 * G_LR` i uruchom ponownie. Jak szybko strata G zwija się do stałej?
2. **Średnie.** Zastąp funkcję straty Goodfellow BCE stratą WGAN: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, i przytnij wagi D do `[-0.01, 0.01]`. Czy trening jest bardziej stabilny? Porównaj wall-clock convergence.
3. **Trudne.** Rozszerz 1-D przykład do danych 2-D (mieszanina 8 Gaussians na okręgu). Śledź ile z 8 trybów generator przechwytuje w krokach 1k, 5k, 10k. Zaimplementuj minibatch discrimination i zmierz ponownie.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Generator | "G" | Sieć noise-to-sample, `G: z → x̂`. |
| Dyskryminator | "D" | Klasyfikator `D: x → [0, 1]`, real vs fake. |
| Minimax | "Gra" | `min_G max_D` wspólnego celu. |
| Non-saturating loss | "Poprawka" | Użyj `-log D(G(z))` dla G zamiast `log(1 - D(G(z)))`. |
| Mode collapse | "G zapamiętało jedną rzecz" | Generator produkuje mało różnych输出ów mimo zróżnicowanych danych. |
| WGAN | "Wasserstein" | Zastąp BCE odległością Earth-Mover + gradient penalty; gładszy gradient. |
| Spectral norm | "Sztuczka Lipschitz" | Wiąże normy wag D, aby ograniczyć jego nachylenie; stabilizuje trening. |
| StyleGAN | "Ten, który działa" | Mapping network + AdaIN; najlepszy w klasie dla twarzy, wciąż w 2026. |

## Uwaga produkcyjna: one-shot inference to trwała przewaga GAN

GANy nie wygrywają już na jakości próbek dla generacji open-domain, ale wciąż wygrywają na kosztach inference. W słownictwie produkcyjnej literatury inference GAN ma:

- **Brak prefill, brak etapów decode.** Pojedynczy forward pass `G(z)`. TTFT ≈ całkowite opóźnienie.
- **Brak presji KV-cache.** Jedyny stan to wagi. Batch size jest ograniczony przez pamięć aktywacji, nie cache.
- **Trywialne continuous batching.** Ponieważ każde żądanie zajmuje te same stałe FLOPSy, statyczny batch przy docelowym occupancy serwera jest zwykle optymalny. Nie potrzeba in-flight schedulera.

Dlatego destylacja GAN (SDXL-Turbo, SD3-Turbo, ADD, LCM) jest dominującą techniką szybkiego text-to-image w 2026: zwija 20-50-krokowy pipeline diffusion do 1-4 forward passów stylu GAN, zachowując rozkład bazy diffusion. Funkcja straty adversarial przeżywa jako pokrętło czasu treningu do zamieniania wolnych generatorów na szybkie.

## Dalsze czytanie

- Goodfellow et al. (2014). Generative Adversarial Nets — oryginalny artykuł o GAN.
- Radford et al. (2015). Unsupervised Representation Learning with DCGAN — pierwsza stabilna architektura.
- Arjovsky, Chintala, Bottou (2017). Wasserstein GAN — WGAN.
- Miyato et al. (2018). Spectral Normalization for GANs — SN.
- Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN — StyleGAN2.
- Karras et al. (2021). Alias-Free Generative Adversarial Networks — StyleGAN3.
- Sauer et al. (2023). Adversarial Diffusion Distillation — SDXL-Turbo.