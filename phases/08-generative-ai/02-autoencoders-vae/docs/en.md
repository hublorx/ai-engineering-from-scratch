# Autoencodery i Variational Autoencodery (VAE)

> Zwykły autoencoder kompresuje, a następnie rekonstruuje. Uczy się na pamięćć. Nie generuje. Dodaj jedną sztuczkę — wymuś, żeby kod wyglądał jak rozkład Gaussa — a dostajesz sampler. Ta jedna sztuczka, reparametryzacja `z = μ + σ·ε`, jest powodem, dla którego każdy model obrazu latent-diffusion i flow-matching, którego używasz w 2026 roku, ma VAE na wejściu.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Szacowany czas:** ~75 minut

## Problem

Skompresuj 784-pikselową cyfrę MNIST do 16-liczbowego kodu, a następnie zrekonstruuj. Zwykły autoencoder zda egzamin z reconstruction MSE, ale przestrzeń kodu jest nierówna. Wybierz losowy punkt w przestrzeni kodu, zdekoduj go, a dostaniesz szum. Nie ma samplera. To model kompresji przebrany za coś innego.

Czego tak naprawdę chcesz: (a) przestrzeń kodu jest czystym, gładkim rozkładem, z którego możesz próbkować — powiedzmy izotropowy Gaussian `N(0, I)`, (b) dekodowanie dowolnej próbki produkuje wiarygodną cyfrę, i (c) encoder i decoder nadal dobrze kompresują. Trzy cele, jedna architektura, jedna funkcja loss.

VAE Kingmy z 2013 rozwiązuje to przez trenowanie encodera, żeby outputował *rozkład* `q(z|x) = N(μ(x), σ(x)²)`, ściągając ten rozkład w kierunku prior `N(0, I)` przez karę KL, a następnie próbkując `z` z `q(z|x)` przed dekodowaniem. W czasie inferencji usuń encoder, próbkuj `z ~ N(0, I)`, dekoduj. Kara KL jest tym, co wymusza strukturę przestrzeni kodu.

W 2026 VAE rzadko są używane samodzielnie — zostały zdominowane przez diffusion pod względem jakości surowego obrazu — ale są encoderem wyboru dla każdego modelu latent-diffusion (SD 1/2/XL/3, Flux, AudioCraft). Naucz się VAE, a nauczysz się niewidocznej pierwszej warstwy każdego potoku obrazu, którego używasz.

## Koncepcja

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. Przestrzeń kodu nieustrukturyzowana.

**VAE encoder.** Outputuje dwa wektory: `μ(x)` i `log σ²(x)`. Definiują one `q(z|x) = N(μ, diag(σ²))`.

**Reparametryzacja.** Próbkowanie z `q(z|x)` nie jest różniczkowalne. Przepisz próbkę jako `z = μ + σ·ε` gdzie `ε ~ N(0, I)`. Teraz `z` jest deterministyczną funkcją `(μ, σ)` plus szumem bez parametrów — gradienty płyną przez `μ` i `σ`.

**Loss.** Evidence Lower BOund (ELBO), dwa składniki:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Rekonstrukcja pcha `x̂` w kierunku `x`. KL pcha `q(z|x)` w kierunku prior. Konkurują ze sobą. Małe β (<1) = ostrzejsze próbki, przestrzeń kodu mniej Gaussowska. Duże β (>1) = czystsza przestrzeń kodu, bardziej rozmazane próbki. β-VAE (Higgins 2017) spopularyzowało ten knob i zapoczątkowało badania nad disentanglement.

**Próbkowanie.** W czasie inferencji: draw `z ~ N(0, I)`, forward przez decoder. Jeden forward pass — bez iteracyjnego próbkowania jak w diffusion.

## Zbuduj to

`code/main.py` implementuje miniaturowy VAE bez numpy ani torch. Input to 8-wymiarowe syntetyczne dane z dwuskładnikowej mieszaniny Gaussowskiej w 8-D. Encoder i decoder to jednowarstwowe ukryte MLP. Implementujemy aktywację tanh, forward pass, loss i ręcznie napisany backward pass. Nie produkcyjnie — pedagogicznie.

### Krok 1: encoder forward

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²` zamiast `σ`, żeby output sieci był nieograniczony (softplus σ to pułapka — gradienty giną przy σ ≈ 0).

### Krok 2: reparametryzuj i dekoduj

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Krok 3: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

Dokładna analityczna KL, bo oba rozkłady są Gaussowskie. Nie całkuj numerycznie. Ludzie w 2026 wciąż wysyłają kod z monte-carlo estymatami KL — to 3x wolniejsze bez powodu.

### Krok 4: generuj

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

To jest model generatywny. Pięć linii.

## Pułapki

- **Posterior collapse.** Składnik KL pcha `q(z|x) → N(0, I)` tak agresywnie, że `z` nie niesie żadnej informacji o `x`. Fix: β-annealing (zacznij od β=0, zwiększaj do 1), free bits, albo pomijaj KL na nieaktywnych wymiarach.
- **Rozmazane próbki.** Gaussowski decoder likelihood implikuje reconstruction MSE, który jest Bayes-optymalny dla L2 (średniej) — średnia zestawu wiarygodnych cyfr jest rozmyta cyfrą. Fix: dyskretny decoder (VQ-VAE, NVAE), albo używaj VAE tylko jako encodera i stackuj diffusion na latentach (tak robi Stable Diffusion).
- **β za duże, za wcześnie.** Patrz posterior collapse. Zacznij od β≈0.01 i zwiększaj.
- **Zbyt mały latent dim.** 16-D działa dla MNIST, 256-D dla ImageNet 256², 2048-D dla ImageNet 1024². VAE Stable Diffusion kompresuje 512×512×3 → 64×64×4 (32x downsample factor w przestrzeni, 32x w kanałach).

## Użyj tego

Stos VAE w 2026:

| Sytuacja | Wybierz |
|-----------|---------|
| Image-latent encoder dla diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) lub Flux VAE |
| Audio-latent encoder | Encodec (Meta), SoundStream, lub DAC (Descript) |
| Video latents | Sora's spatiotemporal patches, Latte VAE, WAN VAE |
| Disentangled representation learning | β-VAE, FactorVAE, TCVAE |
| Dyskretne latenty (dla modelowania transformerem) | VQ-VAE, RVQ (ResidualVQ) |
| Ciągłe latenty do generacji | Plain VAE, potem warunkuj flow/diffusion model w tej przestrzeni latent |

Latent-diffusion model to VAE z modelem diffusion żyjącym między encoderem a decoderem. VAE robi grubą kompresję, model diffusion robi ciężką pracę. Ten sam wzorzec dla wideo (VAE + video-diffusion DiT) i audio (Encodec + MusicGen transformer).

## Wyślij to

Zapisz `outputs/skill-vae-trainer.md`.

Skill przyjmuje: profil datasetu + docelowy latent-dim + downstream use (rekonstrukcja, próbkowanie, lub input do latent-diffusion) i outputuje: wybór architektury (plain/β/VQ/RVQ), harmonogram β, latent dim, decoder likelihood (Gaussian vs categorical), i plan ewaluacji (recon MSE, KL per dim, Fréchet distance między `q(z|x)` a `N(0, I)`).

## Ćwiczenia

1. **Łatwe.** Zmień `β` w `code/main.py` na `0.01`, `0.1`, `1.0`, `5.0`. Zapisz końcowe reconstruction MSE i KL. Które β jest Pareto-najlepsze dla twoich syntetycznych danych?
2. **Średnie.** Zamień Gaussowski decoder likelihood na Bernoulliego likelihood (cross-entropy loss). Porównaj jakość próbek na binarized wersji tych samych syntetycznych danych.
3. **Trudne.** Rozszerz `code/main.py` o mini VQ-VAE: zamień ciągłe `z` na nearest-neighbour lookup w codebooku K=32 wpisów. Porównaj reconstruction MSE i raportuj ile wpisów codebooku jest używanych (codebook collapse jest realny).

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to tak naprawdę oznacza |
|--------|-----------------|---------------------------|
| Autoencoder | Sieć encode-decode | `x → z → x̂`, uczy się MSE. Nie generatywny. |
| VAE | AE ze samplerem | Encoder outputuje rozkład, kara KL kształtuje przestrzeń kodu. |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`; tight gdy `q = p(z|x)`. |
| Reparametryzacja | `z = μ + σ·ε` | Przepisuje stochastic node jako deterministyczny + czysty szum. Umożliwia backprop przez sampling. |
| Prior | `p(z)` | Docelowy rozkład dla latentu, typowo `N(0, I)`. |
| Posterior collapse | "Składnik KL wygrywa" | Encoder ignoruje `x`, outputuje prior; decoder musi halucynować. |
| β-VAE | Konfigurowalna waga KL | `loss = recon + β·KL`. Wyższe β = bardziej disentangled ale bardziej rozmazane. |
| VQ-VAE | Dyskretny latent | Zamień ciągłe `z` na nearest codebook vector; umożliwia modelowanie transformerem. |

## Uwaga produkcyjna: VAE to najgorętsza ścieżka w serwerze diffusion

W potoku Stable Diffusion / Flux / SD3 VAE jest wywoływany dwukrotnie na request — raz do encode (jeśli robisz img2img / inpainting) i raz do decode. Przy 1024² pass decode często jest pojedynczym największym szczytem activation-memory w całym potoku, bo upsampluje `128×128×16` latenty z powrotem do `1024×1024×3`. Dwie praktyczne konsekwencje:

- **Slice albo tile the decode.** `diffusers` udostępnia `pipe.vae.enable_slicing()` i `pipe.vae.enable_tiling()`. Tiling wymienia mały seam artifact na `O(tile²)` pamięci zamiast `O(H·W)`. Niezbędne dla 1024²+ na consumer GPU.
- **bf16 decoder, fp32 numerics dla finalnego resize.** SD 1.x VAE został wydany we fp32 i *cicho produkuje NaNy* gdy castowane do fp16 przy 1024²+. SDXL dostarcza `madebyollin/sdxl-vae-fp16-fix` — zawsze preferuj variant fp16-fix albo używaj bf16.

## Dalsza lektura

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — artykuł o VAE.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — state-of-the-art image VAE.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAE jako encoder.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, audio VAE standard.