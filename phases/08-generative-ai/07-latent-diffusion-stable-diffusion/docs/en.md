# Latent Diffusion & Stable Diffusion

> dyfuzja w przestrzeni pikseli na obrazach 512×512 to zbrodnia wojskowa obliczeniowa. Rombach i in. (2022) zauważyli, że do wygenerowania obrazu nie potrzebujesz wszystkich 786k wymiarów — wystarczy tyle, żeby uchwycić strukturę semantyczną, a osobny dekoder dla reszty. Uruchom dyfuzję w latentnej przestrzeni VAE. Ta jedna idea to Stable Diffusion.

**Typ:** Budowa
**Języki:** Python
**Wymagania wstępne:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Czas:** ~75 minut

## Problem

Dyfuzja w przestrzeni pikseli przy 512² oznacza, że U-Net działa na tensorach o kształcie `[B, 3, 512, 512]`. Każdy krok próbkowania to ~100 GFLOPS dla U-Net z 500M parametrów. Pięćdziesiąt kroków to 5 TFLOPS na obraz. Trenowanie na miliardzie obrazów sprawia, że rachunek za obliczenia jest absurdalny.

Większość tych FLOP idzie na przetwarzanie przez sieć perceptually nieistotnych detali — tekstury wysokiej częstotliwości, którą stratny VAE mógłby skompresować. Idea Rombacha: trenuj VAE raz (pierwszy etap), zamroź go i uruchom dyfuzję całkowicie w 4-kanałowej latentnej przestrzeni 64×64 (drugi etap). 

Uruchamiam tę samą architekturę z 1/16 pikseli i ~64x mniejszą liczbą operacji przy porównywalnej jakości. To jest przepis Stable Diffusion. SD 1.x / 2.x używały U-Net 860M na latentach `64×64×4`, SDXL używał U-Net 2.6B na `128×128×4`, SD3 zamienił U-Net na Diffusion Transformer (DiT) z flow matching. Flux.1-dev (Black Forest Labs, 2024) zawiera DiT-MMDiT z 12B parametrów.

Wszystkie te modele działają na tej samej dwuetapowej strukturze.

Dyfuzja w przestrzeni latentnej: kompresja VAE + dyfuzja w przestrzeni latentnej

Model składa się z dwóch oddzielnie trenowanych etapów. Pierwszy to VAE z enkoderem i dekoderem, które kompresują obraz 8-krotnie w każdym wymiarze przestrzennym, zmniejszając rozmiar latentny do około 1/16 oryginalnej liczby pikseli. Drugi etap przeprowadza dyfuzję na reprezentacji latentnej, trenując U-Net do denoisingu latentnej przestrzeni.

Model wykorzystuje różne enkodery tekstu (CLIP-L dla SD 1.x, CLIP-L+OpenCLIP-G dla SD 2/XL, T5-XXL dla SD3 i Flux) oraz cross-attention injection, gdzie każdy blok U-Net przetwarza cechy obrazu z tokenami tekstowymi jako kluczami i wartościami. Funkcja strat pozostaje identyczna do Lesson 06 — ten sam DDPM lub flow matching MSE na szumie, tylko zmienia się domena danych.

Modele różnią się architekturą i rozmiarem. SD 1.5 i 2.1 używają U-Net z 860-865M parametrów na latentach 64×64×4. SDXL komponuje U-Net z refinerem na latentach 128×128×4, osiągając 2.6B + 6.6B parametrów. SD3 i Flux.1 zamieniają U-Net na MMDiT z 2B do 12B parametrów na latentach 128×128×16, wykorzystując T5-XXL jako enkoder tekstu.

Kierunek rozwoju idzie w stronę DiT zamiast U-Net, skaling enkoderów tekstu (T5 lepsze od CLIP dla adherence promptu), zwiększanie kanałów latentnych (4 → 16 dla większego zapasu detali), oraz distillation do minimalnych kroków próbkowania (1-4 kroki).

Implementacja w `code/main.py` buduje "VAE" w przestrzeni 1D jako dowód koncepcji — enkoder i dekoder jako tożsamość, z DDPM z Lesson 06 jako bazą. Dodaje conditioning klasowe z classifier-free guidance, pokazując że ta sama dyfuzja loss działa niezależnie od tego, czy operuje na surowych wartościach 1D czy na wartościach zakodowanych.

Enkoder i dekoder to proste transformacje skali — `encode(x)` zwraca `x * 0.5`, `decode(z)` zwraca `z * 2.0`. Używam liniowej mapy jako uproszczenia dla pokazania zasady, choć prawdziwy VAE zawierałby w pełni wytrenowane wagi konwolucyjne.

W przestrzeni latentnej Z uruchamiam ten sam proces dyfuzji, traktując `z = E(x)` jako dane wejściowe. Po próbkowaniu `z_0` dekóduję wynik przez `D(z_0)`.

Podczas treningu upuszczam label klasy 10% czasu, wstawiając null token. Przy wnioskowaniu obliczam oba warianty — warunkowy i bezwarunkowy szum — a następnie skaluję je przez współczynnik `w`. Gdy `w = 0`, brak prowadzenia zachowuje pełną różnorodność, `w = 3` to domyślne ustawienie, a wyższe wartości prowadzą do nasycenia i nadmiernej ostrości.

Zamiast labelu klasy wprowadzam reprezentację tekstu z zamrożonego enkodera tekstowego. W każdym bloku U-Net dodaję cross-attention gdzie zapytanie pochodzi z cech obrazu, a klucz i wartość z osadzenia tekstu. To fundamentalna różnica między dyfuzją warunkową a klasycznym Stable Diffusion.

Mogę też manipulować skalą VAE — w SD 1.x mnożę latenty przez stałą, co wpływa na wariancję danych wejściowych dla U-Neta.

Błąd pojawia się też przy złym enkoderze tekstowym — SD3 wymaga T5-XXL z minimum 128 tokenami, a alternatywa tylko z CLIP mocno traci na jakości. Modele używają różnych przestrzeni latentnych, więc LoRA wytrenowana na jednym VAE nie zadziała z innym — nowsze wersje biblioteki diffusers odmawiają ładowania niezgodnych checkpointów. Zbyt wysokie CFG sprawia, że obrazy stają się nasycone i tłuste przy jednoczesnej utracie różnorodności. Ujemne prompty też wymagają ostrożności, bo pusty prompt domyślnie staje się tokenem neutralnym, a wypełniony zachowuje się inaczej przy obliczaniu szumu.

W 2026 roku do dyspozycji mam kilka opcji: SDXL fine-tune najszybciej wdrożysz, jeśli masz wąską domenę z danymi; Flux.1-dev i SD3.5-Large świetnie sprawdzają się w otwartym generowaniu obrazów; Flux.1-schnell oferuje najszybsze próbkowanie; dla jakości prompt adherence są GPT-Image, DALL-E 3 czy Midjourney v7; a Flux.1-Kontext radzi sobie z edycją obrazów. SD 1.5 pozostaje przydatnym punktem odniesienia, choć jest przestarzały.

Tworzę plik `outputs/skill-sd-prompter.md`, który zawiera instrukcje generowania obrazów na podstawie opisu i stylu — określa model, checkpoint, skalę CFG, sampler, negatywny prompt, rozdzielczość i ewentualne kombinacje ControlNet czy IP-Adapter.

W ćwiczeniu testuję wpływ współczynnika `w` na próbki z `code/main.py`, obserwując przy jakiej wartości średnie klasowe przestają odpowiadać rzeczywistym danym. Następnie zamieniam prosty enkoder na MLP z tangensem hiperbolicznym i przebudowuję stratę rekonstrukcji, sprawdzając czy jakość próbek się poprawia. Trzecie zadanie wymaga uruchomienia pełnej infrastruktury z `sdxl-base` przez 30 kroków z classifier-free guidance, a potem porównania z wersją zoptymalizowaną.

Porównuję różnice między tymi podejściami, aby zrozumieć kompromisy między szybkością a jakością generowanych obrazów. Kolejna sekcja przedstawia kluczowe pojęcia związane z modelami dyfuzji — wyjaśnia co oznaczają terminy takie jak pierwszy i drugi etap, classifier-free guidance, null token, cross-attention i architektury DiT oraz MMDiT w kontekście ich praktycznego zastosowania. Opisuje też czynniki skalowania VAE jako magiczne liczby normalizujące przestrzeń latentną. Następnie omawia praktyczne wyzwania uruchomienia modelu Flux-12B na konsumenckiej karcie graficznej z 8GB pamięci — wymaga to trójstopniowego podejścia z sekwencyjnym ładowaniem komponentów, ponieważ model ma rozdzielone sekcje: enkoder tekstowy T5-XXL zajmujący około 10GB w pełnej precyzji, niewielki CLIP-L, główny DiT z 12 miliardami parametrów oraz dekoder VAE. Najpierw koduję prompt używając enkoderów, usuwam je z pamięci, ładuję DiT do denoisingu, ponownie zwalniam zasoby, a na końcu uruchamiam dekoder VAE do finalnego przetworzenia.

Do redukcji zużycia pamięci stosuję kwantyzację 4-bitową z biblioteką bitsandbytes, co zmniejsza rozmiar modelu ośmiokrotnie przy minimalnej utracie jakości dla zadań text-to-image. Dodatkowo wykorzystuję offload na CPU, który automatycznie przenosi moduły między pamięcią GPU a RAM-em podczas przetwarzania — choć dodaje 10-20% opóźnienia, pozwala zmieścić się w limitach konsumenckich kart graficznych. T5 w fp32 wymaga około 10GB, a po kwantyzacji spada do 1.25GB, natomiast DiT z 12 miliardami parametrów potrzebuje około 6GB po kompresji, co razem z aktywacjami mieści się w dostępnej pamięci przy konfiguracji TP=1 z maksymalną kwantyzacją.</think>

# Latent Diffusion i Stable Diffusion

> Dyfuzja w przestrzeni pikseli na obrazach 512×512 to zbrodnia wojskowa obliczeniowa. Rombach i in. (2022) zauważyli, że do wygenerowania obrazu nie potrzebujesz wszystkich 786k wymiarów — wystarczy tyle, żeby uchwycić strukturę semantyczną, a osobny dekoder dla reszty. Uruchom dyfuzję w latentnej przestrzeni VAE. Ta jedna idea to Stable Diffusion.

**Typ:** Budowa
**Języki:** Python
**Wymagania wstępne:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Czas:** ~75 minut

## Problem

Dyfuzja w przestrzeni pikseli przy 512² oznacza, że U-Net działa na tensorach o kształcie `[B, 3, 512, 512]`. Każdy krok próbkowania to ~100 GFLOPS dla U-Net z 500M parametrów. Pięćdziesiąt kroków to 5 TFLOPS na obraz. Trenowanie na miliardzie obrazów sprawia, że rachunek za obliczenia jest absurdalny.

Większość tych FLOP idzie na przetwarzanie przez sieć perceptually nieistotnych detali — tekstury wysokiej częstotliwości, którą stratny VAE mógłby skompresować. Idea Rombacha: trenuj VAE raz (pierwszy etap), zamroź go i uruchom dyfuzję całkowicie w 4-kanałowej latentnej przestrzeni 64×64 (drugi etap). Ten sam U-Net. 1/16 pikseli. ~64x mniej FLOP dla porównywalnej jakości.

To jest przepis Stable Diffusion. SD 1.x / 2.x używały U-Net 860M nad `64×64×4` latentami, SDXL używał U-Net 2.6B nad `128×128×4`, SD3 zamienił U-Net na Diffusion Transformer (DiT) z flow matching. Flux.1-dev (Black Forest Labs, 2024) dostarcza DiT-MMDiT z 12B parametrów. Wszystkie działają na tym samym dwuetapowym podłożu.

## Koncepcja

![Latent diffusion: kompresja VAE + dyfuzja w przestrzeni latentnej](../assets/latent-diffusion.svg)

**Dwa etapy, trenowane oddzielnie.**

1. **Etap 1 — VAE.** Enkoder `E(x) → z`, dekoder `D(z) → x`. Docelowa kompresja: 8× downsample w każdej osi przestrzennej + dostosowanie kanałów tak, że całkowity rozmiar latentny to ~1/16 liczby pikseli. Loss = rekonstrukcja (L1 + LPIPS perceptually) + KL (mała waga, żeby `z` nie był zbyt wymuszony jako Gaussian, bo nie potrzebujemy dokładnego próbkowania z `z`). Często trenowane z adversarial loss, żeby dekodowane obrazy były ostre.

2. **Etap 2 — dyfuzja na `z`.** Traktuj `z = E(x_real)` jako dane. Trenuj U-Net (lub DiT) do denoise `z_t`. Na inferencji: próbkuj `z_0` przez dyfuzję, potem `x = D(z_0)`.

**Warunkowanie tekstowe.** Dwa dodatkowe komponenty. Zamrożony enkoder tekstowy (CLIP-L dla SD 1.x, CLIP-L+OpenCLIP-G dla SD 2/XL, T5-XXL dla SD3 i Flux). Wstrzykiwanie przez cross-attention: każdy blok U-Net przyjmuje `[Q = cechy obrazu, K = V = tokeny tekstowe]` i miesza je. Tokeny są jedynym sposobem, w jaki tekst wpływa na obraz.

**Funkcja loss jest identyczna jak w Lesson 06.** Ten sam DDPM / flow matching MSE na szumie. Tylko zmieniasz domenę danych.

## Warianty architektury

| Model | Rok | Backbone | Kształt latentny | Enkoder tekstowy | Parametry |
|-------|------|----------|------------------|-------------------|-----------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokenów) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | ten sam | 1-4 krokowe próbkowanie |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 kroków |

Trend: zamiana U-Net na DiT (transformer nad latent patches), skalowanie enkodera tekstowego (T5 bije CLIP pod względem adherence promptu), zwiększanie kanałów latentnych (4 → 16 daje więcej miejsca na detale).

## Zbuduj to

`code/main.py` stackuje zabawkowy VAE 1-D (enkoder + dekoder identity, dla demonstracji; prawdziwy VAE byłby siecią konwolucyjną) na szczycie DDPM z Lesson 06 i dodaje warunkowanie klasowe z classifier-free guidance. Pokazuje, że ten sam dyfuzyjny loss działa niezależnie od tego, czy działa na surowych wartościach 1-D czy na zakodowanych wartościach — to kluczowy wgląd.

### Krok 1: enkoder/dekoder

```python
def encode(x):    return x * 0.5          # zabawkowa "kompresja" do mniejszej skali
def decode(z):    return z * 2.0
```

Prawdziwy VAE ma trenowane wagi. Dla pedagogiki, ta liniowa mapa jest wystarczająca, żeby pokazać, że dyfuzja operuje na `z` nie przejmując się oryginalną przestrzenią danych.

### Krok 2: dyfuzja w przestrzeni `z`

Ten sam DDPM co w Lesson 06. Dane, które widzi sieć, to `z = E(x)`. Po próbkowaniu `z_0`, dekoduj z `D(z_0)`.

### Krok 3: classifier-free guidance

Podczas treningu, porzuć label klasy 10% czasu (zastąp null tokenem). Na inferencji, oblicz zarówno `ε_cond` jak i `ε_uncond`, potem:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = brak guidance (pełna różnorodność), `w = 3` = domyślne, `w = 7+` = nasycone / nadmiernie ostre.

### Krok 4: warunkowanie tekstowe (koncepcja, nie kod)

Zamień label klasy na wyjście zamrożonego enkodera tekstowego. Wprowadź embedding tekstowy do U-Net przez cross-attention:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

To jest jedyna istotna różnica między dyfuzją warunkową klasą a Stable Diffusion.

## Pułapki

- **Niedopasowanie skali VAE.** VAE SD 1.x mają stałą skalowania (`scaling_factor ≈ 0.18215`) aplikowaną po kodowaniu. Zapomnienie tego sprawia, że U-Net trenuje na latentach z totalnie złym wariancem. Każdy checkpoint to dostarcza.
- **Enkoder tekstowy cicho błędny.** SD3 potrzebuje T5-XXL z >=128 tokenami, a fallback do CLIP-only jest stratny. Zawsze sprawdź `use_t5=True` albo fidelity promptów się pogarsza.
- **Mieszanie przestrzeni latentnych.** SDXL, SD3, Flux używają różnych VAE. LoRA wytrenowana na latentach SDXL nie zadziała na SD3. Hugging Face diffusers 0.30+ odmawia ładowania niedopasowanych checkpointów.
- **CFG zbyt wysokie.** `w > 10` produkuje nasycone, tłuste obrazy i over-fituje prompt kosztem różnorodności. Słodki punkt to `w = 3-7`.
- **Wyciek negative prompts.** Pusty negative prompt staje się null tokenem; wypełniony negative prompt staje się `ε_uncond`. To nie jest to samo; niektóre pipeline'y cicho defaultują do null.

## Użyj tego

Produkcyjne stacki w 2026:

| Cel | Polecany backbone |
|--------|----------------------|
| Wąska domena, sparowane dane, trenowanie modelu od zera | SDXL fine-tune (LoRA / full) — najszybsze do wysyłki |
| Otwarta domena text-to-image, otwarte wagi | Flux.1-dev (12B, Apache / non-commercial) lub SD3.5-Large |
| Najszybsza inferencja, otwarte wagi | Flux.1-schnell (1-4 krok, Apache) lub SDXL-Lightning |
| Najlepsze adherence promptu, hosted | GPT-Image / DALL-E 3 (nadal), Midjourney v7, Imagen 4 |
| Workflowy edycji | Flux.1-Kontext (grudzień 2024) — natywnie przyjmuje obraz + tekst |
| Badania, baseline | SD 1.5 — starożytny, ale dobrze zbadany |

## Wyślij to

Zapisz `outputs/skill-sd-prompter.md`. Skill bierze text prompt + target style i wyświetla: model + checkpoint, skala CFG, sampler, negative prompt, rozdzielczość, opcjonalna kombinacja ControlNet/IP-Adapter, i per-step QA checklist.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` z guidance `w ∈ {0, 1, 3, 7, 15}`. Zapisz średnią próbkę przez klasę. Przy jakim `w` średnie klasowe rozchodzą się poza średnie rzeczywistych danych?
2. **Średnie.** Zamień zabawkowy liniowy enkoder na parę enkoder/dekoder tanh-MLP z reconstruction loss. Przetrenuj dyfuzję na nowych latentach. Czy jakość próbek się zmienia?
3. **Trudne.** Ustaw realną inferencję Stable Diffusion z diffusers: załaduj `sdxl-base`, uruchom 30 kroków Euler z CFG=7, zmierz czas. Teraz przełącz na `sdxl-turbo` z 4 krokami i CFG=0. Ten sam temat, inna jakość — opisz co się zmieniło i dlaczego.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| First stage | "The VAE" | Trenowana para enkoder/dekoder; kompresuje 512² do 64². |
| Second stage | "The U-Net" | Model dyfuzyjny nad przestrzenią latentną. |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`; dostraja siłę warunkowania. |
| Null token | "Empty prompt embed" | Bezwarunkowy embed używany dla `ε_uncond`. |
| Cross-attention | "How text gets in" | Każdy blok U-Net attenduje do tokenów tekstowych jako K i V. |
| DiT | "Diffusion Transformer" | Zamień U-Net na transformer nad latent patches; lepiej skaluje. |
| MMDiT | "Multi-modal DiT" | Architektura SD3: strumienie tekstu i obrazu z joint attention. |
| VAE scaling factor | "Magic number" | Dzieli latenty przez ~5.4 żeby dyfuzja operowała w przestrzeni jednostkowej wariancji. |

## Nota produkcyjna: uruchamianie Flux-12B na konsumenckiej GPU 8GB

Referencyjna integracja Flux to kanoniczny "mam konsumencką GPU, czy mogę to wysłać?" przepis. Trik to ten sam trój-gałkowy przepis, który produkcyjna literatura inferencyjna podaje dla dyfuzyjnego DiT:

1. **Sekwencyjne ładowanie.** Flux ma trzy sieci, które nigdy nie muszą współistnieć w VRAM: enkoder tekstowy T5-XXL (~10 GB w fp32), CLIP-L (mały), 12B MMDiT i VAE. Koduj prompt najpierw, *usuń* enkoder, załaduj DiT, denoise, *usuń* DiT, załaduj VAE, dekoduj. Konsumenckie GPU 8GB mieszczą tylko jeden etap na raz.
2. **Kwantyzacja 4-bitowa przez bitsandbytes.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` na zarówno enkoderze T5 jak i DiT. Obcina pamięć 8×, spadek jakości jest niepostrzegalny dla text-to-image per benchmarki Aritry (linkowane w notebooku).
3. **CPU offload.** `pipe.enable_model_cpu_offload()` auto-swapuje moduły między CPU a GPU przy każdym forward passie. Dodaje 10-20% latency ale sprawia, że pipeline w ogóle działa.

Rachunkowość pamięciowa: `10 GB T5 / 8 = 1.25 GB` skwantowany, `12 B params × 0.5 bytes = ~6 GB` skwantowany DiT, plus aktywacje. W terminach stas00 to extreme-end TP=1 inference — bez parallelismu modelu, maximum kwantyzacja. Dla produkcji uruchomiłbyś TP=2 lub TP=4 na H100s; dla pojedynczego dev laptopa, to jest przepis.

## Dalsze Czytanie

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 family.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — referencyjna implementacja dla każdego checkpointa powyżej.