# ControlNet, LoRA i warunkowanie

> Sam tekst jest niezdarnym sygnałem sterującym. ControlNet pozwala sklonować wstępnie wytrenowany model dyfuzyjny i kierować nim za pomocą mapy głębi, szkieletu pozy, szkicu lub obrazu krawędzi. LoRA pozwala dostroić model z 2B parametrów, trenując 10 milionów parametrów. Razem zamieniły Stable Diffusion z zabawki w potok obrazów 2026, który trafia do każdej agencji.

**Typ:** Buduj
**Języki:** Python
**Wymagania wstępne:** Faza 8 · 07 (Dyfuzja latentna), Faza 10 (LLM-y od podstaw — podstawy LoRA)
**Czas:** ~75 minut

## Problem

Prompt jak "kobieta w czerwonej sukience prowadząca psa po ruchliwej ulicy" nie dostarcza modelowi żadnych informacji o *tym, gdzie* jest pies, *w jakiej pozie* jest kobieta, lub *z jakiej perspektywy* jest ulica. Tekst określa około 10% tego, co musisz wskazać, by opisać obraz. Reszta jest wizualna i nie może być efektywnie opisana słowami.

Trenowanie nowego modelu warunkowego od zera dla każdego sygnału (poza, głębia, canny, segmentacja) jest niedopuszczalne. Chcesz zamrozić rdzeń SDXL z 2,6B parametrów, dołączyć małą sieć boczną, która odczytuje warunkowanie, i sprawić, by delikatnie nudge'owała pośrednie cechy rdzenia. To jest ControlNet.

Chcesz też nauczyć model nowych koncepcji (swojej twarzy, swojego produktu, swojego stylu) bez ponownego treningu pełnego modelu. Chcesz delty 100x mniejszej. To jest LoRA — adaptery niskiego rzędu, które wpinają się w istniejące wagi uwagi.

ControlNet + LoRA + tekst = toolkit praktyka 2026. Większość produkcyjnych potoków obrazów warstwuje 2-5 LoRA, 1-3 ControlNety i IP-Adapter na szczycie bazy SDXL / SD3 / Flux.

## Koncepcja

![ControlNet klonuje enkoder; LoRA dodaje niskie rzędy delty](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

Weź wstępnie wytrenowany SD. *Sklonuj* połowę enkodera U-Net. Zamroź oryginał. Trenuj klona, by akceptował dodatkowe wejście warunkowe (krawędzie, głębia, poza). Połącz klona z powrotem z częścią dekodera oryginału za pomocą *skip connections* zero-konwolucyjnych (konwolucje 1×1 zainicjowane zerowo — startują jako no-op, uczą się delty).

```
Dekoder SD U-Net:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Inicjalizacja zero-conv oznacza, że ControlNet startuje jako identyczność — bez szkody nawet przed treningiem. Trenuj na 1M trójek (prompt, warunek, obraz) ze standardową stratą dyfuzyjną.

ControlNety per-modalność są dostarczane jako małe modele boczne (~360M dla SDXL, ~70M dla SD 1.5). Możesz je komponować podczas wnioskowania:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

Dla dowolnej warstwy liniowej `W ∈ R^{d×d}` w modelu, zamroź `W` i dodaj deltę niskiego rzędu:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

gdzie `r << d`. Rząd 4-16 jest standardowy dla uwagi, rząd 64-128 dla ciężkich dostrojek. Liczba nowych parametrów: `2 · d · r` zamiast `d²`. Dla uwagi SDXL z `d=640`, `r=16`: 20k parametrów na adapter zamiast 410k — redukcja 20x. W całym modelu: LoRA ma zwykle 20-200MB vs baza 5GB.

Podczas wnioskowania możesz skalować LoRA: `W' = W + α · B @ A`. `α = 0.5-1.5` jest normalne. Wiele LoRA składa się addytywnie (z usual caveat że wchodzą w interakcje w nieliniowy sposób).

### IP-Adapter (Ye et al., 2023)

Mały adapter, który akceptuje *obraz* jako warunkowanie (obok tekstu). Używa enkodera obrazów CLIP do tworzenia tokenów obrazu, wstrzykuje je do cross-attention obok tokenów tekstowych. ~20MB na bazowy model. Pozwala na "generuj obraz w stylu tego referencyjnego" bez LoRA.

## Macierz kompozycyjności

| Narzędzie | Co kontroluje | Rozmiar | Kiedy używać |
|------|------------------|------|-------------|
| ControlNet | Strukturę przestrzenną (poza, głębia, krawędzie) | 70-360MB | Dokładny układ, kompozycja |
| LoRA | Styl, podmiot, koncepcję | 20-200MB | Personalizacja, styl |
| IP-Adapter | Styl lub podmiot z obrazu referencyjnego | 20MB | Żaden tekst nie może opisać wyglądu |
| Textual Inversion | Pojedynczą koncepcję jako nowy token | 10KB | Starsze, głównie zastąpione przez LoRA |
| DreamBooth | Pełne dostrojenie podmiotu | 2-5GB | Silna tożsamość, wysokie obliczenia |
| T2I-Adapter | Lżejsza alternatywa dla ControlNet | 70MB | Urządzenia brzegowe, budżet wnioskowania |

ControlNet ≈ przestrzenne. LoRA ≈ semantyczne. Używaj obu.

## Zbuduj to

`code/main.py` symuluje oba mechanizmy na 1-D:

1. **LoRA.** Wstępnie trenowana warstwa liniowa `W`. Zamroź ją. Trenuj niskiego rzędu `B @ A` tak, by `W + BA` pasowało do docelowej warstwy liniowej. Pokaż, że `r = 1` wystarczy, by idealnie nauczyć się poprawki rzędu 1.

2. **ControlNet-lite.** "Zamrożona baza" predyktor i "sieć boczna", która odczytuje dodatkowy sygnał. Wyjście sieci bocznej jest bramkowane przez skalarny parametr uczący się zainicjowany zerem (nasza wersja zero-conv). Trenuj i obserwuj, jak bramka się rampuje.

### Krok 1: Matematyka LoRA

```python
def lora(W, A, B, x, alpha=1.0):
    # W jest zamrożona; A, B to trenowalne czynniki niskiego rzędu.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Krok 2: sieć boczna z zerową inicjalizacją

```python
side_out = control_net(x, condition)
gated = gate * side_out  # bramka zainicjowana 0
h = base(x) + gated
```

W kroku 0 wyjście jest identyczne z bazą. Wczesny trening aktualizuje `gate` powoli — brak katastroficznego dryfu.

## Pułapki

- **Nadmierne skalowanie LoRA.** `α = 2` lub `α = 3` to popularny hack "zrób to silniejszym", który produkuje nadmiernie stylizowane / złamane wyjścia. Utrzymuj `α ≤ 1.5`.
- **Konflikt wag ControlNet.** Używanie ControlNet-Pozy przy wadze 1.0 i ControlNet-Głębi przy wadze 1.0 zwykle przekracza. Suma wag ≈ 1.0 to bezpieczny default.
- **LoRA na niewłaściwej bazie.** LoRA SDXL cicho no-op na SD 1.5 bo wymiary uwagi nie pasują. Diffusers ostrzeże od wersji 0.30+.
- **Dryf Textual Inversion.** Tokeny trenowane na jednym checkpoincie dryfują fatalnie na innym. LoRA jest bardziej przenośna.
- **Łączenie wag i przechowywanie LoRA.** Możesz wypiec LoRA w wagi modelu bazowego dla szybszego wnioskowania (brak runtime'owego dodawania), ale tracisz możliwość skalowania `α` w runtime. Trzymaj obie wersje.

## Użyj tego

| Cel | Potok 2026 |
|------|---------------|
| Odtworzyć styl artystyczny marki | LoRA trenowana na ~30 kuratorowanych obrazach przy rzędzie 32 |
| Wstawić moją twarz w wygenerowany obraz | DreamBooth lub LoRA + IP-Adapter-FaceID |
| Konkretna poza + prompt | ControlNet-Openpose + SDXL + tekst |
| Kompozycja świadoma głębi | ControlNet-Depth + SD3 |
| Referencja + prompt | IP-Adapter + tekst |
| Dokładny układ | ControlNet-Scribble lub ControlNet-Canny |
| Zamiana tła | ControlNet-Seg + Inpainting (Lekcja 09) |
| Szybki styl 1-krokowy | LCM-LoRA na SDXL-Turbo |

## Wyślij to

Zapisz `outputs/skill-sd-toolkit-composer.md`. Umiejętność przyjmuje zadanie (aktywa wejściowe: prompt, opcjonalny obraz referencyjny, opcjonalna poza, opcjonalna głębia, opcjonalny szkic) i wyprowadza stos narzędzi, wagi i odtwarzalny protokół seed.

## Ćwiczenia

1. **Łatwe.** W `code/main.py`, zmień rząd LoRA `r` od 1 do 4. Przy jakim rzędzie LoRA dokładnie dopasuje docelową deltę rzędu 2?
2. **Średnie.** Trenuj dwie osobne LoRA na dwóch transformacjach docelowych. Załaduj je razem i pokaż ich addytywną interakcję. Kiedy interakcja łamie liniowość?
3. **Trudne.** Użyj diffusers do stosu: SDXL-base + Canny-ControlNet (waga 0.8) + styl LoRA (α 0.8) + IP-Adapter (waga 0.6). Zmierz kompromis FID-vs-przestrzeganie-promptu jak wagi stosu się zmieniają.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| ControlNet | "Kontrola przestrzenna" | Sklonowany enkoder + zero-conv skips; odczytuje obraz warunkowy. |
| Zero convolution | "Startuje jako identyczność" | Konwolucja 1×1 zainicjowana zerowo; ControlNet startuje jako no-op. |
| LoRA | "Adapter niskiego rzędu" | `W + B @ A`, `r << d`; 100x mniej parametrów niż pełne dostrojenie. |
| rząd r | "Pokrętło" | Kompresja LoRA; 4-16 typowe, 64+ dla ciężkiej personalizacji. |
| α | "Siła LoRA" | Runtime skalowanie delty LoRA. |
| IP-Adapter | "Obraz referencyjny" | Mały adapter warunkowania obrazem przez tokeny CLIP-image. |
| DreamBooth | "Pełne dostrojenie podmiotu" | Trenuj pełny model na ~30 obrazach podmiotu. |
| Textual Inversion | "Nowy token" | Naucz się tylko nowego word embedding; starsze, głównie zastąpione. |

## Uwaga produkcyjna: Zamiany LoRA, paski ControlNet, wielodostępne serwowanie

Prawdziwy SaaS text-to-image serwuje setki LoRA i tuzin ControlNetów na tym samym checkpoincie bazowym. Problem serwowania wygląda bardzo podobnie do wielodostępności LLM (literatura produkcyjna pokrywa przypadek LLM pod continuous batching i LoRAX / S-LoRA):

- **Hot-swap LoRA, nie łącz.** Łączenie `W' = W + α·B·A` w bazę daje ~3-5% szybsze wnioskowanie na krok ale zamraża `α` i bazę. Trzymaj LoRA gorące w VRAM jako delty rzędu-r; diffusers eksponuje `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` dla aktywacji per-request. Koszt zamiany to wagi `2 · d · r · num_layers` — skala MB, pod-sekunda.
- **ControlNet jako drugi pas uwagi.** Sklonowany enkoder działa równolegle z bazą. Dwa ControlNety przy wadze 1.0 każdy = dwa dodatkowe przebiegi forward na krok, nie jeden scalony przebieg. Głowa batch size spada kwadratowo. Budżetuj na ~1.5× koszt kroku na aktywny ControlNet.
- **Skwantowane LoRA też.** Jeśli skwantowałeś bazę (patrz Lekcja 07, Flux na 8GB), delta LoRA też skwantowuje się czysto do 8-bit lub 4-bit. Ładowanie w stylu QLoRA pozwala stosować 5-10 LoRA na szczycie 4-bitowego Flux bez rozsadzenia pamięci.

Specyficzne dla Flux: notebook Nielsa Flux-on-8GB kwantuje bazę do 4-bit; stosowanie styl LoRA (`pipe.load_lora_weights("user/style-lora")`) na tej skwantowanej bazie przy `weight_name="pytorch_lora_weights.safetensors"` nadal działa. To jest przepis, który większość SaaS agencji wysyła w 2026.

## Dalsze czytanie

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (pierwotnie dla LLM; porty na dyfuzję).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — lżejsza alternatywa dla ControlNet.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — referencyjne potoki.