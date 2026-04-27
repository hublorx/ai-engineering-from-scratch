# Generacja 3D

> 3D to modalność, w której wykorzystanie wiedzy z 2D do 3D jest najsilniejsze. Przełomem 2023 roku była metoda 3D Gaussian Splatting. Generatywny postęp w latach 2024-2026 łączy dyfuzję wielowidokową z rekonstrukcją 3D, aby produkować obiekty i sceny z pojedynczego prompta lub zdjęcia.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 4 (Wizja), Faza 8 · 07 (Dyfuzja latentna)
**Czas:** ~45 minut

## Problem

Treści 3D są bolesne:

- **Reprezentacja.** Siatki, chmury punktów, siatki vokseli, funkcje odległości ze znakiem (SDF), pola promieniowania neuronowego (NeRF), 3D Gaussy. Każda ma kompromisy.
- **Niedobór danych.** ImageNet ma 14M zdjęć. Największy czysty zbiór danych 3D (Objaverse-XL, 2023) ma ~10M obiektów, w większości niskiej jakości.
- **Pamięć.** Siatka vokseli 512³ to 128M vokseli; użyteczny NeRF sceny wymaga 1M próbek/promień. Generacja jest trudniejsza niż rekonstrukcja.
- **Nadzór.** Dla obrazu 2D masz piksele. Dla 3D zazwyczaj masz kilka widoków 2D i musisz je przenieść do 3D.

Stos w 2026 rozdziela te dwa problemy. Po pierwsze, generuj *wielowidokowe obrazy 2D* za pomocą modelu dyfuzji. Po drugie, dopasuj *reprezentacje 3D* (zazwyczaj Gaussian splatting) do tych obrazów.

## Koncepcja

![Generacja 3D: dyfuzja wielowidokowa + rekonstrukcja 3D](../assets/3d-generation.svg)

### Reprezentacja: 3D Gaussian Splatting (Kerbl et al., 2023)

Reprezentuj scenę jako chmurę ~1M 3D Gaussian. Każda ma 59 parametrów: pozycja (3), kowariancja (6, lub kwaternion 4 + skala 3), krycie (1), kolor z harmonicznymi sferycznymi (48 przy stopniu 3, 3 przy stopniu 0).

Rendering = projekcja + kompozytowanie alfa. Szybki (~100 fps przy 1080p na 4090). Różniczkowalny. Dopasowany przez spadek gradientu względem zdjęć referencyjnych. Scena dopasowuje się w 5-30 minut na konsumenckim GPU.

Dwie innowacje 2023-2024 na wierzchu:
- **Generatywne Gaussian splats.** Modele takie jak LGM, LRM, InstantMesh przewidują chmurę Gaussian bezpośrednio z jednego lub kilku obrazów.
- **4D Gaussian Splatting.** Gaussy z przesunięciami na klatkę dla scen dynamicznych.

### Dyfuzja wielowidokowa

Dostroj pretrained model dyfuzji obrazu do generowania wielu spójnych widoków tego samego obiektu z prompta tekstowego lub pojedynczego obrazu. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). Zazwyczaj wyprowadza 4-16 widoków wokół obiektu, przeniesione do 3D przez Gaussian splatting lub NeRF.

### Potoki text-to-3D

| Model | Wejście | Wyjście | Czas |
|-------|---------|---------|------|
| DreamFusion (2022) | tekst | NeRF przez SDS | ~1 godz. na asset |
| Magic3D | tekst | siatka + tekstura | ~40 min |
| Shap-E (OpenAI, 2023) | tekst | niejawne 3D | ~1 min |
| SJC / ProlificDreamer | tekst | NeRF / siatka | ~30 min |
| LRM (Meta, 2023) | obraz | triplane | ~5 s |
| InstantMesh (2024) | obraz | siatka | ~10 s |
| SV3D (Stability, 2024) | obraz | nowe widoki | ~2 min |
| CAT3D (Google, 2024) | 1-64 obrazy | 3D NeRF | ~1 min |
| TripoSR (2024) | obraz | siatka | ~1 s |
| Meshy 4 (2025) | tekst + obraz | siatka PBR | ~30 s |
| Rodin Gen-1.5 (2025) | tekst + obraz | siatka PBR | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | obraz | siatka | ~30 s |

Kierunek 2025-2026: bezpośrednie modele text-to-mesh z materiałami PBR odpowiednimi dla silników gier. Środkowy krok dyfuzji wielowidokowej jest wciąż najlepszym przepisem pod względem wydajności dla obiektów ogólnych.

### NeRF (dla kontekstu)

Neural Radiance Field (Mildenhall et al., 2020). Mała sieć MLP bierze `(x, y, z, kierunek widoku)` i wyprowadza `(kolor, gęstość)`. Renderuj przez całkowanie wzdłuż promieni. Przewyższa syntezę nowych widoków opartą na siatkach, ale jest 100-1000x wolniejsza w renderowaniu. Zastąpiona przez Gaussian splatting dla większości zastosowań czasu rzeczywistego, ale wciąż dominuje w badaniach.

## Zbuduj to

`code/main.py` implementuje dopasowanie zabawki "Gaussian splatting" w 2D: reprezentuj syntetyczny obraz docelowy (gładki gradient) jako sumę 2D Gaussian splats. Optymalizuj pozycje, kolory i kowariancje przez spadek gradientu, aby dopasować do celu. Widzisz dwie podstawowe operacje: render w przód (splat + kompozytowanie alfa) i dopasowanie przez spadek gradientu.

### Krok 1: 2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Krok 2: renderuj przez sumowanie splats

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

Prawdziwy 3D Gaussian splatting sortuje Gaussy według głębokości i kompozytuje alfa po kolei. Nasza zabawka 2D po prostu sumuje.

### Krok 3: dopasuj przez spadek gradientu

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Pułapki

- **Niespójność widoków.** Jeśli generujesz 4 widoki niezależnie i nie zgadzają się względem struktury obiektu, dopasowanie 3D jest rozmyte. Poprawka: dyfuzja wielowidokowa z uwagą wspólną.
- **Halucynacje strony tylnej.** Single-image → 3D musi wymyślić niezobaczoną stronę. Jakość bardzo różna.
- **Eksplozja Gaussian splats.** Nie kontrolowane trenowanie rośnie do 10M splats i przeucza. Heurystyki gęstości + przycinania (z oryginalnego artykułu 3D-GS) są niezbędne.
- **Problemy topologiczne.** Siatki z pół niejawnych (SDF) często mają dziury lub samoprzecięcia. Uruchom remesher (np. voxel remesh bladera) przed wysłaniem.
- **Licencja danych treningowych.** Objaverse ma mieszane licencje; komercyjne użycie różni się w zależności od modelu.

## Użyj tego

| Zadanie | Wybór 2026 |
|---------|------------|
| Rekonstrukcja sceny ze zdjęć | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| Text-to-3D obiekt dla gier | Meshy 4 lub Rodin Gen-1.5 (wyjście PBR) |
| Image-to-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Synteza nowych widoków z mało obrazów | CAT3D, SV3D |
| Rekonstrukcja sceny dynamicznej | 4D Gaussian Splatting |
| Avatar / ubrany człowiek | Gaussian Avatar, HUGS |
| Badania / SOTA | Cokolwiek wypadło w ostatnim tygodniu |

Dla wysłania produkcyjnego 3D w potoku gier lub e-commerce: Meshy 4 lub Rodin Gen-1.5 wyprowadzają siatki PBR, które idą wprost do Unity / Unreal.

## Wyślij to

Zapisz `outputs/skill-3d-pipeline.md`. Skill bierze brief 3D (wejście: tekst / jeden obraz / kilka obrazów; wyjście: siatka / splat / NeRF; użycie: render / gra / VR) i wyprowadza: potok (dyfuzja wielowidokowa + dopasowanie, lub bezpośredni model siatki), model bazowy, budżet iteracji, pozaprocesowanie topologii, potrzebne kanały materiałów.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` z 4, 16, 64 Gaussianami. Zgłoś końcowy MSE względem celu.
2. **Średnie.** Rozszerz do kolorowych Gaussian (RGB). Potwierdź, że rekonstrukcja pasuje do docelowego wzorca kolorów.
3. **Trudne.** Używając gsplat lub Nerfstudio, zrekonstruuj prawdziwy obiekt z capture'u 50 zdjęć. Zgłoś czas dopasowania i końcowy SSIM na zdjęciach walidacyjnych.

## Kluczowe terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | Scena jako chmura 3D Gaussian; różniczkowalne kompozytowanie alfa renderu. |
| NeRF | "Neural radiance field" | MLP które wyprowadza kolor + gęstość w punkcie 3D; render przez całkowanie promieni. |
| Triplane | "Trzy płaszczyzny 2-D" | Czynnik 3D na trzy dwuwymiarowe siatki cech wyrównane do osi; tańsze niż wolumetryczne. |
| SDS | "Score distillation sampling" | Trenuj model 3D używając wynik dyfuzji 2D jako pseudogradient. |
| Multi-view diffusion | "Wiele widoków na raz" | Model dyfuzji który wyprowadza partie spójnych widoków kamery. |
| PBR | "Physically-based rendering" | Materiał z kanałami albedo, chropowatość, metaliczność, normalne. |
| Gęstość | "Rosnij splats" | Heurystyka treningu 3DGS: dziel / klonuj splats w regionach o wysokim gradiencie. |

## Uwaga produkcyjna: 3D nie ma jeszcze wspólnego podłoża

W przeciwieństwie do obrazu (dyfuzja latentna + DiT) i wideo (spaciotemporal DiT), 3D nie ma jednego dominującego środowiska wykonawczego w 2026. Drzewo decyzyjne produkcji rozgałęzia się na reprezentacji:

- **NeRF / triplane.** Wnioskowanie to ray-marching + przelot MLP na próbę. Render 512² wymaga milionów przelotów MLP. Partuj próby promieni agresywnie; SDPA/xformers się stosuje.
- **Dyfuzja wielowidokowa + rekonstrukcja LRM.** Potok dwuetapowy. Etap 1 (wielowidokowy DiT) to serwer dyfuzji jak Lekcja 07. Etap 2 (transformer LRM) to jednorazowy przelot przez widoki. Ogólny profil opóźnienia to "dyfuzja + jednorazowy" — wybieraj primitywy serwowania per-etap odpowiednio.
- **SDS / DreamFusion.** Optymalizacja per-asset, nie wnioskowanie. Buduj joby, nie handlery requestów.

Dla większości produktów 2026, prawidłowa odpowiedź to "uruchom model dyfuzji wielowidokowej na zadanie, rekonstruuj do 3DGS asynchronicznie, serwuj 3DGS dla czasu rzeczywistego". To czysto rozdziela workload między serwer inferencji GPU (szybki) a optymalizator offline (wolny).

## Dalsze czytanie

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — dyfuzja wielowidokowa.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.