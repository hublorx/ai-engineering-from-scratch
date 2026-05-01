# Vision Transformers (ViT)

> Obraz to siatka patchy. Zdanie to siatka tokenów. Ten sam transformer je oba zjada.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 05 (Full Transformer), Faza 4 · 03 (CNNs), Faza 4 · 14 (Vision Transformers intro)
**Czas:** ~45 minut

## Problem

Przed 2020 computer vision oznaczało convolutions. Każdy SOTA na ImageNet, COCO i detection benchmarks używał CNN backbone. Transformers były dla języka.

Dosovitskiy et al. (2020) — "An Image is Worth 16x16 Words" — pokazali, że można completely porzucić convolutions. Pokrój obraz na fixed-size patches, linearly project każdy patch do embeddingu, podaj sekwencję do vanilla transformer encoder. Przy wystarczającej skali (ImageNet-21k pretraining lub większej), ViT dorównuje lub przebija ResNet-based models.

ViT był początkiem szerszego pattern w 2026: jedna architektura, wiele modalności. Whisper tokenizuje audio. ViT tokenizuje obrazy. Action tokens dla robotyki. Pixel tokens dla wideo. Transformer nie dba — podawaj mu sekwencję, a on się uczy.

Do 2026, ViT i jego descendants (DeiT, Swin, DINOv2, ViT-22B, SAM 3) są w większości vision. CNNs nadal wygrywają na edge devices i latency-sensitive tasks. Wszystko inne ma ViT gdzieś w stacku.

## Koncepcja

![Obraz → patches → tokens → transformer](../assets/vit.svg)

### Krok 1 — patchify

Podziel obraz `H × W × C` na sekwencję `N × (P·P·C)` flat patches. Typowy setup: `224 × 224` obraz, `16 × 16` patches → 196 patches każdy po 768 wartości.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Rozmiar patcha to dźwignia. Mniejsze patches = więcej tokenów, lepsza rozdzielczość, kwadratowy koszt attention. Większe patches = grubsze, tańsze.

### Krok 2 — linear embedding

Pojedyncza nauczona macierz projektuje każdy flat patch do `d_model`. Równoważne convolution z kernel size `P` i stride `P`. W PyTorch to dosłownie `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — implementacja w 2 linijkach.

### Krok 3 — prepend `[CLS]` token, dodaj positional embeddings

- Prepend learnable `[CLS]` token. Jego final hidden state to reprezentacja obrazu używana do klasyfikacji.
- Dodaj learnable positional embeddings (ViT-original) lub sinusoidal 2D (later variants).
- W 2024+ RoPE rozszerzone do 2D dla position, czasem bez explicit embeddings.

### Krok 4 — standard transformer encoder

Złóż L blocks `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Identyczne jak BERT. Żadnych vision-specific layers. To jest pedagogical punchline paperu.

### Krok 5 — head

Dla klasyfikacji: weź `[CLS]` hidden state → linear → softmax. Dla DINOv2 lub SAM, odrzuć `[CLS]`, użyj patch embeddings bezpośrednio.

### Warianty, które miały znaczenie

| Model | Rok | Zmiana |
|-------|------|--------|
| ViT | 2020 | Oryginał. Fixed patch size, full global attention. |
| DeiT | 2021 | Distillation; trainable na ImageNet-1k tylko. |
| Swin | 2021 | Hierarchiczny z shifted windows. Fixed sub-quadratic cost. |
| DINOv2 | 2023 | Self-supervised (bez labels). Najlepsze general vision features. |
| ViT-22B | 2023 | 22B params; scaling laws apply. |
| SigLIP | 2023 | ViT + language pair, sigmoid contrastive loss. |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder. |

### Dlaczego to tyle trwało

ViT potrzebuje *dużo* danych żeby dorównać CNNs bo nie ma żadnych CNN inductive biases (translation invariance, locality). Bez >100M labeled images lub strong self-supervised pretraining, CNNs nadal wygrywają przy matched compute. DeiT to naprawił w 2021 z distillation tricks; DINOv2 naprawił to permanentnie w 2023 z self-supervision.

## Zbuduj to

Zobacz `code/main.py`. Pure-stdlib patchify + linear embedding + sanity checks. Bez treningu — ViT w jakiejkolwiek realistic skali potrzebuje PyTorch i godzin GPU time.

### Krok 1: fake image

Obraz 24 × 24 RGB jako lista wierszy `(R, G, B)` tuples. Używamy 6×6 patches → 16 patches, 108-d embedding vector każdy.

### Krok 2: patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Raster order: row-major across the grid. Każdy ViT używa tego ordering.

### Krok 3: linear embed

Pomnóż każdy flat patch przez random `(patch_flat_size, d_model)` matrix. Zweryfikuj że output shape to `(N_patches + 1, d_model)` po prependowaniu `[CLS]`.

### Krok 4: policz parametry dla realistic ViT

Wydrukuj param count dla ViT-Base: 12 layers, 12 heads, d=768, patch=16. Porównaj do ResNet-50 (~25M). ViT-Base ląduje na ~86M. ViT-Large ~307M. ViT-Huge ~632M.

## Użyj tego

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings to 2026 default dla image features.** Zamroź backbone, trenuj tiny head. Działa dla klasyfikacji, retrieval, detection, captioning. Meta's DINOv2 checkpoints outperform CLIP na każdym non-text vision task.

**Patch-size picking.** Małe modele używają 16×16 (ViT-B/16). Dense prediction (segmentation) używa 8×8 lub 14×14 (SAM, DINOv2). Bardzo duże modele używają 14×14.

## Wyślij to

Zobacz `outputs/skill-vit-configurator.md`. Skill wybiera ViT variant i patch size dla nowego vision task przy given dataset size, resolution i compute budget.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Zweryfikuj że number of patches equals `(H/P) * (W/P)` a flat patch dimension equals `P*P*C`.
2. **Średnie.** Zaimplementuj 2D sinusoidal positional embeddings — dwa niezależne sinusoidal codes dla `row` i `col` każdego patcha, concatenated. Podaj je do tiny PyTorch ViT i porównaj accuracy vs learnable positional embeddings na CIFAR-10.
3. **Trudne.** Zbuduj 3-warstwowy ViT (PyTorch), trenuj na 1,000 obrazach MNIST z 4×4 patches. Zmierz test accuracy. Teraz dodaj DINOv2 pretraining na tych samych 1,000 obrazach (uproszczone: po prostu trenuj encoder żeby predict patch embeddings z masked patches). Czy accuracy się poprawia?

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|--------------------------|
| Patch | "Vision-transformer token" | Flat vector wartości pikseli dla regionu `P × P × C` obrazu. |
| Patchify | "Chop + flatten" | Pokrój obraz na non-overlapping patches, spłaszcz każdy do wektora. |
| `[CLS]` token | "Podsumowanie obrazu" | Prepended learnable token; jego final embedding to reprezentacja obrazu. |
| Inductive bias | "Co model zakłada" | ViT ma mniej priors niż CNNs; potrzebuje więcej danych żeby wypełnić lukę. |
| DINOv2 | "Self-supervised ViT" | Trenowany bez labels używając image augmentation + momentum teacher. Najlepsze general image features w 2026. |
| SigLIP | "Następca CLIP" | ViT + text encoder trenowany z sigmoid contrastive loss; lepszy niż CLIP przy matched compute. |
| Swin | "Windowed ViT" | Hierarchiczny ViT z local attention + shifted windows; sub-quadratic. |
| Register tokens | "Sztuczka z 2023" | Kilka dodatkowych learnable tokens które absorbją attention sinks; poprawia DINOv2 features. |

## Dalsze Czytanie

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — paper ViT.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — fix register-token dla DINOv2.