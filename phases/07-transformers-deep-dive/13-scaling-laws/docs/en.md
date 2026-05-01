# Prawa Skalowania

> Artykuł Kaplana z 2020 powiedział: większy model, niższy loss. Artykuł Hoffmanna z 2022 powiedział: niedoszkalałeś. Compute dzieli się na dwa kubełki — parametry i tokeny — i podział nie jest oczywisty.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 05 (Full Transformer), Faza 7 · 07 (GPT)
**Czas:** ~45 minut

## Problem

Gdy masz C FLOPs treningowego compute i chcesz najlepszy model, masz dwa pokrętła:

1. **Ile parametrów (N)?** Większy model, większa pojemność.
2. **Ile tokenów treningowych (D)?** Więcej danych, lepsze wykorzystanie pojemności.

FLOPy skalują się w przybliżeniu jako `6 × N × D`. Możesz pchać N w górę i D w dół, lub D w górę i N w dół. Co jest lepsze?

Przed 2022 odpowiedź brzmiała "pchać N mocno." GPT-3 (2020) miał 175B parametrów trenowanych na ~300B tokenach. Stosunek około 1.7 tokenów na parametr. Prawa skalowania Kaplana to potwierdzały.

Hoffmann et al. (2022), trenując małą rodzinę modeli o nazwie Chinchilla, znaleźli coś innego: optymalny stosunek jest bliższy **20 tokenów na parametr**. GPT-3 był 10× niedoszkolony. Chinchilla (70B params, 1.4T tokenów) pobił GPT-3 (175B, 300B tokenów) na każdym benchmarku przy 2.5× mniejszym koszcie inferencji.

2026 to świat Chinchilli — z jednym ważnym twistem. Llama 3 8B była trenowana na 15 bilionów tokenów, stosunek 1,875 tokenów na parametr. Dziewięćdziesiąt cztery razy powyżej Chinchilla-optymalnego. Koszt inferencji ma większe znaczenie niż koszt treningu dla modeli które będą używane na skali, więc over-training (ponad Chinchilla) dla mniejszego deployowalnego footprintu to domyślne 2026.

## Koncepcja

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Prawo Hoffmanna

Z papieru Chinchilla, loss wynika:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parametry (non-embedding).
- `D` = tokeny treningowe.
- `α ≈ 0.34`, `β ≈ 0.28` (w przybliżeniu symetryczne).
- `E ≈ 1.69`, sufit irreducible loss.
- `A ≈ 406`, `B ≈ 411`.

Dwa wyrazy handlują przeciwko sobie jak skalujesz. Weź pochodną względem `N` przy ustalonym compute (C = 6ND) i rozwiąż:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Compute-optymalne: 20 tokenów na parametr.

### Dlaczego over-training anyway

Chinchilla-optymalne minimalizuje training loss per training FLOP. Ale płacisz koszt treningu raz; koszt inferencji na zawsze.

Dla czata który serwuje bilion tokenów miesięcznie, inferencja dominuje całkowity koszt. Podejście Llamy: trenuj mniejszy, dłużej. 8B przy 15T tokenów jest głęboko zoptymalizowane pod inferencję:

- Wszechstronne na GPU konsumentów.
- Latency to ułamek 70B Chinchilla-optymalnego.
- Jakość jest wystarczająco blisko dla większości zadań.

Papier DeepMind z 2024 ("Over-training is the new optimal") to sformalizował. Dla workloadów zdominowanych przez inferencję, właściwy stosunek jest bliższy 100–500 tokenów na parametr zależnie od wolumenu serwowania.

### Emergence vs smoothness

Claim: pewne zdolności (arytmetyka, wielo-krokowe rozumowanie, chain-of-thought following) "emergują" nagle na pewnej skali.

Schaeffer et al. (2023) argumentowali że to artifact pomiarowy: emergent metrics używają discontinuous scoring (exact match, accuracy at threshold) które ukrywają smooth improvement w underlying logits. Continuous metrics (cross-entropy) pokazują smooth curves.

W 2026 consensus jest: predictions via continuous loss są reliable. Benchmark jumps są często scorer artifacts. Plan budgets przeciwko continuous metrics.

### Obraz 2026

Prawa skalowania nadal działają, ale:

| Factor | Changed how |
|--------|-------------|
| Data quality | Curating "good" tokens (Phi-style) shifts curves by >2× effective compute |
| MoE | Total params decouple from active FLOPs; scaling laws per-active-FLOP |
| Post-training | Some capabilities (instruction following, code) shift with SFT+RLHF more than pretraining |
| Multimodality | Image + text tokens scale together; separate curves per modality |
| Synthetic data | Models generate training data; effective compute can compound |

Muon optimizer (Kimi Moonlight, 2024) pokazał ~2× effective-compute gain over AdamW at matched data. Niektóre treningi 2026 używają Muon domyślnie. Zmienia absolutną stałą w prawie skalowania, nie jego kształt.

## Zbuduj To

Zobacz `code/main.py`. Implementujemy równanie loss Chinchilla i rozwiązujemy dla compute-optymalnego `(N, D)` przy każdym z kilku budżetów compute.

### Krok 1: Chinchilla loss

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

Plot `L` jako kontur nad `(N, D)` przy ustalonym `C = 6ND`. Znajdź minimum.

### Krok 2: compute-optimal frontier

Dla budżetów compute od `1e17` do `1e25` FLOPs, znajdź `(N, D)` które minimalizują loss przy `6ND = C`. Zweryfikuj stosunek `D/N ≈ 20`.

### Krok 3: over-training cost

Oblicz dodatkowy loss który płacisz żeby trenować model 10× mniejszy (1/10 optimal N, 10× optimal D). Reports the inference FLOP savings (proportional to N) w zamian.

### Krok 4: compare to real models

Wstaw znane pary `(N, D)` dla GPT-3, Chinchilla, Llama 3 8B, DeepSeek-V3 (active params), i porównaj predicted vs reported loss.

## Użyj To

Nie będziesz prawdopodobnie trenował frontu modelu sam. Ale prawa skalowania mówią ci:

1. **Czy twój fine-tune ma wystarczająco dużo danych.** Jeśli twoje dane specyficzne dla zadania są poniżej 20 tokenów na parametr base modelu, spodziewaj się saturacji przy jakimś loss floor.
2. **Czy wybrać większy base model.** Jeśli wydajesz cały budżet na inferencję, wolisz mniejszy, dłużej trenowany model.
3. **Gdzie zwroty maleją.** Powyżej 1000× Chinchilla-optymalnego, zmiany log-loss stają się szumem.

**Trajektoria badań w 2026:**

- **Data-constrained regime.** Web ma skończoną liczbę high-quality tokens (~5–10 trillion English after filtering). Frontier pretraining zbliża się do tego sufitu. Synthetic data, multilingual, multimodal, i RLHF-scaled fine-tuning to następne dźwignie.
- **Compute-multiplier tricks.** Muon optimizer, MoE, better data curation — każdy przesuwa absolute constants, nie asymptote.
- **Prawa skalowania dla RL.** Otwarte pytanie. Wczesne dowody sugerują power-law w RL samples ale z bardzo różnymi exponents niż pretraining.

## Wyślij To

Zobacz `outputs/skill-training-budget-estimator.md`. Skill wybiera `(N, D, hours, GPU)` dla nowego treningu przy danym budżecie compute, deployment constraints, i target loss.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Wydrukuj Chinchilla-optymalne `(N, D)` dla budżetów compute `1e20`, `1e22`, `1e24`. Porównaj do real model table.
2. **Średnie.** Zaimplementuj Hoffmann loss-as-function-of-compute curve. Plot loss vs `log10(C)` dla compute-optimal frontier. Zidentyfikuj kiedy pravo przewiduje że potrzebowalibyśmy `>10^28` FLOPs dla następnego 0.1 redukcji w cross-entropy.
3. **Trudne.** Dopasuj własne prawo skalowania na 5 tiny models (100K do 10M params) trenowanych na tym samym datasetcie. Oszacuj `α` i `E`. Jak dobrze twoje exponents pasują do opublikowanych?

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Parameters (N) | "Model size" | Non-embedding weight count; określa pojemność. |
| Tokens (D) | "Training data" | Liczba tokenów treningowych seen; określa jak dobrze parametry są wykorzystane. |
| Compute (C) | "FLOPs spent" | W przybliżeniu `6 × N × D` dla standardowego transformerа. |
| Chinchilla-optimal | "D/N ≈ 20" | Stosunek który minimalizuje loss per FLOP pretrainingu. |
| Over-training | "Past Chinchilla" | Wydaj dodatkowe treningowe FLOPy żeby zaoszczędzić inference FLOPy; D/N >> 20. |
| Irreducible loss | "The floor" | Wyraz `E` w prawie skalowania; entropia samych danych. |
| Emergent capability | "Sudden jumps at scale" | Często artifact scorera; continuous loss jest gładki. |
| Effective compute | "Training-efficiency multiplier" | Lepsze dane / optimizer / architecture mnożą jak daleko idzie FLOP. |

## Dalsze Czytanie

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — pierwszy paper o prawach skalowania; niedoszkolony.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — emergence jako artifact pomiarowy.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — dlaczego over-training Llamy jest właściwy dla jej workloadu.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× compute multiplier.