# Ewaluacja — FID, CLIP Score, Preferencje Ludzkie

> Każda tablica wyników modeli generatywnych podaje FID, wynik CLIP i współczynnik wygranych z areny preferencji ludzkich. Każda liczba ma tryb awarii, który zdeterminowany badacz może wykorzystać. Jeśli nie znasz trybów awarii, nie możesz odróżnić prawdziwej poprawy od zmanipulowanego wyniku.

**Typ:** Zbuduj
**Języki:** Python
**Wymagania wstępne:** Phase 8 · 01 (Taxonomia), Phase 2 · 04 (Metryki Ewaluacyjne)
**Czas:** ~45 minut

## Problem

Model generatywny jest oceniany pod względem *jakości próbek* i *przestrzegania warunkowania*. Żadna z tych miar nie ma postaci zamkniętej. Twój model musi wyrenderować 10 000 obrazów; coś musi przypisać im liczby; musisz zaufać tym liczbom w obrębie rodzin modeli, rozdzielczości i architektur. Trzy metryki przeszły przez okres 2014-2026:

- **FID (Fréchet Inception Distance).** Odległość między dwoma rozkładami — rzeczywistym i wygenerowanym — w przestrzeni cech sieci Inception. Niższy jest lepszy.
- **CLIP score.** Podobieństwo cosinusowe między osadzeniem CLIP-obraz wygenerowanego obrazu a osadzeniem CLIP-tekst promptu. Wyższy jest lepszy. Mierzy przestrzeganie promptu.
- **Preferencja ludzka.** Zestaw dwa modele człowiek w człowieka na tym samym promptcie, daj ludziom (lub modelowi klasy GPT-4) wybrać lepszy, agreguj do wyniku Elo.

Zobaczysz też: IS (inception score, w dużej mierze wycofany), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Każda koryguje jedną wadę poprzedniego.

## Koncepcja

![FID, CLIP i preferencje: trzy osie, różne tryby awarii](../assets/evaluation.svg)

### FID — jakość próbki

Heusel i in. (2017). Kroki:

1. Wydobądź cechy Inception-v3 (2048-D) dla N rzeczywistych obrazów i N wygenerowanych.
2. Dopasuj rozkład Gaussa do każdej puli: oblicz średnią `μ_r, μ_g` i kowariancje `Σ_r, Σ_g`.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

Interpretacja: odległość Frécheta między dwoma wielowymiarowymi rozkładami Gaussa w przestrzeni cech. Niższy = bardziej podobne rozkłady.

Tryby awarii:
- **Obciążony przy małym N.** FID to średnia kwadratowa nad rozkładem cech — małe N niedoszacowuje kowariancje, daje fałszywie niski FID. Zawsze używaj N ≥ 10 000.
- **Zależny od Inception.** Inception-v3 była trenowana na ImageNet. Domeny dalekie od ImageNet (twarze, sztuka, obrazy tekstowe) produkują bezmeaningowy FID. Użyj ekstraktora cech specyficznego dla domeny.
- **Gaming.** Przeuczenie na prior Inception daje niski FID bez poprawy jakości wizualnej. Pokonaj to za pomocą CMMD (poniżej).

### CLIP score — przestrzeganie promptu

Radford i in. (2021). Dla wygenerowanego obrazu + promptu:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Średnia z 30k wygenerowanych obrazów → skalar porównywalny między modelami.

Tryby awarii:
- **Ślepe punkty CLIP.** CLIP ma słabą umiejętność rozumowania kompozycyjnego ("czerwony sześcian na niebieskiej kuli" często zawodzi). Modele mogą dobrze rankować wg CLIP score bez faktycznego śledzenia złożonych promptów.
- **Obciążenie krótkimi promptami.** Krótkie prompty mają więcej dopasowań CLIP-image w obiegu. Dłuższe prompty mechanicznie mają niższe CLIP score.
- **Gaming promptu.** Włączenie "wysoka jakość, 4k, arcydzieło" do promptu zawyża CLIP score bez poprawy wiązania obraz-tekst.

CMMD (Jayasumana i in., 2024) naprawia część z nich: używa cech CLIP zamiast Inception, maximum-mean discrepancy zamiast Frécheta. Lepsze wykrywanie subtelnych różnic jakości.

### Preferencja ludzka — ground truth

Wybierz pulę promptów. Generuj z modelem A i modelem B. Pokazuj pary ludziom (lub silnemu sędziemu LLM). Agreguj wygrane do wyniku Elo lub Bradley-Terry. Benchmarki:

- **PartiPrompts (Google)**: 1 600 zróżnicowanych promptów, 12 kategorii.
- **HPSv2**: 107k adnotacji ludzkich, szeroko używane jako zautomatyzowany substytut.
- **ImageReward**: 137k par preferencji prompt-obraz, licencja MIT.
- **PickScore**: trenowany na Pick-a-Pic 2.6M preferencji.
- **Areny obrazowe w stylu Chatbot Arena**: https://imagearena.ai/ i inne.

Tryby awarii:
- **Wariancja sędziów.** Niespecjaliści mają inne preferencje niż eksperci. Użyj obu.
- **Rozkład promptów.** Wyselekcjonowane prompty faworyzują jedną rodzinę. Zawsze dokumentuj.
- **Gaming nagrody sędziego LLM.** GPT-4-sędzia daje się oszukać ładnym, ale błędnym wynikom. Trianguluj z ludźmi.

## Używaj razem

Raport ewaluacji produkcyjnej powinien zawierać:

1. FID na 10-30k próbek względem wstrzymanego rozkładu rzeczywistego (jakość próbki).
2. CLIP score / CMMD na tych samych próbkach vs ich prompty (przestrzeganie).
3. Współczynnik wygranych w zaślepionej arenie vs poprzedni model (ogólna preferencja).
4. Analiza trybów awarii: 50 losowo probkowanych wyników, oflagowanych dla znanych problemów (anatomia rąk, renderowanie tekstu, spójna liczba obiektów).

Jakakolwiek pojedyncza metryka to kłamstwo. Trzy wspierające metryki + przegląd jakościowy to twierdzenie.

## Zbuduj to

`code/main.py` implementuje FID, CLIP-score-like i agregacje Elo na syntetycznych "wektorach cech" (używamy wektorów 4-D jako zamienników dla cech Inception). Zobaczysz:

- Obliczanie FID na małym N i dużym N — obciążenie.
- "CLIP score" jako podobieństwo cosinusowe między pulami cech.
- Regułę aktualizacji Elo ze syntetycznego strumienia preferencji.

### Kroki 1: FID w czterech liniach

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Kroki 2: CLIP-style cosine-similarity

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Kroki 3: Agregacja Elo

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Pułapki

- **FID przy N=1000.** Heurystyka jest niewiarygodna poniżej N=10k. Artykuły raportujące niskie-N FID to gaming.
- **Porównywanie FID w rozdzielczościach.** Resize 299x299 Inception zmienia rozkład cech. Porównuj tylko przy dopasowanej rozdzielczości.
- **Raportowanie jednego seeda.** Uruchom min. 3 seedy. Raportuj odchylenie standardowe.
- **Inflacja CLIP score przez negatywne prompty.** Niektóre pipeliny podnoszą CLIP przez przeuczenie promptu. Sprawdź nasycenie wizualne.
- **Obciążenie Elo przez nachodzenie promptów.** Jeśli oba modele widziały prompt benchmarkowy podczas treningu, Elo jest bezmeaningowe. Użyj wstrzymanych zestawów promptów.
- **Sklonowanie ewaluacji ludzkiej przez płatną firmę.** Prolific, MTurk adnotatorzy są młodsi / tech-friendly. Mieszaj z zrekrutowanymi ekspertami sztuki/projektowania.

## Użyj tego

Protokół ewaluacji produkcyjnej w 2026:

| Filar | Minimum | Zalecane |
|--------|---------|-------------|
| Jakość próbki | FID na 10k vs wstrzymany rzeczywisty | + CMMD na 5k + FID na podzbiorze dla kategorii |
| Przestrzeganie promptu | CLIP score na 30k | + HPSv2 + ImageReward + Q&A stylu VQA |
| Preferencja | 200 zaślepionych par vs baseline | + 2000 sparowanych ludzi + sędzia LLM + Chatbot Arena |
| Analiza awarii | 50 ręcznie oflagowanych | 500 ręcznie oflagowanych + automatyczny klasyfikator bezpieczeństwa |

Wszystkie cztery filary w jednym raporcie = twierdzenie. Jakikolwiek sam = marketing.

## Wyślij to

Zapisz `outputs/skill-eval-report.md`. Skill bierze nowy checkpoint modelu + baseline i generuje pełny plan ewaluacji: wielkości próbek, metryki, sondy trybów awarii, kryteria akceptacji.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Porównaj FID przy N=100 vs N=1000 na tych samych syntetycznych rozkładach. Zgłoś wagę obciążenia.
2. **Średnie.** Implementuj CMMD z syntetycznych cech w stylu CLIP (zobacz Jayasumana i in., 2024 za formułą). Porównaj czułość na różnice jakości vs FID.
3. **Trudne.** Zreplikuj setup HPSv2: weź 1000 par obraz-prompt z podzbioru Pick-a-Pic, dostroj małą scorerkę opartą na CLIP na preferencjach, i zmierz jej zgodność ze wstrzymanym zestawem.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie znaczy |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | Odległość Frécheta dopasowanego Gaussa do cech Inception real vs gen. |
| CLIP score | "Podobieństwo tekst-obraz" | Podobieństwo cosinusowe między osadzeniami CLIP obrazu i tekstu. |
| CMMD | "Zamiennik FID" | MMD oparte na cechach CLIP; mniej obciążone, bez założenia Gaussa. |
| IS | "Inception score" | Exp KL(p(y|x) || p(y)); słabo koreluje na nowoczesnych modelach, wycofany. |
| HPSv2 / ImageReward / PickScore | "Nauczone substytuty preferencji" | Małe modele trenowane na preferencjach ludzkich; używane jako automatyczni sędziowie. |
| Elo | "Ranking szachowy" | Agregacja Bradley-Terry parek wygranych. |
| PartiPrompts | "Zestaw promptów benchmarkowych" | 1 600 kuratorowanych przez Google promptów w 12 kategoriach. |
| FD-DINO | "Zamiennik samo-nadzorowany" | FD z cechami DINOv2; lepszy dla domen poza ImageNet. |

## Uwaga produkcyjna: ewaluacja to też workload inference

Uruchomienie FID na 10k próbek oznacza generowanie 10k obrazów. Dla bazy SDXL przy 50 krokach w 1024² na pojedynczej L4, to ~11 godzin inference jednego requestu. Budżety ewaluacji są realne, a framing to dokładnie scenariusz offline-inference (maksymalizuj przepustowość, ignoruj TTFT):

- **Batchuj mocno, zapomnij latency.** Offline eval = statyczny batching przy największym rozmiarze mieszczącym się w pamięci. `pipe(...).images` z `num_images_per_prompt=8` na 80GB H100 chodzi 4-6x szybciej ścieżkowo niż single-request.
- **Cacheuj rzeczywiste cechy.** Ekstrakcja cech Inception (FID) lub CLIP (CLIP-score, CMMD) na rzeczywistym zestawie referencyjnym jest uruchamiana *raz*, zapisywana jako `.npz`. Nie przeliczaj przy każdej ewaluacji.

Dla CI / bramek regresji: uruchom FID + CLIP score na podzbiorze 500 próbek na PR (~30 min); uruchom pełny 10k FID + HPSv2 + Elo nocą.

## Dalsze czytanie

- [Heusel i in. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — artykuł o FID.
- [Jayasumana i in. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford i in. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu i in. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu i in. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu i in. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein i in. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — przegląd trybów awarii.
```