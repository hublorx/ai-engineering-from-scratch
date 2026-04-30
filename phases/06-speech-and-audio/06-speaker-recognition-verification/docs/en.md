# Rozpoznawanie i weryfikacja mówcy

> ASR pyta „co powiedzieli?". Rozpoznawanie mówcy pyta „kto to powiedział?". Matematyka wygląda tak samo — embeddingi plus cosinus — ale każda decyzja produkcyjna zależy od jednej liczby EER.

**Typ:** Budowa
**Języki:** Python
**Wymagania wstępne:** Faza 6 · 02 (Spektrogramy i Mel), Faza 5 · 22 (Modele osadzania)
**Szacowany czas:** ~45 minut

## Problem

Użytkownik mówi hasło. Chcesz wiedzieć: czy to osoba, za którą się podaje (*weryfikacja*, 1:1), czy pierwsza osoba w twojej bazie enrollment (*identyfikacja*, 1:N)? A może żadna z nich — czy to nieznany mówca (*open-set*)?

Przed 2018: GMM-UBM + i-vectors. Przyzwoity EER, ale kruchy przy zmianie kanału (telefon vs laptop) i emocjach. 2018–2022: x-vectors (rdzeń TDNN trenowany z angular margin). 2022+: ECAPA-TDNN i embeddingi WavLM-large. W 2026 roku dziedzina jest zdominowana przez trzy modele i jedną metrykę.

Metryka to **EER** — Equal Error Rate. Ustaw próg decyzyjny tak, żeby False Accept Rate = False Reject Rate. Punkt przecięcia to EER. Używany w każdej pracy, każdym leaderboardzie, każdej rozmowie procurement.

## Koncepcja

![Potok enrollment + weryfikacja z embedding + cosinus + EER](../assets/speaker-verification.svg)

**Potok.** Enrollment: nagraj 5–30 sekund docelowego mówcy; oblicz embedding o ustalonej wymiarowości (192-d dla ECAPA-TDNN, 256-d dla WavLM-large). Weryfikacja: pobierz embedding wypowiedzi testowej; oblicz podobieństwo cosinusowe; porównaj z progiem.

**ECAPA-TDNN (2020, wciąż dominujący w 2026).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. Bloki konwolucyjne 1D z squeeze-excitation, multi-head attention pooling, a po nich warstwa liniowa do 192-d. Trenowany na VoxCeleb 1+2 (2700 mówców, 1,1M wypowiedzi) z Additive Angular Margin loss (AAM-softmax).

**WavLM-SV (2022+).** Fine-tune pretrained backbone WavLM-large SSL z AAM loss. Wyższa jakość, ale wolniejszy — 300+ MB vs 15 MB.

**x-vector (baseline).** TDNN + statistics pooling. Klasyczny; wciąż użyteczny na CPU / edge.

**AAM-softmax.** Standardowy softmax z dodanym marginesem `m` w przestrzeni kątowej: `cos(θ + m)` dla poprawnej klasy. Wymusza separację między klasami w przestrzeni kątowej. Typowy `m=0.2`, scale `s=30`.

### Scoring

- **Cosinus** między embeddingami enrollment i test. Decyzja oparta na progu.
- **PLDA (Probabilistic LDA).** Projektuje embeddingi w przestrzeń latentną, gdzie same-speaker vs different-speaker ma formę zamkniętą likelihood ratio. Dodawane na wierzch cosinusa dla redukcji EER o +10–20%. Standard sprzed 2020; teraz używany tylko w zamkniętych setupach.
- **Normalizacja wyników.** `S-norm` lub `AS-norm`: normalizuje każdy wynik przeciwko średnim i odchyleniom cohorty impostorów. Niezbędne do cross-domain eval.

### Liczby, które powinieneś znać (2026)

| Model | EER na VoxCeleb1-O | Parametry | Throughput (A100) |
|-------|--------------------|-----------|-------------------|
| x-vector (klasyczny) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarizacja

„Kto mówił kiedy" w nagraniu z wieloma mówcami. Potok: VAD → segment → embed każdy segment → klasteryzacja (agglomerative lub spectral) → wygładź granice. Nowoczesny stack: `pyannote.audio` 3.1, który łączy speaker segmentation + embedding + clustering w jednym wywołaniu. SOTA DER na AMI w 2026 to ~15% (spadek z 23% w 2022).

## Konstrukcja

### Krok 1: toy embedding z MFCC statistics

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Nie SOTA bynajmniej — tylko do nauczania. `code/main.py` używa tego jako proof-of-concept na syntetycznych danych mówcy.

### Krok 2: cosine similarity + threshold

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Krok 3: EER z par podobieństw

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Zwraca (eer, threshold_at_eer). Raportuj oba.

### Krok 4: produkcja ze SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: uśrednij embeddingi z 3-5 czystych próbek
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # typowy próg ECAPA; dostrój na swoich danych
```

### Krok 5: diarizacja z pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Zastosowanie

Stack na 2026:

| Sytuacja | Wybierz |
|----------|---------|
| Weryfikacja 1:1 zamknięta, edge | ECAPA-TDNN + próg cosinus |
| Weryfikacja open-set, chmura | WavLM-SV + AS-norm |
| Diarizacja (spotkania, podcasty) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (wykrywanie replay / deepfake) | AASIST lub RawNet2 |
| Małe embedded (KWS + enrollment) | Titanet-Small (NeMo) |

## Pułapki

- **Niedopasowanie kanału.** Model trenowany na VoxCeleb (web video) ≠ audio z rozmowy telefonicznej. Zawsze ewaluuj na docelowym kanale.
- **Krótkie wypowiedzi.** EER dramatycznie się pogarsza poniżej 3 sekund audio testowego.
- **Enrollment z szumem.** Jedno zaszumione enrollment zatruwa anchor. Użyj ≥3 czystych próbek i uśrednij.
- **Stały próg między warunkami.** Zawsze dostrajaj próg na held-out dev secie z docelowej domeny.
- **Cosinus na nieznormalizowanych embeddingach.** Najpierw L2-normalizuj; w przeciwnym razie magnitude dominuje.

## Wyślij to

Zapisz jako `outputs/skill-speaker-verifier.md`. Wybierz model, protokół enrollment, plan dostrajania progu i zabezpieczenia przed fraud.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Buduje syntetycznych „mówców" (różne profile tonów), enrollment, oblicza EER na liście 100 par trial.
2. **Średnie.** Użyj SpeechBrain ECAPA na 30 wypowiedziach VoxCeleb1 (5 mówców × 6 każdy). Oblicz EER z cosinus vs PLDA.
3. **Trudne.** Zbuduj pełny potok enroll → diarize → verify z `pyannote.audio`. Ewaluuj DER na AMI dev set.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| EER | Główna metryka | Próg gdzie False Accept = False Reject. |
| Verification | 1:1 | „Czy to Alice?" |
| Identification | 1:N | „Kto mówi?" |
| Open-set | Możliwy nieznany | Zbiór testowy może zawierać niezarejestrowanych mówców. |
| Enrollment | Rejestracja | Obliczanie referencyjnego embeddingu mówcy. |
| AAM-softmax | Loss | Softmax z addytywnym marginesem kątowym; wymusza separację klastrów. |
| PLDA | Klasyczny scoring | Probabilistic LDA; likelihood-ratio scoring na wierzchu embeddingów. |
| DER | Metryka diarizacji | Diarization Error Rate — miss + false alarm + confusion. |