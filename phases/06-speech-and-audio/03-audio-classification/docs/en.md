# Klasyfikacja dźwięku — Od k-NN na MFCC do AST i BEATs

> Wszystko od "szczekanie psa vs syrena" do "jaki to język" to klasyfikacja dźwięku. Cechy to mels. Architektura przesuwa się co dekadę. Ewaluacja pozostaje: AUC, F1 i recall per klasa.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 6 · 02 (Spektrogramy i Mel), Faza 3 · 06 (CNN), Faza 5 · 08 (CNN i RNN dla tekstu)
**Szacowany czas:** ~75 minut

## Problem

Dostajesz 10-sekundowy klip. Chcesz wiedzieć: "co to jest?" Dźwięk miejski (syrena, wiertarka, pies), komendy głosowe (tak/nie/stop), ID języka (en/es/ar), emocja mówcy (zły/neutralny) lub dźwięk środowiskowy (wnętrze/na zewnątrz, gwar). Wszystko to jest *klasyfikacją dźwięku*, a w 2026 baseline architektura jest dojrzała:

Główna trudność nie polega na sieci. Polega na danych. Zbiory audio mają brutalne niezbalansowanie klas, silne przesunięcie domeny (czyste vs zaszumione) i szum w etykietach (kto zdecydował "gwar miejski" vs "hałas restauracji"?). 80% problemu to kuracja, augmentacja i ewaluacja, nie zamiana CNN na Transformer.

## Koncepcja

![Drabina klasyfikacji audio: k-NN na MFCC do AST do BEATs](../assets/audio-classification.svg)

**k-NN na MFCC (baseline z lat 90.).** Spłaszcz MFCC dla klipu, oblicz podobieństwo cosinusowe do banku z etykietami, zwróć głosowanie większościowe top K najbliższych. Zaskakująco skuteczny na czystych, małych zbiorach (Speech Commands, ESC-50). Działa bez GPU.

**2D CNN na log-mels (2015-2019).** Traktuj `(T, n_mels)` log-mel jako obraz. Zastosuj ResNet-18 lub VGG-style. Global mean pool na osi czasu. Softmax na klasach. Wciąż baseline w większości konkursów kaggle w 2026.

**Audio Spectrogram Transformer, AST (2021-2024).** Patchyfikuj log-mel (np. 16×16 patches), dodaj position embeddings, wrzuć do ViT. State of the art na AudioSet (mAP 0.485) dla supervised learning.

**BEATs i WavLM-base (2024-2026).** Samonadzorowany pretraining na milionach godzin. Fine-tune na swoim zadaniu z 1-10% supervised danych których potrzebowałbyś normalnie. W 2026 to domyślny punkt startowy dla audio nie-mową. BEATs-iter3 bije AST o 1-2 mAP na AudioSet używając 1/4 compute.

**Whisper-encoder jako zamrożony szkielet (2024).** Weź encoder Whispera, odłącz dekoder, doczep liniowy klasyfikator. Near-SOTA na ID języka i prostej klasyfikacji zdarzeń z zerową augmentacją audio. "Free lunch" baseline.

### Niezbalansowanie klas to prawdziwe wyzwanie

ESC-50: 50 klas, 40 klipów każda — zbalansowane, łatwe. UrbanSound8K: 10 klas, niezbalansowane 10:1. AudioSet: 632 klasy z ogonem długości 100,000:1. Techniki które działają:

- Zbalansowane sampling podczas treningu (nie w ewaluacji).
- Mixup: liniowo interpoluj dwa klipy (i ich etykiety) jako augmentacja.
- SpecAugment: maszkuj losowe pasma czasowe i częstotliwościowe. Proste; krytyczne.

### Ewaluacja

- Wieloklasowa exclusive (Speech Commands): top-1 accuracy, top-5 accuracy.
- Wieloklasowa multi-label (AudioSet, UrbanSound-style): mean average precision (mAP).
- Silnie niezbalansowane: recall per klasa + macro F1.

Liczby 2026 które powinieneś znać:

| Benchmark | Baseline | SOTA 2026 | Źródło |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

## Zbuduj to

### Krok 1: tworzenie cech

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Krok 2: podsumowanie o stałej długości

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Proste ale skuteczne: średnia + wariancja w czasie daje 26-dim stały embedding dla 13-coef MFCC. Działa natychmiast. Pokonuje NN baselines na ESC-50 jeszcze w 2017.

### Krok 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Krok 4: modernizacja do CNN na log-mels

W PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M parametrów. Trenuje się w ~10 min na ESC-50 z pojedynczą RTX 4090. 80%+ accuracy.

### Krok 5: domyślny w 2026 — fine-tune BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

Dla BEATs użyj `microsoft/BEATs-base` przez bibliotekę `beats`; API transformers jest ten sam kształt.

## Użyj tego

Stack 2026:

| Sytuacja | Zacznij od |
|-----------|-----------|
| Mały zbiór danych (<1000 klipów) | k-NN na średnich MFCC (twój baseline) + augmentacja audio |
| Średni zbiór danych (1K–100K) | Fine-tune BEATs lub AST |
| Duży zbiór danych (>100K) | Trenuj od zera lub fine-tune Whisper-encoder |
| Czas rzeczywisty, edge | 40-MFCC CNN, skwantowany do int8 (KWS-style) |
| Multi-label (AudioSet) | BEATs-iter3 z BCE loss + mixup + SpecAugment |
| ID języka | MMS-LID, SpeechBrain VoxLingua107 baseline |

Zasada decyzyjna: **zaczynaj od zamrożonego szkieletu, nie świeżego modelu**. Fine-tune głowy BEATs daje ci 95% SOTA w godzinach, nie tygodniach.

## Wyślij to

Zapisz jako `outputs/skill-classifier-designer.md`. Wybierz architekturę, augmentacje, strategię balansu klas i metrykę ewaluacji dla danego zadania klasyfikacji audio.

## Ćwiczenia

1. **Łatwy.** Uruchom `code/main.py`. Trenuje k-NN MFCC baseline na syntetycznym zbiorze 4 klas (czyste tony w różnych pitchach). Zgłoś macierz pomyłek.
2. **Średni.** Zastąp `summarize` [mean, var, skew, kurtosis]. Czy 4-moment pooling bije mean+var na tym samym syntetycznym zbiorze?
3. **Trudny.** Używając `torchaudio`, trenuj 2D CNN na ESC-50 fold 1. Zgłoś 5-fold cross-validation accuracy. Dodaj SpecAugment (time mask = 20, freq mask = 10) i zgłoś deltę.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| AudioSet | ImageNet audio | Zbiór Google: 2M klipów, 632 klasy, weakly-labeled z YouTube. |
| ESC-50 | Mały benchmark klasyfikacyjny | 50 klas × 40 klipów dźwięków środowiskowych. |
| AST | Audio Spectrogram Transformer | ViT na log-mel patches; SOTA 2021-2024. |
| BEATs | Self-supervised audio | Model Microsoft, iter3 prowadzi AudioSet w 2026. |
| Mixup | Pair augmentation | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Mask-based augmentation | Zeruj losowe pasma czasowe i częstotliwościowe spektrogramu. |
| mAP | Główna metryka multi-label | Mean average precision przez klasy i progi. |

## Dalsze czytanie

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — architektura stanu wiedzy 2021–2024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — domyślny od 2024+.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — dominująca augmentacja audio.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — benchmark 50-klasowy który przetrwał.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — taksonomia YouTube 632 klas; wciąż złoty standard.