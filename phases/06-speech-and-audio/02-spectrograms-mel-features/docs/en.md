# Spektrogramy, skala mel i cechy audio

> Neural networks nie konsumują dobrze surowych waveformów. Konsumują spektrogramy. Jeszcze lepiej konsumują mel-spektrogramy. Każdy ASR, TTS i klasyfikator audio w 2026 roku żyje lub umiera przez ten jeden wybór preprocessingu.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 6 · 01 (Audio Fundamentals)
**Szacowany czas:** ~45 minut

## Problem

Weźmy 10-sekundowy klip 16 kHz. To jest 160 000 floatów, wszystkie w `[-1, 1]`, prawie idealnie nieskorelowane z etykietą "pies szczeka" lub "słowo kot". Surowy waveform ma informację, ale w formie, z której model nie może łatwo jej wyciągnąć. Dwa identyczne fonemy wypowiedziane 100 ms od siebie mają zupełnie inne surowe próbki.

Spektrogram to naprawia. Zgniata szczegóły czasowe, których percepcja ludzka ignoruje (mikrosekundowy jitter) i zachowuje strukturę, na którą percepcja zwraca uwagę (które częstotliwości są energetyczne, w oknach czasowych ~10–25 ms).

Mel-spektrogramy idą dalej. Ludzie postrzegają wysokość dźwięku logarytmicznie: 100 Hz vs 200 Hz brzmi "tak samo oddalone" jak 1000 Hz vs 2000 Hz. Skala mel zakrzywia oś częstotliwości, żeby pasowała do percepcji. Mel-spektrogram to najważniejsza cecha w speech ML od 2010 do 2026 roku.

## Koncept

![Od waveformu do STFT do mel-spektrogramu do MFCC ladder](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform).** Pokrój waveform na nakładające się ramki (typowo: okno 25 ms, hop 10 ms = 400 próbek / 160 próbek przy 16 kHz). Pomnóż każdą ramkę przez funkcję okna (Hann to domyślna; Hamming ma nieco inny kompromis). Wykonaj FFT każdej ramki. Złóż widma magnitudy w macierz o kształcie `(n_frames, n_freq_bins)`, która jest twój spektrogram.

**Log-magnitudy.** Surowe magnitudy obejmują 5-6 rzędów wielkości. Weź `log(|X| + 1e-6)` lub `20 * log10(|X|)`, żeby skompresować zakres dynamiczny. Każdy produkcyjny pipeline używa log-magnitude, nie surowej magnitudy.

**Skala Mel.** Częstotliwość `f` w Hz mapuje na mel `m` przez `m = 2595 * log10(1 + f / 700)`. Mapowanie jest w przybliżeniu liniowe poniżej 1 kHz i w przybliżeniu logarytmiczne powyżej. 80 mel binów obejmujących 0–8 kHz to standardowe wejście ASR.

**Mel filterbank.** Zbiór trójkątnych filtrów rozmieszczonych równo na skali mel. Każdy filtr to ważona suma sąsiednich binów FFT. Mnożenie magnitudy STFT przez macierz filterbank daje mel-spektrogram w jednym matmul.

**Log-mel spectrogram.** `log(mel_spec + 1e-10)`. Wejście Whispera. Wejście Parakeeta. Wejście SeamlessM4T. Uniwersalny frontend audio 2026.

**MFCCs.** Weź log-mel spectrogram, zastosuj DCT (typ II), zachowaj pierwsze 13 współczynników. Dekoreluje cechy i kompresuje dalej. Dominująca cecha do około 2015 roku, kiedy CNN/Transformery na surowych log-melach dogoniły. Wciąż używane w rozpoznawaniu mówcy (x-vectors, ECAPA).

**Kompromis rozdzielczości.** Większe FFT = lepsza rozdzielczość częstotliwościowa, ale gorsza rozdzielczość czasowa. 25 ms / 10 ms to domyślne audio-ML; 50 ms / 12.5 ms dla muzyki; 5 ms / 2 ms dla detekcji transjentów (uderzenia bębna, plosywy).

## Zbuduj to

### Krok 1: ramkowanie waveformu

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

10-sekundowy klip 16 kHz z `frame_len=400, hop=160` daje 998 ramek.

### Krok 2: okno Hann

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Mnożenie element-wise przed FFT. Usuwa przeciek widmowy spowodowany obcięciem na niezerowych końcach.

### Krok 3: magnitude STFT

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

Produkcja używa `torch.stft` lub `librosa.stft` (FFT-backed, wektoryzowane). Pętla tutaj jest dydaktyczna, działa na krótkich klipach w `code/main.py`.

### Krok 4: mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mels obejmujących 0–8 kHz z `n_fft=400` daje macierz `(80, 201)`. Pomnóż magnitudę STFT `(n_frames, 201)` przez transpozycję, żeby dostać mel-spektrogram `(n_frames, 80)`.

### Krok 5: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

Alternatywy: `librosa.power_to_db` (reference-normalized dB), `10 * log10(power + eps)`. Whisper używa bardziej złożonej procedury clip + normalize (zobacz Whisper's `log_mel_spectrogram`).

### Krok 6: MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Zastosuj DCT do każdej ramki log-mel, zachowaj pierwsze 13 współczynników. To jest twoja macierz MFCC. Pierwszy współczynnik zwykle się porzuca (koduje ogólną energię).

## Użyj tego

Stack w 2026:

| Zadanie | Cechy |
|---------|-------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels, 10 ms hop, 25 ms okno |
| TTS acoustic model (VITS, F5-TTS, Kokoro) | 80 mels, 5–12 ms hop dla precyzyjnej kontroli czasowej |
| Audio classification (AST, PANNs, BEATs) | 128 log-mels, 10 ms hop |
| Speaker embedding (ECAPA-TDNN, WavLM) | 80 log-mels lub raw-waveform SSL |
| Muzyka (MusicGen, Stable Audio 2) | EnCodec discrete tokens (nie mels) |
| Keyword spotting | 40 MFCCs dla malutkich urządzeń |

Zasada kciukowa: **jeśli nie pracujesz nad muzyką, zacznij od 80 log-mels.** Ciężar dowodu spoczywa na każdym odstępstwie.

## Pułapki, które wciąż trafiają do produkcji w 2026

- **Niezgodność liczby mel.** Trening z 80 mels, inferencja z 128 mels — cicha porażka. Loguj kształt cechy na obu końcach.
- **Niezgodność sample-rate wcześniej w pipeline.** Mels obliczone przy 22.05 kHz wyglądają inaczej niż przy 16 kHz. Napraw SR *przed* featurizacją.
- **dB vs log.** Whisper oczekuje log-mel, nie dB-mel. Niektóre HF pipeline autodetekują; twój custom code nie.
- **Dryft normalizacji.** Normalizacja per-utterance podczas treningu, globalna normalizacja podczas inferencji. Produkcyjny bug, który podwaja WER.
- **Przeciek z paddingu.** Zero-padding końca klipu produkuje płaskie widmo w końcowych ramkach. Paddinguj symetrycznie lub replikuj.

## Wyślij to

Zapisz jako `outputs/skill-feature-extractor.md`. Skill wybiera typ cechy, liczbę mel, frame/hop i normalizację dla danego celu modelu.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Synthesizuje chirp (częstotliwość przemiata 200 → 4000 Hz) i drukuje argmax mel bin dla każdej ramki. Zapisz (opcjonalnie), potwierdź, że pasuje do przemiatania.

2. **Średnie.** Uruchom ponownie z `n_mels` w `{40, 80, 128}` i `frame_len` w `{200, 400, 800}`. Zmierz ostra szerokość pasma szczytu wzdłuż osi czasu. Która kombinacja najlepiej rozdziela chirp?

3. **Trudne.** Zaimplementuj `power_to_db` i porównaj dokładność ASR tiny CNN classifiera na AudioMNIST używając (a) surowego log-mel, (b) dB-mel z `ref=max`, (c) MFCC-13 + delta + delta-delta. Zgłoś top-1 accuracy.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Frame | Plaster | Kawałek waveformu 25 ms wrzucany do jednego FFT. |
| Hop | Stride | Próbki między kolejnymi ramkami; 10 ms to domyślne ASR. |
| Window | Rzecz Hann/Hamming | Mnożnik point-wise, który zwęża krawędzie ramki do zera. |
| STFT | Generator spektrogramu | Ramkowany + okienkowany FFT; daje macierz czas × częstotliwość. |
| Mel | Zakrzywiona częstotliwość | Skala log-percepcyjna; `m = 2595·log10(1 + f/700)`. |
| Filterbank | Macierz | Trójkątne filtry które projektują STFT na mel bin. |
| Log-mel | Wejście Whispera | `log(mel_spec + eps)`; standaryzowane w 2026. |
| MFCC | Stara szkoła cech | DCT log-mel; 13 coeffów, zdekorelowane. |

## Dalsze czytanie

- Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition.
- Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch.
- OpenAI — Whisper source, log_mel_spectrogram.
- librosa feature extraction docs.
- NVIDIA NeMo — audio preprocessing.