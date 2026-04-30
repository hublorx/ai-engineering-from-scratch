# Podstawy Audio — Przebiegi, Próbkowanie, Transformata Fouriera

> Przebiegi to surowy sygnał. Spektrogramy to reprezentacja. Cechy Mel to forma przyjazna dla ML. Każdy nowoczesny potok ASR i TTS przechodzi przez tę drabinę, a pierwszym szczeblem jest zrozumienie próbkowania i transformaty Fouriera.

**Typ:** Poznawcze
**Języki:** Python
**Wymagania wstępne:** Faza 1 · 06 (Wektory i Macierze), Faza 1 · 14 (Rozkłady Prawdopodobieństwa)
**Szacowany czas:** ~45 minut

## Problem

Mikrofon wytwarza sygnał ciśnienie-czas. Twoja sieć neuronowa konsumuje tensory. Między nimi znajduje się stos konwencji, które przy naruszeniu produkują ciche błędy: model uczy się poprawnie, ale WER podwaja się, albo TTS wysyła syk, albo system klonowania głosu zapamiętuje mikrofon zamiast mówcy.

Każdy błąd w systemach mowy prowadzi do jednego z trzech pytań:

1. Przy jakiej częstotliwości próbkowania dane zostały nagrane i czego model oczekuje?
2. Czy sygnał jest zaliasowany?
3. Czy operujesz na surowych próbkach czy na reprezentacji częstotliwościowej?

Jeśli odpowiesz na nie poprawnie, reszta Fazy 6 staje się wykonalna. Jeśli popełnisz błąd, nawet Whisper-Large-v4 produkuje śmieci.

## Koncepcja

![Przebieg, próbkowanie, DFT i koszyki częstotliwości wizualizowane](../assets/audio-fundamentals.svg)

**Przebieg (Waveform).** Jednowymiarowa tablica floatów w `[-1.0, 1.0]`. Indeksowana numerem próbki. Aby przekonwertować na sekundy, podziel przez częstotliwość próbkowania: `t = n / sr`. 10-sekundowy klip przy 16 kHz to tablica 160 000 floatów.

**Częstotliwość próbkowania (sr).** Ile próbek na sekundę. Typowe częstotliwości w 2026:

| Częstotliwość | Zastosowanie |
|------|-----|
| 8 kHz | Telefonia, legacy VOIP. Nyquist przy 4 kHz zabija spółgłoski. Unikaj dla ASR. |
| 16 kHz | Standard ASR. Whisper, Parakeet, SeamlessM4T v2 wszystkie konsumują 16 kHz. |
| 22.05 kHz | Trenowanie vocodera TTS dla starszych modeli. |
| 24 kHz | Nowoczesne TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | Audio CD, muzyka. |
| 48 kHz | Film, profesjonalne audio, wysokiej wierności TTS (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** Częstotliwość próbkowania `sr` może jednoznacznie reprezentować częstotliwości do `sr/2`. Granica `sr/2` to *częstotliwość Nyquista*. Energia powyżej Nyquista jest *aliasowana* — składana w dół do niższych częstotliwości — i psuje sygnał. Zawsze filtruj dolnoprzepustowo przed downsamplingiem.

**Głębia bitowa.** 16-bitowy PCM (signed int16, zakres ±32 767) to uniwersalny format wymiany. 24-bitowy dla muzyki, 32-bitowy float dla wewnętrznego DSP. Biblioteki takie jak `soundfile` czytają int16, ale eksponują tablice float32 w `[-1, 1]`.

**Transformata Fouriera.** Każdy sygnał skończony jest sumą sinusoid przy różnych częstotliwościach. Dyskretna Transformata Fouriera (DFT) oblicza, dla `N` próbek, `N` współczynników zespolonych — jeden na koszyk częstotliwości. `koszyk k` odwzorowuje na częstotliwość `k · sr / N` Hz. Magnituda to amplituda przy tej częstotliwości, kąt to faza.

**FFT.** Szybka Transformata Fouriera: algorytm `O(N log N)` dla DFT gdy `N` jest potęgą dwóch. Każda biblioteka audio używa FFT pod spodem. FFT o 1024 próbkach przy 16 kHz daje 512 użytecznych koszyków częstotliwości obejmujących 0–8 kHz przy rozdzielczości 15.6 Hz.

**Ramkowanie + okno.** Nie robimy FFT na całym klipie. Dzielimy go na zachodzące na siebie *ramki* (typowo 25 ms z przeskokiem 10 ms), mnożymy każdą ramkę przez funkcję okna (Hann, Hamming) aby zlikwidować nieciągłości na krawędziach, a następnie FFT każdą ramkę. To jest Krótkoczasowa Transformata Fouriera (STFT). Lekcja 02 wychodzi od tego punktu.

## Zbuduj To

### Krok 1: wczytaj klip i narysuj przebieg

`code/main.py` używa tylko modułu stdlib `wave` aby demo nie miało zależności. W produkcji będziesz używać `soundfile` lub `torchaudio.load` (obie zwracają krotki `(waveform, sr)`):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Krok 2: zsyntezuj falę sinusoidalną od podstaw

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

Fala sinusoidalna 440 Hz (concert A) przy 16 kHz przez 1 sekundę to 16 000 floatów. Zapisz za pomocą `wave.open(..., "wb")` używając kodowania 16-bitowego PCM.

### Krok 3: oblicz DFT ręcznie

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — w porządku dla `N=256` aby potwierdzić poprawność, bezużyteczne dla prawdziwego audio. Prawdziwy kod wywołuje `numpy.fft.rfft` lub `torch.fft.rfft`.

### Krok 4: znajdź dominującą częstotliwość

Indeks szczytowy magnitudy `k_star` odwzorowuje na częstotliwość `k_star * sr / N`. Uruchomienie tego na fali sinusoidalnej 440 Hz powinno zwrócić szczyt przy koszyku `440 * N / sr`.

### Krok 5: zademonstruj aliasing

Próbkuj sinusoidę 7 kHz przy 10 kHz (Nyquist = 5 kHz). Ton 7 kHz jest powyżej Nyquista i składa się do `10 − 7 = 3 kHz`. Szczyt FFT pojawia się przy 3 kHz. To klasyczny demo aliasingu i powód, dla którego każdy DAC/ADC jest dostarczany z filtrem dolnoprzepustowym typu brick-wall.

## Użyj To

Stos, który faktycznie wyślesz w 2026:

| Zadanie | Biblioteka | Dlaczego |
|------|---------|-----|
| Odczyt/zapis WAV/FLAC/OGG | `soundfile` (wrapper libsndfile) | Najszybsza, stabilna, zwraca float32. |
| Resampling | `torchaudio.transforms.Resample` lub `librosa.resample` | Wbudowany poprawny anty-aliasing. |
| STFT / Mel | `torchaudio` lub `librosa` | Przyjazne dla GPU; ekosystem PyTorch. |
| Streaming w czasie rzeczywistym | `sounddevice` lub `pyaudio` | Wieloplatformowe wiązania PortAudio. |
| Inspect plik | `ffprobe` lub `soxi` | CLI, szybkie, raportuje sr/kanały/kodek. |

Reguła decyzyjna: **dopasuj częstotliwość próbkowania zanim dopasujesz cokolwiek innego**. Whisper oczekuje 16 kHz mono float32. Przekaż mu 44.1 kHz stereo i dostaniesz śmieci, które wyglądają jak błąd modelu.

## Wyślij To

Zapisz jako `outputs/skill-audio-loader.md`. Skill pomaga sprawdzić, czy wejście audio pasuje do oczekiwań modelu downstream i poprawnie resampluje gdy nie pasuje.

## Ćwiczenia

1. **Łatwe.** Zsyntezuj 1-sekundową mieszankę 220 Hz + 440 Hz + 880 Hz przy 16 kHz. Uruchom DFT. Potwierdź trzy szczyty przy oczekiwanych koszykach.
2. **Średnie.** Nagraj 3-sekundowy WAV swojego głosu przy 48 kHz. Downsampluj do 16 kHz używając `torchaudio.transforms.Resample` (z anty-aliasing), a potem do 16 kHz używając naiwnej decymacji (co trzecią próbkę). FFT oba. Gdzie pojawia się aliasing?
3. **Trudne.** Zbuduj STFT od podstaw używając tylko `math` i DFT z Kroku 3. Rozmiar ramki 400, hop 160, okno Hann. Narysuj magnitudy z `matplotlib.pyplot.imshow`. To jest spektrogram z Lekcji 02.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Częstotliwość próbkowania | Ile próbek na sekundę | Częstotliwość w Hz przy której ADC mierzy sygnał. |
| Nyquist | Maksymalna częstotliwość którą możesz reprezentować | `sr/2`; energia powyżej niej aliasuje w dół. |
| Głębia bitowa | Rozdzielczość każdej próbki | `int16` = 65 536 poziomów; `float32` = 24-bitowa precyzja w `[-1, 1]`. |
| DFT | Transformata Fouriera dla sekwencji | `N` próbek → `N` zespolonych współczynników częstotliwości. |
| FFT | Szybka DFT | Algorytm `O(N log N)` wymagający `N` = potęgi dwóch. |
| Koszyk (Bin) | Kolumna częstotliwości | `k · sr / N` Hz; rozdzielczość = `sr / N`. |
| STFT | Pod spodem spektrogramu | Ramkowane + oknowane FFT w czasie. |
| Aliasing | Dziwne duchy częstotliwości | Energia powyżej Nyquista odzwierciedlająca w dół do niższych koszyków. |