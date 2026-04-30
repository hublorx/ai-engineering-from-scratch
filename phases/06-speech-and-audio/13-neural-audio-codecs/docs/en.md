# Neuralne kodeki audio — EnCodec, SNAC, Mimi, DAC i podział semantyczno-akustyczny

> Audio generation w 2026 to prawie w całości tokeny. EnCodec, SNAC, Mimi i DAC zamieniają ciągłe przebiegi falowe w dyskretne sekwencje, które transformer może przewidywać. Podział semantyczno-akustyczny — pierwszy codebook jako semantyczny, reszta jako akustyczny — to najważniejsza zmiana architektoniczna od czasu Transformera dla audio.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Phase 6 · 02 (Spektrogramy), Phase 10 · 11 (Kwantyzacja), Phase 5 · 19 (Subword Tokenization)
**Szacowany czas:** ~60 minut

## Problem

Modele językowe operują na dyskretnych tokenach. Audio jest ciągłe. Jeśli chcesz model typu LLM dla mowy / muzyki — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — najpierw potrzebujesz **neuronalnego kodeka audio**: nauczonego enkodera, który dyskretyzuje audio do małego słownictwa tokenów, i pasującego dekodera, który rekonstruuje przebieg falowy.

Wykrystalizowały się dwie rodziny:

1. **Kodeki oparte na rekonstrukcji** — EnCodec, DAC. Optymalizują percepcyjną jakość audio. Tokeny są „akustyczne" — przechwytują wszystko, w tym tożsamość mówcy, barwę dźwięku, szum tła.
2. **Kodeki oparte na semantyce** — Mimi (Kyutai), SpeechTokenizer. Zmuszają pierwszy codebook do kodowania treści lingwistycznej/fonetycznej (często przez destylację z WavLM). Kolejne codebooki są rezyduami akustycznymi.

Odkrycie lat 2024-2026: **czysto rekonstrukcyjny kodek daje rozmyte audio przy próbie generowania z tekstu.** LLM nad tokenami kodeka musi nauczyć się zarówno struktury językowej JAK I struktury akustycznej w tym samym codebooku, co nie skaluje się. Rozdzielenie ich — semantyczny codebook 0, akustyczne codebooki 1-N — to dlatego Moshi i Sesame CSM działają.

## Koncepcja

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### Główny trick: Residual Vector Quantization (RVQ)

Zamiast jednego dużego codebooku (który potrzebowałby milionów kodów dla dobrej jakości), wszystkie nowoczesne kodeki audio używają **RVQ**: kaskady małych codebooków. Pierwszy codebook kwantuje wyjście enkodera, drugi kwantuje rezyduum; i tak dalej. Każdy codebook ma 1024 kody. 8 codebooków = efektywne słownictwo 1024^8 = 10^24.

W czasie inferencji dekoder sumuje wszystkie wybrane kody na ramkę, aby zrekonstruować.

### Cztery kodeki, które mają znaczenie w 2026

**EnCodec (Meta, 2022).** Baseline. Enkoder-dekoder nad przebiegiem falowym, bottleneck RVQ. 24 kHz, 32 codebooki możliwe, domyślnie 4 codebooki @ 1.5 kbps. Używa architektury `1D conv + transformer + 1D conv`. Używany przez MusicGen.

**DAC (Descript, 2023).** RVQ z znormalizowanymi L2 codebookami, okresowymi funkcjami aktywacji, ulepszonymi stratami. Najwyższa wierność rekonstrukcji spośród otwartych kodeków — czasem nie do odróżnienia od oryginalnej mowy przy 12 codebookach. 44.1 kHz full-band.

**SNAC (Hubert Siuzdak, 2024).** Wieloskalowy RVQ — coarse codebooki operują z niższą częstotliwością klatek niż fine. Efektywnie modeluje audio hierarchicznie: coarse „szkic" przy ~12 Hz plus detale przy 50 Hz. Używany przez Orpheus-3B, ponieważ struktura hierarchiczna dobrze mapuje na LM-based generation.

**Mimi (Kyutai, 2024).** Game-changer 2026. Częstotliwość klatek 12.5 Hz (ekstremalnie niska), 8 codebooków @ 4.4 kbps. Codebook 0 jest **destylowany z WavLM** — trenowany do przewidywania cech speech-content z WavLM. Codebooki 1-7 to rezydua akustyczne. Ten podział napędza Moshi (Lesson 15) i Sesame CSM.

### Częstotliwości klatek mają znaczenie dla language modeling

Niższa częstotliwość klatek = krótsza sekwencja = szybszy LM.

| Kodek | Częstotliwość klatek | 1 s = N klatek | Dobry dla |
|-------|---------------------|----------------|-----------|
| EnCodec-24k | 75 Hz | 75 | muzyka, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

Przy 12.5 Hz, 10-sekundowa wypowiedź to tylko 125 klatek kodeka — transformer może je łatwo przewidywać.

### Tokeny semantyczne vs akustyczne

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Token semantyczny (codebook 0 w Mimi).** Koduje, co zostało powiedziane — fonemy, słowa, treść. Destylowany z WavLM przez pomocniczą stratę predykcyjną.
- **Tokeny akustyczne (codebooki 1-7).** Kodują barwę głosu, tożsamość mówcy, prozodię, szum tła, drobne detale.

AR LM najpierw przewiduje token semantyczny (warunkowany tekstem), potem przewiduje tokeny akustyczne (warunkowane semantycznym + referencją mówcy). Ta faktoryzacja to dlaczego nowoczesny TTS może zero-shot-clone voices: model semantyczny obsługuje treść, model akustyczny obsługuje barwę.

### Jakość rekonstrukcji 2026 (bits per sec, niższy bitrate jest lepszy)

| Kodek | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Tradycyjne kodeki jak Opus nadal wygrywają per bit na percepcyjnej jakości. Neuronalne kodeki wygrywają na **dyskretnych tokenach** (których Opus nie produkuje) i **jakości generatywnego modelu** (co LM może zrobić z tymi tokenami).

## Zbuduj to

### Krok 1: enkoding z EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` przy 6 kbps. Każdy code to 0-1023 (10-bit).

### Krok 2: dekodowanie i pomiar rekonstrukcji

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Krok 3: podział semantyczno-akustyczny (styl Mimi)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantyczny codebook 0 jest wyrównany z WavLM. Możesz trenować text-to-semantic transformer — znacznie mniejsze słownictwo niż idąc direct-to-audio. Potem osobny acoustic-to-waveform decoder warunkuje się na referencję mówcy.

### Krok 4: dlaczego AR LM nad tokenami kodeka działa

Dla 10 s clipu mowy przy Mimi 12.5 Hz × 8 codebooków:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 tokenów to trywialny kontekst dla transformera. Transformer 256M-parametrów może wygenerować 10 sekund mowy w milisekundach na nowoczesnym GPU.

## Użyj tego

Mapuj problem → kodek:

| Zadanie | Kodek |
|---------|-------|
| General music generation | EnCodec-24k |
| Najwyższa wierność rekonstrukcji | DAC-44.1k |
| AR LM over speech (TTS) | SNAC lub Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| Biblioteka efektów dźwiękowych z tekstem | EnCodec + T5 condition |
| Fine-grained audio editing | DAC + inpainting |

Zasada kciuka: **jeśli budujesz generatywny model, zacznij od Mimi lub SNAC. Jeśli budujesz pipeline kompresji, użyj Opus.**

## Pułapki

- **Za dużo codebooków.** Dodawanie codebooków zwiększa wierność liniowo, ale sekwencja LM też liniowo rośnie. Zatrzymaj się na 8-12.
- **Niezgodność częstotliwości klatek.** Trenowanie LM na 12.5 Hz Mimi, potem fine-tuning na 50 Hz EnCodec fails silently.
- **Założenie, że wszystkie codebooki są równe.** W Mimi, codebook 0 niesie treść; utrata go niszczy zrozumiałość. Utrata codebooka 7 jest prawie niezauważalna.
- **Używanie jakości rekonstrukcji jako jedynej metryki.** Kodek może mieć świetną rekonstrukcję, ale być bezużyteczny dla LM-based generation, jeśli struktura semantyczna jest zła.

## Wyślij to

Zapisz jako `outputs/skill-codec-picker.md`. Wybierz kodek dla danego zadania generatywnego lub kompresji.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Implementuje toy scalar + residual quantizer i mierzy błąd rekonstrukcji przy dodawaniu codebooków.
2. **Średnie.** Zainstaluj `encodec` i porównaj 1, 4, 8, 32 codebooki na held-out speech clip. Zrób wykres PESQ lub MSE vs bitrate.
3. **Trudne.** Załaduj Mimi. Enkoding clipu. Zamień codebook 0 na losowe liczby całkowite; dekoduj. Potem zamień codebook 7 podobnie. Porównaj dwa zepsucia — codebook 0 powinien zniszczyć zrozumiałość; codebook 7 powinien prawie nic nie zmienić.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|-------|-----------------|--------------------------|
| RVQ | Residual quantization | Kaskada małych codebooków; każdy kwantuje poprzednie rezyduum. |
| Frame rate | Codec speed | Ile token-frame'ów na sekundę. Niższy = szybszy LM. |
| Semantic codebook | Codebook 0 (Mimi) | Codebook destylowany z SSL features; koduje treść. |
| Acoustic codebooks | Wszystko inne | Barwa głosu, prozodia, szum, drobne detale. |
| PESQ / ViSQOL | Perceptual quality | Obiektywne metryki korelujące z MOS. |
| EnCodec | Meta codec | RVQ baseline; używany przez MusicGen. |
| Mimi | Kyutai codec | 12.5 Hz frame rate; podział semantyczno-akustyczny; napędza Moshi. |

## Dalsze czytanie

- Défossez et al. (2023). EnCodec — RVQ baseline.
- Kumar et al. (2023). Descript Audio Codec (DAC) — highest-fidelity open.
- Siuzdak (2024). SNAC — multi-scale RVQ.
- Kyutai (2024). Mimi codec — semantic-acoustic split, WavLM distillation.
- Borsos et al. (2023). AudioLM — two-stage semantic/acoustic paradigm.
- Zeghidour et al. (2021). SoundStream — oryginalny streamable RVQ codec.