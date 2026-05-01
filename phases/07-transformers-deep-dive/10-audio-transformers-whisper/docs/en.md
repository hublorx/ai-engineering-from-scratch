# Audio Transformers — Architektura Whisper

> Audio to obraz częstotliwości w czasie. Whisper to ViT który je mel spectrograms i mówi.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 05 (Full Transformer), Faza 7 · 08 (Encoder-Decoder), Faza 7 · 09 (ViT)
**Czas:** ~45 minut

## Problem

Przed Whisper (OpenAI, Radford et al. 2022), state-of-the-art automatic speech recognition (ASR) oznaczało wav2vec 2.0 i HuBERT — self-supervised feature extractors plus fine-tuned head. Wysoka jakość, drogie data pipelines, domain-brittle. Multilingual speech recognition potrzebował separate models per language family.

Whisper postawił na trzy rzeczy:

1. **Trenuj na wszystkim.** 680,000 godzin weakly-labeled audio scraped z internetu w 97 językach. Żaden clean academic corpus. Żadnych phoneme labels.
2. **Multi-task single model.** Jeden decoder trenowany jointly na transcription, translation, voice activity detection, language ID i timestamping przez task tokens.
3. **Standard encoder-decoder transformer.** Encoder konsumuje log-mel spectrograms. Decoder produkuje text tokens autoregresywnie. Żaden vocoder, żaden CTC, żaden HMM.

Rezultat: Whisper large-v3 jest robust przez accents, noise i języki które mają zero clean labeled data. To jest domyślny speech front-end dla każdego open-source voice assistant i większości komercyjnych w 2026.

## Koncepcja

![Pipeline Whisper: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Krok 1 — resample + window

Audio przy 16 kHz. Clip/pad do 30 sekund. Oblicz log-mel spectrogram: 80 mel bins, 10 ms stride → ~3,000 frames × 80 features. To jest "obraz input" który Whisper widzi.

### Krok 2 — convolutional stem

Dwie Conv1D warstwy z kernel 3 i stride 2 redukują 3,000 frames do 1,500. Połowa długości sekwencji bez dodawania dużo parametrów.

### Krok 3 — encoder

24-warstwowy (dla large) transformer encoder over 1,500 timesteps. Sinusoidal positional encoding, self-attention, GELU FFN. Produkuje 1,500 × 1,280 hidden states.

### Krok 4 — decoder

24-warstwowy transformer decoder. Autoregresywnie produkuje tokens z BPE vocabulary który jest superset GPT-2's z kilkoma audio-specific special tokens.

### Krok 5 — task tokens

Decoder prompt zaczyna się od control tokens które mówią modelowi co robić:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

lub

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

Model był trenowany na tej konwencji. Kontrolujesz task przez prefix. 2026 equivalent instruction-tuning, ale applied to speech.

### Krok 6 — output

Beam search (width 5) z log-prob threshold. Timestamps są predicted co 0.02 sekundy audio gdy `<|notimestamps|>` token jest absent.

### Rozmiary Whisper

| Model | Params | Warstwy | d_model | Heads | VRAM (fp16) |
|-------|--------|---------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo (2024) zmniejszył decoder z 32 warstw do 4. 8× szybsze dekodowanie z <1 WER point regression. Ten unlock dekodowania to dlaczego Whisper-turbo jest domyślny dla real-time voice agents w 2026.

### Czego Whisper nie robi

- No diarization (kto mówi). Sparuj z pyannote.
- No real-time streaming natively — 30-sekundowe okno jest fixed. Nowoczesne wrappers (`faster-whisper`, `WhisperX`) bolt on streaming przez VAD + overlap.
- No long-form context beyond 30 s bez external chunking. Działa dobrze w praktyce bo ludzka mowa rzadko potrzebuje long-range context dla transcription.

### Krajobraz 2026

| Zadanie | Model | Uwagi |
|---------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine jest 4× szybszy na edge |
| Multilingual ASR | Whisper-large-v3 | 97 języków |
| Streaming ASR | faster-whisper + VAD | 150 ms latency targets achievable |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder pattern, ale Whisper-shaped |
| Audio + language | AudioLM, SeamlessM4T | Text tokens + audio tokens w jednym transformerze |

## Zbuduj to

Zobacz `code/main.py`. Nie trenujemy Whisper — budujemy log-mel spectrogram pipeline + task-token prompt formatter. To są części których faktycznie dotykasz w produkcji.

### Krok 1: synthesize audio

Generuj 1-sekundową sinusoidę przy 440 Hz sampled at 16 kHz. 16,000 samples.

### Krok 2: log-mel spectrogram (uproszczony)

Pełny mel spectrogram potrzebuje FFT. Robimy uproszczoną wersję framing + per-frame energy która pokazuje pipeline bez wymagania `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms, hop = 10 ms. Pasuje do Whisper's windowing. Per-frame energy stands in dla mel bins for pedagogy.

### Krok 3: pad do 30 s

Whisper zawsze przetwarza 30-sekundowe chunks. Pad (lub clip) spectrogram do 3,000 frames.

### Krok 4: zbuduj prompt tokens

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

To jest cała powierzchnia kontroli task. 4-token prefix.

## Użyj tego

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Szybciej, OpenAI-compatible:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**Kiedy wybrać Whisper w 2026:**

- Multilingual ASR z jednym modelem.
- Robust transcription noisy, diverse audio.
- Research / prototype ASR — fastest starting point.

**Kiedy wybrać coś innego:**

- Ultra-low latency streaming on edge — Moonshine beats Whisper at matched quality.
- Real-time conversational AI needing <200 ms — dedicated streaming ASR.
- Speaker diarization — Whisper tego nie robi; bolt on pyannote.

## Wyślij to

Zobacz `outputs/skill-asr-configurator.md`. Skill wybiera ASR model, decoding parameters i preprocessing pipeline dla nowej speech application.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Potwierdź że frame count dla 1-sekundowego sygnału przy 16 kHz z 10 ms hop to ~100 frames. Dla 30 sekund: ~3,000 frames.
2. **Średnie.** Zbuduj full log-mel spectrogram używając `numpy.fft`. Zweryfikuj że 80 mel bins match `librosa.feature.melspectrogram(n_mels=80)` within numerical error.
3. **Trudne.** Zaimplementuj streaming inference: chunk audio na 10 s windows z 2 s overlap, uruchom Whisper na każdym chunk, merge transcripts. Zmierz word-error rate vs single-pass na 5-minutowym podcaście.

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|--------------------------|
| Mel spectrogram | "Obraz audio" | 2D representation: frequency bins na jednej osi, time frames na drugiej; log-scaled energy per cell. |
| Log-mel | "Co Whisper widzi" | Mel spectrogram passed przez log; approximates human perception of loudness. |
| Frame | "Jeden time slice" | 25 ms window of samples; overlapping at 10 ms stride. |
| Task token | "Prompt prefix dla speech" | Specjalne tokens jak `<|transcribe|>` / `<|translate|>` w decoder prompt. |
| Voice activity detection (VAD) | "Znajdź mowę" | Gate który removes silence przed ASR; cuts cost massively. |
| CTC | "Connectionist Temporal Classification" | Classic ASR loss dla alignment-free training; Whisper NIE używa tego. |
| Whisper-turbo | "Mały decoder, pełny encoder" | large-v3 encoder + 4-layer decoder; 8× szybsze dekodowanie. |
| Faster-whisper | "Production wrapper" | CTranslate2 reimplementation; int8 quantization; 4× szybszy niż OpenAI's reference. |

## Dalsze Czytanie

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — paper Whisper.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — reference code + model weights. Przeczytaj `whisper/model.py` żeby zobaczyć Conv1D stem + encoder + decoder top-to-bottom w ~400 linijkach.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — logika beam-search + task-token opisana w Krokach 5–6 jest tutaj; 500 linijek, fully readable.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — precursor; still SOTA features w niektórych settings.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — production wrapper, 4× szybszy niż reference.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 edge-friendly ASR, Whisper-shaped ale mniejszy.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — kanoniczny przepis fine-tuning włączając mel spectrogram preprocessor i token-timestamp handling.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — full implementation (encoder, decoder, cross-attention, generation) która mirroruje diagram architektury z lekcji.