# Text-to-Speech (TTS) — od Tacotrona do F5 i Kokoro

> ASR odwraca mowę na tekst; TTS odwraca tekst na mowę. Stack 2026 składa się z trzech części: tekst → tokeny, tokeny → mel, mel → waveform. Każda część ma domyślny model, który mieści się na laptopie.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 6 · 02 (Spektrogramy i Mel), Faza 5 · 09 (Seq2Seq), Faza 7 · 05 (Pełny Transformer)
**Szacowany czas:** ~75 minut

## Problem

Masz string: "Please remind me to water the plants at 6 pm." Potrzebujesz 3-sekundowego klipu audio, który brzmi naturalnie, ma poprawną prozodię (pauzy, akcent), wymawia "plants" z właściwą samogłoską i działa w mniej niż 300 ms na CPU dla live voice assistant. Potrzebujesz też zmieniać głosy, obsługiwać code-switched input ("remind me at 6 pm, daijoubu?") i nie robić wstydu przy nazwiskach.

Nowoczesne pipeline'y TTS wyglądają tak:

1. **Text frontend.** Normalizuj tekst (daty, liczby, emaile), konwertuj na fonemy lub subword tokeny, przewiduj cechy prozodii.
2. **Model akustyczny.** Tekst → mel spectrogram. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Vocoder.** Mel → waveform. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), neural codec vocodery w 2024+.

W 2026 roku podział acoustic + vocoder się zaciera wraz z end-to-end modelami diffusion i flow-matching. Ale mental model trzech części nadal obowiązuje przy debugowaniu.

## Koncepcja

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder emituje mel frames. Wolny (AR), niestabilny przy długim tekście. Wciąż cytowany jako baseline.

**FastSpeech 2 (2020).** Non-autoregressive. Duration predictor输出uje, ile mel frames przypada na każdy fonem. 1-pass, 10× szybszy niż Tacotron. Traci trochę naturalności (monotonic alignment), ale działa wszędzie.

**VITS (2021).** Wspólnie trenuje encoder + flow-based duration + HiFi-GAN vocoder end-to-end z wnioskowaniem wariacyjnym. Wysoka jakość, jeden model. Dominujący open-source TTS 2022–2024. Warianty: YourTTS (multi-speaker zero-shot), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Diffusion transformer over flow matching. Naturalna prozodia, zero-shot voice cloning z 5 sekund reference audio. Szczyt open-source TTS leaderboards w 2026. 335M params.

**Kokoro (2024).** Mały (82M), działa na CPU, najlepszy angielski TTS klasy real-time. Zamknięty vocabulary English-only, apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** Komercyna state of the art. ElevenLabs v2.5 emotion tags ("[whispered]", "[laughing]") i character voices dominują produkcję audiobooków w 2026.

### Ewolucja vocodera

| Era | Vocoder | Latency | Jakość |
|-----|---------|---------|--------|
| 2016 | WaveNet | tylko offline | SOTA przy wydaniu |
| 2018 | WaveRNN | ~realtime | dobra |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizuje across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | zintegrowany z AR models | discrete tokens, bit-efficient |

Do 2026 większość modeli "TTS" to end-to-end z tekstu na waveform; mel spectrogram to internal representation.

### Ewaluacja

- **MOS (Mean Opinion Score).** Skala 1–5, crowd-sourced. Wciąż gold standard; boleśnie wolny.
- **CMOS (Comparative MOS).** Preferencja A-vs-B. Zawężone confidence intervals per annotation.
- **UTMOS, DNSMOS.** Reference-free neural MOS predictors. Używane w leaderboards.
- **CER (Character Error Rate) via ASR.** Przepuść TTS output przez Whisper, oblicz CER względem input text. Proxy for intelligibility.
- **SECS (Speaker Embedding Cosine Similarity).** Voice-clone quality.

Wyniki 2026 na LibriTTS test-clean:

| Model | UTMOS | CER (via Whisper) | Rozmiar |
|-------|-------|-------------------|---------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## Budowanie

### Krok 1: fonemizacja inputu

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Fonemy to uniwersalny pomost. Unikaj podawania surowego tekstu do wszystkiego poniżej jakości VITS.

### Krok 2: uruchom Kokoro (2026 CPU default)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Działa offline, single file, 82M params.

### Krok 3: uruchom F5-TTS z voice cloning

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Przekaż 5-sekundowy klip referencyjny + jego transkrypcję; F5 klonuje prozodię i timbre.

### Krok 4: HiFi-GAN vocoder od zera

Zbyt duży, żeby zmieścić się w tutorial script, ale kształt jest taki:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Trening: adversarial (discriminator on short windows) + mel-spectrogram reconstruction loss + feature-matching loss. Skomodytyzowane — używaj pretrained checkpoints z `hifi-gan` repo lub nvidia-NeMo.

### Krok 5: pełny pipeline (pseudokod)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Użycie

Stack 2026:

| Sytuacja | Wybierz |
|---------|--------|
| Real-time English voice assistant | Kokoro (CPU) lub XTTS v2 (GPU) |
| Voice cloning z 5 s reference | F5-TTS |
| Komercynne character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 lub XTTS v2 + fine-tune |
| Low-resource language | Trenuj VITS na 5–20 h target-lang data |
| Expressive / emotion tags | ElevenLabs v2.5 lub StyleTTS 2 fine-tune |

Open-source leader w 2026: **F5-TTS dla jakości, Kokoro dla efektywności**. Nie sięgaj po Tacotrona, chyba że jesteś historykiem.

## Pułapki

- **Brak text normalizera.** "Dr. Smith" czyta się jako "Doctor" czy "Drive"? "2026" jako "twenty twenty six" czy "two zero two six"? Normalizuj PRZED phonemizer.
- **OOV proper nouns.** "Ghumare" → "ghyu-mair"? Dostarcz fallback grapheme-to-phoneme model dla nieznanych tokenów.
- **Clipping.** Vocoder output rzadko clipuje, ale mel scaling mismatch przy inference może przekroczyć ±1.0. Zawsze `np.clip(wav, -1, 1)`.
- **Sample-rate mismatch.** Kokoro outputuje 24 kHz; twój downstream pipeline oczekuje 16 kHz → resampluj albo dostaniesz aliasing.

## Dostarcz to

Zapisz jako `outputs/skill-tts-designer.md`. Zaprojektuj pipeline TTS dla danego głosu, latency i target language.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Buduje phoneme dictionary z toy vocab, szacuje duration per fonem i drukuje fake "mel" schedule.
2. **Średnie.** Zainstaluj Kokoro, zsyntezuj to samo zdanie w voice `af_bella` i `am_adam`. Porównaj audio durations i subiektywną jakość.
3. **Trudne.** Nagraj 5-sekundowy klip referencyjny siebie. Użyj F5-TTS do sklonowania. Zgłoś SECS między reference a cloned output.

## Kluczowe terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Phoneme | Jednostka dźwięku | Abstrakcyjna klasa dźwięku; 39 w języku angielskim (ARPABet). |
| Duration predictor | Jak długo trwa każdy fonem | Non-AR model output; integer frames per phoneme. |
| Vocoder | Mel → waveform | Neural net mapujący mel-spec na raw samples. |
| HiFi-GAN | Standardowy vocoder | GAN-based; dominujący 2020–2024. |
| MOS | Subiektywna jakość | Średni wynik opinii 1–5 od ludzkich oceniających. |
| SECS | Metryka voice-clone | Cosine similarity między target a output speaker embedding. |
| F5-TTS | 2024 open-source SOTA | Flow-matching diffusion; zero-shot cloning. |
| Kokoro | CPU English leader | 82M-param model, Apache 2.0. |

## Dalsze czytanie

- Shen i in. (2017). Tacotron 2 — baseline seq2seq.
- Kim, Kong, Son (2021). VITS — end-to-end flow-based.
- Chen i in. (2024). F5-TTS — current open-source SOTA.
- Kong, Kim, Bae (2020). HiFi-GAN — vocoder, który wciąż się używa w 2026.
- Kokoro-82M na HuggingFace — 2024 CPU-friendly English TTS.