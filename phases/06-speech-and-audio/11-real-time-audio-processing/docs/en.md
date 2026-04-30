# Przetwarzanie Audio w Czasie Rzeczywistym

> Batchowe pipeline'y przetwarzają plik. Pipeline'y czasu rzeczywistego przetwarzają kolejne 20 milisekund zanim nadejdą kolejne 20. Każda konwersacyjna AI, studio nadawcze i bot telefoniczny żyje i umiera przez ten budżet latency.

**Typ:** Budowa
**Języki:** Python, Rust
**Wymagania wstępne:** Faza 6 · 02 (Spektrogramy), Faza 6 · 04 (ASR), Faza 6 · 07 (TTS)
**Czas:** ~75 minut

## Problem

Chcesz asystenta głosowego, który sprawia wrażenie żywego. Latency ludzkiej konwersacyjnej zmiany tur wynosi ~230 ms (cisza-do-odpowiedzi). Cokolwiek powyżej 500 ms sprawia wrażenie robotycznego, powyżej 1500 ms sprawia wrażenie zepsutego. Budżet na pełną pętlę **usłysz → zrozum → odpowiedz → powiedz** w 2026 roku:

| Etap | Budżet |
|------|--------|
| Mikrofon → bufor | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (pierwszy token) | 100 ms |
| TTS (pierwszy chunk) | 100 ms |
| Render → głośnik | 20 ms |
| **Razem** | **~400 ms** |

Moshi (Kyutai, 2024) osiągnął 200 ms full-duplex. GPT-4o-realtime (2024) osiąga ~320 ms. Kaskadowe pipeline'y w 2022 roku dostarczały przy 2500 ms. 10× poprawa pochodzi od trzech technik: (1) streaming wszędzie, (2) asynchroniczne potokowanie z częściowymi wynikami, (3) przerywalna generacja.

## Koncepcja

![Potok audio streaming z ring buffer, bramką VAD, przerwaniem](../assets/real-time.svg)

**Ramka / chunk / okno.** Audio w czasie rzeczywistym płynie jako bloki o stałym rozmiarze. Typowy wybór: 20 ms (320 próbek przy 16 kHz). Wszystko downstream musi nadążać za tym rytmem.

**Ring buffer.** Bufor kołowy o stałym rozmiarze. Wątek producenta zapisuje nowe ramki, wątek konsumenta odczytuje. Zapobiega alokacjom na gorącej ścieżce. Rozmiar ≈ maksymalna-latency × sample-rate; ring 2-sekundowy przy 16 kHz = 32,000 próbek.

**VAD (Voice Activity Detection).** Bramkuje downstream gdy nikt nie mówi. Silero VAD 4.0 (2024) działa <1 ms na ramkę 30 ms na CPU. `webrtcvad` to starsza alternatywa.

**Streaming ASR.** Modele, które emitują częściowe transkrypcje w miarę napływu audio. Parakeet-CTC-0.6B w trybie streaming (NeMo, 2024) osiąga 2–5% WER przy 320 ms latency. Whisper-Streaming (Macháček et al., 2023) dzieli Whisper na near-streaming przy ~2 s latency.

**Przerwanie.** Gdy użytkownik mówi podczas gdy asystent mówi, musisz (a) wykryć barge-in, (b) zatrzymać TTS, (c) odrzucić pozostały wynik LLM. Wszystko w ciągu 100 ms, albo użytkownik postrzega głuchego asystenta.

**WebRTC Opus transport.** Ramki 20 ms, 48 kHz, adaptacyjny bitrate 8–128 kbps. Standard dla przeglądarek i mobile. LiveKit, Daily.co, Pion to stosy 2026 do budowania aplikacji głosowych.

**Jitter buffer.** Pakiety sieciowe docierają w złej kolejności / spóźnione. Jitter buffer przestawia i wygładza, za mały → słyszalne przerwy, za duży → latency. 60–80 ms typowo.

### Typowe pułapki

- **Thread contention.** Python GIL + ciężkie modele mogą zagłodzić wątek audio. Użyj biblioteki audio z C-callback (sounddevice, PortAudio) i trzymaj Pythona z dala od gorącej ścieżki.
- **Sample-rate conversion latency.** Resampling wewnątrz pipeline dodaje 5–20 ms. Albo resampluj od razu albo użyj resamplera zero-latency (PolyPhase, `soxr_hq`).
- **TTS priming.** Nawet szybki TTS jak Kokoro ma 100–200 ms rozgrzewki przy pierwszym żądaniu.-cache'uj model + rozgrzej go dummy run przed pierwszą prawdziwą turą.
- **Echo cancellation.** Bez AEC wyjście TTS ponownie wchodzi do mikrofonu i uruchamia ASR na głosie bota. WebRTC AEC3 to domyślny open-source.

## Zbuduj to

### Krok 1: ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Capacity określa max buffering latency. 32,000 próbek przy 16 kHz = 2 s.

### Krok 2: VAD gate

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Zamień na Silero VAD w produkcji:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Krok 3: streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Krok 4: interruption handler

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Opiera się na async I/O i cancellable TTS streaming. WebRTC peerconnection.stop() na audio track to kanoniczny sposób.

## Użyj tego

Stos 2026:

| Warstwa | Wybór |
|--------|-------|
| Transport | LiveKit (WebRTC) lub Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B lub Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro lub ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API lub Moshi |

## Pułapki

- **Buffering 500 ms żeby być bezpiecznym.** Buffer *to* twoja latency floor. Zmniejsz go.
- **Nie pinning threads.** Audio callback na wątku o niższym priorytecie niż UI = glitche pod obciążeniem.
- **TTS chunks za małe.** Sub-200 ms chunks sprawiają, że artefakty vocodera są słyszalne. 320 ms chunks to optimum.
- **Brak jitter buffer.** Realne sieci są jittery, bez wygładzania dostajesz trzaski.
- **Single-shot error handling.** Audio pipeline musi być crash-proof. Jednen wyjątek zabija sesję.

## Wyślij to

Zapisz jako `outputs/skill-realtime-designer.md`. Zaprojektuj pipeline audio czasu rzeczywistego z konkretnymi budżetami latency na etap.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Symuluje ring buffer + energy VAD, drukuje stage latencies dla fikcyjnego 10-sekundowego streamu.
2. **Średnie.** Używając `sounddevice`, zbuduj passthrough loop który przetwarza twój mikrofon w ramkach 20 ms i drukuje stan VAD dla każdej ramki.
3. **Trudne.** Zbuduj full duplex echo test z `aiortc`: przeglądarka → WebRTC → Python → WebRTC → przeglądarka. Zmierz glass-to-glass latency z impulsem 1 kHz.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Ring buffer | Kolejka kołowa | FIFO o stałym rozmiarze, lock-free (lub SPSC-locked) dla audio frames. |
| VAD | Brama ciszy | Model lub heurystyka oznaczająca speech vs non-speech. |
| Streaming ASR | Real-time STT | Emituje częściowy tekst w miarę napływu audio, ograniczony lookahead. |
| Jitter buffer | Wygładzacz sieciowy | Kolejka przestawiająca pakiety poza kolejnością, 60–80 ms typowo. |
| AEC | Echo cancellation | Odejmuje ścieżkę sprzężenia speaker-to-mic. |
| Barge-in | Przerwanie użytkownika | System wykrywa mowę użytkownika w połowie TTS, musi anulować odtwarzanie. |
| Full duplex | Jednocześnie w obie strony | Użytkownik i bot mogą mówić jednocześnie, Moshi jest full duplex. |

## Dalsze czytanie

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — chunked near-streaming Whisper.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex 200 ms latency.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — production audio agent orchestration.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — echo cancellation under open source.