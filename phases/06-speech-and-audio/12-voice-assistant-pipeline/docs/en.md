# Budowanie potoku asystenta głosowego — Fase 6 Projekt końcowy

> Wszystko z lekcji 01-11, zszyte razem. Zbuduj asystenta głosowego, który słucha, rozumuje i odpowiada. W 2026 roku to rozwiązany problem inżynieryjny, nie problem badawczy — ale szczegóły integracji decydują o tym, czy produkt trafia na rynek.

**Typ:** Projekt
**Języki:** Python
**Wymagania wstępne:** Faz 6 · 04, 05, 06, 07, 11; Faz 11 · 09 (Function Calling); Faz 14 · 01 (Agent Loop)
**Szacowany czas:** ~120 minut

## Problem

Zbuduj asystenta end-to-end:

1. Przechwytuje wejście z mikrofonu (16 kHz mono).
2. Wykrywa początek/koniec mowy użytkownika.
3. Transkrybuje strumieniowo.
4. Przekazuje transkrypcję do LLM, który może wywoływać narzędzia (timer, pogoda, kalendarz).
5. Strumieniuje tekst LLM do TTS.
6. Odtwarza audio z powrotem do użytkownika.
7. Zatrzymuje się, jeśli użytkownik przerwie w połowie odpowiedzi.

Cel opóźnienia: pierwszy bajt audio TTS w ciągu 800 ms od zakończenia wypowiedzi użytkownika na laptopowym CPU. Cel jakości: brak pominiętych słów, brak halucynacji podtekstów na ciszy, brak wycieku klonowania głosu, brak powodzenia wstrzyknięcia prompta.

## Koncepcja

![Potok asystenta głosowego: mikrofon → VAD → STT → LLM+narzędzia → TTS → głośnik](../assets/voice-assistant.svg)

### Siedem komponentów

1. **Przechwytywanie audio.** Mikrofon → 16 kHz mono → fragmenty 20 ms. Zwykle `sounddevice` w Pythonie lub natywne AudioUnit/ALSA/WASAPI w produkcji.
2. **VAD (Lekcja 11).** Silero VAD @ próg 0.5, min mowa 250 ms, cisza hang-over 500 ms. Sygnalizuje "start" i "koniec."
3. **Streaming STT (Lekcja 4-5).** Whisper-streaming, Parakeet-TDT lub Deepgram Nova-3 (API). Częściowe + finalne transkrypcje.
4. **LLM z wywoływaniem narzędzi.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. JSON schema dla narzędzi. Strumieniuj tokeny.
5. **Streaming TTS (Lekcja 7).** Kokoro-82M (najszybszy open) lub Cartesia Sonic (komercyjny). Start TTS po 20 tokenach LLM.
6. **Odtwarzanie.** Wyjście na głośnik; opus-encode dla sieci niskiej przepustowości.
7. **Handler przerwań.** Jeśli VAD zadziała podczas odtwarzania TTS, stop odtwarzania, cancel LLM, restart STT.

### Trzy tryby awarii, które napotkasz

1. **Przycięcie pierwszego słowa.** VAD startuje trochę za późno. "Hej" użytkownika jest brakujący. Start threshold na 0.3, nie 0.5.
2. **Pomylenie przerwania w połowie odpowiedzi.** LLM nadal generuje po przerwaniu przez użytkownika; asystent mówi przez użytkownika. Podłącz VAD → cancel-LLM.
3. **Halucynacja ciszy.** Whisper generuje "Dziękujemy za oglądanie" na cichych frame'ach rozgrzewki. Zawsze VAD-gate.

### Referencyjne stosy produkcyjne 2026

| Stos | Opóźnienie | Licencja | Uwagi |
|------|------------|----------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | komercyjne API | Domyślne branżowe 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | głównie open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Single-model; inna architektura, lekcja 15 |
| Vapi / Retell (managed) | 300-500 ms | komercyjne | Najszybsze do uruchomienia; ograniczona personalizacja |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Prywatność / edge |

## Zbuduj to

### Krok 1: przechwytywanie mikrofonu z chunking (pseudokod)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Krok 2: VAD-gated turn capture

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Krok 3: streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Krok 4: tool calling wewnątrz pętli LLM

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Krok 5: obsługa przerwań

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Użyj tego

Zobacz `code/main.py` dla symulacji z stub models, więc możesz zobaczyć kształt potoku nawet bez sprzętu. Dla prawdziwej implementacji, zamień stubs na:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` lub `openai-whisper`
- `openai` (`gpt-4o`) lub `anthropic`
- `kokoro` lub `cartesia`
- `sounddevice` dla I/O

## Pułapki

- **Rejestrowanie PII na zawsze.** Pełny turn audio to PII w większości jurysdykcji. 30-dniowe przechowywanie, encrypted at rest.
- **Brak barge-in.** Użytkownicy będą przerywać. Twój asystent musi przestać mówić.
- **TTS który blokuje.** Synchroniczny TTS blokuje event loop. Użyj async lub oddzielnego wątku.
- **Brak obsługi błędów tool-call.** Narzędzia się psują. LLM musi otrzymać błąd + retry raz, potem graceful degrade.
- **Zbyt gorliwe filtry halucynacji.** Over-filter i asystent powtarza "Nie mogę pomóc z tym." Under-filter i mówi cokolwiek. Kalibruj na held-out set.
- **Brak opcji wake-word.** Zawsze słuchający to odpowiedzialność za prywatność. Dodaj wake-word gate (Porcupine lub openWakeWord).

## Wyślij to

Zapisz jako `outputs/skill-voice-assistant-architect.md`. Biorąc pod uwagę budżet + skalę + język + ograniczenia compliance, wyprodukuj pełną specyfikację stosu.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Symuluje jeden pełny turn end-to-end ze stub modules i打印uje opóźnienie per-stage.
2. **Średnie.** Zastąp STT stub prawdziwym modelem Whisper na pre-recorded `.wav`. Zmierz WER i end-to-end opóźnienie.
3. **Trudne.** Dodaj tool calling: zaimplementuj `get_weather` (dowolne API) i `set_timer`. Przeprowadź LLM przez narzędzia i zweryfikuj, że gdy użytkownik powie "ustaw 5-minutowy timer" prawidłowa funkcja się zapala i ustna odpowiedź to potwierdza.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Turn | Jedna wymiana user + assistant | Jeden VAD-bounded user speech + jedna odpowiedź LLM-TTS. |
| Barge-in | Przerwanie | Użytkownik mówi podczas gdy asystent mówi; asystent przestaje. |
| Wake word | "Hej asystent" | Short keyword detector; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Koniec tury | VAD + min-silence decision że użytkownik skończył. |
| Pre-roll | Pre-speech buffer | Keep 200-400 ms of audio przed VAD fire żeby uniknąć first-word clip. |
| Tool call | Wywołanie funkcji | LLM emituje JSON; runtime despatchuje; result wraca w pętlę. |

## Dalsze czytanie

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — reference produkcyjna.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY-friendly framework.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — managed voice-native path.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — full-duplex reference (Lekcja 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — wake-word gating.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM function calling.