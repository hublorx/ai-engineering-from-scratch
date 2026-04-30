# Streaming Speech-to-Speech — Moshi, Hibiki i Full-Duplex Dialogue

> 2024–2026 na nowo zdefiniowało voice AI. Moshi dostarcza jeden model, który słucha i mówi jednocześnie z opóźnieniem 200 ms. Hibiki wykonuje tłumaczenie speech-to-speech chunk po chunku. Oba porzucają potok ASR → LLM → TTS na rzecz ujednoliconej architektury full-duplex przez tokeny kodeka Mimi. To jest nowy wzorzec projektowy.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 6 · 13 (Neural Audio Codecs), Faza 6 · 11 (Real-Time Audio), Faza 7 · 05 (Full Transformer)
**Szacowany czas:** ~75 minut

## Problem

Każdy voice agent zbudowany z Lekcji 11 + 12 ma fundamentalny próg opóźnienia na poziomie 300–500 ms: VAD się uruchamia, STT przetwarza, LLM przetwarza, TTS generuje. Każdy etap ma własne minimalne opóźnienie. Możesz dostrajać i równoległościć, ale kształt potoku ogranicza Cię.

Moshi (Kyutai, 2024–2026) zadaje inne pytanie: co jeśli nie ma potoku? Co jeśli jeden model przyjmuje audio na wejściu i emituje audio na wyjściu bezpośrednio, ciągle, z tekstem jako pośrednim „wewnętrznym monologiem" zamiast wymaganego etapu?

Odpowiedź to **full-duplex speech-to-speech**. Opóźnienie teoretyczne 160 ms (80 ms ramka Mimi + 80 ms opóźnienie akustyczne). Opóźnienie praktyczne 200 ms na pojedynczym GPU L4. To połowa tego, co osiąga najlepszy w swojej klasie pipelined voice agent.

## Koncepcja

![Architektura Moshi: dwa równoległe strumienie Mimi + wewnętrzny monolog tekstowy](../assets/moshi-hibiki.svg)

### Architektura Moshi

**Wejścia.** Dwa strumienie kodeka Mimi, oba przy 12,5 Hz × 8 codebooks:

- Strumień 1: audio użytkownika (zakodowane przez Mimi, ciągle napływające)
- Strumień 2: własne audio Moshi (generowane przez Moshi)

**Transformer.** 7B-parametrowy Temporal Transformer przetwarza oba strumienie i strumień tekstowy „wewnętrznego monologu". Przy każdym kroku 80 ms wykonuje:

1. Konsumuje najnowsze tokeny Mimi użytkownika (8 codebooks).
2. Konsumuje najnowsze tokeny Mimi Moshi (8 codebooks, jak zostały wyprodukowane).
3. Generuje następny token tekstowy Moshi (wewnętrzny monolog).
4. Generuje następne tokeny Mimi Moshi (8 codebooks przez mały Depth Transformer).

Wszystkie trzy strumienie — audio użytkownika, audio Moshi, tekst Moshi — działają równoległe. Moshi może słyszeć użytkownika podczas mówienia, może przerywać siebie, gdy użytkownik przerywa, może wysyłać sygnały potwierdzenia („mhm") bez przerywania głównej wypowiedzi.

**Depth transformer.** W ramach ramki 8 codebooks nie jest przewidywanych równoległe — mają między-codebookowe zależności. Mały 2-warstwowy „depth transformer" przewiduje je sekwencyjnie w ciągu 80 ms. To standardowy faktor dla AR codec LMs, który używany jest również przez VALL-E, VibeVoice.

### Dlaczego wewnętrzny monolog tekstowy pomaga

Bez jawnego tekstu model musi implicitnie modelować język w strumieniu akustycznym. Wgląd Moshi: zmusza go do emitowania tokenów tekstowych obok audio. Strumień tekstowy to w istocie transkrypcja tego, co Moshi mówi. To poprawia spójność semantyczną, ułatwia wymianę głowy modelu językowego i daje transkrypcje za darmo.

### Hibiki: streaming speech-to-speech translation

Ta sama architektura, trenowana na parach tłumaczeniowych. Audio źródłowe na wejściu, audio w języku docelowym na wyjściu, ciągle. Hibiki-Zero (luty 2026) eliminuje potrzebę danych treningowych wyrównanych na poziomie słów — używa danych na poziomie zdań + GRPO reinforcement learning dla optymalizacji opóźnienia.

Czas, gdy modele uczą się tłumaczyć ciągle, bez kar za opóźnienia, które są mniejsze niż u człowieka.

Cztery pary językowe obsługiwane początkowo; można zaadaptować do nowego języka z ≈1000 godzin.

### Szerszy stos Kyutai (2026)

- **Moshi** — full-duplex dialog (najpierw francuski, angielski dobrze obsługiwany)
- **Hibiki / Hibiki-Zero** — jednoczesne tłumaczenie mowy
- **Kyutai STT** — streaming ASR (wyprzedzenie 500 ms lub 2,5 s)
- **Kyutai Pocket TTS** — 100M-param TTS działa na CPU (styczeń 2026)
- **Unmute** — pełny potok łączący te komponenty na publicznych serwerach

Przepustowość na GPU L40S: 64 równoczesne sesje przy 3× czasie rzeczywistego.

### Sesame CSM — kuzyn

Sesame CSM (2025) używa podobnej koncepcji — backbone Llama-3 z głową kodeka Mimi. Ale CSM jest jednokierunkowy (przyjmuje kontekst + tekst, produkuje mowę) zamiast full-duplex. Ma najlepszy na rynku „voice presence" TTS; nie do końca to samo, co możliwości full-duplex Moshi.

### Liczby wydajności 2026

| Model | Opóźnienie | Przypadek użycia | Licencja |
|-------|------------|------------------|----------|
| Moshi | 200 ms (L4) | full-duplex dialog angielski / francuski | CC-BY 4.0 |
| Hibiki | 12,5 Hz framerate | jednoczesne tłumaczenie francuski ↔ angielski | CC-BY 4.0 |
| Hibiki-Zero | to samo | 5 par językowych, bez danych wyrównanych | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | zamknięte, OpenAI API | komercyjne |
| Gemini 2.5 Live | ~350 ms | zamknięte, Google API | komercyjne |

## Zbuduj To

### Krok 1: interfejs

Moshi udostępnia serwer WebSocket, który przyjmuje chunki 80 ms zakodowanego audio Mimi i zwraca chunki 80 ms zakodowanego audio Mimi. W obie strony. Ciągle.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Krok 2: pełna pętla full-duplex

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Oba kierunki działają jednocześnie. Python asyncio lub Rust futures to standardowy transport.

### Krok 3: cel treningowy (koncepcyjny)

Dla każdej ramki 80 ms `t`:

- Wejście: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Predykcja: `moshi_text[t]`, a potem `moshi_mimi[t, codebook_0..7]`

Tekst jest przewidywany przed audio (wewnętrzny monolog); audio jest przewidywane sekwencyjnie między codebooks w depth transformer.

### Krok 4: gdzie Moshi wygrywa i gdzie nie

Moshi wygrywa:

- Sub-250 ms end-to-end na tanim sprzęcie.
- Naturalne sygnały potwierdzenia i przerwania.
- Brak kodu sklejającego potok.

Moshi nie wygrywa:

- Tool calling (nie trenowane; potrzebujesz osobnej ścieżki LLM).
- Długie wnioskowanie (Moshi to ok. 8B model dialogowy, nie Claude/GPT-4).
- Faktualna dokładność w niszowych tematach.
- Większość produkcyjnych przypadków enterprise (wciąż używają potoków w 2026).

## Użyj

| Sytuacja | Wybierz |
|----------|---------|
| Najniższe opóźnienie voice companion | Moshi |
| Live translation call | Hibiki |
| Voice demo / research | Moshi, CSM |
| Enterprise agent z narzędziami | Potok (Lekcja 12), nie Moshi |
| Custom-voice TTS w kontekście | Sesame CSM |
| Speech-to-speech, dowolne języki | GPT-4o Realtime lub Gemini 2.5 Live (komercyjne) |

## Pułapki

- **Ograniczone tool calling.** Moshi to model dialogowy, a framework agentowy. Połącz z potokiem dla narzędzi.
- **Kondycjonowanie na konkretny głos.** Moshi używa pojedynczej wytrenowanej persony; klonowanie to osobny cykl treningowy.
- **Pokrycie językowe.** Francuski + angielski jest doskonałe; inne ograniczone. Hibiki-Zero pomaga, ale nadal potrzebujesz danych treningowych.
- **Koszt zasobów.** Pełna sesja Moshi, która zajmuje slot GPU; nie jest to tani wzorzec wdrożenia dla shared-tenant.

## Wyślij

Zapisz jako `outputs/skill-duplex-pipeline.md`. Wybierz architekturę potokową vs full-duplex dla obciążenia voice-agent, z uzasadnieniem.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Symuluje architekturę dwóch strumieni + wewnętrzny monolog symbolicznie.
2. **Średnie.** Pobierz Moshi z HuggingFace, uruchom serwer, przetestuj jedną rozmowę. Zmierz opóźnienie zegarka od końca mowy użytkownika do początku odpowiedzi Moshi.
3. **Trudne.** Weź swojego potokowego agenta z Lekcji 12 i porównaj P50 latency vs Moshi na 20 dopasowanych wypowiedziach testowych. Napisz, kiedy architektura potokowa i tak wygrywa.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|------------------------|
| Full-duplex | Słuchaj-i-mów jednocześnie | Dwa strumienie audio aktywne jednocześnie na tym samym modelu. |
| Inner monologue | Strumień tekstowy modelu | Moshi emituje tokeny tekstowe obok swojego wyjścia audio. |
| Depth transformer | Predyktor między-codebook | Mały transformer, który przewiduje 8 codebooks w jednej ramce 80 ms. |
| Mimi | Kodek Kyutai | 12,5 Hz × 8 codebooks; semantyczny+akustyczny; napędza Moshi. |
| Streaming S2S | Audio → audio na żywo | Tłumaczenie/dialog chunk po chunku, bez etapów potoku. |
| Back-channeling | Reakcje „Mhm" | Moshi może emitować małe potwierdzenia bez przerywania swojej kolejki. |

## Dalsza Lecja

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — artykuł.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — streaming translation bez wyrównanych danych.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — specyfikacja CSM.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — instalacja + serwer.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — zamknięty komercyjny odpowiednik.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — framework STT/TTS pod spodem.

---

**Wprowadzone poprawki:**

1. **Przecinek przed „w"** (sekcja „Dlaczego wewnętrzny monolog tekstowy pomaga"): Usunięto przecinek w wyrażeniu „model musi implicitnie modelować język w strumieniu akustycznym" — fraza przyimkowa „w strumieniu akustycznym" pełni rolę okolicznika sposobu/miejsca i nie wymaga przecinka.

2. **Termin techniczny** (sekcja „Szerszy stos Kyutai"): „look-ahead 500 ms" → „wyprzedzenie 500 ms" — spolszczono termin techniczny.

3. **Wielka litera w tabeli** (sekcja „Liczby wydajności 2026"): „Jednoczesne tłumaczenie" → poprawione na „jednoczesne" z małej litery (termin techniczny)

4. **Kod** — wszystkie fragmenty pozostawione w oryginalnej formie angielskiej.