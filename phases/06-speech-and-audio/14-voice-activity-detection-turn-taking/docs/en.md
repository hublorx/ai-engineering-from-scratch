```markdown
# Voice Activity Detection i wykrywanie zmiany turny — Silero, Cobra i sztuczka z flush

> Każdy voice agent żyje lub umiera z dwóch decyzji: czy użytkownik teraz mówi i czy skończył? VAD odpowiada na pierwsze. Wykrywanie zmiany turny (VAD + cisza po aktywności + semantyczny model końca wypowiedzi) odpowiada na drugie. Zrób którekolwiek źle, a Twój asystent albo przerywa użytkownikowi, albo nigdy nie zamknie dzioba.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Szacowany czas:** ~45 minut

## Problem

Trzy różne decyzje, które voice agent podejmuje dla każdego fragmentu 20 ms:

1. **Czy ta ramka to mowa?** — VAD. Binarna, na ramkę.
2. **Czy użytkownik zaczął nową wypowiedź?** — wykrywanie początku.
3. **Czy użytkownik skończył?** — wskazanie punktu końcowego (koniec tury).

Naiwna odpowiedź (próg energii) zawodzi przy jakimkolwiek szumie — ruch uliczny, klawiatury, gwar tłumu. Odpowiedź 2026: Silero VAD (open, deep-learned) + model wykrywania zmiany turny (semantyczne wskazanie końca) + VAD-kalibrowane ciszy po aktywności.

## Koncepcja

![Kaskada VAD: energia → Silero → detektor turn → sztuczka z flush](../assets/vad-turn-taking.svg)

### Trzypoziomowa kaskada VAD

**Poziom 1: bramka energetyczna.** Najtańsza. Próg RMS przy -40 dBFS. Filtruje oczywistą ciszę, ale wyzwala przy dowolnym szumie powyżej progu.

**Poziom 2: Silero VAD** (2020-2026, MIT). 1M parametrów. Trenowany na 6000+ języków. Działa w ~1 ms na fragment 30 ms na jednym wątku CPU. 87.7% TPR przy 5% FPR. Domyślny open-source.

**Poziom 3: semantyczny detektor turn.** Model LiveKit do wykrywania turn (2024-2026) lub własna mała klasyfikator. Rozróżnia "pauza w środku zdania" od "skończyłem mówić." Używa kontekstu lingwistycznego (intonacja + ostatnie słowa), nie tylko ciszy.

### Kluczowe parametry i ich wartości domyślne

- **Threshold.** Silero zwraca prawdopodobieństwo; klasyfikuj mowę przy > 0.5, (domyślne) lub > 0.3 (czułe). Niższy próg = mniej obciętych pierwszych słów, więcej fałszywych pozytywów.
- **Minimalny czas trwania mowy.** Odrzuć mowę krótszą niż 250 ms — zwykle kaszel lub dźwięk krzesła.
- **Cisza po aktywności (wskazanie końca).** Po powrocie VAD do 0, poczekaj 500-800 ms przed ogłoszeniem końca tury. Za krótko → przerywasz użytkownikowi. Za długo → sprawia wrażenie opieszałości.
- **Bufor pre-roll.** Trzymaj 300-500 ms audio przed wyzwoleniem VAD. Zapobiega obcięciu "hej".

### Sztuczka z flush (Kyutai 2025)

Strumieniowe modele STT mają opóźnienie lookahead (500 ms dla Kyutai STT-1B, 2.5 s dla STT-2.6B). Normalnie czekałbyś tak długo po końcu mowy na transkrypt. Sztuczka z flush: gdy VAD wykrywa koniec mowy, **wyślij sygnał wymuszenia do STT**, który wymusza natychmiastowy wynik. STT typu flush przetwarza w ~4× realtime, więc bufor 500 ms kończy się w ~125 ms.

Od końca do końca: 125 ms VAD + flush STT = latencja konwersacyjna.

### Porównanie VAD 2026

| VAD | TPR @ 5% FPR | Latencja | Licencja |
|-----|--------------|---------|----------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | komercyjna |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero to właściwy domyślny wybór. Cobra to ulepszenie dokładności / zgodności**,** Energy-only VAD nie ma miejsca w produkcji 2026.

## Zbuduj to

### Krok 1: bramka energetyczna

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Krok 2: Silero VAD w Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Krok 3: automat stanów wykrywania końca tury

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Krok 4: szkielet sztuczki z flush

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT (Kyutai, Deepgram, AssemblyAI) musi obsługiwać flush, żeby to działało. Whisper streaming nie — jest block-based i zawsze czeka na chunki.

## Użyj tego

| Sytuacja | Wybór VAD |
|----------|-----------|
| Open, fast, general | Silero VAD |
| Komercyjne call center | Cobra VAD |
| On-device (telefon) | Silero VAD ONNX |
| Research / diarization | pyannote segmentation |
| Zero-dependency fallback | WebRTC VAD (legacy) |
| Potrzebujesz jakości końca tury | Silero + LiveKit turn-detector layered |

Zasada kciuka: nigdy nie wysyłaj energy-only VAD, chyba że naprawdę nie masz innej opcji.

## Pułapki

- **Stały próg.** Działa w ciszy, zawodzi w hałasie. Albo kalibruj on-device, albo przełącz na Silero.
- **Za krótka cisza po aktywności.** Agent przerywa w połowie zdania. 500-800 ms to optimum dla mowy konwersacyjnej.
- **Za długa cisza po aktywności.** Sprawia wrażenie opieszałości. Testuj A/B z docelowymi użytkownikami.
- **Brak bufora pre-roll.** Pierwsze 200-300 ms audio użytkownika tracone. Zawsze trzymaj rolling pre-roll.
- **Ignorowanie semantycznego wskazania końca.** "Hmm, daj pomyśleć..." zawiera długie pauzy. Użytkownicy nienawidzą, gdy przerywa im w połowie myśli. Używaj LiveKit turn-detector lub podobnego.

## Wyślij to

Zapisz jako `outputs/skill-vad-tuner.md`. Wybierz model VAD, threshold, hangover, pre-roll i strategię wykrywania turn dla swojego obciążenia.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Symuluje sekwencję mowa + cisza + mowa + kaszel i testuje trzy poziomy VAD.
2. **Średnie.** Zainstaluj `silero-vad`, przetwórz nagranie 5-min, dostrój threshold, żeby zminimalizować obcięte pierwsze słowa i fałszywe wyzwalania. Raportuj precision/recall.
3. **Trudne.** Zbuduj mini turn-detector: Silero VAD + 3-warstwowy MLP na ostatnich 10 słowach embeddings (użyj sentence-transformers). Trenuj na ręcznie oznakowanym datasetcie końca tury. Przebij Silero-only o 10% F1.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| VAD | Wykrywacz głosu | Binarny na ramkę: czy to mowa? |
| Turn detection | Wskazanie końca | VAD + cisza po aktywności + semantyczny endpoint. |
| Silence hangover | Czekanie po mowie | Czas przed ogłoszeniem końca tury; 500-800 ms. |
| Pre-roll | Bufor przed mową | Trzymaj 300-500 ms audio przed wyzwoleniem VAD. |
| Flush trick | Hack Kyutai | VAD → flush-STT → 125 ms zamiast 500 ms opóźnienia. |
| Semantic endpoint | "Czy zamierzał się zatrzymać?" | Klasyfikator ML, który patrzy na słowa, nie tylko ciszę. |
| TPR @ FPR 5% | Punkt ROC | Standard VAD benchmark; 87.7% dla Silero, 50% WebRTC. |
```