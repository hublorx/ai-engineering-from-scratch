# Generowanie Audio

> Audio to sygnał 1-D o częstotliwości 16-48 kHz. Pięciosekundowy klip to 80-240 tys. próbek. Żaden transformer nie przetwarza bezpośrednio takiej sekwencji. Rozwiązanie dla każdego produkcyjnego modelu audio w 2026 jest takie samo: neuronowy kodek (Encodec, SoundStream, DAC) kompresuje audio do dyskretnych tokenów przy 50-75 Hz, a transformer lub model dyfuzyjny generuje tokeny.

**Typ:** Zbuduj to
**Języki:** Python
**Wymagania wstępne:** Faza 6 · 02 (Funkcje Audio), Faza 6 · 04 (ASR), Faza 8 · 06 (DDPM)
**Czas:** ~45 minut

## Problem

Trzy zadania generowania audio:

1. **Synteza mowy (Text-to-speech).** Na podstawie tekstu wygeneruj mowę. Czysta mowa jest wąskopasmowa i ma silną strukturę fonetyczną — dobrze rozwiązana przez transformer-over-tokens. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Generowanie muzyki.** Na podstawie promptu (tekst, melodia, progresja akordów, gatunek) wygeneruj muzykę. Znacznie szerszy rozkład. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **Efekty dźwiękowe / projektowanie dźwięku.** Na podstawie promptu wygeneruj dźwięk ambientowy lub Foley. AudioGen, AudioLDM 2, Stable Audio Open.

Wszystkie trzy działają na tym samym podłożu: neuronowy kodek audio + generator token-AR lub dyfuzyjny.

## Koncepcja

![Generowanie audio: tokeny kodeka + transformer lub dyfuzja](../assets/audio-generation.svg)

### Neuronowe kodeki audio

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). Konwolucyjny enkoder kompresuje waveform do wektora na krok czasowy; residual vector quantization (RVQ) konwertuje każdy wektor do kaskady K indeksów codebook. Dekoder odwraca proces. Audio 24 kHz przy 2 kbps używając 8 codebooków RVQ przy 75 Hz = 600 tokenów/sec.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Dwa paradygmaty generatywne na górze

**Token-autoregresywny.** Spłaszcz tokeny RVQ do sekwencji, uruchom decoder-only transformer. MusicGen używa "opóźnionego równoległego" do emitowania K strumieni codebook równolegle z offsetami per-stream. VALL-E generuje tokeny mowy z promptu tekstowego + 3-sekundowej próbki głosu.

**Latent diffusion.** Pakuj tokeny kodeka jako ciągłe latenty lub modeluj je z categorical diffusion. Stable Audio 2.5 używa flow matching na ciągłych latentach audio. AudioLDM 2 używa text-to-mel-to-audio diffusion.

Trend 2024-2026: flow matching wygrywa w muzyce (szybsza inferencja, czystsze próbki) podczas gdy token-AR nadal dominuje w mowie, ponieważ jest naturalnie kauzalny i dobrze się strumieniuje.

## Krajobraz produkcyjny

| System | Zadanie | Backbone | Latencja |
|--------|---------|----------|----------|
| ElevenLabs V3 | TTS | Token-AR + neuronowy vocoder | ~300ms pierwszy token |
| OpenAI GPT-4o audio | Pełny dupleks mowy | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Muzyka / SFX | DiT + flow matching na latentach audio | ~10s dla klipu 1-minutowego |
| Suno v4 | Pełne piosenki | Niepubliczne; podejrzewany token-AR | ~30s na piosenkę |
| Udio v1.5 | Pełne piosenki | Niepubliczne | ~30s na piosenkę |
| MusicGen 3.3B | Muzyka | Token-AR na Encodec 32kHz | Real-time |
| AudioCraft 2 | Muzyka + SFX | Flow matching | ~5s dla klipu 5s |
| Riffusion v2 | Muzyka | Spectrogram diffusion | ~10s |

## Zbuduj to

`code/main.py` symuluje główną ideę: trenuj maleńki next-token transformer na syntetycznych sekwencjach "tokenów audio" wygenerowanych z dwóch distinct "stylów" (naprzemienne niskie i wysokie tokeny dla stylu A, monotoniczny ramp dla stylu B). Warunkuj na stylu i próbkuj.

### Krok 1: syntetyczne tokeny audio

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Krok 2: trenuj maleńki predyktor tokenów

Predyktor typu bigram warunkowany na stylu. Punkt to wzorzec: tokeny kodeka → trening cross-entropy → autoregresyjne próbkowanie.

### Krok 3: próbkuj warunkowo

Mając token stylu i token startowy, próbkuj następny token z przewidzianego rozkładu. Kontynuuj dla 20-40 tokenów.

## Pułapki

- **Jakość kodeka ogranicza jakość wyjścia.** Jeśli kodek nie może wiernie reprezentować dźwięku, żadna ilość jakości generatora nie pomoże. DAC to obecny najlepszy open source.
- **Akumulacja błędów RVQ.** Każda warstwa RVQ modeluje residuum poprzedniej. Błędy na warstwie 1 się propagują. Próbkowanie z temperaturą 0 na wyższych warstwach pomaga.
- **Struktura muzyczna.** 30 sekund tokenów to 20k+ tokenów przy 75 Hz. Trudne dla transformerów. MusicGen używa sliding window + kontynuacji promptu; Stable Audio używa krótszych klipów + crossfading.
- **Artefakty na granicach.** Crossfading między generowanymi klipami wymaga starannego overlap-add.
- **Apetyt na czyste dane.** Generatory muzyki potrzebują dziesiątek tysięcy godzin licencjonowanej muzyki. Pozew RIAA Suno / Udio (2024) to ujawnił.
- **Etyka klonowania głosu.** 3-sekundowa próbka plus prompt tekstowy wystarczy dla VALL-E / XTTS / ElevenLabs do sklonowania głosu. Każdy produkcyjny model potrzebuje wykrywania nadużyć + list opt-out.

## Użyj tego

| Zadanie | Stack 2026 |
|---------|------------|
| Komercyjny TTS | ElevenLabs, OpenAI TTS, lub Azure Neural |
| Klonowanie głosu (z weryfikacją zgody) | XTTS v2 (open) lub ElevenLabs Pro |
| Muzyka w tle, szybka | Stable Audio 2.5 API, Suno, lub Udio |
| Muzyka z tekstami | Suno v4 lub Udio v1.5 |
| Efekty dźwiękowe / Foley | AudioCraft 2, ElevenLabs SFX, lub Stable Audio Open |
| Agent głosowy w czasie rzeczywistym | GPT-4o realtime lub Gemini Live |
| Badania muzyczne open-weights | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Dubbing / tłumaczenie | HeyGen, ElevenLabs Dubbing |

## Wyślij to

Zapisz `outputs/skill-audio-brief.md`. Skill przyjmuje brief audio (zadanie, czas trwania, styl, głos, licencja) i wyprowadza: model + hosting, format promptu (tagi gatunku, deskryptory stylu, markery strukturalne), łańcuch kodek + generator + vocoder, protokół seeda, i plan ewaluacji (MOS / CLAP score / CER dla TTS / A/B użytkowników).

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` i ustaw styl jawnie. Zweryfikuj, że generowane sekwencje odpowiadają wzorcowi stylu.
2. **Średnie.** Dodaj opóźnione równoległe dekodowanie: symuluj 2 strumienie tokenów, które muszą pozostać offsetowane o 1 krok. Trenuj wspólny predyktor.
3. **Trudne.** Użyj HuggingFace transformers do uruchomienia MusicGen-small lokalnie. Wygeneruj klip 10-sekundowy z trzema różnymi promptami; A/B dla przestrzegania stylu.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Codec | "Kompresja neuronaowa" | Enkoder/dekoder dla audio; typyczne wyjście to tokeny 50-75 Hz. |
| RVQ | "Residual VQ" | Kaskada K kwantyzatorów; każdy modeluje residuum poprzedniego. |
| Token | "Jeden symbol kodeka" | Dyskretny indeks do codebooka; 1024 lub 2048 typowe. |
| Delayed parallel | "Offset codebooki" | Emituj K strumieni tokenów ze staggered offsetami, aby skrócić długość sekwencji. |
| Flow matching | "Wygrana 2024 dla audio" | Alternatywa dla dyfuzji ze straighter path; szybsze próbkowanie. |
| Voice prompt | "3-sekundowa próbka" | Speaker embedding lub prefix tokenów, który kieruje klonowanym głosem. |
| Mel spectrogram | "Obraz" | Log-magnitude perceptual spectrogram; używany przez wiele systemów TTS. |
| Vocoder | "Mel to wave" | Neuronowy komponent konwertujący mel spectrogramy z powrotem do audio. |

## Uwaga produkcyjna: audio to problem streamingu

Audio to jedyna modalność wyjściowa, której użytkownicy oczekują, że dotrze *w miarę generowania*, nie wszystko naraz. W kategoriach produkcyjnych oznacza to, że TPOT ma znaczenie (Time Per Output Token), ponieważ prędkość słuchania użytkownika to docelowa przepustowość — nie jego prędkość czytania. Dla audio 16kHz tokenizowanego przy ~75 tokenach/sec (Encodec), serwer musi generować ≥75 tokenów/sec na użytkownika, aby odtwarzanie było płynne.

Dwa konsekwencje architektoniczne:

- **Modele audio z flow matching nie mogą trywialnie streamować.** Stable Audio 2.5 i AudioCraft 2 renderują ustaloną długość klipu w jednym przejściu. Aby streamować, dzielisz klip na chunki i overlapujesz granice — pomyśl sliding-window diffusion — dodając 100-300ms overhead latencji vs codec AR model.

Jeśli produkt to "live voice chat" lub "real-time music continuation", wybierz ścieżkę codec AR. Jeśli to "renderuj 30-sekundowy klip przy submit", flow-matching wygrywa na jakości i całkowitej latencji.

## Dalsze czytanie

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — standard kodeka.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — pierwszy szeroko używany neuronowy kodek audio.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — tekst-do-muzyki 2025 z flow matching.