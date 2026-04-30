# Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> Modele audio-językowe z 2026 roku przeprowadzają rozumowanie na podstawie mowy, dźwięków środowiskowych i muzyki. Qwen2.5-Omni-7B dorównuje GPT-4o Audio na MMAU-Pro. Audio Flamingo Next przewyższa Gemini 2.5 Pro na LongAudioBench. Przewaga modeli zamkniętych nad otwartymi została zasadniczo zlikwidowana — z wyjątkiem zadań multi-audio, gdzie wszyscy osiągają wyniki bliskie losowym.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Szacowany czas:** około 45 minut

## Problem

Masz 5 sekund audio: psy szczekają, ktoś krzyczy „stop!", a potem cisza. Przydatne pytania obejmują wiele osi:

- **Transkrypcja.** „Co zostało powiedziane?" — teren ASR.
- **Rozumowanie semantyczne.** „Czy osoba jest w niebezpieczeństwie?" — wymaga wspólnego zrozumienia szczekania + krzyku + ciszy.
- **Rozumowanie muzyczne.** „Jakie instrumenty grają melodię?"
- **Wyszukiwanie w długim audio.** „Gdzie w tym 90-minutowym wykładzie instruktor wyjaśnił gradient descent?"

Pojedynczy model odpowiadający na wszystkie te pytania jednym promptem to **audio-language model** (LALM / ALM). Inny niż czyste ASR: LALM generują odpowiedzi w wolnej formie w języku naturalnym, nie tylko transkrypcje.

## Koncepcja

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### Szablon trójskładnikowy

Każdy LALM z 2026 roku ma ten sam szkielet:

1. **Audio encoder.** Whisper encoder · BEATs · CLAP · WavLM · lub niestandardowy encoder dla danego modelu.
2. **Projector.** Warstwa liniowa lub MLP łącząca cechy audio encoder z przestrzenią osadzania tokenów LLM.
3. **LLM.** Dekoder oparty na Llamie / Qwen / Gemmie. Przyjmuje przeplatane tokeny tekstowe + audio; generuje tekst.

Trening:

- **Etap 1.** Zamroź encoder + LLM; trenuj projector tylko na danych ASR / captioning.
- **Etap 2.** Pełny / LoRA fine-tune na zadaniach instruction-following z audio (QA, rozumowanie, rozumienie muzyki).
- **Etap 3 (opcjonalny).** Voice-in / voice-out dodaje speech decoder. Qwen2.5-Omni i AF3-Chat to robią.

### Mapa modeli z 2026

| Model | Backbone | Audio encoder | Modalność wyjściowa | Dostęp |
|-------|----------|---------------|----------------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Sprawdzenie rzeczywistości benchmarków (2026)

**MMAU-Pro.** 1800 par QA obejmujących speech / sound / music / mixed. Podzbiór multi-audio włączony.

| Model | Ogółem | Speech | Sound | Music | Multi-audio |
|-------|--------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA na LongAudioBench | — | — | — | — |

**Kolumna multi-audio jest kompromitująca dla wszystkich.** Losowy traf na wielokrotnym wyborze 4 opcji = 25%; większość modeli osiąga właśnie tyle. LALM wciąż mają trudności z porównywaniem dwóch klipów.

### Gdzie LALM są przydatne w 2026

- **Audit zgodności nagrań call-center.** „Czy agent wspomniał o wymaganym ujawnieniu?"
- **Dostępność.** Opisuj zdarzenia dźwiękowe dla głuchych użytkowników (nie tylko transkrypcja).
- **Moderacja treści.** Wykrywaj przemoc werbalną + groźący ton + kontekst tła.
- **Rozdziałowanie podcastów / spotkań.** Semantyczne podsumowanie, nie tylko zmiany mówców.
- **Analiza katalogu muzycznego.** „Znajdź wszystkie utwory z zmianą tonacji w części B."

### Gdzie NIE są (jeszcze) przydatne

- Szczegółowa teoria muzyki (poniżej poziomu akordów).
- Rozumowanie z atrybuacją mówcy w długich rozmowach (pogarsza się po 10 minutach).
- Porównywanie multi-audio (22-26% to ledwo powyżej losowego).
- Rozumowanie w czasie rzeczywistym streaming (większość to offline batch inference).

## Zbuduj to

### Krok 1: zapytanie do Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Krok 2: wzorzec projectora

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

I to wszystko. Projector to zwykle 1-3 warstwy liniowe. Trenowanie go na parach ASR (audio → transkrypcja) to pretext task Etapu 1.

### Krok 3: benchmarking MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

Raportuj osobno według kategorii (speech / sound / music / multi-audio). Zagregowane liczby ukrywają, gdzie model się nie sprawdza.

## Użyj tego

| Zadanie | Wybór na 2026 |
|---------|---------------|
| Free-form audio QA (open) | Qwen2.5-Omni-7B |
| Najlepszy open na długie audio | Audio Flamingo Next |
| Najlepszy closed | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni lub GPT-4o Audio |
| Rozumowanie muzyczne | Audio Flamingo 3 lub 2 (music-specialized AF-CLAP) |
| Audit call-center | Gemini 2.5 Pro via API, z RAG na twoich dokumentach polityki |

## Pułapki

- **Nadmierne zaufanie do multi-audio.** Jeśli twoje zadanie wymaga „który klip ma X," wydajność na poziomie losowym to rzeczywistość.
- **Pogorszenie na długim audio.** Po 10 minutach atrybuacja mówcy w większości modeli się psuje. Najpierw diarizuj (Lesson 6), potem podsumuj.
- **Halucynacje na ciszy.** Ten sam problem co w Whisper-style jest dziedziczony przez LALM używające Whisper encoder. Bramkuj VAD.
- **Cherry-picking benchmarków.** Posty na blogach vendorów podkreślają najlepsze przypadki w kategoriach. Uruchom podzbiór MMAU-Pro multi-audio samodzielnie.

## Wyślij to

Zapisz jako `outputs/skill-alm-picker.md`. Wybierz LALM + podzbiór benchmark + modalność wyjściowa (text vs speech) dla danego zadania rozumienia audio.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`, aby zobaczyć wzorzec toy projector + fake LALM routing (audio-embedding, text-tokens) → output tokens.
2. **Średnie.** Oceń Qwen2.5-Omni-7B na 100 elementach MMAU-Pro speech. Porównaj z liczbą podaną w artykule.
3. **Trudne.** Zbuduj minimalny baseline audio-captioning: BEATs encoder + 2-warstwowy projector + zamrożony Llama-3.2-1B. Fine-tune tylko projector na AudioCaps. Porównaj z SALMONN na Clotho-AQA.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | Małe MLP mapujące cechy audio w przestrzeń osadzania LLM. |
| MMAU | Benchmark | 10k par audio-QA obejmujących speech, sound, music. |
| MMAU-Pro | Trudniejszy MMAU | 1800 pytań multi-audio / intensywnych rozumieniowo. |
| LongAudioBench | Ewaluacja długich form | Wielominutowe klipy z zapytaniami semantycznymi. |
| Voice-in / voice-out | Speech-native | Model przyjmuje mowę i emituje mowę bez pośrednictwa tekstu. |

## Dalsze czytanie

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — architektura referencyjna.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — otwarty lider long-audio.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — SOTA na LongAudioBench.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — pionier dual-encoder.
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — live rankings na 2026.