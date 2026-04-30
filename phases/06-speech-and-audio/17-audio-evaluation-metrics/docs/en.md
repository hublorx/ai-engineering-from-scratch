# Ocena Audio — WER, MOS, UTMOS, MMAU, FAD i tablice wyników (Leaderboardy)

> Nie możesz wydać tego, czego nie możesz zmierzyć. Ta lekcja przedstawia metryki 2026 dla każdego zadania audio: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), audio-language (MMAU, LongAudioBench), muzykę (FAD, CLAP) i weryfikację mówcy (EER). Plus tablice wyników, gdzie możesz porównać.

**Typ:** Ucz się
**Języki:** Python
**Wymagania wstępne:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Szacowany czas:** ~60 minut

## Problem

Każde zadanie audio ma wiele metryk, z których każda mierzy inny wymiar. Użycie niewłaściwej metryki to sposób na wydanie modelu, który wygląda świetnie na twoim dashboardzie, a tragicznie w produkcji. Kanoniczna lista 2026:

| Zadanie | Podstawowa | Wtórna |
|---------|------------|--------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Klonowanie głosu | SECS (ECAPA cosine) | MOS · CER |
| Weryfikacja mówcy | EER | minDCF · FAR / FRR at operating point |
| Diarizacja | DER | JER · speaker confusion |
| Klasyfikacja audio | top-1 · mAP | macro F1 · per-class recall |
| Generowanie muzyki | FAD | CLAP · listening panel MOS |
| Model audio-językowy | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Strumieniowe S2S | latency P50/P95 | WER · MOS |

## Koncepcja

![Macierz oceny audio — metryki vs zadania vs tablice wyników 2026](../assets/eval-landscape.svg)

### Metryki ASR

**WER (Word Error Rate).** `(S + D + I) / N`. Małe litery, usuń interpunkcję, normalizuj liczby przed scoringiem. Użyj `jiwer` lub OpenAI `whisper_normalizer`. < 5% = ludzka równość dla mowy odczytanej.

**CER (Character Error Rate).** Ten sam wzór, na poziomie znaków. Używany dla języków tonalnych (Mandaryński, Kantoński), gdzie segmentacja słów jest niejednoznaczna.

**RTFx (inverse real-time factor).** Sekundy audio przetworzone na sekundę zegarową. Wyższy jest lepszy. Parakeet-TDT osiąga 3380×. Whisper-large-v3 to ~30×.

**First-token latency.** Zegar od wejścia audio do pierwszego tokena transkrypcji. Krytyczne dla streamingu. Deepgram Nova-3: ~150 ms.

### Metryki TTS

**MOS (Mean Opinion Score).** Ocena ludzka 1-5. Złoty standard, ale wolna. Zbierz 20+ słuchaczy na próbkę, 100+ próbek na model.

**UTMOS (2022-2026).** Nauczony predyktor MOS. Koreluje ~0.9 z ludzkim MOS na standardowych benchmarkach. F5-TTS: UTMOS 3.95; ground truth: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** Dla klonowania głosu. Cosine ECAPA embedding między referencją a sklonowanym wyjściem. > 0.75 = rozpoznawalny klon.

**WER-on-ASR-round-trip.** Uruchom Whisper na wyjściu TTS, oblicz WER wobec tekstu wejściowego. Łapie regresje zrozumiałości. SOTA 2026: < 2% CER.

**TTFA (time-to-first-audio).** Latencja zegarowa. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.

### Klonowanie głosu — specyficzne

**SECS + MOS + CER** jako trójka. Klonowanie z wysokim SECS ale niskim MOS oznacza barwa-poprawna-ale-nienaturalna; przeciwieństwo oznacza naturalny głos ale zły mówca.

### Weryfikacja mówcy

**EER (Equal Error Rate).** Próg, gdzie False Accept Rate równa się False Reject Rate. ECAPA na VoxCeleb1-O: 0.87%.

**minDCF (min Detection Cost).** Ważony koszt w wybranym punkcie operacyjnym (często FAR=0.01). Bardziej produkcyjnie-relewantny niż EER.

### Diarizacja

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. Pominięta mowa + fałszywy alarm mowa + konfuzja mówcy, każdy jako ułamek. AMI meetings: DER ~10-20% jest realistyczny. pyannote 3.1 + Precision-2 commercial: <10% DER na dobrze-nagranym audio.

**JER (Jaccard Error Rate).** Alternatywa do DER, odporna na bias krótkich segmentów.

### Klasyfikacja audio

Multi-label: **mAP (mean Average Precision)** przez wszystkie klasy. AudioSet: 0.548 mAP dla BEATs-iter3.

Multi-class exclusive: **top-1, top-5 accuracy**. Speech Commands v2: 99.0% top-1 (Audio-MAE).

Imbalanced: **macro F1** + **per-class recall**. Raportuj per-class — zagregowana accuracy ukrywa, które klasy zawodzą.

### Generowanie muzyki

**FAD (Fréchet Audio Distance).** Odległość między rozkładami embeddingów VGGish prawdziwego vs wygenerowanego audio. MusicGen-small na MusicCaps: 4.5. MusicLM: 4.0. Niższy jest lepszy.

**CLAP Score.** Wynik alignowania text-audio używając CLAP embeddings. > 0.3 = rozsądny alignment.

**Listening panel MOS.** Wciąż ostatnie słowo dla konsumenckiej muzyki. Suno v5 ELO 1293 na TTS Arena (z paretowych preferencji ludzkich).

### Audio-language benchmarks

**MMAU (Massive Multi-Audio Understanding).** 10k audio-QA par.

**MMAU-Pro.** 1800 trudnych elementów, cztery kategorie: speech / sound / music / multi-audio. Losowy traf 25% na 4-way. Gemini 2.5 Pro overall ~60%; multi-audio ~22% wśród wszystkich modeli.

**LongAudioBench.** Klipy wielominutowe z zapytaniami semantycznymi. Audio Flamingo Next pokonuje Gemini 2.5 Pro.

**AudioCaps / Clotho.** Benchmarki podpisu. Metryki SPICE, CIDEr, FENSE.

### Strumieniowe S2S

**Latency P50 / P95 / P99.** Zegar od końca mowy użytkownika do pierwszego słyszalnego odpowiedzi. Moshi: 200 ms; GPT-4o Realtime: 300 ms.

**WER / MOS** na wyjściu.

**Barge-in responsiveness.** Czas od przerwania użytkownika do wyciszenia asystenta. Cel: < 150 ms.

### Tablice wyników 2026

| Tablica wyników | Ślady | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO z paretowych głosów | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (wewnątrz MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

## Zbuduj to

### Krok 1: WER z normalizacją

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Krok 2: TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Krok 3: SECS dla voice cloning

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Krok 4: FAD dla music generation

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Krok 5: EER dla speaker verification (ten sam kod co w Lesson 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Użyj tego

Sparuj każdy deploy z ustaloną infrastrukturą eval, która uruchamia się przy każdej aktualizacji modelu. Trzy kardynalne zasady:

1. **Normalizuj przed scoringiem.** Małe litery, usuń interpunkcję, rozwinięcie liczb. Raportuj regułę normalizacji.
2. **Raportuj rozkłady, nie średnie.** P50/P95/P99 dla latencji. Per-class recall dla klasyfikacji. Per-category dla MMAU.
3. **Uruchom jeden kanoniczny publiczny benchmark.** Nawet jeśli twoje dane produkcyjne różnią się, raportowanie na Open ASR / TTS Arena / MMAU pozwala recenzentom porównywać jabłka do jabłek.

## Pułapki

- **Ekstrapolacja UTMOS.** Trenowany na czystej mowie stylu VCTK; źle ocenia szum / sklonowane / emocjonalne audio.
- **Bias panelu MOS.** 20 pracowników Amazon Mechanical Turk ≠ 20 docelowych użytkowników. Zapłać za panel domenowy jeśli stawki są wysokie.
- **FAD zależy od zestawu referencyjnego.** Porównuj z tym samym rozkładem referencyjnym między modelami.
- **Agregowany WER.** Ogólny WER 5% może ukryć WER 30% na mowie z akcentem. Raportuj według wycinka demograficznego.
- **Nasycenie publicznych benchmarków.** Większość modeli frontier jest blisko sufitu na standardowych benchmarkach. Zbuduj wewnętrzny held-out set, który odzwierciedla twój ruch.

## Wyślij to

Zapisz jako `outputs/skill-audio-evaluator.md`. Wybierz metryki, benchmarki i format raportowania dla każdego release modelu audio.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Oblicz WER / CER / EER / SECS / FAD-ish / MMAU-ish na przykładowych inputach.
2. **Średnie.** Zbudź harness TTS round-trip WER. Uruchom wyjście twojego Kokoro lub F5-TTS przez Whisper. Oblicz WER przez 50 promptów. Oznacz prompty z WER > 10%.
3. **Trudne.** Oceń wybór LALM z Lesson 10 na podzbiorach MMAU-Pro speech + multi-audio (po 50 elementów). Raportuj per-category accuracy i porównaj z opublikowaną liczbą.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| WER | Wynik ASR | `(S+D+I)/N` na poziomie słowa po normalizacji. |
| CER | Character WER | Dla języków tonalnych lub systemów na poziomie znaków. |
| MOS | Opinia ludzka | Ocena 1-5; 20+ słuchaczy × 100 próbek. |
| UTMOS | ML MOS predictor | Nauczony model; koreluje ~0.9 z ludzkim MOS. |
| SECS | Podobieństwo voice-clone | Cosine ECAPA między referencją a klonem. |
| EER | Wynik speaker verif | Próg gdzie FAR = FRR. |
| DER | Wynik diarization | (FA + Miss + Confusion) / total. |
| FAD | Jakość music-gen | Fréchet distance na VGGish embeddings. |
| RTFx | Throughput | Sekundy audio na sekundę zegarową. |

## Dalsze czytanie

- [jiwer](https://github.com/jitsi/jiwer) — biblioteka WER/CER z narzędziami normalizacji.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — nauczony predyktor MOS.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — standard music-gen.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — live rankingi 2026.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — leaderboard TTS głosowany przez ludzi.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — leaderboard LALM reasoning.
- [HEAR benchmark](https://hearbenchmark.com/) — audio SSL benchmarks.