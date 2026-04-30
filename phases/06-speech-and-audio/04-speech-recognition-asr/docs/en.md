# Rozpoznawanie mowy (ASR) — CTC, RNN-T, Attention

> Rozpoznawanie mowy to klasyfikacja audio w każdej ramce czasowej, połączona w całość przez model sekwencyjny, który zna język angielski i ciszę. CTC, RNN-T i attention to trzy sposoby na to. Wybierz jeden i zrozum dlaczego.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Szacowany czas:** ~45 minut

## Problem

Masz 10-sekundowy klip 16 kHz. Chcesz string: "turn on the kitchen lights". Wyzwanie jest strukturalne: ramki audio nie są wyrównane jeden do jednego ze znakami. Słowo "okay" może trwać 200 ms lub 1200 ms. Cisza przerywa wypowiedź. Niektóre fonemy są dłuższe od innych. Liczba tokenów wyjściowych nie jest znana z góry.

Trzy sformułowania to rozwiązują:

1. **CTC (Connectionist Temporal Classification).** Emituje tokeny per-ramka z prawdopodobieństwem, w tym specjalny *blank*. Zwija powtórzenia i blanki przy dekodowaniu. Non-autoregressive, szybkie. Używane przez wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer).** Sieć predykcyjna przewiduje następny token biorąc pod uwagę ramkę enkodera i poprzednich tokenów. Strumieniowalne. Używane przez ASR Google na urządzeniach, NVIDIA Parakeet.
3. **Attention encoder-decoder.** Enkoder kompresuje audio do stanów ukrytych, dekoder stosuje cross-attention do wyjść enkodera, by generować tokeny autoregresywnie. Używane przez Whisper i SeamlessM4T.

W 2026 SOTA WER na LibriSpeech test-clean to 1.4% (Parakeet-TDT-1.1B, NVIDIA) i 1.58% (Whisper-Large-v3-turbo). Różnice są minimalne, więc różnice we wdrożeniu są ogromne.

## Koncepcja

![Trzy sformułowania ASR: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Intuicja CTC.** Niech enkoder wyprowadza `T` dystrybucji per-ramka na `V+1` tokenach (V znaków + blank). Dla stringa docelowego `y` o długości `U < T`, każde wyrównanie ramek, które zwija się do `y`, się liczy. Funkcja strat CTC sumuje po wszystkich takich wyrównaniach, co odpowiada marginalizacji. Inferencja: per-ramka argmax, zwijanie powtórzeń, usuwanie blanków.

Zalety: non-autoregressive, strumieniowalne, zero lookahead. Wadą jest *założenie warunkowej niezależności* — każda predykcja ramki jest niezależna od innych, więc nie ma wewnętrznego modelu językowego. Napraw to zewnętrznym LM przez beam search lub shallow fusion.

**Intuicja RNN-T.** Dodaje sieć *predykcyjną*, która osadza historię tokenów i *joiner*, który łączy stan predyktora z ramką enkodera w wspólną dystrybucję na `V+1` (`+1` to null / brak-emitu). Jawnie modeluje warunkową zależność, którą CTC zignorowało. Strumieniowalne, bo każdy krok warunkuje tylko na przeszłych ramkach i przeszłych tokenach.

Zalety: strumieniowalne + wewnętrzny LM. Wadą jest bardziej złożone trening i większe zużycie pamięci (3D lattice strat); RNN-T loss kernels to cała kategoria bibliotek.

**Attention encoder-decoder.** Enkoder (6-32 warstwy transformer) nad log-mel ramkami. Dekoder (6-32 warstwy transformer) cross-attends do wyjść enkodera, by generować tokeny autoregresywnie. Brak ograniczenia wyrównania — attention może patrzeć gdziekolwiek w audio. Non-strumieniowalne, chyba że ograniczysz attention (chunked Whisper-Streaming, 2024).

Zalety: najwyższa jakość na offline ASR, łatwe trenowanie standardowymi narzędziami seq2seq. Wadą jest autoregresyjne opóźnienie proporcjonalne do długości wyjściowej; nie może być strumieniowe bez inżynierii.

### WER: jedna liczba

**Word Error Rate** = `(S + D + I) / N`, gdzie S=substytucje, D=usunięcia, I=wstawienia, N=liczba słów referencyjnych. Odpowiada odległości edycyjnej Levenshteina na poziomie słów, niższy jest lepszy. WER powyżej 20% jest generalnie bezużyteczny; poniżej 5% to ludzka równość dla mowy czytanej. Liczby z 2026 na standardowych benchmarkach:

| Model | LibriSpeech test-clean | LibriSpeech test-other | Rozmiar |
|-------|------------------------|------------------------|---------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

Wszystkie to encoder-decoder lub RNN-T. Czyste systemy CTC (wav2vec 2.0) osiągają około 1.8–2.1% na test-clean.

## Build It

### Step 1: greedy CTC decode

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Dwie reguły: zwijaj kolejne powtórzenia, usuń blanki. Przykład: `a a _ _ a b b _ c` → `a a b c`.

### Step 2: beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

Produkcja używa prefix tree beam search z LM fusion; to jest konceptualny szkielet.

### Step 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Step 4: inferencja przeciwko Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

One-liner dla najsilniejszego ASR ogólnego przeznaczenia w 2026. Działa na GPU 24 GB w ~20× realtime.

### Step 5: strumieniowanie z Parakeet lub wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Strumieniowe ASR potrzebuje chunked attention enkodera i carryover state; użyj biblioteki, która to wspiera (NeMo dla Parakeet, pipeline `transformers` z `chunk_length_s`).

## Zastosowanie

Stack w 2026:

| Sytuacja | Wybierz |
|---------|---------|
| Angielski, offline, max jakość | Whisper-large-v3-turbo |
| Wielojęzyczny, robust | SeamlessM4T v2 |
| Strumieniowanie, niskie opóźnienie | Parakeet-TDT-1.1B lub Riva |
| Edge, mobile, <500 ms opóźnienia | Whisper-Tiny skwantowany lub Moonshine (2024) |
| Długa forma | Whisper z VAD-based chunking (WhisperX) |
| Specyficzny domenowo (medyczny, prawny) | Fine-tune wav2vec 2.0 + domain LM fusion |

## Pułapki, które nadal trafiają do produkcji w 2026

- **Brak VAD.** Zawsze bramkuj z VAD, uruchomienie Whisper na ciszy produkuje halucynacje ("Thanks for watching!").
- **WER na znak vs słowo vs subword.** Raportuj WER na poziomie słowa *po* normalizacji (lowercase, usunięty punctuation).
- **Dryf LID.** Dryf LID, auto LID Whisper myli noisy klipy do języka japońskiego lub walijskiego; wymuś `language="en"` gdy wiesz.
- **Długie klipy bez chunking.** Whisper ma okno 30 sekund. Użyj `chunk_length_s=30, stride=5` dla czegokolwiek dłuższego.

## Ship It

Zapisz jako `outputs/skill-asr-picker.md`. Wybierz model, strategię dekodowania, chunking i LM fusion dla danego celu wdrożenia.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Greedy dekoduje ręcznie stworzone wyjście CTC i oblicza WER względem referencji.
2. **Średnie.** Zaimplementuj poprawnie prefix-tree beam search w Step 2 (uwzględnij regułę blank merge). Porównaj z greedy na 10-przykładowym syntetycznym dataset.
3. **Trudne.** Użyj `whisper-large-v3-turbo` na [LibriSpeech test-clean](https://www.openslr.org/12). Oblicz WER na pierwszych 100 utterances. Porównaj z opublikowanymi liczbami.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| CTC | The blank-token loss | Margines po wszystkich wyrównaniach frame-to-token; non-AR. |
| RNN-T | The streaming loss | CTC + next-token predictor; obsługuje kolejność słów. |
| Attention enc-dec | Whisper-style | Enkoder + cross-attending dekoder; najlepsza offline jakość. |
| WER | The number you report | `(S+D+I)/N` na poziomie słowa. |
| Blank | The emptiness | Specjalny token w CTC sygnalizujący, że "brak emisji w tej ramce". |
| LM fusion | External language model | Dodaj ważone LM log-proby podczas beam search. |
| VAD | The silence gate | Voice activity detector; przycina non-speech, wyłączyć cichy szum. |

## Dalsza lektura

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — paper CTC.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — paper RNN-T.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — kanoniczny paper z 2022; rozszerzenie v3-turbo w 2024.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — lider Open ASR Leaderboard 2026.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — live benchmark ponad 25+ modeli.