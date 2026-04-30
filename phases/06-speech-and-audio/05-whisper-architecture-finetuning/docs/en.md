# Whisper — Architektura i Fine-Tuning

> Whisper to transformer encoder-decoder z oknem 30-sekundowym, trenowany na 680 000 godzin wielojęzycznych, słabo nadzorowanych par audio-tekst. Jedna architektura, wiele zadań, odporna na 99 języków. Referencyjny ASR 2026.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Szacowany czas:** ~75 minut

## Problem

Whisper, wydany przez OpenAI we września 2022, był pierwszym modelem ASR, który trafił do masowej sprzedaży: wklejasz audio, otrzymujesz tekst, 99 języków, odporny na szum, działa na laptopie. Do 2024 OpenAI wydał warianty Large-v3 i Turbo; do 2026 Whisper jest domyślnym baseline'em dla wszystkiego — od transkrypcji podcastów po asystentów głosowych po napisy YouTube.

Ale Whisper nie jest pipeline'em, który można traktować jako czarną skrzynkę przez wieczność. Przesunięcie domenowe go zabija — żargon techniczny, akcenty mówców, nazwy własne, krótkie klipy, cisza. Musisz wiedzieć:

1. Co tak naprawdę jest w środku.
2. Jak poprawnie podawać mu audio w chunkach, streamingowe lub długie.
3. Kiedy fine-tunować i jak.

## Koncepcja

![Whisper encoder-decoder, zadania, chunked inference, fine-tune](../assets/whisper.svg)

**Architektura.** Standardowy transformer encoder-decoder.

- Input: 30-sekundowe log-mel spectrogram, 80 mels, 10 ms hop → 3000 klatek. Klipy krótsze są dopełniane zerami, dłuższe są chunkowane.
- Encoder: conv-downsample (stride 2) + `N` transformer blocks. Dla Large-v3: 32 warstwy, dim 1280, 20 głów.
- Decoder: `N` transformer blocks z causal self-attn + cross-attn do outputu encodera. Ten sam rozmiar co encoder.
- Output: tokeny BPE ze słownikiem 51 865 tokenów.

Large-v3 ma 1,55B parametrów. Turbo używa 4-warstwowego decodera (z 32), zmniejszając latency 8× z <1% spadkiem WER.

**Format promptu.** Whisper jest modelem multitask sterowanym przez specjalne tokeny w decoder prompt:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — tag języka; wymusza zachowanie tłumaczenie-vs-transkrypcja.
- `<|transcribe|>` lub `<|translate|>` — tłumacz output po angielsku z dowolnego języka input, albo wiernie.
- `<|notimestamps|>` — pomijaj timestampy na poziomie słów (szybciej).

Prompt jest tym, co pozwala jednemu modelowi wykonywać wiele zadań. Zmień `<|en|>` na `<|fr|>` i transkrybuje po francusku.

**30-sekundowe okno.** Wszystko jest przypięte do 30 sekund. Dłuższe klipy wymagają chunkowania; krótsze są dopełniane. Okna nie są natywnie streamowane — dlatego istnieją WhisperX, Whisper-Streaming i faster-whisper.

**Log-mel normalizacja.** `(log_mel - mean) / std` gdzie statystyki pochodzą z własnego korpusu treningowego Whisper. Musisz użyć preprocessingu Whisper (`whisper.audio.log_mel_spectrogram`), nie `librosa.feature.melspectrogram`.

### Warianty w 2026

| Wariant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|-------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Fine-tuning

Kanoniczny workflow w 2026:

1. Zbierz 10–100 godzin audio z docelowej domeny z wyrównanymi transkryptami.
2. Uruchom `transformers.Seq2SeqTrainer` z callbackiem `generate_with_loss`.
3. Parametric-efficient: LoRA na `q_proj`, `k_proj`, `v_proj` warstw attn zmniejsza pamięć GPU 4× z <0.3 kosztem WER.
4. Zamroź encoder jeśli masz <10 godzin. Tunuj tylko decoder.
5. Użyj własnego tokenizera i formatu promptu Whisper; nigdy nie zamieniaj tokenizera.

Wyniki społeczności: fine-tune Medium na 20 godzinach dyktowania medycznego obniża WER z 12% do 4.5% na słownictwie medycznym. Fine-tune Turbo na 4 godzinach islandzkiego obniża WER z 18% do 6%.

## Zbuduj to

### Krok 1: uruchom Whisper out of the box

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Kluczowe domyślne wartości które zawsze powinieneś nadpisać: `temperature=0.0` (sampling domyślnie do fallback chain 0.0 → 0.2 → 0.4 …), `condition_on_previous_text=False` (zapobiega kaskadowemu problemowi halucynacji), i `no_speech_threshold=0.6` (wykrywanie ciszy).

### Krok 2: chunked long-form

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX dodaje (1) bramkowanie Silero VAD, (2) wyrównanie na poziomie słów przez wav2vec 2.0, (3) diarizację przez `pyannote.audio`. Workhorse 2026 do produkcyjnej transkrypcji.

### Krok 3: fine-tune z LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

Potem standardowa pętla Trainera. Checkpoint co 1000 kroków. Ewaluuj z WER na held-out.

### Krok 4: sprawdź czego każda warstwa się uczy

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Wizualizuj heatmapą — zobaczysz diagonal alignment gdy decoder steps skanują przez encoder frames. Ta diagonala to pojęcie Whisper o timestampach słów.

## Użyj tego

Stack 2026:

| Sytuacja | Wybierz |
|----------|---------|
| Ogólny angielski, offline | Large-v3-turbo przez `whisperx` |
| Mobile / edge | Whisper-Tiny quantized (int8) lub Moonshine |
| Wielojęzyczny long-form | Large-v3 przez `whisperx` + diarization |
| Low-resource language | Fine-tune Medium lub Turbo z LoRA |
| Streaming (latency 2 s) | Whisper-Streaming lub Parakeet-TDT |
| Timestamps na poziomie słów | WhisperX (forced alignment przez wav2vec 2.0) |

`faster-whisper` (backend CTranslate2) to najszybszy runtime inferencyjny CPU+GPU w 2026 — 4× szybszy niż vanilla z identycznym outputem.

## Pułapki które nadal są w wysyłce w 2026

- **Zhalucynowany tekst na ciszy.** Whisper trenowany na captionach zawiera "Thanks for watching!", "Subscribe!", teksty piosenek. Zawsze bramkuj przez VAD przed wywołaniem.
- **Kaskada `condition_on_previous_text`.** Jedna halucynacja zanieczyszcza kolejne okna. Ustaw `False` chyba że potrzebujesz płynności między chunkami.
- **Dopełnianie krótkich klipów.** Klip 2-sekundowy dopełniony do 30 sekund może halucynować w ciszy na końcu. Użyj `pad=False` lub bramkuj przez VAD.
- **Złe statystyki mel.** Używanie mels librosy zamiast Whisper produkuje prawie losowy output. Użyj `whisper.audio.log_mel_spectrogram`.

## Wyślij to

Zapisz jako `outputs/skill-whisper-tuner.md`. Zaprojektuj pipeline fine-tune lub inference Whisper dla danej domeny.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Tokenizuje Whisper-style prompt, oblicza budżety kształtów dekodowania i drukuje schedule chunków dla 10-minutowego klipu.
2. **Średnie.** Zainstaluj `faster-whisper`, transkrybuj 10-minutowy podcast, porównaj WER z ludzkim transkryptem. Wypróbuj `language="auto"` vs wymuszone `language="en"`.
3. **Trudne.** Używając HF `datasets`, wybierz język z którym Whisper sobie nie radzi (np. Urdu), fine-tune Medium z LoRA przez 2 epoki na 2 godzinach i raportuj delta WER.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| 30-sec window | Limit Whisper | Twardy cap input; chunkuj dłuższe audio. |
| SOT | Start-of-transcript | `<|startoftranscript|>` kick-startuje decoder prompt. |
| Timestamps token | Wyrównanie temporalne | Co 0.02 s offset to specjalny token w 51k vocab. |
| Turbo | Szybki wariant | 4 warstwy decodera, 8× szybszy, <1% regresja WER. |
| WhisperX | Wrapper long-form | VAD + Whisper + wyrównanie wav2vec + diarization. |
| LoRA fine-tune | Efficient tuning | Dodaj low-rank adapters do attn; trenuj ~0.3% parametrów. |
| Hallucination | Cicha porażka | Whisper produkuje płynny angielski z szumu/ciszy. |

## Dalsze czytanie

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — oryginalna architektura i recipe treningowy.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4-warstwowy decoder, 8× speedup.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — long-form, word-aligned, diarized.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — backed CTranslate2, 4× szybszy.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — kanoniczny LoRA / full-FT walkthrough.