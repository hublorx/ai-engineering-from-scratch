# T5, BART — Modele Encoder-Decoder

> Enkodery rozumieją. Dekodery generują. Połącz je i dostajesz model zbudowany dla zadań input → output: tłumaczenie, streszczenie, przepisywanie, transkrypcja.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 05 (Full Transformer), Faza 7 · 06 (BERT), Faza 7 · 07 (GPT)
**Czas:** ~45 minut

## Problem

Decoder-only GPT i encoder-only BERT każdy zrzuca architekturę z 2017 dla innego celu. Ale wiele zadań jest naturalnie input-output:

- Tłumaczenie: English → French.
- Streszczenie: artykuł 5000 tokenów → streszczenie 200 tokenów.
- Rozpoznawanie mowy: audio tokens → text tokens.
- Strukturalna ekstrakcja: proza → JSON.

Dla tych, encoder-decoder pasuje najczystej. Encoder produkuje gęstą reprezentację źródła. Decoder generuje output, cross-attending do tej reprezentacji na każdym kroku. Trening to shift-by-one po stronie outputu. Ta sama strata co GPT, tylko warunkowana na encoder output.

Dwa paper'y zdefiniowały nowoczesny playbook:

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." Każde NLP task przeprowadzony jako text-in, text-out. Pojedyncza architektura, pojedynczy vocabulary, pojedyncza strata. Pretrained na masked span prediction (uszkodzone spans w input, dekoduj je w output).
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." Denoising autoencoder: uszkadzaj input na wiele sposobów (shuffle, mask, delete, rotate), poproś decoder o rekonstrukcję oryginału.

W 2026 format encoder-decoder żyje tam, gdzie struktura inputu ma znaczenie:

- Whisper (speech → text).
- Stack tłumaczeniowy Google.
- Niektóre modele code-completion / repair, które mają distinct context-and-edit structures.
- Flan-T5 i warianty dla structured reasoning tasks.

Decoder-only wygrał spotlight, ale encoder-decoder nigdy nie odszedł.

## Koncepcja

![Encoder-decoder z cross-attention](../assets/encoder-decoder.svg)

### Pętla forward

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

Kluczowe: encoder uruchamia się raz per input. Decoder uruchamia się autoregresywnie ale cross-attends do *tego samego* encoder output na każdym kroku. Cacheowanie encoder output to darmowy speedup dla długich inputów.

### T5 pretraining — span corruption

Wybierz losowe spans inputu (średnia długość 3 tokeny, 15% total). Zamień każdy span unikalnym sentinel: `<extra_id_0>`, `<extra_id_1>`, etc. Decoder outputuje tylko uszkodzone spans z ich sentinel prefix:

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

Tańszy sygnał niż predykcja całej sekwencji. Konkurencyjny z MLM (BERT) i prefix-LM (UniLM) w ablacji paperu T5.

### BART pretraining — multi-noise denoising

BART próbuje pięciu funkcji noising:

1. Token masking.
2. Token deletion.
3. Text infilling (mask span, decoder wstawia poprawną długość).
4. Sentence permutation.
5. Document rotation.

Łączenie text infilling + sentence permutation dało najlepsze downstream numbers. Decoder zawsze rekonstruuje oryginał. Output BART to pełna sekwencja, nie tylko uszkodzone spans — więc pretraining compute jest wyższy niż T5.

### Inferencja

Ta sama autoregressive generation co GPT. Greedy / beam / top-p sampling apply. Beam search (width 4–5) to standard dla tłumaczenia i streszczania, bo rozkład outputu jest węższy niż chat.

### Kiedy wybrać każdy wariant w 2026

| Zadanie | Encoder-decoder? | Dlaczego |
|---------|------------------|----------|
| Tłumaczenie | Tak, zwykle | Jasna sekwencja źródłowa; fixed output distribution; beam search działa |
| Speech-to-text | Tak (Whisper) | Input modality różni się od output; encoder kształtuje audio features |
| Chat / reasoning | Nie, decoder-only | Brak persistent "input" — konwersacja to sekwencja |
| Code completion | Zwykle nie | Decoder-only z long context wygrywa; code models jak Qwen 2.5 Coder są decoder-only |
| Streszczenie | Oba działają | BART, PEGASUS biją wcześniejsze decoder-only baselines; nowoczesne decoder-only LLM je dorównują |
| Strukturalna ekstrakcja | Oba | T5 jest czyste bo "text → text" absorbuje dowolny format outputu |

Trend od ~2022: decoder-only przejmuje zadania, które encoder-decoder wcześniej miał, bo (a) instruction-tuned decoder-only LLM uogólnia się do wszystkiego przez prompting, (b) jedna architektura skaluje się łatwiej niż dwie, (c) RLHF zakłada decoder. Encoder-decoder trzyma się tam, gdzie input modality różni się (speech, images) albo gdzie beam search quality ma znaczenie.

## Zbuduj to

Zobacz `code/main.py`. Implementujemy T5-style span corruption dla toy corpus — najbardziej użyteczny kawałek tej lekcji, bo pojawia się w każdym encoder-decoder pretraining recipe od tamtego czasu.

### Krok 1: span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Format celu to konwencja T5: `<sent0> span0 <sent1> span1 ...`. Uszkodzony input przeplata niezmienione tokeny z sentinel tokens na lokalizacjach span.

### Krok 2: zweryfikuj round-trip

Mając uszkodzony input i target, zrekonstruuj oryginalne zdanie. Jeśli twoja corruption jest odwracalna, forward pass jest well-defined. To jest sanity check — prawdziwy trening nigdy tego nie robi, ale test jest tani i łapie off-by-one bugs w twoim span bookkeeping.

### Krok 3: BART noising

Pięć funkcji: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. Złóż dwie z nich i pokaż wynik.

## Użyj tego

HuggingFace reference:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

Sztuczka T5: nazwa zadania idzie do input text. Ten sam model obsługuje dziesiątki zadań bo każde zadanie to text-in, text-out. W 2026 ten pattern został uogólniony przez instruction-tuned decoder-only models, ale T5 go spisał pierwszy.

## Wyślij to

Zobacz `outputs/skill-seq2seq-picker.md`. Skill wybiera między encoder-decoder a decoder-only dla nowego zadania przy given input-output structure, latency i quality targets.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`, zastosuj span corruption do 30-tokenowego zdania, zweryfikuj, że concatenating non-sentinel source tokens z decoded target spans reprodukuje oryginał.
2. **Średnie.** Zaimplementuj BART `text_infill` noise: zamień losowe spans jednym `<mask>` tokenem, a decoder musi wywnioskować poprawną długość span plus contents. Pokaż jeden przykład.
3. **Trudne.** Fine-tune `flan-t5-small` na tiny English → pig-Latin corpus (200 pairs). Zmierz BLEU na held-out 50-pair set. Porównaj z fine-tuning `Llama-3.2-1B` na tych samych danych z tym samym compute.

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|--------------------------|
| Encoder-decoder | "Seq2seq transformer" | Dwa stosy: bidirectional encoder dla inputu, causal decoder z cross-attention dla outputu. |
| Cross-attention | "Gdzie source mówi do target" | Decoder's Q × encoder's K/V. jedyne miejsce gdzie encoder information wchodzi do decodera. |
| Span corruption | "Trick pretraining T5" | Zamień losowe spans sentinel tokens; decoder outputuje spans. |
| Denoising objective | "Gra BART" | Zastosuj funkcję noise do inputu, trenuj decoder do rekonstrukcji czystej sekwencji. |
| Sentinel token | "Placeholder `<extra_id_N>`" | Specjalne tokeny które tagują uszkodzone spans w source i re-tagują je w target. |
| Flan | "Instruction-tuned T5" | T5 fine-tuned na >1,800 zadań; sprawił że encoder-decoder jest konkurencyjny przy instruction-following. |
| Beam search | "Strategia dekodowania" | Trzymaj top-k partial sequences na każdym kroku; standard dla tłumaczenia/streszczania. |
| Teacher forcing | "Input w czasie treningu" | Podczas treningu podawaj prawdziwy poprzedni output token do decodera, nie sampled one. |

## Dalsze Czytanie

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, kanoniczny encoder-decoder 2026.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — implementacja referencyjna.