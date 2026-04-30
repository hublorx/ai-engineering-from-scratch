# Tokenizacja pod-słów — BPE, WordPiece, Unigram, SentencePiece

> Tokenizery słów na pojedynczych słowach duszą się na nieznanych słowach. Tokenizery znakowe rozdmuchują długość sekwencji. Tokenizery pod-słów znajdują złoty środek. Każdy nowoczesny LLM jest na jednym z nich zbudowany.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 01 (Przetwarzanie tekstu), Faza 5 · 04 (GloVe / FastText / Pod-słowa)
**Szacowany czas:** ~60 minut

## Problem

Twoja mapa słów ma 50 000 słów. Użytkownik wpisuje „untokenizable". Twój tokenizer zwraca `[UNK]`. Model teraz nie ma żadnego sygnału o tym słowie. Co gorsza, dokument na 90. percentylu w Twoim korpusie ma 40 rzadkich słów, co oznacza 40 bitów utraconej informacji na dokument.

Tokenizacja pod-słów to rozwiązuje. Częste słowa pozostają pojedynczymi tokenami. Rzadkie słowa rozkładają się na znaczące części: `untokenizable` → `un`, `token`, `izable`. Dane treningowe pokrywają wszystko, bo każdy ciąg znaków ostatecznie jest sekwencją bajtów.

Każdy frontowy LLM w 2026 jest zbudowany na jednym z trzech algorytmów (BPE, Unigram, WordPiece), opakowanym w jednej z trzech bibliotek (tiktoken, SentencePiece, HF Tokenizers). Nie możesz wdrożyć modelu językowego bez wybrania jednego.

## Koncepcja

![BPE vs Unigram vs WordPiece, znak po znaku](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Zacznij od słownika na poziomie znaków. Policz każdą parę sąsiadujących znaków. Połącz najczęstszą parę w nowy token. Powtarzaj, aż osiągniesz docelowy rozmiar słownika. Dominujący algorytm: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**BPE na poziomie bajtów.** Ten sam algorytm, ale na surowych bajtach (256 tokenów bazowych) zamiast znaków Unicode. Gwarantuje zero tokenów `[UNK]` — każdy ciąg bajtów da się zakodować. GPT-2 używa 50 257 tokenów (256 bajtów + 50 000 scalonych + 1 specjalny).

**Unigram.** Zacznij od ogromnego słownika. Przypisz każdemu tokenowi prawdopodobieństwo unigramowe. Iteracyjnie przycinaj tokeny, których usunięcie najmniej zwiększa log-likelihood korpusu. Probabilistyczny podczas wnioskowania: może próbkować tokenizacje (użyteczne do augmentacji danych poprzez regularyzację pod-słów). Używany przez T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.** Łącz pary, które maksymalizują likelihood korpusu treningowego zamiast surowej częstotliwości. Używany przez BERT, DistilBERT, ELECTRA.

**SentencePiece vs tiktoken.** SentencePiece to biblioteka, która *trenuje* słowniki (BPE lub Unigram) bezpośrednio na surowym tekście Unicode, kodując spację jako `▁`. tiktoken to szybki *enkoder* OpenAI na predefiniowanych słownikach; nie trenuje.

Zasada kciuka:

- **Trenowanie nowego słownika:** SentencePiece (wielojęzyczny, bez pre-tokenizacji) lub HF Tokenizers.
- **Szybkie wnioskowanie przeciwko słownikowi GPT:** tiktoken (cl100k_base, o200k_base).
- **Oba:** HF Tokenizers — jedna biblioteka, trening + serwowanie.

## Zbuduj to

### Krok 1: BPE od zera

Zobacz `code/main.py`. Pętla:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Trzy fakty, które algorytm koduje. `</w>` oznacza koniec słowa, więc „low" (przyrostek) i „lower" (przedrostek) pozostają różne. Waga częstotliwości sprawia, że pary o wysokiej częstotliwości wygrywają wcześnie. Lista scalonych jest uporządkowana — wnioskowanie stosuje scalenia w kolejności treningu.

### Krok 2: kodowanie za pomocą nauczonych scalonych

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Naiwny O(n·|merges|). Produkcjne implementacje (tiktoken, HF Tokenizers) używają lookupu rangi scalania z kolejkami priorytetowymi i działają w czasie niemal liniowym.

### Krok 3: SentencePiece w praktyce

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Zauważ: nie wymaga pre-tokenizacji, spacja kodowana jako `▁`, `character_coverage` kontroluje, jak agresywnie rzadkie znaki są zachowywane vs mapowane na `<unk>`.

### Krok 4: tiktoken dla słowników kompatybilnych z OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Tylko kodowanie. Szybki (backend Rust). Dokładne dopasowanie do tokenizacji GPT-4/5 do liczenia bajtów, szacowania kosztów, planowania okna kontekstowego.

## Pułapki, które wciąż trafiają do produkcji w 2026

- **Tokenizer drift.** Trenowanie na słowniku A, wdrożenie przeciwko słownikowi B. ID tokenów się różnią, model generuje śmieci. Sprawdź hash `tokenizer.json` w CI.
- **Ambigwacja białych znaków.** BPE „hello" vs „ hello" produkują różne tokeny. Zawsze podawaj `add_special_tokens` i `add_prefix_space` jawnie.
- **Niedotrenowanie wielojęzyczne.** Korpusy zdominowane przez angielski produkują słowniki, które rozbijają nie-łacińskie pisma na 5-10x więcej tokenów. Ten sam prompt kosztuje 5-10x więcej w japońskim/arabskim na GPT-3.5. o200k_base częściowo to naprawiło.
- **Rozbijanie emoji.** Pojedyncze emoji może zająć 5 tokenów. Sprawdź obsługę emoji podczas planowania budżetu kontekstu.

## Użyj tego

Stack na 2026:

| Sytuacja | Wybierz |
|-----------|------|
| Trenowanie monolingwalnego modelu od zera | HF Tokenizers (BPE) |
| Trenowanie wielojęzycznego modelu | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serwowanie API kompatybilnego z OpenAI | tiktoken (`o200k_base` dla GPT-4+) |
| Słownik domenowy (kod, matma, białko) | Trenuj własny BPE na korpusie domenowym, scal ze słownikiem bazowym |
| Wnioskowanie na brzegu, mały model | Unigram (mniejsze słowniki działają lepiej) |

Rozmiar słownika to decyzja skalowania, nie stała. Zgrubna heurystyka: 32k dla <1B parametrów, 50-100k dla 1-10B, 200k+ dla wielojęzycznych/frontowych.

## Wdroży to

Zapisz jako `outputs/skill-tokenizer-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Ćwiczenia

1. **Łatwe.** Trenuj BPE z 500 scaleniami na małym korpusie z `code/main.py`. Koduj trzy słowa z holdout. Ile z nich wygenerowało dokładnie 1 token vs >1 token?
2. **Średnie.** Porównaj liczbę tokenów na 100 zdaniach angielskiej Wikipedii między `cl100k_base`, `o200k_base`, a BPE SentencePiece, który trenujesz z vocab=32k. Raportuj współczynnik kompresji każdego.
3. **Trudne.** Trenuj ten sam korpus z BPE, Unigram i WordPiece. Zmierz downstream accuracy przy użyciu każdego na małym klasyfikatorze sentymentu. Czy wybór ma większy wpływ niż 1 punkt F1?

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Zachłanne łączenie najczęstszych par znaków, aż osiągnięty docelowy rozmiar słownika. |
| Byte-level BPE | Nigdy żadnych nieznanych tokenów | BPE na surowych 256 bajtach; GPT-2 / Llama tego używają. |
| Unigram | Probabilistyczny tokenizer | Przycina z dużego zestawu kandydującego używając log-likelihood; używany przez T5, Gemma. |
| SentencePiece | Ten od białych znaków | Biblioteka treningowa BPE/Unigram na surowym tekście; spacja kodowana jako `▁`. |
| tiktoken | Ten szybki | BPE enkoder OpenAI z backendem Rust na pre-definiowanych słownikach. Bez treningu. |
| Merge list | Magiczne liczby | Uporządkowana lista scalonych `(a, b) → ab`; wnioskowanie stosuje w kolejności. |
| Character coverage | Jak rzadkie to za rzadkie? | Frakcja znaków w korpusie treningowym, którą tokenizer musi pokryć; ~0.9995 typowo. |