# Tłumaczenie maszynowe

> Tłumaczenie to zadanie, które przez trzydzieści lat finansowało badania nad NLP i nadal to robi.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 5 · 10 (Mechanizm Attention), Phase 5 · 04 (GloVe, FastText, Subword)
**Szacowany czas:** ~75 minut

## Problem

Model odczytuje zdanie w jednym języku i produkuje zdanie w innym. Długość się różni. Kolejność słów się różni. Niektóre słowa źródłowe mapują na wiele słów docelowych i odwrotnie. Idiomy odmawiają jedno-do-jednego mapowania. "I miss you" po francusku to "tu me manques" — dosłownie "tobie brakuje mi". Żadne mapowanie na poziomie słów tego nie przetrwa.

Tłumaczenie maszynowe to zadanie, które zmusiło NLP do wynalezienia enkoderów-dekoderów, attention, transformerów i ostatecznie całego paradygmatu LLM. Każdy krok naprzód pojawił się dlatego, że jakość tłumaczenia była mierzalna, a luka między człowiekiem a maszyną była uporczywa.

Ta lekcja pomija lekcję historii i uczy działającego pipeline'u 2026: pretrained multilingual encoder-decoder (NLLB-200 lub mBART), subword tokenization, beam search, ewaluację BLEU i chrF oraz garść trybów awarii, które nadal trafiają do produkcji niewykryte.

## Koncepcja

![Pipeline MT: tokenizacja → enkoding → dekodowanie z attention → detokenizacja](../assets/mt-pipeline.svg)

Nowoczesne MT to transformer encoder-decoder trenowany na równoległym tekście. Encoder odczytuje źródło w tokenizacji jego języka. Decoder generuje cel, jeden subword na raz, używając wyjścia enkodera przez cross-attention (lekcja 10). Dekodowanie używa beam search, żeby uniknąć pułapki greedy-decoding. Wyjście jest detokenizowane, detruecase'owane i oceniane względem referencji.

Trzy operacyjne wybory napędzają jakość MT w realnym świecie.

- **Tokenizer.** SentencePiece BPE trenowany na korpusie mieszanym językowo. Współdzielony słownik między językami to coś, co umożliwia zero-shot pary w NLLB.
- **Rozmiar modelu.** NLLB-200 distilled 600M mieści się na laptopie. NLLB-200 3.3B to opublikowany domyślny produkcyjny. 54.5B to sufit badawczy.
- **Dekodowanie.** Szerokość beam 4-5 dla ogólnej treści. Length penalty, żeby uniknąć zbyt krótkiego wyjścia. Constrained decoding, gdy potrzebujesz spójności terminologii.

## Zbuduj to

### Krok 1: wywołanie pretrained MT

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Trzy rzeczy tu mają znaczenie. `src_lang` mówi tokenizerowi, który skrypt i segmentację zastosować. `forced_bos_token_id` mówi decoderowi, który język generować. Oba to triki specyficzne dla NLLB; mBART i M2M-100 używają własnych konwencji i nie są zamienne.

### Krok 2: BLEU i chrF

BLEU mierzy n-gram overlap między wyjściem a referencją. Cztery rozmiary referencyjnych n-gramów (1-4), geometryczna średnia precyzji, brevity penalty za zbyt krótkie wyjście. Wynik jest w [0, 100]. Powszechnie używany. Frustrujący w interpretacji: 30 BLEU to "używalne"; 40 to "dobre"; 50 to "wyjątkowe"; różnice poniżej 1 BLEU to szum.

chrF mierzy character-level F-score. Bardziej czuły na morfologicznie bogate języki, gdzie BLEU niedoszacowuje dopasowań. Często raportowany obok BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Zawsze używaj `sacrebleu`. Normalizuje tokenizację, więc wyniki są porównywalne między publikacjami. Pisać własną implementację BLEU to sposób na mylące benchmarki.

### Hierarchia ewaluacji trójpoziomowa (2026)

Nowoczesna ewaluacja MT używa trzech komplementarnych rodzin metryk. Shipuj z przynajmniej dwoma.

- **Heuristic** (BLEU, chrF). Szybkie, reference-based, interpretowalne, niewrażliwe na parafrazę. Używaj do legacy porównań i wykrywania regresji.
- **Learned** (COMET, BLEURT, BERTScore). Modele neuronowe trenowane na ludzkim osądzie; porównują semantyczne podobieństwo tłumaczenia do źródła i referencji. COMET ma najwyższą korelację z badaniami MT od 2023 i jest produkcyjnym domyślnym 2026, gdzie jakość ma znaczenie.
- **LLM-as-judge** (reference-free). Promptuj duży model, żeby oceniał tłumaczenia na płynność, adekwatność, ton, kulturową odpowiedniość. GPT-4-as-judge zgadza się z ludźmi w ~80% przypadków, gdy rubryka jest dobrze zaprojektowana. Używaj dla otwartej treści, gdzie nie ma referencji.

Praktyczny stack 2026: `sacrebleu` dla BLEU i chrF, `unbabel-comet` dla COMET i prompted LLM dla końcowego sygnału dla ludzi. Kalibruj każdą metrykę na 50-100 przykładach z etykietami ludzkimi przed zaufaniem jej na danych produkcyjnych.

Metryki reference-free (COMET-QE, BLEURT-QE, LLM-as-judge) pozwalają ewaluować tłumaczenia bez referencji, co ma znaczenie dla długoogonowych par językowych, gdzie referencyjne tłumaczenia nie istnieją.

### Krok 3: co się psuje w produkcji

Działający pipeline powyżej będzie tłumaczył płynnie w 80% przypadków i cicho zawodził w pozostałych 20%. Nazwane tryby awarii:

- **Hallucination.** Model wymyśla treść, której nie było w źródle. Powszechne w nieznanej domenowej terminologii. Symptom: wyjście jest płynne, ale twierdzi fakty, których źródło nie podawało. Mitygacja: constrained decoding na terminach domenowych, human review na regulowanej treści, monitoring wyjścia znacznie dłuższego niż wejście.
- **Off-target generation.** Model tłumaczy na zły język. NLLB jest zaskakująco podatny na to przy rzadkich parach językowych. Mitygacja: weryfikuj `forced_bos_token_id` i zawsze dekoduj z language-ID model check na wyjściu.
- **Terminology drift.** "Sign up" staje się "s'inscrire" w dokumencie 1 i "créer un compte" w dokumencie 2. Dla UI text i user-facing strings spójność ma większe znaczenie niż surowa jakość. Mitygacja: glossary-constrained decoding lub post-edit dictionary.
- **Formality mismatch.** Francuskie "tu" vs "vous", japońskie poziomy grzeczności. Model wybiera tę formę, która była częstsza w treningu. Dla customer-facing treści to zwykle błąd. Mitygacja: prompt prefix z formality tokenem, jeśli model to wspiera, lub fine-tune małego modelu na korpusach tylko formalnych.
- **Length explosion on short input.** Bardzo krótkie zdania wejściowe często produkują zbyt długie tłumaczenia, bo length penalty spada z klifu poniżej ~5 tokenów źródłowych. Mitygacja: twardy max-length cap proporcjonalny do długości źródła.

### Krok 4: fine-tuning dla domeny

Pretrained modele to generaliści. Tłumaczenie prawne, medyczne lub dialogów z gier mierzalnie korzysta z fine-tuningu na domenowych danych równoległych. Przepis nie jest egzotyczny:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

Kilka tysięcy wysokiej jakości równoległych przykładów bije kilkaset tysięcy zaszumionych ze scrapowania sieci. Jakość danych treningowych to pojedyncza największa dźwignia produkcyjna.

## Użyj tego

Produkcyjny stack MT na 2026:

| Przypadek użycia | Polecany punkt wyjścia |
|---------|---------------------------|
| Dowolny-do-dowolnego, 200 języków | `facebook/nllb-200-distilled-600M` (laptop) lub `nllb-200-3.3B` (produkcja) |
| English-centric, wysoka jakość, 50 języków | `facebook/mbart-large-50-many-to-many-mmt` |
| Krótkie uruchomienia, tania inferencja, angielsko-francusko-niemiecko-hiszpański | Helsinki-NLP / Marian models |
| Latency-critical browser-side | ONNX-quantized Marian (~50 MB) |
| Maksymalna jakość, gotowość płacić | GPT-4 / Claude / Gemini z translation prompts |

LLMy od 2026 przewyższają specjalizowane modele MT na kilku parach językowych, szczególnie na idiomatycznej treści i długim kontekście. Tradeoff to koszt per-token i latency. Wybieraj LLM, gdy długość kontekstu, stylistyczna spójność lub domenowa adaptacja przez prompting ma większe znaczenie niż throughput.

## Wyślij to

Zapisz jako `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Ewaluuj wyjście tłumaczenia maszynowego do wysyłki.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Ćwiczenia

1. **Łatwe.** Przetłumacz 5-zdaniowy angielski paragraf na francuski i z powrotem na angielski używając `nllb-200-distilled-600M`. Zmierz, jak blisko round-trip jest oryginału. Powinieneś zobaczyć semantyczne zachowanie z dryfem wyboru słów.
2. **Średnie.** Zaimplementuj language-ID check na tłumaczeniach używając `fasttext lid.176` lub `langdetect`. Zintegruj z MT call, żeby off-target generations były łapane przed zwróceniem.
3. **Trudne.** Fine-tune'uj `nllb-200-distilled-600M` na 5000-parowym korpusie domenowym do wyboru. Zmierz BLEU na held-out set przed i po fine-tuningu. Raportuj, które rodzaje zdań się poprawiły, a które zregresowały.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| BLEU | Translation score | N-gram precision z brevity penalty. [0, 100]. |
| chrF | Character F-score | Character-level F-score. Bardziej czuły dla morfologicznie bogatych języków. |
| NMT | Neural MT | Transformer encoder-decoder trenowany na równoległym tekście. Default od 2017+. |
| NLLB | No Language Left Behind | Rodzina modeli MT Meta obsługująca 200 języków. |
| Constrained decoding | Controlled output | Wymuszenie konkretnych tokenów lub n-gramów, żeby pojawiały się / nie pojawiały się w wyjściu. |
| Hallucination | Wymyślona treść | Wyjście modelu, które nie jest wspierane przez źródło. |

## Dalsza lektura

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — artykuł o NLLB.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — dlaczego `sacrebleu` to jedyny poprawny sposób raportowania BLEU.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — artykuł o chrF.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — praktyczny przewodnik fine-tuningu.