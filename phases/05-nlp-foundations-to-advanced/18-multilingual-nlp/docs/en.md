# Wielojęzyczne NLP

> Jeden model, ponad 100 języków, zero danych treningowych dla większości z nich. Cross-lingual transfer to praktyczny cud lat 2020.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## The Problem

Angielski ma miliardy oznaczonych przykładów. Urdu ma tysiące. Maithili prawie nic. Każdy praktyczny system NLP, który obsługuje globalną publiczność, musi radzić sobie z długim ogonem języków, gdzie dane treningowe specyficzne dla zadania nie istnieją.

Wielojęzyczne modele rozwiązują to, trenując jeden model na wielu językach jednocześnie. Wspólna reprezentacja pozwala modelowi przenosić umiejętności nabyte w językach o dużych zasobach na języki o małych zasobach. Dostrój model do analizy sentymentu po angielsku, a produkuje on zaskakująco dobre predykcje sentymentu po urdu out of the box. To jest zero-shot cross-lingual transfer i zmieniło to sposób, w jaki NLP trafia do świata.

Ta lekcja wymienia kompromisy, kanoniczne modele i jedną decyzję, która sprawia kłopoty zespołom nowym w wielojęzycznej pracy: wybór języka źródłowego do transferu.

## The Concept

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**Shared vocabulary.** Wielojęzyczne modele wykorzystują tokenizer SentencePiece lub WordPiece trenowany na tekście ze wszystkich języków docelowych. Słownik jest współdzielony: ta sama jednostka subword reprezentuje ten sam morfem w powiązanych językach. `anti-` w angielskim i włoskim otrzymuje ten sam token.

**Shared representation.** Transformer wstępnie trenowany na maskowanym modelowaniu językowym w wielu językach uczy się, że semantycznie podobne zdania w różnych językach produkują podobne ukryte stany. mBERT, XLM-R i NLLB to wykazują. Embeddingi dla "cat" po angielsku grupują się blisko "chat" po francusku i "gato" po hiszpańsku, podobnie jak embeddingi pełnych zdań.

**Zero-shot transfer.** Dostrój model na oznaczonych danych w jednym języku (zwykle angielskim). Podczas wnioskowania uruchom go na dowolnym innym języku obsługiwanym przez model. Etykiety w języku docelowym nie są potrzebne. Wyniki są silne dla typologicznie powiązanych języków i słabsze dla odległych.

**Few-shot fine-tuning.** Dodaj 100-500 oznaczonych przykładów w języku docelowym. Dokładność skacze do 95-98% angielskiego baseline'a w zadaniach klasyfikacji. To jest najbardziej opłacalny mechanizm w wielojęzycznym NLP.

## The models

| Model | Year | Coverage | Notes |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | Trained on Wikipedia. First practical multilingual LM. Weak on low-resource. |
| XLM-R | 2019 | 100 languages | Trained on CommonCrawl (much larger than Wikipedia). Sets the cross-lingual baseline. Base 270M, Large 550M. |
| XLM-V | 2023 | 100 languages | XLM-R with 1M-token vocabulary (vs 250k). Better on low-resource. |
| mT5 | 2020 | 101 languages | T5 architecture for multilingual generation. |
| NLLB-200 | 2022 | 200 languages | Meta's translation model; includes 55 low-resource languages. |
| BLOOM | 2022 | 46 languages + 13 programming | Open 176B LLM trained multilingually. |
| Aya-23 | 2024 | 23 languages | Cohere's multilingual LLM. Strong on Arabic, Hindi, Swahili. |

Pick by use case. Classification works well with XLM-R-base as the sane default. Generation tasks call for mT5 or NLLB depending on translation vs open generation. LLM-style work pairs with Aya-23 lub Claude przy użyciu jawnego wielojęzycznego promptowania.

## The source-language decision (2026 research)

Większość zespołów domyślnie wybiera angielski jako źródło dostrajania. Najnowsze badania (2026) pokazują, że to często błąd.

Podobieństwo językowe lepiej przewiduje jakość transferu niż surowy rozmiar korpusu. Dla celów słowiańskich, niemiecki lub rosyjski często pokonują angielski. Dla celów indyjskich, hindi często pokonuje angielski. Metryka podobieństwa **qWALS** (2026, oparta na cechach World Atlas of Language Structures) to kwantyfikuje. **LANGRANK** (Lin et al., ACL 2019) to oddzielna, wcześniejsza metoda, która rankinguje kandydackie języki źródłowe na podstawie kombinacji podobieństwa językowego, rozmiaru korpusu i pokrewieństwa genetycznego.

Praktyczna zasada: jeśli twój język docelowy ma typologicznie bliskiego krewniaka o dużych zasobach, najpierw spróbuj dostroić na nim, a potem porównaj z dostrajaniem po angielsku.

## Build It

### Step 1: zero-shot cross-lingual classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

Jeden model, trzy języki, ten sam API. XLM-R trenowany na danych NLI dobrze transferuje do klasyfikacji przez entailment trick.

### Step 2: multilingual embedding space

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Tłumaczenia lądują blisko w przestrzeni embeddingów. Inne angielskie zdanie ląduje dalej. To jest to, co sprawia, że cross-lingual retrieval, clustering i podobieństwo działają.

### Step 3: few-shot fine-tuning strategy

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

Dla 100-500 przykładów w języku docelowym, `num_train_epochs=5` i `learning_rate=2e-5` to bezpieczne domyślne wartości. Wyższe learning rates powodują, że wielojęzyczne wyrównanie się załamuje i otrzymujesz model tylko po angielsku.

## Evaluation that actually works

- **Dokładność per-język na held-out sets.** Nie agregowana. Agregat ukrywa długi ogon.
- **Benchmarkuj przeciwko monolingwalnemu baseline'owi.** Dla języków z wystarczającą ilością danych, monolingwalny model wytrenowany od zera czasem pokonuje wielojęzyczny. Testuj.
- **Testy na poziomie encji.** Nazwane encje w języku docelowym. Wielojęzyczne modele często mają słabą tokenizację dla pism dalekich od łacińskiego.
- **Cross-lingual consistency.** To samo znaczenie w dwóch językach powinno produkować tę samą predykcję. Mierz różnicę.

## Use It

The 2026 stack:

| Zadanie | Rekomendowane |
|-----|-------------|
| Klasyfikacja, 100 języków | XLM-R-base (~270M) fine-tuned |
| Zero-shot text classification | `joeddav/xlm-roberta-large-xnli` |
| Wielojęzyczne sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Tłumaczenie, 200 języków | `facebook/nllb-200-distilled-600M` (see lesson 11) |
| Wielojęzyczna generacja | Claude, GPT-4, Aya-23, mT5-XXL |
| Low-resource language NLP | XLM-V lub domenowo-specyficzny fine-tune na powiązanym języku o dużych zasobach |

Zawsze planuj budżet na fine-tuning w języku docelowym, jeśli wydajność ma znaczenie. Zero-shot to punkt wyjścia, nie ostateczna odpowiedź.

### The tokenization tax (co idzie źle dla języków o małych zasobach)

Wielojęzyczne modele współdzielą jeden tokenizer na wszystkie swoje języki. Ten słownik jest trenowany na korpusie zdominowanym przez angielski, francuski, hiszpański, chiński, niemiecki. Dla każdego języka poza dominującym zestawem, trzy podatki kumulują się po cichu:

- **Fertility tax.** Tekst w języku o małych zasobach tokenizuje się do znacznie większej liczby tokenów na słowo niż po angielsku. Zdanie w hindi może potrzebować 3-5x więcej tokenów niż równoważne zdanie po angielsku. Te 3-5x zjada twój context window, efektywność treningu i latency.
- **Variant recovery tax.** Każda literówka, wariant z dykrytykiem, niezgodność normalizacji Unicode lub wariant wielkości liter staje się cold-startem niepowiązanej sekwencji w przestrzeni embeddingów. Model nie może nauczyć się odpowiedników ortograficznych, które native speaker bierze za oczywiste.
- **Capacity spillover tax.** Podatki 1 i 2 konsumują context positions, głębokość warstw i wymiary embeddingów. To, co pozostaje dla rzeczywistego rozumowania, jest systematycznie mniejsze niż to, co język o dużych zasobach otrzymuje z tego samego modelu.

Praktyczny symptom: twój model trenuje normalnie na hindi, krzywa loss wygląda dobrze, eval perplexity wygląda rozsądnie, a produkcyjne outputs są subtelnie złe. Morfologia załamuje się w połowie zdania. Rzadkie odmiany pozostają nieodwracalne. **Nie możesz data-scale'ować się z popsutego tokenizera.**

Środki łagodzące: wybierz tokenizer z dobrą pokrycią dla twojego języka docelowego (1M-token vocabulary XLM-V to bezpośrednia poprawka); weryfikuj tokenization fertility na held-out target text przed treningiem; użyj byte-level fallback (SentencePiece `byte_fallback=True`, GPT-2-style byte-level BPE) dla naprawdę long-tail scripts, żeby nic nigdy nie było OOV.

## Ship It

Save as `outputs/skill-multilingual-picker.md`:

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Ćwiczenia

1. **Łatwe.** Uruchom zero-shot classification pipeline na 10 zdaniach na język w angielskim, francuskim, hindi i arabskim. Zgłoś dokładność dla każdego. Powinieneś zobaczyć silny francuski, przyzwoity hindi, zmienny arabski.
2. **Średnie.** Użyj `paraphrase-multilingual-MiniLM-L12-v2` do zbudowania cross-lingual retrievara na małym mixed-language corpusie. Query po angielsku, retrieve documents w dowolnym języku. Zmierz recall@5.
3. **Trudne.** Porównaj English-source i Hindi-source fine-tuning dla hindi classification task. Użyj 500 target-language examples dla few-shot fine-tuningu w obu reżimach. Zgłoś, który source produkuje lepszą hindi accuracy i o ile. To jest LANGRANK thesis w miniaturze.

## Key Terms

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Multilingual model | Jeden model, wiele języków | Współdzielony słownik i parametry między językami. |
| Cross-lingual transfer | Trenuj na jednym języku, uruchom na innym | Dostrój na źródle, ewaluuj na celu bez etykiet w języku docelowym. |
| Zero-shot | Brak etykiet w języku docelowym | Transfer bez fine-tuningu na języku docelowym. |
| Few-shot | Małe etykiety w docelowym | 100-500 przykładów w języku docelowym użytych do fine-tuningu. |
| mBERT | Pierwszy wielojęzyczny LM | 104-języczny BERT wstępnie trenowany na Wikipedii. |
| XLM-R | Standardowy cross-lingual baseline | 100-języczny RoBERTa wstępnie trenowany na CommonCrawl. |
| NLLB | Meta's 200-języczne MT | No Language Left Behind. Zawiera 55 języków o małych zasobach. |

## Dalsze czytanie

- Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale — artykuł o XLM-R.
- Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT? — artykuł analityczny, który zapoczątkował linię badań nad cross-lingual transfer.
- Costa-jussà et al. (2022). No Language Left Behind — artykuł o NLLB-200.
- Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model — Aya, wielojęzyczny LLM Cohere.
- Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026) — artykuł o qWALS / LANGRANK source language.