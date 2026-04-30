# Streszczanie tekstu

> Systemy ekstrakcyjne mówią ci, co dokument powiedział. Systemy abstrakcyjne mówią ci, co autor miał na myśli. Różne zadania, różne pułapki.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 02 (BoW + TF-IDF), Faza 5 · 11 (Machine Translation)
**Szacowany czas:** ~75 minut

## Problem

Artykuł informacyjny o 2000 słowach trafia do twojego feeda. Potrzebujesz 120 słów, które go opiszą. Możesz albo wybrać trzy najważniejsze zdania z artykułu (ekstrakcyjne), albo przepisać treść własnymi słowami (abstrakcyjne). Oba nazywają się streszczaniem. To są całkowicie różne problemy.

Streszczanie ekstrakcyjne to problem rankingowy. Ocenasz każde zdanie, zwracasz top-`k`. Wynik zawsze jest gramatyczny, bo pochodzi wprost ze źródła. Ryzyko polega na tym, że możesz przeoczyć treść rozproszoną w całym artykule.

Streszczanie abstrakcyjne to problem generacyjny. Transformer produkuje nowy tekst warunkowany na podstawie danych wejściowych. Wynik jest płynny i kompresyjny, ale może halucynować fakty, których nie było w źródle. Ryzyko polega na pewnym fabricowaniu.

Ta lekcja buduje oba podejścia, wraz z trybem awarii, jaki każde z nich posiada.

## Koncepcja

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Ekstrakcyjne.** Traktuj artykuł jako graf, gdzie węzłami są zdania, a krawędziami podobieństwa. Uruchom PageRank (lub coś podobnego) na grafie, aby ocenić zdania na podstawie tego, jak połączone są z wszystkim innym. Zdania z najwyższą punktacją stanowią podsumowanie. Kanoniczna implementacja to **TextRank** (Mihalcea and Tarau, 2004).

**Abstrakcyjne.** Dostrój transformer encoder-decoder (BART, T5, Pegasus) na parach dokument-podsumowanie. Podczas inference, model czyta dokument i generuje podsumowanie token po tokenie poprzez cross-attention. Pegasus w szczególności używa pretraining objective gap-sentence, co czyni go doskonałym w streszczaniu bez dużego dostrajania.

Ewaluacja za pomocą **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation). ROUGE-1 i ROUGE-2 oceniają overlap unigramów i bigramów. ROUGE-L ocenia longest common subsequence. Wyższy wynik jest lepszy, ale 40 ROUGE-L to "dobry" wynik, a 50 to "wyjątkowy." Każda praca raportuje wszystkie trzy. Użyj pakietu `rouge-score`.

## Zbuduj to

### Krok 1: TextRank (ekstrakcyjny)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

Dwie rzeczy warte nazwania. Funkcja podobieństwa używa log-normalized word overlap, co jest oryginalnym wariantem TextRank. Cosine TF-IDF vectors też działa. Damping factor 0.85 i liczba iteracji to domyślne wartości PageRank.

### Krok 2: abstrakcyjny z BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN jest dostrojony na korpusie CNN/DailyMail. Produkuje podsumowania w stylu newsowym out of the box. Dla innych domen (dokumenty naukowe, dialog, prawo), użyj odpowiedniego Pegasus checkpoint lub dostrój na swoich danych docelowych.

### Krok 3: ewaluacja ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Zawsze używaj stemming. Bez niego "running" i "run" liczą się jako różne słowa, a ROUGE niedoszacowuje.

### Poza ROUGE (ewaluacja streszczania 2026)

ROUGE był dominującą metryką streszczania przez dwadzieścia lat i sam w sobie jest niewystarczający w 2026. Dużej skali meta-analiza prac NLG pokazała:

- **BERTScore** (similarity kontekstowych embeddingów) zyskał popularność przez 2023 i jest teraz raportowany obok ROUGE w większości prac o streszczaniu.
- **BARTScore** traktuje ewaluację jako generację: ocenia podsumowanie na podstawie tego, jak bardzo prawdopodobne jest, że pretrained BART przypisze mu przy danym źródle.
- **MoverScore** (Earth Mover's Distance na kontekstowych embeddingach) osiągnął top spot w benchmarkach streszczania w 2025, ponieważ capture semantic overlap lepiej niż ROUGE.
- **FactCC** i **QA-based faithfulness** były popularne w 2021-2023, teraz często zastępowane przez **G-Eval** (łańcuch promptów GPT-4, który ocenia spójność, consistency, fluency, relevance z chain-of-thought reasoning).
- **G-Eval** i podobne podejścia LLM-judge pasują do ludzkiego osądu ~80% czasu, gdy rubryki są dobrze zaprojektowane.

Production recommendation: raportuj ROUGE-L dla legacy comparison, BERTScore dla semantic overlap, G-Eval dla spójności i faktuarności. Kalibruj na 50-100 podsumowaniach oznaczonych przez ludzi.

### Krok 4: problem faktuarności

Abstrakcyjne podsumowania są podatne na halucynacje. Streszczenia ekstrakcyjne mają znacznie niższe ryzyko halucynacji, bo wynik jest liftowany wprost ze źródła, choć mogą nadal wprowadzać w błąd, jeśli zdania źródłowe są decontextualized, outdated, lub cytowane poza kolejnością. To jest największy powód, dla którego systemy produkcyjne nadal preferują metody ekstrakcyjne dla treści compliance-adjacent.

Typy halucynacji do nazwania:

- **Entity swap.** Źródło mówi "John Smith." Podsumowanie mówi "John Brown."
- **Number drift.** Źródło mówi "25,000." Podsumowanie mówi "25 million."
- **Polarity flip.** Źródło mówi "odrzucił ofertę." Podsumowanie mówi "przyjął ofertę."
- **Fact invention.** Źródło nie wspomina o CEO. Podsumowanie mówi, że CEO approve'ował.

Podejścia ewaluacyjne, które działają:

- **FactCC.** Binarny klasyfikator trenowany na entailment między zdaniem źródłowym a zdaniem podsumowania. Przewiduje factual/not-factual.
- **QA-based factuality.** Zadaj modelowi QA pytania, których odpowiedzi są w źródle. Jeśli podsumowanie wspiera inne odpowiedzi, flaguj.
- **Entity-level F1.** Porównaj named entities w źródle vs podsumowaniu. Entities obecne tylko w podsumowaniu są suspect.

Dla wszystkiego user-facing, gdzie faktuarność ma znaczenie (news, medycyna, prawo, finanse), ekstrakcyjne jest bezpieczniejszym domyślnym wyborem. Abstrakcyjne potrzebuje factuality check w pętli.

## Użyj tego

Stack 2026:

| Przypadek użycia | Polecane |
|-----------------|----------|
| Newsy, 3-5 sentence summary, angielski | `facebook/bart-large-cnn` |
| Dokumenty naukowe | `google/pegasus-pubmed` lub dostrojony T5 |
| Multi-document, long-form | Dowolny LLM z 32k+ context, prompted |
| Streszczanie dialogów | `philschmid/bart-large-cnn-samsum` |
| Ekstrakcyjne, niskie ryzyko halucynacji by design | TextRank lub `sumy`'s LSA / LexRank |

LLMy z długim kontekstem często pokonują specjalizowane modele w 2026, gdy compute nie jest ograniczeniem. Tradeoff to koszt i reproducibility; specjalizowane modele dają bardziej spójne wyniki.

## Wyślij to

Zapisz jako `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Ćwiczenia

1. **Łatwe.** Uruchom TextRank na 5 artykułach informacyjnych. Porównaj top-3 zdania do referencyjnego podsumowania. Zmierz ROUGE-L. Powinieneś zobaczyć 30-45 ROUGE-L na artykułach w stylu CNN/DailyMail.
2. **Średnie.** Zaimplementuj entity-level factuality: wyekstruuj named entities ze źródła i podsumowania (spaCy), oblicz recall source entities w podsumowaniu i precision summary entities przeciwko źródłu. Wysoka precision i niski recall oznaczają bezpieczne ale terse; niska precision oznacza halucynowane entities.
3. **Trudne.** Porównaj BART-large-CNN przeciwko LLM (Claude lub GPT-4) na 50 artykułach CNN/DailyMail. Raportuj ROUGE-L, factuality (przez entity F1), i koszt per summary. Udokumentuj, gdzie każdy wygrywa.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Extractive | Pick sentences | Zwracaj zdania wprost ze źródła. Nigdy nie halucynuje. |
| Abstractive | Rewrite | Generuj nowy tekst warunkowany na źródle. Może halucynować. |
| ROUGE | Summary metric | N-gram / LCS overlap między outputem systemu a referencją. |
| TextRank | Graph-based extractive | PageRank over sentence similarity graph. |
| Factuality | Is it right | Czy twierdzenia podsumowania są wspierane przez źródło. |
| Hallucination | Made-up content | Treść w podsumowaniu, której źródło nie wspiera. |

## Dalsze czytanie

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — kanoniczny artykuł o ekstrakcyjnym podejściu.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — artykuł o BART.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus i gap-sentence objective.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — artykuł o ROUGE.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — artykuł o krajobrazie faktuarności.