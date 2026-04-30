# Systemy Question Answering (QA)

> Trzy systemy ukształtowały nowoczesne QA. Ekstrakcyjne znajdowały fragmenty. Wspomagane retrieval zakorzeniły je w dokumentach. Generatywne produkowały odpowiedzi. Każdy nowoczesny asystent AI to mieszanka tych trzech.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Szacowany czas:** ~75 minut

## Problem

Użytkownik wpisuje „Kiedy zadebiutował pierwszy iPhone?" i oczekuje „29 czerwca 2007". Nie „Historia Apple jest długa i zróżnicowana." Nie „2007" w izolacji bez zdania. Bezpośrednia, ugruntowana, poprawna odpowiedź.

Trzy architektury dominowały w QA przez ostatnią dekadę.

- **QA ekstrakcyjne.** Mając pytanie i passzę, o której wiadomo, że zawiera odpowiedź, znajdź indeksy początkowe i końcowe fragmentu odpowiedzi w passzie. SQuAD to kanoniczny benchmark.
- **QA typu open-domain.** Passza nie jest podana. Najpierw pobierz odpowiednią passzę, a następnie wyodrębnij lub wygeneruj odpowiedź. To fundament każdego pipeline'u RAG dzisiaj.
- **QA generatywne / closed-book.** Duży model językowy odpowiada z swojej pamięci parametrycznej. Bez retrieval. Najszybsze przy wnioskowaniu, najmniej wiarygodne w kwestii faktów.

Trend w 2026 to hybryda: pobierz najlepsze kilka passz, następnie promptuj model generatywny, żeby odpowiedział na podstawie tych passz. To RAG, a lekcja 14 omawia połowę związaną z retrieval w szczegółach. Ta lekcja buduje połowę QA.

## Koncepcja

![Architektury QA: ekstrakcyjna, wspomagana retrieval, generatywna](../assets/qa.svg)

**Ekstrakcyjne.** Koduj pytanie i passzę razem z transformerem (rodzina BERT). Trenuj dwa heady, które przewidują indeksy tokenów początkowego i końcowego odpowiedzi. Loss to cross-entropy pozycji ważnych. Wynik to fragment z passzy. Nigdy nie halucynuje (z konstrukcji), nie obsługuje pytań, na które passza nie może odpowiedzieć (z konstrukcji).

**Wspomagane retrieval (RAG).** Dwa etapy. Pierwszy, retriever znajduje top-`k` passz z korpusu. Drugi, reader (ekstrakcyjny lub generatywny) produkuje odpowiedź używając tych passz. Podział retriever-reader pozwala na niezależne trenowanie i ewaluację. Nowoczesny RAG często dodaje reranker między nimi.

**Generatywne.** Decoder-only LLM (GPT, Claude, Llama) odpowiada z wyuczonych wag. Bez etapu retrieval. Świetne na powszechną wiedzę, katastrofalne na rzadkie lub recent facts. Współczynnik halucynacji jest odwrotnie skorelowany z częstością faktu w danych pretreningowych.

## Zbuduj to

### Krok 1: QA ekstrakcyjne z pretrained modelem

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` jest trenowany na SQuAD 2.0, który zawiera niemożliwe do odpowiedzenia pytania. Domyślnie, pipeline `question-answering` zwraca najwyżej punktowany fragment nawet gdy null score wygrywa — *nie* zwraca automatycznie pustej odpowiedzi. Żeby uzyskać explicit zachowanie „no answer", przekaż `handle_impossible_answer=True` do wywołania pipeline'u: pipeline wtedy zwraca pustą odpowiedź tylko gdy null score przewyższa każdy wynik fragmentu. Zawsze sprawdzaj pole `score`.

### Krok 2: pipeline wspomagany retrieval (szkic)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Pipeline dwuetapowy. Dense retriever (Sentence-BERT) znajduje relevantne passze przez semantic similarity. Extractive reader (RoBERTa-SQuAD) wyciąga fragment odpowiedzi z połączonych top passz. Działa na małych korpusach. Dla korpusu z milionem dokumentów, użyj FAISS lub wektorowej bazy danych.

### Krok 3: generatywne z RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Wzorzec promptu ma znaczenie. Explicitne powiedzenie modelowi, żeby zakorzenił odpowiedź w kontekście i zwracał „I don't know" gdy kontekst jest niewystarczający, obcina współczynnik halucynacji o 40-60% w porównaniu do naiwnego promptowania. Bardziej elaborate wzorce dodają cytaty, confidence scores i structured extraction.

### Krok 4: ewaluacja odzwierciedlająca realny świat

SQuAD używa **Exact Match (EM)** i **token-level F1**. EM to ścisłe dopasowanie po normalizacji (lowercase, strip punctuation, remove articles) — albo predykcja pasuje dokładnie albo dostaje 0. F1 jest obliczane przez overlap tokenów między predykcją a referencją i daje częściowy credit. Oba niedoceniają paraphras: „June 29, 2007" vs „June 29th, 2007" typowo dostaje 0 EM (ordinal łamie normalizację) ale wciąż zarabia substancjalny F1 z overlapping tokens.

Dla produkcyjnego QA:

- **Answer accuracy** (LLM-judged lub human-judged, bo metrics nie łapią semantic equivalence).
- **Citation accuracy.** Czy cytowana passza faktycznie wspiera odpowiedź? Trywialne do automatycznego sprawdzenia string match między wygenerowanymi cytatami a pobranymi passzami.
- **Refusal calibration.** Gdy odpowiedź nie jest w pobranych passzach, czy system poprawnie mówi „I don't know"? Mierz false confidence rate.
- **Retrieval recall.** Przed ewaluacją readera, zmierz czy retriever wprowadził właściwą passzę do top-`k`. Reader nie naprawi brakującej passzy.

### RAGAS: framework ewaluacyjny produkcyjnej 2026

`RAGAS` jest purpose-built dla systemów RAG i jest shipping default w 2026. Ocenia cztery wymiary bez wymagania gold references:

- **Faithfulness.** Czy każde twierdzenie w odpowiedzi pochodzi z pobranego kontekstu? Mierzone przez NLI-based entailment. Twój primary hallucination metric.
- **Answer relevance.** Czy odpowiedź adresuje pytanie? Mierzone przez generowanie hypothetical questions z odpowiedzi i porównywanie do realnego pytania.
- **Context precision.** Z pobranych chunków, jaka frakcja była faktycznie relevant? Low precision = szum w promptcie.
- **Context recall.** Czy pobrany zbiór zawierał wszystkie potrzebne informacje? Low recall = reader nie może odnieść sukcesu.

Reference-free scoring pozwala ewaluować na live production traffic bez curated gold answers. Warstwa LLM-as-judge na wierzchu dla open-ended questions gdzie exact-match metrics są bezużyteczne.

`pip install ragas`. Podłącz retriever + reader. Dostajesz cztery skalary na zapytanie. Alertuj na regresje.

## Użyj tego

Stack 2026.

| Przypadek użycia | Polecane |
|---------|-------------|
| Dana passza, znajdź fragment odpowiedzi | `deepset/roberta-base-squad2` |
| Nad ustalonym korpusem, closed-book nieakceptowalne | RAG: dense retriever + LLM reader |
| Real-time nad dokument store | RAG z hybrid (BM25 + dense) retriever + reranker (lesson 14) |
| Conversational QA (pytania follow-up) | LLM z conversation history + RAG na każdy turn |
| Wysoce faktyczne, regulowane domeny | Extractive nad authoritative corpus; nigdy same generative |

QA ekstrakcyjne jest niemodne w 2026 bo RAG z LLM-ami obsługuje więcej przypadków. Wciąż shipped w kontekstach gdzie wymagany jest literal quotation: prawnicze badania, regulatory compliance, narzędzia audytowe.

## Wyślij to

Zapisz jako `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Ćwiczenia

1. **Łatwe.** Skonfiguruj pipeline SQuAD extractive powyżej na 10 Wikipedia passzach. Ręcznie stwórz 10 pytań. Zmierz jak często odpowiedź jest poprawna. Powinieneś zobaczyć 7-9 poprawnych jeśli passze i pytania są czyste.
2. **Średnie.** Dodaj refusal classifier. Gdy top retrieval score jest poniżej progu (powiedzmy 0.3 cosine), zwróć „I don't know" zamiast wywoływać readera. Dostrój próg na held-out set.
3. **Trudne.** Zbuduj pipeline RAG nad korpusem 10 000 dokumentów do wyboru. Zaimplementuj hybrid retrieval (BM25 + dense) z RRF fusion (zobacz lekcja 14). Zmierz answer accuracy z i bez kroku hybrid. Zdokumentuj które typy pytań korzystają najbardziej.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Extractive QA | Znajdź fragment odpowiedzi | Przewiduj indeksy początkowe i końcowe odpowiedzi w ramach danej passzy. |
| Open-domain QA | QA nad korpusem | Brak danej passzy; musisz retrieve potem odpowiedzieć. |
| RAG | Retrieve then generate | Retrieval-augmented generation. Pipeline retriever + reader. |
| SQuAD | Kanoniczny benchmark | Stanford Question Answering Dataset. Metryki EM + F1. |
| Hallucination | Zmyślona odpowiedź | Wyjście readera nie wspierane przez pobrany kontekst. |
| Refusal calibration | Wiedzieć kiedy się zamknąć | System poprawnie mówi „I don't know" gdy nie może odpowiedzieć. |

## Dalsze czytanie

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — the benchmark paper.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, the canonical dense retriever for QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — the paper that named RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — comprehensive RAG survey.