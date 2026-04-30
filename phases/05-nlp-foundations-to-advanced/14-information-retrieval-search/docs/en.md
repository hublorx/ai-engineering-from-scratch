# Pobieranie informacji i wyszukiwanie

> BM25 jest precyzyjny, ale kruchy. Dense rzuca szeroką sieć, ale chybia słów kluczowych. Hybrid to domyślne rozwiązanie 2026. Wszystko inne to dostrajanie.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Szacowany czas:** ~75 minut

## Problem

Użytkownik wpisuje "what happens if someone lies to get money" i oczekuje, że znajdzie statut, który to faktycznie obejmuje: "Section 420 IPC." Wyszukiwanie słowne chybia całkowicie (brak wspólnego słownictwa). Wyszukiwanie semantyczne chybia, jeśli embeddingi nie były trenowane na tekstach prawnych. Prawdziwe wyszukiwanie musi obsłużyć oba przypadki.

IR to potok pod każdym systemem RAG, każdym paskiem wyszukiwania, każdym rozmytym wyszukiwaniem na stronach dokumentacji. Architektura 2026, która działa w produkcji, nie jest jedną metodą. Jest łańcuchem komplementarnych metod, z których każda łapie błędy poprzedniej.

Ta lekcja buduje każdy element i nazywa, jakie błędy każdy z nich łapie.

## Koncepcja

![Hybrydowe pobieranie: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

Cztery warstwy. Wybierz te, których potrzebujesz.

1. **Sparse retrieval (BM25).** Szybkie, precyzyjne przy dokładnych dopasowaniach, okropne przy semantyce. Działa na odwróconym indeksie. Poniżej 10ms na zapytanie na milionach dokumentów. Zapewnia referencje statutowe, kody produktów, komunikaty błędów, nazwy własne.
2. **Dense retrieval.** Koduj zapytanie i dokumenty do wektorów. Wyszukiwanie najbliższego sąsiada. Przechwytuje parafrazy i podobieństwo semantyczne. Chybia dokładnych dopasowań słów kluczowych różniących się jednym znakiem. 50-200ms na zapytanie z FAISS lub wektorową bazą danych.
3. **Fuzja.** Scal listy rankingowe z sparse i dense. Reciprocal Rank Fusion (RRF) to łatwy domyślny wybór, ponieważ ignoruje surowe wyniki (które żyją w różnych skalach) i używa tylko pozycji rankingu. Ważona fuzja to opcja, gdy wiesz, że jeden sygnał dominuje w Twojej domenie.
4. **Cross-encoder rerank.** Weź top-30 z fuzji. Uruchom cross-encoder (zapytanie + dokument razem, oceniając każdą parę). Zatrzymaj top-5. Cross-encodery są wolniejsze na parę niż bi-encodery, ale daleko dokładniejsze. Amortyzujesz, uruchamiając je tylko na top-30.

Three-way retrieval (BM25 + dense + learned-sparse jak SPLADE) przewyższa two-way w benchmarkach 2026, ale wymaga infrastruktury dla learned-sparse indexes. Dla większości zespołów, two-way plus cross-encoder rerank to optimum.

## Zbuduj to

### Krok 1: BM25 od zera

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Dwa parametry, które warto znać. `k1=1.5` kontroluje nasycenie term-frequency; wyższy oznacza większą wagę na powtórzenia terminów. `b=0.75` kontroluje normalizację długości; 0 ignoruje długość dokumentu, 1 całkowicie normalizuje. Wartości domyślne to rekomendacje Robertsona z oryginalnego artykułu i rzadko wymagają dostrajania.

### Krok 2: dense retrieval z bi-encoderem

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2-normalizuj embeddingi, żeby iloczyn skalarny równał się cosinusowemu. `all-MiniLM-L6-v2` ma 384 wymiary, jest szybki i wystarczająco dobry dla większości angielskiego pobierania. Dla pracy wielojęzycznej użyj `paraphrase-multilingual-MiniLM-L12-v2`. Dla najwyższej dokładności: `bge-large-en-v1.5` lub `e5-large-v2`.

### Krok 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Stała `k=60` pochodzi z oryginalnego artykułu o RRF. Wyższy `k` spłaszcza wkład różnic w rankingu; niższy `k` sprawia, że top ranki dominują. 60 to opublikowany domyślny i rzadko wymaga dostrajania.

### Krok 4: hybrid search + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

Trzy etapy złożone. BM25 znajduje leksykalne dopasowania. Dense znajduje semantyczne dopasowania. RRF scala dwa rankingi bez potrzeby kalibracji wyników. Cross-encoder ponownie ocenia top-30 używając par zapytanie-dokument razem, co przechwytuje drobną granularność relewancji, którą bi-encoder przeoczył. Zatrzymaj top-5.

### Krok 5: ewaluacja

| Metryka | Znaczenie |
|---------|-----------|
| Recall@k | Z zapytań, gdzie poprawny dokument istnieje, jak często jest w top-k? |
| MRR (Mean Reciprocal Rank) | Średnia z 1/rank pierwszego relewantnego dokumentu. |
| nDCG@k | Uwzględnia gradacje relewancji, nie tylko binarne relevant/not. |

Dla RAG konkretnie, **Recall@k** retrievera to najważniejsza liczba. Twój reader nie może odpowiedzieć, jeśli poprawny pass nie jest w pobranym zestawie.

Wskazówka debugowania: dla zapytań które zawodzą, porównaj sparse i dense rankingi. Jeśli jeden znajduje poprawny dokument, a drugi nie, masz mismatch słownictwa (poprawka: dodaj brakującą połowę) lub dwuznaczność semantyczną (poprawka: lepsze embeddingi lub reranker).

## Użyj tego

Stack 2026:

| Skala | Stack |
|-------|-------|
| 1k-100k dokumentów | BM25 w pamięci + `all-MiniLM-L6-v2` embeddingi + RRF. Bez osobnej bazy danych. |
| 100k-10M dokumentów | FAISS lub pgvector dla dense + Elasticsearch / OpenSearch dla BM25. Uruchom równolegle. |
| 10M+ dokumentów | Qdrant / Weaviate / Vespa / Milvus z hybrydowym wsparciem. Cross-encoder rerank na top-30. |
| Najlepsza jakość na frontier | Three-way (BM25 + dense + SPLADE) + ColBERT late-interaction reranking |

Niezależnie od wyboru, zaplanuj ewaluację. Benchmarkuj recall retrievera przed benchmarkowaniem end-to-end RAG accuracy. Reader nie może naprawić tego, co retriever przeoczył.

### Trudno wywalczone lekcje z produkcyjnego RAG 2026

- **80% błędów RAG sięga do ingestii i chunkowania, nie modelu.** Zespoły spędzają tygodnie na zamienianiu LLM i dostrajaniu promptów, podczas gdy retrieval cicho zwraca zły kontekst co trzecie zapytanie. Napraw chunkowanie pierwsze.
- **Strategia chunkowania ma większe znaczenie niż rozmiar chunku.** Chunki o stałym rozmiarze łamią tabele, kod i zagnieżdżone nagłówki. Sentence-aware to domyślne; semantic lub LLM-based chunking zwraca się dla dokumentacji technicznej i podręczników produktowych.
- **Pattern parent-doc.** Pobieraj małe "child" chunki dla precyzji. Gdy wiele childów z tej samej sekcji nadrzędnej się pojawia, włóż blok nadrzędny, żeby zachować kontekst. To konsekwentnie podnosi jakość odpowiedzi bez ponownego treningu.
- **k_rerank=3 jest zwykle optymalny.** Każdy dodatkowy chunk poza tym dodaje koszt tokenów i opóźnienie generacji bez podnoszenia jakości odpowiedzi. Jeśli k=8 nadal jest lepsze od k=3 dla Ciebie, reranker niedomaga.
- **HyDE / query expansion.** Generuj hipotetyczną odpowiedź z zapytania, embeduj to, pobierz. Zamyka lukę frazeologiczną między krótkimi pytaniami a długimi dokumentami. Darmowy lift precyzji bez treningu.
- **Context budget poniżej 8K tokenów.** Spójne trafienia na tym limicie oznaczają, że próg rerankera jest zbyt luźny.
- **Wersjonuj wszystko.** Prompty, reguły chunkowania, model embeddings, reranker. Każdry dryf cicho łamie jakość odpowiedzi. CI gates na faithfulness, context precision, i unanswered-question rate blokują regresje zanim użytkownicy zobaczą.
- **Three-way retrieval (BM25 + dense + learned-sparse jak SPLADE) przewyższa two-way** w benchmarkach 2026, szczególnie dla zapytań mieszających rzeczowniki własne z semantyką. Wysyłaj, gdy infrastruktura wspiera SPLADE indexes.

Właściwy design retrievera redukuje halucynacje o 70-90% według pomiarów branżowych 2026. Większość zysków wydajności RAG pochodzi z lepszego retrievala, nie fine-tuningu modelu.

## Wyślij to

Zapisz jako `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Ćwiczenia

1. **Łatwe.** Zaimplementuj `hybrid_search` powyżej na korpusie 500 dokumentów. Przetestuj 20 zapytań. Porównaj recall at 5 między BM25-only, dense-only i hybrid.
2. **Średnie.** Dodaj obliczanie MRR. Dla każdego testowego zapytania ze znanym poprawnym dokumentem, znajdź rank poprawnego doc w BM25, dense i hybrid rankingach. Raportuj MRR dla każdego.
3. **Trudne.** Dostrój dense encoder na swojej domenie używając MultipleNegativesRankingLoss (Sentence Transformers). Zbuduj training set z 500 par query-document. Porównaj recall przed i po fine-tuningu.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| BM25 | Wyszukiwanie słowne | Okapi BM25. Ocenia dokumenty przez term frequency, IDF i długość. |
| Dense retrieval | Wyszukiwanie wektorowe | Koduj zapytanie + doc do wektorów, znajdź najbliższych sąsiadów. |
| Bi-encoder | Model embeddingowy | Koduje zapytanie i doc niezależnie. Szybki w czasie zapytania. |
| Cross-encoder | Model rerankera | Koduje zapytanie + doc razem. Wolny, ale dokładny. |
| RRF | Fuzja rankingu | Połącz dwa rankingi sumując `1/(k + rank)`. |
| Recall@k | Metryka pobierania | Ułamek zapytań, gdzie relewantny doc jest w top-k. |

## Dalsze czytanie

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — definitywne opracowanie BM25.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, kanoniczny bi-encoder.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — learned-sparse retriever, który zamyka lukę z dense.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — artykuł o RRF.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — late-interaction retrieval.