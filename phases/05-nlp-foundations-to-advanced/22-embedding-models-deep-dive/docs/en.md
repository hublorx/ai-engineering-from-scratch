# Modele osadzania — dogłębna analiza 2026

> Word2Vec dawał wektor na słowo. Współczesne modele osadzania dają wektor na akapit, wielojęzyczny, z widokami sparse, dense i multi-vector, dopasowane do rozmiaru indeksu. Wybierz źle, a Twój RAG pobiera złą treść.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 03 (Word2Vec), Faza 5 · 14 (Information Retrieval)
**Szacowany czas:** ~60 minut

## Problem

Twój system RAG pobiera zły akapit w 40% przypadków. Winowajcą rzadko jest baza wektorowa lub prompt. Jest nim model osadzania.

Wybieranie osadzania w 2026 roku oznacza wybór w pięciu wymiarach:

1. **Dense vs sparse vs multi-vector.** Jeden wektor na akapit, lub jeden na token, lub sparse ważona torba słów.
2. **Pokrycie językowe.** Monolingwalne modele angielskie wciąż wygrywają w zadaniach tylko angielskich. Wielojęzyczne modele wygrywają, gdy korpusy są mieszane.
3. **Długość kontekstu.** 512 tokenów vs 8 192 vs 32 768 — a rzeczywista efektywna pojemność to często 60-70% reklamowanego maksimum.
4. **Budżet wymiarów.** 3 072 floaty przy pełnej precyzji = 12 KB na wektor. Przy 100M wektorów, storage to 1 300 $/miesiąc. Obcinanie Matryoshka zmniejsza to 4×.
5. **Open vs hosted.** Open-weight oznacza, że kontrolujesz stos i dane. Hosted oznacza, że wymieniasz kontrolę na zawsze-najnowsze.

Ta lekcja nazywa kompromisy, abyś mógł wybierać na podstawie dowodów, a nie tego, co było popularne w ostatnim kwartale.

## Koncepcja

![Dense, sparse i multi-vector embeddings](../assets/embedding-modes.svg)

**Dense embeddings.** Jeden wektor na akapit (zwykle 384-3 072 wymiary). Cosine similarity rankuje akapity według semantycznego podobieństwa. OpenAI `text-embedding-3-large`, BGE-M3 tryb dense, Voyage-3. Domyślny wybór.

**Sparse embeddings.** Styl SPLADE. Transformer przewiduje wagę dla każdego tokena ze słownika, a następnie zeruje większość z nich. Wynikiem jest sparse wektor o rozmiarze |vocab|. Przechwytuje dopasowanie leksykalne (jak BM25), ale z nauczonymi wagami terminów. Silny przy zapytaniach obfitujących w słowa kluczowe.

**Multi-vector (late interaction).** ColBERTv2, Jina-ColBERT. Jeden wektor na token. Scoring z MaxSim: dla każdego tokena zapytania znajdź najbardziej podobny token dokumentu, zsumuj wyniki. Droższe w przechowywaniu i scoringu, ale wygrywa przy długich zapytaniach i korpusach specyficznych dla domeny.

**BGE-M3: wszystko trzy naraz.** Pojedynczy model wyprowadza gęste, sparse i multi-vector reprezentacje jednocześnie. Każda może być odpytywana niezależnie; wyniki łączą się przez ważoną sumę. Dom domyślny 2026, gdy chcesz elastyczności z jednego checkpointa.

**Matryoshka Representation Learning.** Trenowana tak, że pierwsze N wymiarów wektora tworzy użyteczną samodzielną reprezentację. Obetnij wektor 1 536-dim do 256 dim i zapłać ~1% dokładności za 6× oszczędności storage. Wspierane przez OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Leaderboard MTEB opowiada część historii

Massive Text Embedding Benchmark — 56 zadań w 8 typach zadań przy starcie (2022), rozszerzony do 100+ zadań w MTEB v2. Na początku 2026, Gemini Embedding 2 przoduje w retrieval (67.71 MTEB-R). Cohere embed-v4 prowadzi general (65.2 MTEB). BGE-M3 prowadzi open-weight multilingual (63.0). Leaderboard jest konieczny, ale niewystarczający — zawsze testuj benchmark na swojej domenie.

### Wzorzec trójwarstwowy

| Przypadek użycia | Wzorzec |
|----------|---------|
| Szybki pierwszy przebieg | Dense bi-encoder (BGE-M3, text-3-small) |
| Wzmocnienie recall | Sparse (SPLADE, BGE-M3 sparse) + RRF fuse |
| Precyzja na top-50 | Multi-vector (ColBERTv2) lub cross-encoder reranker |

Większość produkcyjnych stacków używa wszystkich trzech.

## Zbuduj to

### Krok 1: baseline — dense embeddings z Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` sprawia, że dot product równa się cosine similarity. Zawsze to ustawiaj.

### Krok 2: Obcinanie Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Normalizuj ponownie po obcięciu. Nomic v1.5, OpenAI text-3 i Voyage-4 są trenowane tak, że to jest bezstratne dla pierwszych kilku poziomów. Modele non-Matryoshka (oryginalny Sentence-BERT) degradują gwałtownie przy obcięciu.

### Krok 3: BGE-M3 multi-funkcjonalność

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Trzy indeksy, jedno wywołanie inference. Fuzja wyników:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Dostrój wagi na swojej domenie.

### Krok 4: MTEB eval na niestandardowym zadaniu

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Uruchom swoje kandydackie modele na *reprezentatywnym* podzbiorze. Nie ufaj samej pozycji w leaderboardzie — twoja domena ma znaczenie.

### Krok 5: ręcznie napisany cosine od zera

Zobacz `code/main.py`. Averaged Hashing Trick embeddings (tylko stdlib). Niekonkurencyjne z transformer embeddings, ale pokazuje kształt: tokenizuj → wektoryzuj → normalizuj → dot product.

## Pułapki

- **Ten sam model dla zapytania i dokumentu.** Niektóre modele (Voyage, Jina-ColBERT) używają asymetrycznego kodowania — zapytanie i dokument przechodzą przez różne ścieżki. Zawsze sprawdź kartę modelu.
- **Brakujący prefix.** Modele `bge-*` wymagają dodania `"Represent this sentence for searching relevant passages: "` do zapytań. Różnica 3-5 punktów recall, jeśli zapomnisz.
- **Przetrimowanie Matryoshki.** 1 536 → 256 jest zwykle bezpieczne. 1 536 → 64 nie jest. Waliduj na swoim eval set.
- **Obcinanie kontekstu.** Większość modeli dyskretnie obcina inputy przekraczające ich max length. Długie dokumenty wymagają chunkowania (zobacz lekcję 23).
- **Ignorowanie latency tail.** Wyniki MTEB ukrywają p99 latency. Model 600M może pokonać model 335M o 2 punkty, ale kosztować 3× więcej per zapytanie.

## Użyj tego

Stack 2026:

| Sytuacja | Wybierz |
|-----------|--------|
| Tylko angielski, szybki, API | `text-embedding-3-large` lub `voyage-3-large` |
| Open-weight, angielski | `BAAI/bge-large-en-v1.5` |
| Open-weight, wielojęzyczny | `BAAI/bge-m3` lub `Qwen3-Embedding-8B` |
| Długi kontekst (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| Tylko CPU deployment | Nomic Embed v2 (137M params, MoE) |
| Ograniczony storage | Matryoshka-truncated + int8 quantization |
| Zapytania obfitujące w słowa kluczowe | Dodaj SPLADE sparse, RRF-fuse z dense |

Wzorzec 2026: zacznij od BGE-M3 lub text-3-large, ewaluuj na swojej domenie z MTEB, wymień jeśli model specyficzny dla domeny wygrywa o więcej niż 3 punkty.

## Wyślij to

Zapisz jako `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Ćwiczenia

1. **Łatwe.** Zakoduj 100 zdań z `bge-small-en-v1.5` przy pełnym wymiarze (384), a następnie przy Matryoshka 128. Zmierz spadek MRR na 10 zapytaniach.
2. **Średnie.** Porównaj BGE-M3 dense, sparse i colbert na 500 akapitach z twojej domeny. Który wygrywa na recall@10? Czy fuzja RRF bije najlepszy pojedynczy tryb?
3. **Trudne.** Uruchom MTEB na trzech kandydackich modelach na twoich top-2 zadaniach domenowych. Raportuj wynik MTEB, p99 latency na batchu 100 zapytań i $/1M zapytań. Wybierz tego Pareto-optymalnego.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Dense embedding | Wektor | Jeden wektor o stałym rozmiarze na tekst. Cosine similarity do rankingu. |
| Sparse embedding | Learned BM25 | Jedna waga na token słownika; głównie zera; trenowane end-to-end. |
| Multi-vector | ColBERT-style | Jeden wektor na token; MaxSim scoring; większy indeks, lepszy recall. |
| Matryoshka | Russian doll trick | Pierwsze N wymiarów to prawidłowy mniejszy embedding sam w sobie. |
| MTEB | Benchmark | Massive Text Embedding Benchmark — 56 zadań przy starcie, 100+ w v2. |
| BEIR | Retrieval benchmark | 18 zero-shot retrieval tasks; często cytowane dla cross-domain robustness. |
| Asymmetric encoding | Ścieżka query ≠ doc | Model używa różnych projekcji dla zapytań i dokumentów. |

## Dalsza lektura

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — artykuł o bi-encoderze.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — artykuł o leaderboardzie.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — jednolity model trzytrybowy.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — cel trenowania dimension-ladder.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — late interaction w produkcji.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — live rankings.