# Strategie chunkingu dla RAG

> Konfiguracja chunkingu wpływa na jakość retrieve'u tak samo bardzo, jak wybór modelu embeddingowego (Vectara NAACL 2025). Jeśli chunking jest błędny, żadne rerankingi cię nie uratują.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 14 (Information Retrieval), Faza 5 · 22 (Embedding Models)
**Szacowany czas:** ~60 minut

## Problem

Wrzuciłeś 50-stronicowy kontrakt do systemu RAG. Użytkownik pyta: "Jaka jest klauzula rozwiązania umowy?" Retriever zwraca stronę tytułową. Dlaczego? Ponieważ model był trenowany na chunkach 512 tokenów, a klauzula rozwiązania znajduje się 20 stron dalej, podzielona przez podział strony, bez lokalnych słów kluczowych łączących ją z zapytaniem.

Rozwiązanie to nie "kup lepszy model embeddingowy". Rozwiązanie to chunking. Jak duży? Overlap? Gdzie dzielić? Z otaczającym kontekstem?

Benchmarki z lutego 2026 pokazują zaskakujące wyniki:

- Badanie Vectara 2026: rekursywny chunking 512-tokenowy pokonał semantic chunking 69% → 54% accuracy.
- SPLADE + Mistral-8B na Natural Questions: overlap nie przyniósł żadnego mierzalnego korzyści.
- **Context cliff** (klif kontekstowy): jakość odpowiedzi gwałtownie spada wokół 2.500 tokenów kontekstu.

"Oczywista" odpowiedź (semantic chunking, 20% overlap, 1000 tokenów) często jest błędna. Ta lekcja buduje intuicję dla sześciu strategii i mówi ci, kiedy po którą sięgnąć.

## Koncepcja

![Sześć strategii chunkingu zwizualizowanych na jednym fragmencie](../assets/chunking.svg)

**Fixed chunking.** Dziel co N znaków lub tokenów. Najprostszy baseline. Łamie w połowie zdania. Dobra kompresja, zła koherencja.

**Recursive.** LangChain's `RecursiveCharacterTextSplitter`. Próbuj dzielić na `\n\n` najpierw, potem `\n`, potem `.`, potem spację. Czysto się cofa. Domyślny w 2026.

**Semantic.** Embeduj każde zdanie. Oblicz cosine similarity między sąsiednimi zdaniami. Dziel tam, gdzie similarity spada poniżej progu. Zachowuje spójność tematyczną. Wolniejsze; czasami produkuje mikroskopijne fragmenty 40 tokenów, które szkodzą retrieve'owi.

**Sentence.** Dziel na granicach zdań. Jedno zdanie na chunk lub okno N zdań. Dorównuje semantic chunking do ~5k tokenów za ułamek kosztu.

**Parent-document.** Przechowuj małe child chunki do retrieve'u *oraz* większy parent chunk do kontekstu. Retrieve przez child; zwracaj parent. Delikatnie się degraduje: złe child chunki nadal zwracają rozsądne parent chunki.

**Późny chunking (2024).** Embeduj cały dokument na poziomie tokenów najpierw, potem pooluj token embeddings w chunk embeddings. Zachowuje kontekst między-chunkami. Działa z long-context embedderami (BGE-M3, Jina v3). Wyższy koszt obliczeniowy.

**Contextual retrieval (Anthropic, 2024).** Poprzedź każdy chunk LLM-generated summary jego pozycji w dokumencie ("This chunk is section 3.2 of the termination clauses..."). 35-50% poprawa retrieve'u w benchmarku Anthropic. Drogi do indeksowania.

### Reguła, która bije każdy domyślny wybór

Dopasuj rozmiar chunka do typu zapytania:

| Typ zapytania | Rozmiar chunka |
|---------------|----------------|
| Factoid ("kim jest CEO?") | 256-512 tokenów |
| Analityczne / multi-hop | 512-1024 tokenów |
| Zrozumienie całej sekcji | 1024-2048 tokenów |

NVIDIA 2026 benchmark. Chunk powinien być wystarczająco duży, żeby zawierać odpowiedź plus lokalny kontekst, wystarczająco mały, żeby retriever's top-K skupiał się na odpowiedzi, a nie na szumie kontekstowym.

## Zbuduj

### Krok 1: fixed i recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Krok 2: semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Dostrój `threshold` do swojej domeny. Za wysoki → fragmenty. Za niski → jeden gigantyczny chunk.

### Krok 3: parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Kluczowy insight: dedupe parents. Wiele children może mapować do tego samego parent; zwracanie wszystkich zmarnowałoby kontekst.

### Krok 4: contextual retrieval (wzorzec Anthropic)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Indeksuj zcontextualizowane chunki. W czasie zapytania, retrieval korzysta z dodatkowego sygnału otoczenia.

### Krok 5: evaluate

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Zawsze benchmarkuj. "Najlepsza" strategia dla twojego corpusu może nie pasować do żadnego blogposta.

## Pułapki

- **Chunking ewaluowany tylko na zapytaniach factoid.** Multi-hop queries ujawniają bardzo różnych zwycięzców. Użyj eval setu stratyfikowanego pod kątem typu zapytania.
- **Semantic bez minimalnego rozmiaru.** Produkuje fragmenty 40-tokenowe, które szkodzą retrieve'owi. Zawsze egzekwuj `min_tokens`.
- **Overlap jako cargo cult.** Badania z 2026 find overlap często nie przynosi żadnej korzyści i podwaja koszt indeksu. Mierz, nie zakładaj.
- **Brak egzekwowania min/max.** Chunki 5 tokenów lub 5000 tokenów oba psują retrieval. Clampuj.
- **Chunking między dokumentami.** Nigdy nie pozwalaj, żeby chunk obejmował dwa dokumenty. Zawsze chunkuj per-doc, potem merguj.

## Użyj tego

Stack 2026:

| Sytuacja | Strategia |
|----------|-----------|
| Pierwszy build, nieznany corpus | Recursive, 512 tokenów, bez overlap |
| Factoid QA | Recursive, 256-512 tokenów |
| Analityczne / multi-hop | Recursive, 512-1024 tokenów + parent-document |
| Ciężkie cross-reference (kontrakty, artykuły) | Late chunking lub contextual retrieval |
| Konwersacyjne / dialog corpus | Turn-level chunks + speaker metadata |
| Krótkie wypowiedzi (tweety, recenzje) | Jeden dokument = jeden chunk |

Zacznij od recursive 512. Zmierz recall@5 na 50-query eval set. Dostrój stamtąd.

## Wyślij to

Zapisz jako `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Ćwiczenia

1. **Łatwe.** Chunkuj jeden 20-stronicowy dokument z fixed(512, 0), recursive(512, 0) i recursive(512, 100). Porównaj liczbę chunków i jakość granic.
2. **Średnie.** Zbuduj 30-query eval set na 5 dokumentach. Zmierz recall@5 dla recursive, semantic i parent-document. Kto wygrywa? Czy pasuje to do blogpostów?
3. **Trudne.** Zaimplementuj contextual retrieval. Zmierz poprawę MRR względem baseline recursive. Raportuj koszt indeksu (wywołania LLM) vs zysk accuracy.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Chunk | Kawałek dokumentu | Jednostka sub-dokumentowa, która jest embedowana, indeksowana i retrievowana. |
| Overlap | Margines bezpieczeństwa | N tokenów dzielonych między sąsiednimi chunkami; często bezużyteczne w benchmarkach 2026. |
| Semantic chunking | Sprytny chunking | Dziel tam, gdzie similarity embeddingów sąsiednich zdań spada. |
| Parent-document | Dwupoziomowy retrieval | Retrieve małych children, zwracaj większe parents. |
| Late chunking | Chunk po embedowaniu | Embed pełny dokument na poziomie tokenów, pooluj w wektory chunków. |
| Contextual retrieval | Trik Anthropic | LLM-generated summary dodawane przed każdym chunkiem przed indeksowaniem. |
| Context cliff | Ściana 2500 tokenów | Spadek jakości obserwowany wokół 2.5k tokenów kontekstu w RAG (sty 2026). |

---

**Zastosowane poprawki:**

| # | Lokalizacja | Przed | Po |
|---|-------------|-------|-----|
| 1 | Nagłówek sekcji | `### Zbuduj to` | `### Zbuduj` |
| 2 | Koncepcja, late chunking | `cross-chunk context` | `kontekst między-chunkami` |
| 3 | Koncepcja, context cliff | `**Context cliff**: jakość` | `**Context cliff** (klif kontekstowy): jakość` |
| 4 | Koncepcja, late chunking | `**Late chunking (2024).**` | `**Późny chunking (2024).**` |
| 5 | Pułapki, semantic | `**Semantic chunking without minimalny rozmiar**` | `**Semantic bez minimalnego rozmiaru**` |
| 6 | Pułapki, cross-doc | `**Cross-doc chunking**` | `**Chunking między dokumentami**` |