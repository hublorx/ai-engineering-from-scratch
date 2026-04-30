# Ekstrakcja relacji i budowa grafu wiedzy

> NER znalazł encje. Entity linking je zakotwiczył. Ekstrakcja relacji znajduje krawędzie między nimi. Graf wiedzy to suma węzłów, krawędzi i ich pochodzenia.

**Typ:** Zbuduj
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 06 (NER), Faza 5 · 25 (Entity Linking)
**Szacowany czas:** ~60 minut

## Problem

Analityk czyta: "Tim Cook became CEO of Apple in 2011." Cztery fakty:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Ekstrakcja relacji (RE) zamienia tekst swobodny w ustrukturyzowane trójki `(subject, relation, object)`. Agregacja w całym korpusie daje graf wiedzy. Agregacja i zapytania dają podłoże do wnioskowania dla RAG, analityki lub audytów zgodności.

Problem 2026: LLM-e ekstrahują relacje entuzjastycznie. Zbyt entuzjastycznie. Halucynują trójki, których tekst źródłowy nie wspiera. Bez pochodzenia nie da się odróżnić prawdziwych trójek od prawdopodobnej fikcji. Odpowiedź 2026 to potoki AEVS-style anchor-and-verify.

## Koncepcja

![Tekst → trójki → graf wiedzy](../assets/relation-extraction.svg)

**Forma trójki.** `(subject_entity, relation_type, object_entity)`. Relacje pochodzą z zamkniętej ontologii (właściwości Wikidata, FIBO, UMLS) lub zbioru otwartego (OpenIE-style, wszystko jest dozwolone).

**Trzy podejścia do ekstrakcji.**

1. **Oparte na regułach / wzorcach.** Wzorce Hearst: "X such as Y" → `(Y, isA, X)`. Plus ręcznie tworzone regex-y. Kruche, precyzyjne, wytłumaczalne.
2. **Supervized classifier.** Mając dwa menciony encji w zdaniu, przewiduj relację ze stałego zestawu. Trenowany na TACRED, ACE, KBP. Standard 2015–2022.
3. **Generatywny LLM.** Wymuś na modelu emisję trójek. Działa out of the box. Wymaga pochodzenia, bo inaczej halucynuje prawdopodobnie wyglądający śmieć.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Obecny framework mitigacji halucynacji:

- **Anchor.** Zidentyfikuj każdy span encji i span frazy relacji z dokładnymi pozycjami.
- **Extract.** Generuj trójki powiązane ze spanami zakotwiczonymi.
- **Verify.** Dopasuj każdy element trójki z powrotem do tekstu źródłowego; odrzuć wszystko co nie jest wspierane.
- **Supplement.** Przejście pokryciowe upewnia się, że żaden zakotwiczony span nie został pominięty.

Halucynacje spadają dramatycznie. Wymaga więcej compute, ale jest audytowalny.

**Kompromis open-vs-closed.**

- **Zamknięta ontologia.** Stała lista właściwości (np. 11 000+ właściwości Wikidata). Przewidywalna. Queryowalna. Trudno wymyślić.
- **Open IE.** Każde werbalne wyrażenie staje się relacją. Wysoki recall. Niska precision. Chaotyczne w query.

Produkcyjne KG zwykle mieszają: open IE do odkrywania, potem kanonikalizacja relacji na zamkniętej ontologii przed mergem do głównego grafu.

## Zbuduj to

### Krok 1: ekstrakcja oparta na wzorcach

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Zobacz `code/main.py` po pełny toy extractor. Wzorce Hearst nadal trafiają do domenowo-specyficznych potoków, bo są debugowalne.

### Krok 2: supervisowana klasyfikacja relacji

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL to seq2seq relation extractor: tekst w, trójki out, już w property ids Wikidata. Fine-tuned na danych distant-supervision. Standardowy open-weights baseline.

### Krok 3: ekstrakcja wymuszona LLM z zakotwiczeniem

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Zweryfikuj każdy zwrócony span względem źródła. Odrzuć wszystko, gdzie `text[start:end] != triple_entity`. To jest krok AEVS "verify" w jego minimalnej formie.

### Krok 4: kanonikalizacja na zamkniętej ontologii

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Kanonikalizacja to często 60-80% pracy inżynieryjnej. Zaplanuj na to budżet.

### Krok 5: zbuduj mały graf i wykonaj zapytanie

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

To jest atom każdego systemu RAG-over-KG. Skaluj to z RDF triple stores (Blazegraph, Virtuoso), property graphs (Neo4j), lub vector-augmented graph stores.

## Pułapki

- **Coreference przed RE.** "He founded Apple" — RE musi wiedzieć, kto to "he". Uruchom coref najpierw (lesson 24).
- **Kanonikalizacja encji.** "Apple Inc" i "Apple" muszą być rozwiązane do tego samego węzła. Entity linking najpierw (lesson 25).
- **Halucynowane trójki.** LLM-e emitują trójki, których tekst nie wspiera. Egzekwuj weryfikację spanów.
- **Dryft kanonikalizacji relacji.** Open IE relations są niespójne ("was born in," "came from," "is a native of"). Zwiń do kanonicznych id albo graf jest niequeryowalny.
- **Błędy temporalne.** "Tim Cook is CEO of Apple" — prawdziwe teraz, fałszywe w 2005. Wiele relacji jest ograniczonych czasowo. Używaj kwalifikatorów (`P580` start time, `P582` end time w Wikidata).
- **Niedopasowanie domeny.** REBEL trenowany na Wikipedia. Tekst prawny, medyczny i naukowy często wymaga domain-fine-tuned RE models.

## Użyj tego

Stack 2026:

| Sytuacja | Wybierz |
|-----------|---------|
| Szybka produkcja, domena ogólna | REBEL lub LlamaPred z kanonikalizacją Wikidata |
| Domenowo-specyficzny (biomed, prawny) | SciREX-style domain fine-tune + custom ontology |
| LLM-prompted, audytowany output | Potok AEVS: anchor → extract → verify → supplement |
| Wysoka objętość news IE | Hybryd oparty na wzorcach + supervised |
| Budowanie KG od zera | Open IE + ręczne przejście kanonikalizacji |
| Temporalny KG | Ekstrakcja z kwalifikatorami (start/end time, point in time) |

Pattern integracji: NER → coref → entity linking → relation extraction → ontology mapping → graph load. Każdy etap jest potencjalną bramką jakości.

## Wyślij to

Zapisz jako `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Zaprojektuj potok ekstrakcji relacji z pochodzeniem i kanonikalizacją.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Mając korpus (domena, język, wolumen) i downstream use (KG-RAG, analytics, compliance), wyślij:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Uzasadnienie powiązane z celem precision vs recall.
2. Ontologia. Zamknięta lista właściwości (Wikidata / domena) lub open IE z przejściem kanonikalizacji.
3. Provenance. Każda trójka niesie source char-span + doc id. Non-negotiable dla audytu.
4. Strategia merge. Kanoniczne entity id + relation id + temporal qualifiers; dedup policy.
5. Ewaluacja. Precision / recall na 200 ręcznie labelowanych trójkach + hallucination-rate na LLM-extracted sample.

Odrzuć każdy LLM-based RE pipeline bez span verification (source provenance). Odrzuć open-IE output płynący do produkcyjnego grafu bez kanonikalizacji. Oznacz potoki bez temporal qualifier na relacjach ograniczonych czasowo (employer, spouse, position).
```

## Ćwiczenia

1. **Łatwe.** Uruchom pattern extractor w `code/main.py` na 5 zdaniach z artykułów news. Ręcznie sprawdź precision.
2. **Średnie.** Użyj REBEL (lub małego LLM) na tych samych zdaniach. Porównaj trójki. Który extractor ma wyższą precision? Wyższy recall?
3. **Trudne.** Zbuduj potok AEVS: extract z LLM + verify spans względem źródła. Zmierz hallucination rate przed vs po kroku verify na 50 zdaniach w stylu Wikipedia.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Triple | Subject-relation-object | Krotka `(s, r, o)` która jest atomową jednostką KG. |
| Open IE | Ekstrahuj wszystko | Otwarte frazy relacji; wysoki recall, niska precision. |
| Zamknięta ontologia | Stały schemat | Ograniczony zbiór typów relacji (Wikidata, UMLS, FIBO). |
| Kanonikalizacja | Normalizuj wszystko | Mapuj nazwy powierzchniowe / relacje na kanoniczne id. |
| AEVS | Ugruntowana ekstrakcja | Potok Anchor-Extraction-Verification-Supplement (2026). |
| Provenance | Link do źródła prawdy | Każda trójka niesie doc id + char-span do jej źródła. |
| Distant supervision | Tanie labele | Wyrównaj tekst z istniejącym KG aby stworzyć dane treningowe. |

## Dalsze czytanie

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — artykuł o distant-supervision.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE workhorse.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — joint IE.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 hallucination-mitigation design.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — kanoniczne zapytania grafowe.