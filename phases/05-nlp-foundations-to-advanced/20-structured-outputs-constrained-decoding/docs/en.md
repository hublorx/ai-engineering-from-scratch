# Structured Outputs i Constrained Decoding

> Poproś LLM o JSON. Otrzymaj JSON przez większość czasu. W produkcji „przez większość czasu" to problem. Constrained decoding zamienia „przez większość" w „zawsze" poprzez edycję logitów przed samplingiem.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 5 · 17 (Chatboty), Phase 5 · 19 (Subword Tokenization)
**Szacowany czas:** ~60 minut

## Problem

Klasyfikator promptuje LLM: „Zwróć jedną z {pozytywna, negatywna, neutralna}." Model zwraca „Sentiment jest pozytywny — ta recenzja jest zdecydowanie korzystna, ponieważ klient wyraźnie stwierdza, że ...". Twój parser się zawiesza. F1 klasyfikatora wynosi 0.0.

Generowanie w wolnej formie nie jest kontraktem. To sugestia. System produkcyjny potrzebuje kontraktu.

W 2026 istnieją trzy warstwy.

1. **Prompting.** Poproś grzecznie. „Zwróć tylko obiekt JSON." Działa w ~80% na frontier models, mniej na mniejszych modelach.
2. **Native structured output APIs.** OpenAI `response_format`, Anthropic tool use, Gemini JSON mode. Niezawodne na wspieranych schematach. Vendor-locked.
3. **Constrained decoding.** Modyfikuj logitty przy każdym kroku generowania, więc model *nie może* wyemitować nieprawidłowych tokenów. 100% poprawności z definicji. Działa na dowolnym modelu lokalnym.

Ta lekcja buduje intuicję dla wszystkich trzech i wskazuje kiedy sięgać po którą.

## Koncept

![Constrained decoding maskujący nieprawidłowe tokeny na każdym kroku](../assets/constrained-decoding.svg)

**Jak działa constrained decoding.** Przy każdym kroku generowania LLM produkuje wektor logitów nad pełnym słownictwem (~100k tokenów). *Logit processor* siedzi między modelem a samplerem. Oblicza, które tokeny są prawidłowe w danym momencie pozycji w docelowej gramatyce — JSON Schema, regex, context-free grammar — i ustawia logitty wszystkich nieprawidłowych tokenów na minus nieskończoność. Softmax na pozostałych logitach przenosi masę prawdopodobieństwa tylko na prawidłowe kontynuacje.

Implementacje w 2026:

- **Outlines.** Kompiluje JSON Schema lub regex do finite-state machine. Każdy token otrzymuje O(1) lookup prawidłowych następnych tokenów. FSM-based, więc rekurencyjne schematy wymagają spłaszczenia.
- **XGrammar / llguidance.** Context-free grammar engines. Obsługują rekurencyjne JSON Schema. Near-zero decoding overhead. OpenAI przypisał zasługę llguidance w swojej implementacji structured output z 2025.
- **vLLM guided decoding.** Wbudowane `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` przez Outlines, XGrammar lub backendy lm-format-enforcer.
- **Instructor.** Pydantic-based wrapper nad dowolnym LLM. Retry przy błędach walidacji. Cross-provider, ale nie modyfikuje logitów — polega na retry + structured-output-aware prompts.

### Intuicyjny wynik

Constrained decoding jest często *szybszy* niż nieograniczone generowanie. Dwa powody. Po pierwsze, zmniejsza przestrzeń wyszukiwania następnego tokena. Po drugie, sprytne implementacje pomijają generowanie tokenów całkowicie dla wymuszonych tokenów (scaffolding jak `{"name": "` — każdy bajt jest określony).

### Pułapka, która Cię kosztuje

Kolejność pól ma znaczenie. Umieść `answer` przed `reasoning`, a model zobowiązuje się do odpowiedzi zanim pomyśli. JSON jest poprawny. Odpowiedź jest błędna. Żadna walidacja tego nie wychwyci.

```json
// ŹLE
{"answer": "yes", "reasoning": "because ..."}

// DOBRZE
{"reasoning": "... therefore ...", "answer": "yes"}
```

Kolejność pól w schemacie to logika, nie formatowanie.

## Zbuduj To

### Krok 1: regex-constrained generation od zera

Zobacz `code/main.py` dla samodzielnej implementacji FSM. Główna idea w 30 liniach:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM śledzi, które części gramatyki zostały dotychczas spełnione. `valid_tokens(state, tokenizer)` oblicza, które tokeny ze słownictwa mogą przesunąć FSM bez opuszczenia ścieżki akceptującej.

### Krok 2: Outlines dla JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Zero błędów walidacji. Kiedykolwiek. FSM sprawia, że nieprawidłowe wyjście jest nieosiągalne.

### Krok 3: Instructor dla provider-agnostic Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Inny mechanizm. Instructor nie dotyka logitów. Formatuje schemat do promptu, parsuje wyjście i retryje przy błędzie walidacji (domyślnie 3 razy). Działa z dowolnym providerem. Retry dodają latency i koszt. Cross-provider portability to selling point.

### Krok 4: native vendor APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Server-side constrained decoding. Niezawodność na poziomie Outlines dla wspieranych schematów. Bez zarządzania modelem lokalnym. Blokuje Cię do vendora.

## Pułapki

- **Rekurencyjne schematy.** Outlines spłaszcza rekursję do ustalonej głębokości. Drzewiasto strukturyzowane wyjścia (zagnieżdżone komentarze, AST) potrzebują XGrammar lub llguidance (CFG-based).
- **Ogromne enumy.** Enum z 10 000 opcji kompiluje się powoli lub timeoutuje. Przełącz na retriever: najpierw przewiduj top-k kandydatów, potem ogranicz do nich.
- **Gramatyka zbyt restrykcyjna.** Wymuś regex `date: "YYYY-MM-DD"` i model nie może wyprowadzić `"unknown"` dla brakujących dat. Model rekompensuje to wymyślając datę. Pozwól na `null` lub sentinela.
- **Przedwczesne zobowiązanie.** Zobacz pułapkę kolejności pól powyżej. Zawsze najpierw umieszczaj reasoning.
- **Vendor JSON mode bez schematu.** Czysty JSON mode gwarantuje tylko poprawną składnię JSON, nie poprawność *dla Twojego przypadku użycia*. Zawsze podawaj pełny schemat.

## Użyj To

Stack w 2026:

| Sytuacja | Wybierz |
|-----------|------|
| Model OpenAI/Anthropic/Google, prosty schemat | Native vendor structured output |
| Dowolny provider, workflow Pydantic, może tolerować retry | Instructor |
| Model lokalny, potrzebujesz 100% poprawności, płaski schemat | Outlines (FSM) |
| Model lokalny, rekurencyjny schemat | XGrammar lub llguidance |
| Self-hosted inference server | vLLM guided decoding |
| Batch processing z akceptowalnymi retry | Instructor + najtańszy model |

## Wysyłaj To

Zapisz jako `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Ćwiczenia

1. **Łatwe.** Zepytaj mały open-weights model (np. Llama-3.2-3B) bez constrained decoding dla `Review(sentiment, confidence, evidence_span)`. Zmierz frakcję, która parsuje jako poprawny JSON na 100 recenzjach.
2. **Średnie.** Ten sam korpus z Outlines JSON mode. Porównaj compliance rate, latency i semantic accuracy.
3. **Trudne.** Zaimplementuj regex-constrained decoder od zera dla numerów telefonów (`\d{3}-\d{3}-\d{4}`). Zweryfikuj 0 nieprawidłowych wyjść na 1000 próbkach.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| Constrained decoding | Wymuś prawidłowe wyjście | Maskuj logitty nieprawidłowych tokenów przy każdym kroku generowania. |
| Logit processor | To co ogranicza | Funkcja: `(logits, state) -> masked_logits`. |
| FSM | Finite-state machine | Skompilowana reprezentacja gramatyki; O(1) lookup prawidłowych następnych tokenów. |
| CFG | Context-free grammar | Gramatyka obsługująca rekursję; wolniejsza ale bardziej ekspresyjna niż FSM. |
| Schema field order | Czy to ma znaczenie? | Tak — pierwsze pole zobowiązuje; zawsze kładź reasoning przed answer. |
| Guided decoding | Nazwa vLLM dla tego | Ten sam koncept, zintegrowany z inference serverem. |
| JSON mode | Wczesna wersja OpenAI | Gwarantuje składnię JSON; NIE gwarantuje dopasowania schematu. |

## Dalsze Czytanie

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — artykuł Outlines.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — szybki CFG-based constrained decoding.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — integracja z inference serverem.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API reference + gotchas.
- [Instructor library](https://python.useinstructor.com/) — Pydantic + retries między providerami.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — benchmarking 6 frameworków constrained decoding.