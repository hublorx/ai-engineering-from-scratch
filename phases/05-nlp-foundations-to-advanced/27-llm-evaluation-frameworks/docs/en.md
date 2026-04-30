```markdown
# Ewaluacja LLM — RAGAS, DeepEval, G-Eval

> Dokładne dopasowanie i F1 nie oddają semantycznej równoważności. Recenzja ludzka nie skaluje się. LLM-as-judge to produkcyjne rozwiązanie — z wystarczającą kalibracją, by ufać liczbie.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 13 (Question Answering), Faza 5 · 14 (Information Retrieval)
**Szacowany czas:** około 75 minut

## Problem

Twój system RAG odpowiada: „29 czerwca 2007."
Złota referencja to: „29 czerwca 2007."
Dokładne dopasowanie daje 0. F1 około 75%. Człowiek dałby 100%.

Teraz pomnóż to przez 10 000 przypadków testowych. Pomnóż jeszcze raz przez każdą zmianę w retrieverze, chunkowaniu, prompcie lub modelu. Potrzebujesz ewaluatora, który rozumie znaczenie, działa tanio na skalę, nie kłamie w kwestii regresji i wyłania właściwe tryby błędów.

2026 ma trzy frameworki, które dominują w tym problemie.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. Cztery metryki RAG (faithfulness, answer-relevance, context-precision, context-recall) z backendami opartymi na NLI i na LLM-judge. Oparte na badaniach, lekkie.
- **DeepEval.** Pytest dla LLM. G-Eval, task-completion, hallucination, bias metrics. Natywne dla CI/CD.
- **G-Eval.** Metoda (i metryka DeepEval): LLM-as-judge z chain-of-thought, własnymi kryteriami, wynikiem 0-1.

Wszystkie trzy opierają się na LLM-as-judge. Ta lekcja buduje intuicję dla tej metody i warstwy zaufania wokół niej.

## Koncepcja

![Cztery wymiary ewaluacji, architektura LLM-as-judge](../assets/llm-evaluation.svg)

**LLM-as-judge.** Zastąp statyczną metrykę LLM, który ocenia wyniki na podstawie rubryki. Mając `(query, context, answer)`, wyślij prompt do sędziego LLM: „Oceń 0-1 na faithfulness." Zwróć wynik.

Dlaczego to działa: LLM przybliżają ludzką ocenę za ułamek kosztu. GPT-4o-mini przy około 0,003 USD za oceniany przypadek umożliwia uruchomienie ewaluacji regresyjnej na 1000 próbek za mniej niż 5 USD.

Dlaczego zawodzi po cichu:

1. **Bias sędziego.** Sędziowie preferują dłuższe odpowiedzi, odpowiedzi z własnej rodziny modeli i odpowiedzi pasujące do stylu promptu.
2. **Błędy parsowania JSON.** Zły JSON → wynik NaN → po cichu wykluczony z agregatu. Użytkownicy RAGAS to znają. Zabezpiecz za pomocą try/except + jawny tryb błędu.
3. **Dryft między wersjami modelu.** Uaktualnienie sędziego zmienia każdą metrykę. Zamroź model sędziego + wersję.

**Cztery metryki RAG.**

| Metryka | Pytanie | Aparat |
|---------|---------|--------|
| Faithfulness | Czy każde twierdzenie w odpowiedzi pochodzi z pobranego kontekstu? | NLI-based entailment |
| Answer relevance | Czy odpowiedź odnosi się do pytania? | Generuj hipotetyczne pytania z odpowiedzi; porównaj z rzeczywistym pytaniem |
| Context precision | Jaka część pobranych chunków była istotna? | LLM-judge |
| Context recall | Czy pobieranie zwróciło wszystko, co potrzebne? | LLM-judge względem złotej odpowiedzi |

**G-Eval.** Zdefiniuj własne kryterium: „Czy odpowiedź cytuje poprawne źródło?" Framework automatycznie rozwija to w kroki ewaluacji chain-of-thought, a następnie ocenia 0-1. Dobre dla specyficznych dla domeny wymiarów jakości, których RAGAS nie obejmuje.

**Kalibracja.** Nigdy nie ufaj surowemu wynikowi sędziego, dopóki nie masz korelacji z etykietami ludzkimi. Uruchom 100 ręcznie oznakowanych przykładów. Wykreśl judge vs human. Oblicz Spearman rho. Jeśli rho < 0,7, rubryka sędziego wymaga pracy.

## Zbuduj to

### Krok 1: faithfulness z NLI (styl RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` to dowolna funkcja: prompt str -> wygenerowany str.
# Przykład: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Rozłóż odpowiedź na atomowe twierdzenia. Sprawdź NLI każde twierdzenie względem pobranego kontekstu. Faithfulness = frakcja wsparte.

### Krok 2: answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: dowolny model implementujący .encode(texts, normalize_embeddings=True) -> ndarray
# np. encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Jeśli odpowiedź implikuje inne pytania niż to zadane, istotność spada.

### Krok 3: G-Eval custom metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Kroki ewaluacji to rubryka. Jawne kroki są bardziej stabilne niż niejawne „oceniaj 0-1" prompty.

### Krok 4: CI gate

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Dostarcz jako plik pytest. Uruchamiaj przy każdym PR, i blokuj merge przy regresjach.

### Krok 5: toy eval od zera

Zobacz `code/main.py`. Aproksymacje tylko ze stdlib dla faithfulness (overlap twierdzeń odpowiedzi z kontekstem) i relevance (overlap tokenów odpowiedzi z tokenami pytania). Nieprodukcyjne. Pokazuje kształt.

## Pułapki

- **Brak kalibracji.** Sędzia z korelacją 0,3 do etykiet ludzkich to szum. Wymagaj przebiegu kalibracyjnego przed wdrożeniem.
- **Samoewaluacja.** Używanie tego samego LLM do generowania i oceniania zawyża wyniki o 10-20%. Użyj innej rodziny modeli dla sędziego.
- **Bias pozycyjny w ocenie parami.** Sędziowie wolą pierwszą przedstawioną opcję. Zawsze randomizuj kolejność i uruchamiaj obie.
- **Surowy agregat ukrywa błędy.** Średni wynik 0,85 często ukrywa 5% katastrofalnych błędów. Zawsze sprawdzaj dolny kwantyl dla przypadków, które przeszły.
- **Próchnica zbioru golden.** Niewersjonowane zestawy ewaluacyjne, które dryfują z czasem, psują porównania podłużne. Taguj zbiór danych przy każdej zmianie.
- **Koszt LLM.** Na skalę, wywołania sędziego dominują koszt. Używaj najtańszego modelu, który spełnia próg kalibracji. GPT-4o-mini, Claude Haiku, Mistral-small.

## Użyj tego

Stack 2026:

| Przypadek użycia | Framework |
|------------------|-----------|
| Monitorowanie jakości RAG | RAGAS (4 metryki) |
| CI/CD regression gates | DeepEval + pytest |
| Niestandardowe kryteria domenowe | G-Eval w DeepEval |
| Monitorowanie ruchu produkcyjnego | RAGAS z trybem reference-free |
| Human-in-the-loop spot checks | LangSmith lub Phoenix z UI annotation |
| Red-teaming / safety eval | Promptfoo + DeepEval |

Typowy stack: RAGAS do monitorowania, DeepEval do CI, G-Eval do nowych wymiarów. Uruchom wszystkie trzy; rozbieżnie, ale użytecznie.

## Dostarcz to

Zapisz jako `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Ćwiczenia

1. **Łatwe.** Użyj RAGAS na 10 przykładach RAG ze znanymi halucynacjami. Zweryfikuj, że metryka faithfulness wyłapuje każdy z nich.
2. **Średnie.** Ręcznie oznakuj 50 odpowiedzi QA 0-1 pod kątem poprawności. Oceń za pomocą G-Eval. Zmierz Spearman rho między sędzią a człowiekiem.
3. **Trudne.** Zbuduj pytest CI gate z DeepEval. Celowo zregresuj retriever. Zweryfikuj, że bramka się nie powiedzie. Dodaj alertowanie dolnego kwantyla poprzez próg na najniższych 10%.

## Kluczowe terminy

| Termin | Co, ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| LLM-as-judge | Ocena z LLM | Promptuj model sędziego, by oceniał wyniki 0-1, na podstawie rubryki. |
| RAGAS | Biblioteka metryk RAG | Open-source framework ewaluacyjny z 4 metrykami RAG reference-free. |
| Faithfulness | Czy odpowiedź jest ugruntowana? | Frakcja twierdzeń odpowiedzi entailowana przez pobrany kontekst. |
| Context precision | Czy pobrane chunki były istotne? | Frakcja top-K chunków, które faktycznie miały znaczenie. |
| Context recall | Czy pobieranie znalazło wszystko? | Frakcja twierdzeń złotej odpowiedzi wsparciych przez pobrane chunki. |
| G-Eval | Niestandardowy sędzia LLM | Rubryka + kroki ewaluacji chain-of-thought + wynik 0-1. |
| Calibration | Ufać, ale weryfikować | Korelacja Spearmana między wynikiem sędziego a wynikiem ludzkim. |

## Dalsze czytanie

- Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation — artykuł RAGAS.
- Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment — artykuł G-Eval.
- DeepEval docs — otwarty stack produkcyjny.
- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena — biasy, kalibracja, limity.
- MLflow GenAI Scorer — ujednollicający framework integrujący RAGAS, DeepEval, Phoenix.
```