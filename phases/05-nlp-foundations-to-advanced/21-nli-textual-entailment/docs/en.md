# Natural Language Inference — Wnioskowanie tekstowe

> "t entailuje h" oznacza, że człowiek czytający t wywnioskuje, że h jest prawdziwe. NLI to zadanie przewidywania entailmentu / sprzeczności / neutralności. Z pozoru nudne, ale kluczowe w produkcji.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 05 (Analiza sentymentu), Faza 5 · 13 (Question Answering)
**Szacowany czas:** ~60 minut

## Problem

Zbudowałeś podsumowanie. Wygenerowało podsumowanie. Skąd wiesz, że podsumowanie nie zawiera halucynacji?

Zbudowałeś chatbota. Odpowiedział "tak." Skąd wiesz, że odpowiedź jest wsparta przez pobrany fragment?

Musisz sklasyfikować 10 000 artykułów informacyjnych według tematu. Nie masz etykiet treningowych. Czy możesz ponownie wykorzystać model?

Wszystkie trzy problemy sprowadzają się do Natural Language Inference. NLI pyta: mając przesłankę `t` i hipotezę `h`, czy `h` wynika z `t`, jest sprzeczne, czy neutralne (niepowiązane)?

- **Sprawdzanie halucynacji:** `t` = dokument źródłowy, `h` = twierdzenie z podsumowania. Brak entailmentu = halucynacja.
- **Ugruntowane QA:** `t` = pobrany fragment, `h` = wygenerowana odpowiedź. Brak entailmentu = fabricacja.
- **Klasyfikacja zero-shot:** `t` = dokument, `h` = werbalizowana etykieta ("To jest o sporcie"). Entailment = przewidywana etykieta.

Jedno zadanie, trzy zastosowania produkcyjne. Dlatego każdy framework ewaluacyjna RAG ma pod maską model NLI.

## Koncepcja

![NLI: klasyfikacja trójstopniowa, przesłanka vs hipoteza](../assets/nli.svg)

**Trzy etykiety.**

- **Entailment.** `t` → `h`. "Kot śpi na kanapie" entailuje "W pokoju jest kot."
- **Sprzeczność.** `t` → ¬`h`. "Kot śpi na kanapie" przeczy "W pokoju nie ma kota."
- **Neutralny.** Brak wnioskowania w żadną stronę. "Kot śpi na kanapie" jest neutralny wobec "Kot jest głodny."

**Nie logiczny entailment.** NLI to *natural* wnioskowanie językowe — co typowy człowiek by wywnioskował, a nie ścisła logika. "Jan wyprowadził psa" entailuje "Jan ma psa" w NLI, ale ścisła logika pierwszego rzędu przyjęłaby to tylko wtedy, gdybyś aksjomatyzował posiadanie.

**Zbiory danych.**

- **SNLI** (2015). 570k par z adnotacjami ludzkimi, podpisy obrazów jako przesłanki. Wąska domena.
- **MultiNLI** (2017). 433k par z 10 gatunków. Standardowy korpus treningowy w 2026.
- **ANLI** (2019). Adversarial NLI. Ludzie pisali przykłady specjalnie zaprojektowane, żeby łamać istniejące modele. Trudniejsze.
- **DocNLI, ConTRoL** (2020–21). Przesłanki na poziomie dokumentu. Testuje wnioskowanie wieloskokowe i dalekiego zasięgu.

**Architektura.** Enkoder transformer (BERT, RoBERTa, DeBERTa) czyta `[CLS] przesłanka [SEP] hipoteza [SEP]`. Reprezentacja `[CLS]` trafia do 3-kierunkowego softmax. Trenuj na MNLI, ewaluuj na hold-out benchmarkach, uzyskaj 90%+ accuracy na parach z rozkładem.

**Zero-shot przez NLI.** Mając dokument i kandydackie etykiety, zamień każdą etykietę w hipotezę ("Ten tekst jest o sporcie"). Oblicz prawdopodobieństwo entailmentu dla każdej. Wybierz max. To jest mechanizm za pipeline'em `zero-shot-classification` od Hugging Face.

## Zbuduj to

### Krok 1: uruchom pretrained model NLI

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Do produkcyjnego NLI, `facebook/bart-large-mnli` i `microsoft/deberta-v3-large-mnli` to domyślne open-source. DeBERTa-v3 prowadzi na leaderboardach.

### Krok 2: klasyfikacja zero-shot

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

Szablon to "This example is about {label}." domyślnie. Dostosuj z `hypothesis_template`. Nie wymaga danych treningowych. Nie wymaga fine-tuningu. Działa out of the box.

### Krok 3: sprawdzanie wierności dla RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

To jest rdzeń faithfulness w RAGAS. Podziel wygenerowaną odpowiedź na atomowe twierdzenia. Sprawdź każde twierdzenie względem pobranego kontekstu. Raportuj frakcję, która entailuje.

### Krok 4: ręcznie robiony klasyfikator NLI (koncepcyjny)

Zobacz `code/main.py` dla zabawki tylko ze stdlib: przesłanka i hipoteza są porównywane przez zachodzenie leksykalne + wykrywanie negacji. Niekonkurencyjne z modelami transformer — ale pokazuje kształt zadania: dwa teksty w, 3-kierunkowa etykieta out, loss = cross-entropy nad `{entail, contradict, neutral}`.

## Pułapki

- **Skróty tylko od hipotezy.** Modele mogą przewidywać etykietę tylko z hipotezy na ~60% na SNLI, bo "not", "nobody", "never" korelują ze sprzecznością. Silny baseline do wykrywania label leakage.
- **Heurystyka zachodzenia leksykalnego.** Heurystyka podciągu ("każdy podciąg jest entailowany") przechodzi SNLI, ale failuje na HANS/ANLI. Używaj adversarial benchmarków.
- **Degradacja na długości dokumentu.** Modele NLI na pojedynczych zdaniach tracą 20+ F1 na przesłankach na poziomie dokumentu. Używaj modeli trenowanych na DocNLI dla długiego kontekstu.
- **Wrażliwość szablonu zero-shot.** "This example is about {label}" vs "{label}" vs "The topic is {label}" może zmienić accuracy o 10+ punktów. Dostrój szablon.
- **Niedopasowanie domeny.** MNLI trenuje na ogólnym angielskim. Teksty prawne, medyczne i naukowe potrzebują modeli NLI specyficznych dla domeny (np. SciNLI, MedNLI).

## Użyj tego

Stack 2026:

| Przypadek użycia | Model |
|---------|-------|
| NLI ogólnego przeznaczenia | `microsoft/deberta-v3-large-mnli` |
| Szybki / edge | `cross-encoder/nli-deberta-v3-base` |
| Klasyfikacja zero-shot (lekki) | `facebook/bart-large-mnli` |
| NLI na poziomie dokumentu | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Wielojęzyczny | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Wykrywanie halucynacji w RAG | Warstwa NLI w RAGAS / DeepEval |

**Metawzorzec 2026:** NLI to taśma klejąca text understanding. Ilekroć potrzebujesz "czy A wspiera B?" lub "czy A przeczy B?" — sięgnij po NLI zanim sięgniesz po kolejny call LLM.

## Wyślij to

Zapisz jako `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Ćwiczenia

1. **Łatwe.** Uruchom `facebook/bart-large-mnli` na 20 ręcznie stworzonych trójkach (przesłanka, hipoteza, etykieta) pokrywających wszystkie trzy klasy. Zmierz accuracy. Dodaj adversarial "subsequence heuristic" pułapki ("Nie zjadłem ciasta" vs "Zjadłem ciasto") i sprawdź, czy to łamie model.
2. **Średnie.** Porównaj szablon zero-shot `"This text is about {label}"` z `"The topic is {label}"` i `"{label}"` na 100 nagłówkach AG News. Raportuj zmianę accuracy.
3. **Trudne.** Zbuduj checker wierności RAG: dekompozycja na atomowe twierdzenia + NLI per claim. Ewaluuj na 50 wygenerowanych odpowiedziach RAG z gold context. Zmierz false-positive i false-negative rates vs hand labels.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | 3-kierunkowa klasyfikacja relacji przesłanka-hipoteza. |
| RTE | Recognizing Textual Entailment | Starsza nazwa NLI; to samo zadanie. |
| Entailment | "t implikuje h" | Typowy czytelnik wywnioskuje, że h jest prawdziwe mając t. |
| Contradiction | "t wyklucza h" | Typowy czytelnik wywnioskuje, że h jest fałszywe mając t. |
| Neutral | "niezdecydowany" | Brak wnioskowania z t do h w żadną stronę. |
| Zero-shot classification | NLI jako klasyfikator | Werbalizuj etykiety jako hipotezy, wybierz max entailment. |
| Faithfulness | Czy odpowiedź jest wsparta? | NLI nad (pobrany kontekst, wygenerowana odpowiedź). |

## Dalsza lektura

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — benchmark ANLI.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — koń pociągowy NLI 2026.

---

**Poprawka zastosowana:** "Metapattern" → "Metawzorzec" (termin opisowy, nie jest na liście dozwolonych anglicyzmów).

**Uwaga:** Przeanalizowałem pozostałe dwa przypadki. W obu nie ma błędów:
- "jest neutralny wobec" — to fraza z przyimkiem, nie spójnik łączący dwa niezależne zdania, więc przecinek nie jest wymagany.
- "domyślnie" — przysłówek na końcu zdania, nie wymaga przecinka.