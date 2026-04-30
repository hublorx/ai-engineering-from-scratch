```markdown
# Śledzenie Stanu Dialogu

> „Chcę tanią restaurację na północy... tak naprawdę to umiarkowaną... i dodaj włoską." Trzy rundy, trzy aktualizacje stanu. DST utrzymuje słownik slot-wartość w synchronizacji, aby rezerwacja działała.

**Typ:** Budowa
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 17 (Chatboty), Faza 5 · 20 (Strukturyzowane wyniki)
**Czas:** ~75 minut

## Problem

W systemie dialogowym zorientowanym na zadanie, cel użytkownika jest kodowany jako zbiór par slot-wartość: `{cuisine: italian, area: north, price: moderate}`. Każda wypowiedź użytkownika może dodać, zmienić lub usunąć slot. System musi przeczytać całą rozmowę i poprawnie wyprowadzić aktualny stan.

Jeśli jeden slot będzie błędny, system zarezerwuje złą restaurację, zaplanuje zły lot lub obciąży złą kartę. DST jest interfejsem między tym, co użytkownik powiedział, a tym, co wykonuje backend.

Dlaczego to nadal ma znaczenie w 2026 pomimo LLM:

- Domeny wrażliwe na zgodność (bankowość, opieka zdrowotna, rezerwacja lotów) wymagają deterministycznych wartości slotów, nie swobodnego generowania.
- Agenci używający narzędzi nadal potrzebują rozwiązywania slotów przed wywołaniem API.
- Korekta w wielu rundach jest trudniejsza niż się wydaje: „tak naprawdę to w czwartek."

Nowoczesny potok: klasyczne koncepcje DST + ekstraktory LLM + strażnicy strukturyzowanych wyników.

## Koncepcja

![DST: historia dialogu → stan slot-wartość](../assets/dst.svg)

**Struktura zadania.** Schemat definiuje domeny (restauracja, hotel, taksówka) i ich slothy (cuisine, area, price, people). Każdy slot może być pusty, wypełniony wartością z zamkniętego zbioru (price: {cheap, moderate, expensive}) lub wartością w formie dowolnej (name: "The Copper Kettle").

**Dwie formułacje DST.**

- **Klasyfikacja.** Dla każdej pary (slot, wartość_kandydująca) przewiduj tak/nie. Działa dla slotów z zamkniętym słownictwem. Standard sprzed 2020.
- **Generacja.** Mając dialog, generuj wartości slotów jako tekst swobodny. Działa dla slotów z otwartym słownictwem. Współczesny domyślny sposób.

**Metryka.** Joint Goal Accuracy (JGA) — ułamek rund, w których każdy slot jest poprawny. Wszystko-albo-nic. Tablica wyników MultiWOZ 2.4 osiąga około 83% w 2026.

**Architektury.**

1. **Regułowy (wyrażenie regex + słowo kluczowe slot).** Silna baza dla wąskich domen. Możliwy do diagnozowania błędów.
2. **TripPy / BERT-DST.** Generacja oparta na kopiowaniu z kodowaniem BERT. Standard sprzed ery LLM.
3. **LDST (LLaMA + LoRA).** Instruowany LLM z promptem domena-slot. Osiąga jakość na poziomie ChatGPT na MultiWOZ 2.4.
4. **Ontology-free (2024–2026).** Pomiń schemat; generuj nazwy i wartości slotów bezpośrednio. Obsługuje otwarte domeny.
5. **Prompt + strukturyzowany wynik (2024–2026).** LLM ze schematem Pydantic + dekodowanie z ograniczeniami. 5 linii kodu, gotowe do produkcji.

### Klasyczne tryby awarii

- **Odwołanie przez okresy.** „Zostańmy przy pierwszej opcji." Wymaga rozwiązania, która opcja.
- **Nadpisanie vs dołączenie.** Użytkownik mówi „dodaj włoską." Czy zastępujesz cuisine, czy dołączasz?
- **Niejawne potwierdzenia.** „OK fajnie" — czy to zaakceptowało oferowaną rezerwację?
- **Korekta.** „Tak naprawdę to 19:00." Musi zaktualizować czas, bez czyszczenia innych slotów.
- **Odwołanie do poprzedniej wypowiedzi systemu.** „Tak, tę." Które „tę"?

## Zbuduj to

### Krok 1: ekstraktor slotów oparty na regułach

Zobacz `code/main.py`. Regex + słowniki synonimów pokrywają 70% kanonicznych wypowiedzi w wąskich domenach:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Kruchy poza kanonicznym słownictwem. Działa dla deterministycznych potwierdzeń slotów.

### Krok 2: pętla aktualizacji stanu

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Trzy niezmienniki:

- Nigdy nie resetuj slota, którego użytkownik nie dotknął.
- Jawne przeczenie („nie ma znaczenia cuisine") musi czyścić.
- Korekta użytkownika („tak naprawdę...") musi nadpisywać, nie dołączać.

### Krok 3: DST oparty na LLM ze strukturyzowanym wynikiem

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic gwarantuje prawidłowy obiekt stanu. Bez regex, bez niezgodności schematu, bez zmyślonych slotów.

### Krok 4: ewaluacja JGA

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Kalibruj: jaka część rund system ma WSZYSTKIE slothy poprawne? Dla MultiWOZ 2.4, najlepsze systemy 2026: 80-83%. Twój system w domenie powinien to przekroczyć na twoim wąskim słownictwie lub bazowa linia LLM cię pokona.

### Krok 5: obsługa korekty

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

Po wykryciu korekty nadpisz ostatnio zaktualizowany slot zamiast dołączać. Trudne do prawidłowego wykonania bez pomocy LLM. Współczesny wzorzec: zawsze pozwalaj LLM regenerować cały stan z historii zamiast inkrementalnej aktualizacji — to naturalnie obsługuje korekty.

## Pułapki

- **Koszt pełnej regeneracji historii.** Pozwolenie LLM na regenerację stanu każdej rundy kosztuje O(n²) całkowitych tokenów. Ogranicz historię lub podsumowuj starsze rundy.
- **Dryfowanie schematu.** Dodanie nowych slotów po fakcie, łamie stare dane treningowe. Wersjonuj swój schemat.
- **Wielkość liter.** „Italian" vs "italian" vs "ITALIAN" — normalizuj wszędzie.
- **Niejawny transfer.** Jeśli użytkownik wcześniej określił „dla 4 osób", nowe żądanie o innej porze nie powinno czyścić people. Zawsze przekazuj pełną historię.
- **Swobodny vs zamknięty zbiór.** Nazwy, godziny i adresy potrzebują slotów swobodnych; kuchnie i obszary są zamknięte. Mieszaj oba w schemacie.

## Zastosowanie

Stos 2026:

| Sytuacja | Podejście |
|-----------|-----------|
| Wąska domena (jeden lub dwa cele) | Regułowy + regex |
| Szeroka domena, dostępne dane z etykietami | LDST (LLaMA + LoRA na danych w stylu MultiWOZ) |
| Szeroka domena, brak etykiet, prod-gotowe | LLM + Instructor + schemat Pydantic |
| Mówione / głosowe | ASR + normalizer + LLM-DST |
| Wielodomenowy potok rezerwacji | Schema-guided LLM z per-domenowymi modelami Pydantic |
| Wrażliwe na zgodność | Regułowy primary, LLM fallback z potokiem potwierdzenia |

## Dostarcz

Zapisz jako `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Zaprojektuj tracker stanu dialogu — schemat, ekstraktor, polityka aktualizacji, ewaluacja.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Mając przypadek użycia (domena, języki, otwartość słownictwa, potrzeby zgodności), wyprowadź:

1. Schemat. Lista domen, slothy na domenę, otwarte vs zamknięte słownictwo per slot, Uzasadnienie.
2. Ekstraktor. Regułowy / seq2seq / LLM-z-Pydantic.
3. Polityka aktualizacji. Regeneruj-cały-stan / inkrementalny; obsługa korekty; obsługa przecenia.
4. Ewaluacja. Joint Goal Accuracy na hold-out zbiorze dialogów, precision/recall per slot, najtrudniejszy slot.
5. Potok potwierdzenia. Kiedy jawnie pytać użytkownika o potwierdzenie (działania destrukcyjne, niskie pewne ekstrakcje).

Odrzuć LLM-only DST dla slotów wrażliwych na zgodność bez regułowego sprawdzenia pomocniczego, Oznacz schematy bez tagów wersji. Odrzuć każdy DST, który nie może wycofać slota przy korekcie użytkownika, Oznacz schematy bez tagów wersji.
```

## Ćwiczenia

1. **Łatwe.** Zbuduj tracker stanu oparty na regułach w `code/main.py` dla 3 slotów (cuisine, area, price). Przetestuj na 10 ręcznie przygotowanych dialogach. Zmierz JGA.
2. **Średnie.** Ten sam zbiór danych z Instructor + Pydantic + małym LLM. Porównaj JGA, Sprawdź najtrudniejsze rundy.
3. **Trudne.** Zaimplementuj oba i trasuj: regułowy primary, LLM fallback gdy regułowy emituje <2 slothy z pewnością, Zmierz połączony JGA i koszt wnioskowania na rundę.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| DST | Śledzenie stanu dialogu | Utrzymywanie słownika slot-wartość przez rundy dialogu. |
| Slot | Jednostka intencji użytkownika | Nazwany parametr, którego potrzebuje backend (cuisine, date). |
| Domain | Obszar zadania | Restauracja, hotel, taksówka — zbiory slotów. |
| JGA | Joint Goal Accuracy | Ułamek rund, w których każdy slot jest poprawny. Wszystko-albo-nic. |
| MultiWOZ | Benchmark | Wielodomenowy zbiór danych WOZ; standardowa ewaluacja DST. |
| Ontology-free DST | Bez schematu | Generuj nazwy i wartości slotów bezpośrednio, bez ustalonej listy. |
| Correction | „Tak naprawdę..." | Runda, która nadpisuje wcześniej wypełniony slot. |

## Dalsza lektura

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — kanoniczny benchmark.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA instruction tuning dla DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — praca bazowa copy-based DST.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM-based unsupervised TOD.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — kanoniczne wyniki DST.
```