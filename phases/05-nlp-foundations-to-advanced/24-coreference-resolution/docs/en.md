# Rozwązywanie Koreferencji

> „Zadzwoniła do niego. Nie odpowiedział. Lekarz był na lunchu." Trzy odniesienia do dwóch osób i nikt nie jest nazwany. Rozwązywanie koreferencji ustala, kto jest kim.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 06 (NER), Faza 5 · 07 (POS i parsowanie)
**Szacowany czas:** ~60 minut

## Problem

Wyodrębnij każde wystąpienie Apple Inc. z artykułu o 300 słowach. Łatwe, gdy artykuł mówi „Apple". Trudne, gdy mówi „firma", „oni", „technologiczny gigant z Cupertino" lub „firma Jobsa". Bez rozwiązania tych odniesień do tej samej encji, pipeline NER pominie 60-80% wystąpień.

Rozwązywanie koreferencji łączy każde wyrażenie, które odnosi się do tej samej encji ze świata rzeczywistego, w jeden klaster. Jest to spoiwo między przetwarzaniem na poziomie powierzchniowym NLP (NER, parsowanie) a semantyką downstream (IE, QA, summarizacja, KG).

Dlaczego ma to znaczenie w 2026 roku:

- Podsumowanie: „Dyrektor generalny ogłosił..." vs „Tim Cook ogłosił..." — podsumowanie powinno nazwać dyrektora generalnego.
- Odpowiadanie na pytania: „Kogo ona zadzwoniła?" wymaga rozwiązania „ona".
- Ekstrakcja informacji: graf wiedzy z „PER1 założył Apple" i „jobs założył Apple" jako oddzielne wpisy jest błędny.
- Wielodokumentowe IE: łączenie wystąpień między artykułami o tym samym wydarzeniu to koreferencja między dokumentami.

## Koncepcja

![Grupowanie koreferencji: wspomnienia → encje](../assets/coref.svg)

**Zadanie.** Wejście: dokument. Wyjście: grupowanie wspomnień (zakresów), gdzie każdy klaster odnosi się do jednej encji.

**Typy wspomnień.**

- **Encja nazwana.** „Tim Cook"
- **Nominal.** „dyrektor generalny", „firma"
- **Zaimkowy.** „on", „ona", „oni", „ono"
- **Apozycyjny.** „Tim Cook, dyrektor generalny Apple,"

**Architektury.**

1. **Oparty na regułach (Hobbs, 1978).** Rozwiązywanie zaimków oparte na drzewie syntaktycznym przy użyciu reguł gramatycznych. Dobry punkt wyjścia. Zaskakująco trudny do pokonania na zaimkach.
2. **Klasyfikator par wspomnień.** Dla każdej pary wspomnień (m_i, m_j), przewiduj, czy są koreferencyjne. Grupuj przez domknięcie tranzytywne. Standard sprzed 2016.
3. **Ranking wspomnień.** Dla każdego wspomnienia, ranking kandydatów na antecedens (łącznie z „brak antecedensu"). Wybierz najlepszego.
4. **End-to-end oparty na zakresach (Lee et al., 2017).** Enkoder Transformer. Wylicz wszystkie kandydatkie zakresy do ustalonej długości. Przewiduj wyniki wspomnień. Przewiduj prawdopodobieństwo antecedensu dla każdego zakresu. Grupuj zachłannie. Nowoczesny domyślny.
5. **Generatywny (2024+).** Promptuj LLM: „Wymień każdy zaimek w tym tekście i jego antecedens." Działa dobrze w łatwych przypadkach, zmaga się z długimi dokumentami i rzadkimi referentami.

**Metryki ewaluacji.** Pięć standardowych metryk (MUC, B³, CEAF, BLANC, LEA), ponieważ żadna pojedyncza metryka nie oddaje jakości grupowania. Raportuj średnią z pierwszych trzech jako CoNLL F1. Stan techniki w 2026 na CoNLL-2012: ~83 F1.

**Znane trudne przypadki.**

- Opisy definitywne odnoszące się do encji wprowadzonych wcześniej na stronicach.
- Mostkowanie anafor („koła" → wcześniej wspomniany samochód).
- Zero-anafora w językach takich jak chiński i japoński.
- Katafora (zaimek przed referentem): „Gdy **weszła**, Mary się uśmiechnęła."

## Zbuduj To

### Krok 1: wstępnie wytrenowany neuronalny coreference (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # eksperymentalny model
doc = nlp("Apple ogłosiło nowe produkty. Firma powiedziała, że wyśle je wkrótce.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Na dłuższym dokumencie otrzymujesz coś takiego:
- Klaster 1: [Apple, Firma, je]
- Klaster 2: [nowe produkty]

### Krok 2: rule-based pronoun resolver (nauczanie)

Zobacz `code/main.py` dla implementacji stdlib-only:

1. Ekstrahuj wspomnienia: encje nazwane (zakresy z wielką literą), zaimki (wyszukiwanie w słowniku), opisy definitywne („the X").
2. Dla każdego zaimka, sprawdź poprzednie K wspomnień i oceń je przez:
   - zgodność rodzaju/liczby (heurystyka)
   - aktualność (bliższe wygrywa)
   - rola syntaktyczna (podmioty preferowane)
3. Połącz najwyżej punktowany antecedens.

Niekonkurencyjny z modelami neuronowymi. Ale pokazuje przestrzeń przeszukiwania i decyzje, które model end-to-end musi podejmować.

### Krok 3: używanie LLM do koreferencji

```python
prompt = f"""Tekst: {text}

Wymień każdy zaimek i frazę rzeczownikową odnoszącą się do osoby lub firmy.
Grupuj je według tego, do czego się odnoszą. Wyjście JSON:
[{{"entity": "Apple", "mentions": ["Apple", "firma", "ono"]}}, ...]
"""
```

Dwa tryby błędów do obserwacji. Po pierwsze, LLM nadmiernie łączy („on" i „ona" odnoszące się do dwóch różnych osób). Po drugie, LLM milcząco pomija wspomnienia w długich dokumentach. Zawsze weryfikuj za pomocą sprawdzania przesunięcia zakresu.

### Krok 4: ewaluacja

Standardowy skrypt conll-2012 oblicza MUC, B³, CEAF-φ4 i raportuje średnią. Do wewnętrznej ewaluacji, zacznij od precision i recall na poziomie zakresu na twoim oznaczonym zestawie testowym, a następnie dodaj F1 linkowania wspomnień.

## Pułapki

- **Eksplozja singletonów.** Niektóre systemy raportują każde wspomnienie jako własny klaster. B³ jest pobłażliwy. MUC to karze. Zawsze sprawdzaj wszystkie trzy metryki.
- **Zaimki w długim kontekście.** Wydajność spada o ~15 F1 na dokumentach powyżej 2000 tokenów. Fragmentuj ostrożnie.
- **Założenia dotyczące płci.** Zakodowane reguły płci psują się na niebinarnych referentach, organizacjach, zwierzętach. Używaj nauczonych modeli lub neutralnego punktowania.
- **Dryf LLM na długich dokumentach.** Pojedyncze wywołanie API nie może wiarygodnie grupować wspomnień przez 50+ akapitów. Używaj sliding-window + merge.

## Użyj To

Stack 2026:

| Sytuacja | Wybierz |
|----------|---------|
| Angielski, pojedynczy dokument | `en_coreference_web_trf` (spaCy-experimental) lub AllenNLP neural coref |
| Wielojęzyczny | SpanBERT / XLM-R trenowany na OntoNotes lub Multilingual CoNLL |
| Koreferencja zdarzeń między dokumentami | Specjalizowane modele end-to-end (2025-26 SOTA) |
| Szybki baseline LLM | GPT-4o / Claude ze structured-output coref prompt |
| Systemy dialogowe produkcyjne | Fallback oparty na regułach + neuronowy primary + ręczna weryfikacja dla krytycznych slotów |

Wzorzec integracji, który trafia w 2026: uruchom NER najpierw, uruchom coref, scal klastry coref w encje NER. Zadania downstream widzą jedną encję na klaster, nie jedną encję na wspomnienie.

## Wyślij To

Zapisz jako `outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: Wybierz podejście do koreferencji, plan ewaluacji i strategię integracji.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

W podanym przypadku użycia (single-doc / multi-doc, domena, język), wyprowadź:

1. Podejście. Oparte na regułach / neuronowe oparte na zakresach / LLM-promptowane / hybrydowe. Jednozdaniowe uzasadnienie.
2. Model. Nazwany checkpoint jeśli neuronowy.
3. Integracja. Kolejność operacji: tokenizacja → NER → coref → downstream task.
4. Ewaluacja. CoNLL F1 (średnia MUC + B³ + CEAF-φ4) na held-out set + ręczny przegląd klastrów na 20 dokumentach.

Odrzuć koreferencję opartą wyłącznie na LLM dla dokumentów powyżej 2000 tokenów bez sliding-window merge. Odrzuć każdy pipeline, który uruchamia coref bez raportu precision-recall na poziomie wspomnień. Oznacz systemy z heurystyką płci wdrożone na demograficznie zróżnicowanym tekście.
```

## Ćwiczenia

1. **Łatwe.** Uruchom resolver oparty na regułach w `code/main.py` na 5 ręcznie stworzonych akapitach. Zmierz dokładność linkowania wspomnień względem ground truth.
2. **Średnie.** Użyj wstępnie wytrenowanego neuronalnego modelu coref na artykule informacyjnym. Porównaj klastry z własną ręczną adnotacją. Gdzie się nie udało?
3. **Trudne.** Zbuduj pipeline NER wzmocniony coref: najpierw NER, następnie scal przez klastry coref. Zmierz poprawę pokrycia encji vs NER-only na 100 artykułach.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Mention (wspomnienie) | Odniesienie | Zakres tekstu odnoszący się do encji (nazwa, zaimek, fraza rzeczownikowa). |
| Antecedent | Do czego „ono" się odnosi | Wcześniejsze wspomnienie, z którym późniejsze jest koreferencyjne. |
| Cluster (klaster) | Wspomnienia encji | Zbiór wspomnień, które wszystkie odnoszą się do tej samej encji ze świata rzeczywistego. |
| Anaphora (anafora) | Wsteczne odniesienie | Późniejsze wspomnienie odnosi się do wcześniejszego („on" → „Jan"). |
| Cataphora (katafora) | W przód odniesienie | Wcześniejsze wspomnienie odnosi się do późniejszego („Gdy on przybył, Jan..."). |
| Bridging (mostkowanie) | Niejawne odniesienie | „Kupiłem samochód. Koła były złe." (koła TEGO samochodu.) |
| CoNLL F1 | Wynik na tablicach wyników | Średnia wyników F1 MUC, B³, CEAF-φ4. |

## Dalsze Czytanie

- [Jurafsky & Martin, SLP3 Rozdział 26 — Rozwązywanie Koreferencji i Łączenie Encji](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — kanoniczny rozdział podręcznika.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — end-to-end oparty na zakresach.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — pretraining który poprawia coref.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — benchmark.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/articlepii/S0024384178900064) — klasyczny oparty na regułach.