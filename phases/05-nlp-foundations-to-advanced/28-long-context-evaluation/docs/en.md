# Long-Context Evaluation — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro reklamuje 10 milionów tokenów kontekstu. Przy 1 milionie tokenów, MRCR z 8 igłami spada do 26,3%. Reklamowane ≠ użyteczne. Ewaluacja długiego kontekstu pokazuje rzeczywistą pojemność modelu, na którym wdrażasz rozwiązanie.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 13 (Question Answering), Faza 5 · 23 (Chunking Strategies)
**Szacowany czas:** około 60 minut

## Problem

Masz 200-stronicową umowę. Model deklaruje kontekst 1 miliona tokenów. Wklejasz umowę i pytasz: "Jaki jest klauzula rozwiązania umowy?" Model odpowiada — ale odpowiada ze strony tytułowej, ponieważ klauzula rozwiązania znajduje się na głębokości 120k tokenów, poza tym, gdzie model faktycznie prowadzi attention.

To jest luka w pojemności kontekstu w 2026 roku. Specyfikacje mówią 1M lub 10M. Rzeczywistość mówi, że 60-70% tego jest użyteczne, a "użyteczne" zależy od zadania.

- **Pobieranie (pojedyncza igła w stercie):** niemal idealne aż do reklamowanego maksimum na modelach frontowych.
- **Multi-hop / agregacja:** gwałtownie degraduje po ~128k na większości modeli.
- **Rozumowanie na rozproszonych faktach:** pierwsze zadanie, które się nie udaje.

Ewaluacja długiego kontekstu mierzy te osie. Ta lekcja nazywa benchmarki, co każdy z nich faktycznie mierzy i jak zbudować niestandardowy test igłowy dla swojej domeny.

## Koncepcja

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** Umieść fakt ("magiczne słowo to ananas") na kontrolowanej głębokości w długim kontekście. Poproś model o jego pobranie. Przesuwaj głębokość × długość. Oryginalny benchmark długiego kontekstu. Modele frontowe teraz nasycają ten test, więc jest to konieczny, ale niewystarczający baseline.

**RULER (Nvidia, 2024).** 13 typów zadań w 4 kategoriach: pobieranie (pojedyncze / multi-key / multi-value), multi-hop tracing (śledzenie zmiennych), agregacja (częstotliwość wspólnych słów), QA. Konfigurowalna długość kontekstu (4k do 128k+). Ujawnia modele, które nasycają NIAH, ale nie radzą sobie z multi-hop. W wydaniu z 2024 roku tylko połowa z 17 modeli deklarujących 32k+ kontekst utrzymała jakość na poziomie 32k.

**LongBench v2 (2024).** 503 pytania wielokrotnego wyboru, konteksty 8k-2M słów, sześć kategorii zadań: single-doc QA, multi-doc QA, long in-context learning, long dialogue, code repo, long structured data. Produkcyjny benchmark dla rzeczywistego zachowania długiego kontekstu.

**MRCR (Multi-Round Coreference Resolution).** Multi-turn coreference na skali. Warianty z 8, 24 i 100 igłami. Ujawnia, ile faktów model może jednocześnie przetwarzać, zanim attention się zdegraduje.

**NoLiMa.** "Igła nieleksykalna." Igła i zapytanie nie mają dosłownego nakładania się; pobieranie wymaga jednego kroku rozumowania semantycznego. Trudniejsze niż NIAH.

**HELMET.** Łączy wiele dokumentów, zadaje pytanie z dowolnego z nich. Testuje selektywne attention.

**BABILong.** Osadza łańcuchy rozumowania bAbI wewnątrz nieistotnych stert. Testuje rozumowanie w stercie, nie tylko pobieranie.

### Co faktycznie raportować

- **Reklamowane okno kontekstu.** Liczba z specyfikacji.
- **Efektywna długość pobierania.** NIAH pass przy pewnym progu (np. 90%).
- **Efektywna długość rozumowania.** Multi-hop lub agregacja pass przy tym samym progu.
- **Krzywa degradacji.** Dokładność vs długość kontekstu, wykreślona dla każdego typu zadania.

Dwie liczby do specyfikacji: retrieval-effective i reasoning-effective. Zwykle reasoning-effective wynosi 25-50% reklamowanego okna.

## Zbuduj to

### Krok 1: niestandardowy NIAH dla twojej domeny

Zobacz `code/main.py`. Szkielet:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Powtarzaj wypełniacz, aż będzie wystarczająco długi, by wypełnić ciało sterty.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Przesuwaj `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Wykreśl heatmapę. To jest karta NIAH dla twojego modelu docelowego.

### Krok 2: wariant multi-needle

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Pytania takie jak "Jakie są trzy magiczne słowa?" wymagają pobrania wszystkich trzech. Sukces single-needle nie prognozuje sukcesu multi-needle.

### Krok 3: multi-hop variable tracing (styl RULER)

```python
haystack = """X1 = 42. ... (wypełniacz) ... X2 = X1 + 10. ... (wypełniacz) ... X3 = X2 * 2."""
question = "What is X3?"
```

Odpowiedź wymaga łańcucha trzech przypisań. Modele frontowe przy 128k często spadają do 50-70% dokładności tutaj.

### Krok 4: LongBench v2 na twoim stacku

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Raportuj dokładność per kategoria. Zagregowane wyniki ukrywają duże różnice między zadaniami.

## Pułapki

- **Ewaluacja tylko NIAH.** Zdanie testu NIAH przy 1M tokenów nic nie mówi o multi-hop. Zawsze uruchamiaj RULER lub niestandardowy test multi-hop.
- **Jednorodne próbkowanie głębokości.** Wiele implementacji testuje tylko depth=0.5. Testuj depth=0, 0.25, 0.5, 0.75, 1.0 — efekt "zagubiony w środku" jest rzeczywisty.
- **Nakładanie leksykalne z wypełniaczem.** Jeśli igła dzieli słowa kluczowe z wypełniaczem, pobieranie staje się trywialne. Używaj igieł bez nakładania się stylu NoLiMa.
- **Ignorowanie latencji.** Prompty 1M-tokenowe zajmują 30-120 sekund na prefill. Mierz time-to-first-token obok dokładności.
- **Liczby podawane przez vendorów.** OpenAI, Google, Anthropic wszyscy publikują własne wyniki. Zawsze uruchamiaj niezależnie na swoim przypadku użycia.

## Użyj tego

Stack na 2026:

| Sytuacja | Benchmark |
|-----------|-----------|
| Szybki sanity check | Custom NIAH przy 3 głębokościach × 3 długościach |
| Wybór modelu do produkcji | RULER (13 zadań) przy docelowej długości |
| Jakość QA w realnym świecie | LongBench v2 single-doc-QA subset |
| Multi-hop reasoning | BABILong lub niestandardowy variable-tracing |
| Konwersacyjny / dialogue | MRCR 8-needle przy docelowej długości |
| Regresja po upgrade modelu | Stały wewnętrzny NIAH + RULER harness, uruchamiaj na każdym nowym modelu |

Zasada kciuka dla produkcji: nigdy nie ufaj oknu kontekstu, dopóki nie masz NIAH + 1 zadanie rozumowania przy zamierzonej długości.

## Wdróż to

Zapisz jako `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Ćwiczenia

1. **Łatwe.** Zbuduj NIAH z 3 głębokościami (0.25, 0.5, 0.75) × 3 długościami (1k, 4k, 16k). Uruchom na dowolnym modelu. Wykreśl pass rate jako heatmapę 3×3.
2. **Średnie.** Dodaj wariant z 3 igłami. Zmierz pobieranie wszystkich 3 przy każdej długości. Porównaj do single-needle pass rate przy tej samej długości.
3. **Trudne.** Skonstruuj zadanie variable-tracing (X1 → X2 → X3, z 3 skokami) osadzone w 64k wypełniaczu. Zmierz dokładność na 3 modelach frontowych. Raportuj efektywną długość rozumowania per model.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| NIAH | Needle in haystack | Zasadź fakt we wypełniaczu, poproś model o jego pobranie. |
| RULER | NIAH na sterydach | 13 typów zadań obejmujących pobieranie / multi-hop / agregację / QA. |
| Effective context | Rzeczywista pojemność | Długość, przy której dokładność nadal utrzymuje się powyżej progu. |
| Lost in the middle | Bias głębokości | Modele niedostatecznie atencjonują treść w środku długich wejść. |
| Multi-needle | Wiele faktów naraz | Wiele zasadzonych; testuje żonglowanie attention, nie samo pobieranie. |
| MRCR | Multi-round coref | Igły 8, 24 lub 100; ujawnia nasycenie attention. |
| NoLiMa | Igła nieleksykalna | Igła i zapytanie nie dzielą dosłownych tokenów; wymaga rozumowania. |

## Dalsze czytanie

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — oryginalne repo NIAH.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — benchmark multi-task.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — ewaluacja długiego kontekstu w realnym świecie.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — trudniejsze igły.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — rozumowanie w stercie.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — artykuł o biasie głębokości.

---

**Poprawione błędy:**

1. `zdegoraduje` → `zdegraduje` (linia ~38)
2. Komentarz w kodzie `# Repeat filler until long enough to fill the haystack body.` → `# Powtarzaj wypełniacz, aż będzie wystarczająco długi, by wypełnić ciało sterty.`
3. Zamiana angielskiego `filler` → polskie `wypełniacz` we wszystkich komentarzach i opisach tekstowych (w kodzie pozostawiono nazwy zmiennych bez zmian)
4. Dodatkowe poprawki spójności: "Zdanie egzaminu NIAH" → "Zdanie testu NIAH", "(filler)" → "(wypełniacz)" w stringu przykładowym, "wypełniaczem" w opisie Pułapki