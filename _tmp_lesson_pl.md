# Szablon Lekcji

Użyj tego szablonu podczas tworzenia nowej lekcji. Skopiuj strukturę folderów i wypełnij treść.

## Struktura Folderów

```
NN-nazwa-lekcji/
├── code/
│   ├── main.py            (główna implementacja)
│   ├── main.ts            (wersja TypeScript, jeśli dotyczy)
│   ├── main.rs            (wersja Rust, jeśli dotyczy)
│   └── main.jl            (wersja Julia, jeśli dotyczy)
├── notebook/
│   └── lesson.ipynb       (notatnik Jupyter do eksperymentowania)
├── docs/
│   └── en.md              (dokumentacja lekcji)
└── outputs/
    ├── prompt-*.md         (prompty wygenerowane przez tę lekcję)
    └── skill-*.md          (umiejętności wygenerowane przez tę lekcję)
```

## Format Dokumentacji (docs/en.md)

```markdown
# [Tytuł Lekcji]

> [Jednolinijkowe hasło — podstawowa idea, która zostaje w pamięci]

**Typ:** Build | Learn
**Języki:** Python, TypeScript, Rust, Julia (wymień używane)
**Wymagania wstępne:** [Wymień potrzebne wcześniejsze lekcje]
**Czas:** ~[szacowany czas] minut

## Problem

[2-3 akapity. Czego nie możesz zrobić bez tego? Dlaczego powinno cię to obchodzić?
Bądź konkretny — pokaż scenariusz, w którym nieznajomość tego boli.]

## Koncepcja

[Wytłumacz za pomocą diagramów i intuicji. Na razie bez kodu.
Używaj diagramów ASCII, tabel lub linków do grafik w aplikacji webowej.
Buduj mentalne modele przed implementacją.]

## Zbuduj To

[Krok po kroku implementacja od zera.
Zacznij od najprostszej wersji, potem dodawaj złożoność.
Każdy blok kodu powinien działać samodzielnie.]

### Krok 1: [Nazwa]

[Wyjaśnienie]

    [blok kodu]

### Krok 2: [Nazwa]

[Wyjaśnienie]

    [blok kodu]

[...kontynuuj...]

## Użyj Tego

[Teraz pokaż, jak frameworki/biblioteki robią to samo.
Porównaj swoją wersję od zera z wersją biblioteczną.
To dowodzi koncepcji i wprowadza praktyczne narzędzia.]

## Wydaj To

[Jaki wielokrotnego użytku artefakt produkuje ta lekcja?
Może to być prompt, umiejętność, agent, serwer MCP lub narzędzie.
Umieść go tutaj i zapisz w folderze outputs/.]

## Ćwiczenia

1. [Łatwe — utrwal podstawową koncepcję]
2. [Średnie — zastosuj to do innego problemu]
3. [Trudne — rozszerz lub połącz z wcześniejszymi lekcjami]

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| [termin] | [powszechna концепция] | [rzeczywista definicja] |

## Dalsza Lektura

- [Zasób 1](url) — [dlaczego warto przeczytać]
- [Zasób 2](url) — [dlaczego warto przeczytać]
```

## Wytyczne Dotyczące Plików z Kodem

- Kod musi działać bez błędów
- Brak komentarzy — kod powinien być zrozumiały sam w sobie
- Używaj języka, który najlepiej pasuje do tematu
- Dołącz plik `requirements.txt` lub równoważny, jeśli są zależności
- Zaczynaj prosto, buduj złożoność stopniowo
- Każda funkcja i klasa powinny mieć jasny cel

## Format Plików Wyjściowych

### Prompty

```markdown
---
name: nazwa-promptu
description: Co ten prompt robi
phase: [numer fazy]
lesson: [numer lekcji]
---

[Treść promptu]
```

### Umiejętności

```markdown
---
name: nazwa-umiejetnosci
description: Czego uczy ta umiejętność
version: 1.0.0
phase: [numer fazy]
lesson: [numer lekcji]
tags: [relevantne, tagi]
---

[Treść umiejętności]
```