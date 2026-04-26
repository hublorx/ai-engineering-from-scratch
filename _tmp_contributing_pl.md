# Wnoszenie wkładu w Inżynierię AI od Zera

Dziękujemy za chęć tworzenia lepszej edukacji AI dla wszystkich.

## Sposoby wnoszenia wkładu

### 1. Dodaj nową lekcję

Każda lekcja znajduje się w `phases/XX-nazwa-fazy/NN-nazwa-lekcji/` z tą strukturą:

```
NN-nazwa-lekcji/
├── code/           Co najmniej jedna wykonywalna implementacja
├── notebook/       Jupyter notebook do eksperymentowania (opcjonalne)
├── docs/
│   └── en.md       Dokumentacja lekcji (wymagane)
└── outputs/        Prompty, umiejętności lub agenci produkowane przez tę lekcję (jeśli dotyczy)
```

**Format dokumentacji lekcji** (`en.md`):

```markdown
# Tytuł lekcji

> Jednozdaniowe motto — główna idea w jednym zdaniu.

## Problem

Dlaczego to jest ważne? Czego nie możesz zrobić bez tego?

## Koncepcja

Wyjaśnij za pomocą diagramów, wizualizacji i intuicji. Kod przychodzi później.

## Zbuduj to

Implementacja krok po kroku od zera.

## Użyj tego

Teraz użyj prawdziwego frameworka lub biblioteki, żeby zrobić to samo.

## Wyślij to

Prompt, umiejętność, agent lub narzędzie produkowane przez tę lekcję.

## Ćwiczenia

1. Ćwiczenie pierwsze
2. Ćwiczenie drugie
3. Ćwiczenie challenge
```

### 2. Dodaj tłumaczenie

Utwórz nowy plik w folderze `docs/` dowolnej lekcji:

```
docs/
├── en.md    (Angielski — zawsze wymagane)
├── zh.md    (Chiński)
├── ja.md    (Japoński)
├── es.md    (Hiszpański)
├── hi.md    (Hindi)
└── ...
```

Zachowaj tę samą strukturę co w wersji angielskiej. Tłumacz treść, nie kod.

### 3. Dodaj wynik

Jeśli lekcja powinna produkować wielokrotnego użytku prompt, umiejętność, agenta lub serwer MCP:

1. Utwórz go w folderze `outputs/` lekcji
2. Dodaj odniesienie w indeksie `outputs/` na najwyższym poziomie

**Format prompta:**

```markdown
---
name: nazwa-prompta
description: Co ten prompt robi
phase: 14
lesson: 01
---

[System prompt lub szablon tutaj]
```

**Format umiejętności:**

```markdown
---
name: nazwa-umiejętności
description: Czego ta umiejętność uczy
version: 1.0.0
phase: 14
lesson: 01
tags: [agents, loops]
---

[Zawartość umiejętności tutaj]
```

### 4. Napraw błędy lub ulepszy istniejące lekcje

- Napraw kod, który się nie uruchamia
- Ulepsz wyjaśnienia
- Dodaj lepsze diagramy
- Zaktualizuj nieaktualne informacje

### 5. Dodaj ćwiczenia lub projekty

Więcej ćwiczeń i projektów jest zawsze mile widziane, zwłaszcza takie, które łączą wiele faz.

## Wytyczne

- **Kod musi działać.** Każdy plik z kodem powinien się wykonać bez błędów przy podanych zależnościach.
- **Brak komentarzy w kodzie.** Kod powinien być zrozumiały sam w sobie. Używaj dokumentacji do wyjaśnień.
- **Najlepszy język do zadania.** Nie wymuszaj Pythona tam, gdzie TypeScript lub Rust jest lepszym wyborem.
- **Najpierw buduj od zera.** Zawsze implementuj koncepcję od pierwszych zasad przed pokazaniem wersji z frameworkiem.
- **Zachowaj praktyczność.** Teoria służy praktyce, nie na odwrót.
- **Bez AI śmieci.** Pisz jak człowiek. Bądź bezpośredni. Usuń wypełniacz.

## Proces Pull Request

1. Forkuj repozytorium
2. Utwórz gałąź funkcjonalną (`git checkout -b add-lesson-phase3-gradient-descent`)
3. Wprowadź swoje zmiany
4. Upewnij się, że cały kod działa
5. Złóż pull request z jasnym opisem

## Kodeks postępowania

Zobacz [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Bądź miły, bądź pomocny, bądź konstruktywny.