# Dziennik zmian

Nowości w programie nauczania. Najnowsze na górze.

Format luźno wzorowany na [Keep a Changelog](https://keepachangelog.com/). Każdy wpis wymienia fazę, lekcję i co się zmieniło, dzięki czemu uczniowie mogą od razu przejść do różnic.

## [Niewydane]

### Dodane
- `scripts/scaffold-lesson.sh` — szkielet, który tworzy `phases/NN-phase/NN-lesson/` z pełną strukturą folderów i szkieletem `docs/en.md` wypełnionym na podstawie `LESSON_TEMPLATE.md`.
- `.github/PULL_REQUEST_TEMPLATE.md` — lista kontrolna dla współtwórców (kod działa, brak komentarzy w kodzie, najpierw zbudowane od zera, atomowy commit na lekcję, wiersz linku markdown w ROADMAP).
- `.github/ISSUE_TEMPLATE/bug_report.md` i `new_lesson_proposal.md` — ustrukturyzowany formularz zgłoszeń błędów i propozycji lekcji.
- Niniejszy `CHANGELOG.md`.

## 2026-04 — Faza 4: Widzenie komputerowe ukończona

### Dodane
- Wszystkie 28 lekcji Fazy 4, obejmujące fundamenty obrazu przez wielomodalne widzenie (VLM, 3D, wideo, samonadzorowane).
- Wiersze Fazy 4 w `ROADMAP.md` powiązane jako linki markdown do folderów lekcji, dzięki czemu strona internetowa je wyświetla.

### Naprawione
- Przegląd precyzji Fazy 4 w ponad 15 lekcjach:
  - `phase-4/02`: kalkulator kształtu określa obsługę RF/stride dla adaptive pool, flatten i linear.
  - `phase-4/03`: opis selektora backbone wymienia wszystkie omawiane rodziny; dodano wskazówki dla głowy w OCR, medycynie, przemyśle.
  - `phase-4/04`: diagnostyka klasyfikacji używa ilościowych progów dla każdego trybu awarii; `n/a` deklarowane dla niezdefiniowanych metryk; zabezpieczenie dla mniej niż 3 klas.
  - `phase-4/06`: czytnik metryk detekcji używa `AP@0.5` (nie `mAP@0.5`); recall per-klasowy deklarowany opcjonalnie; projektant kotwic clarifikuje obcinanie stride i ścieżkę jednej kotwicy na poziom.
  - `phase-4/10`: wybierak sampler deklaruje `unet_forward_ms` jako wejście; strażnik ControlNet promowany do reguły 0.
  - `phase-4/14`: inspektor ViT wyrównany z regułą odmowy — próby portów są audytowane, nie popierane.
  - `phase-4/24`: wybierak stosu open-vocab ma jawną pierwszeń reguł i semantykę filtra licencji; projektant koncepcji rozwiązuje konflikt step-5/rule-80.
  - `phase-4/25`: dokumentacja VLM `_merge` rzuca opisowy `ValueError` przy niezgodności placeholder; CMER normalizuje wewnętrznie.
  - `phase-4/27`: `synthetic_frames` przycina GT boxes do H/W ramki.
  - `phase-4/28`: `rope_3d` waliduje podział dim; usunięto nieużywany import `F` z przykładu DiT block.

## 2026-Q1 i wcześniej

### Dodane
- Faza 0 (Konfiguracja i narzędzia): wszystkie 12 lekcji.
- Faza 1 (Fundamenty matematyczne): wszystkie 22 lekcje.
- Faza 2 (Fundamenty ML): wszystkie 18 lekcji.
- Faza 3 (Głębokie uczenie — rdzeń): lekcje podstawowe przez perceptron, backprop, optymalizatory.
- Wbudowane umiejętności Claude Code: `find-your-level` (test umieszczający) i `check-understanding` (test na fazę).
- Strona internetowa pod `aiengineeringfromscratch.com`: katalog, strony lekcji, mapa drogowa, 277-terminowy słownik.
- Początkowe szkielety dla wszystkich 20 faz (`phases/00-*` przez `phases/19-*`).
- `LESSON_TEMPLATE.md`, `CONTRIBUTING.md`, `ROADMAP.md`, `README.md`.

[Niewydane]: https://github.com/rohitg00/ai-engineering-from-scratch/compare/HEAD...HEAD