# SESSION_STATE - AI Engineering Course PL

## Projekt
- **Nazwa:** AI Engineering Course PL
- **Lokalizacja:** `/c/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch`
- **GitHub:** https://github.com/hublorx/ai-engineering-from-scratch (fork z rohitg00/ai-engineering-from-scratch)
- **Cel:** Pełne tłumaczenie kursu EN→PL (423 lekcji, 20 faz)

## Resume Prompt
```
Kontynuuj tłumaczenie AI Engineering Course. Status: Fazy 00-02 GOTOWE (52 lekcji).
Uruchom agentów dla Faz 03-04. Skillet: ~/.claude/skills/translate-markdown-course/
```

## Session History

### Session 1 (2026-04-26) - ~1.5h
**Cel:** Przetłumaczyć Fazę 00-02 + root docs + glossary

**Wykonane:**
- Fork repo z GitHub
- Stworzono skill `translate-markdown-course`
- Przetłumaczono 52 lekcje (Fazy 00-02)
- Przetłumaczono 10 plików dokumentacji (root + glossary)
- 14 subagentów MiniMax Sonnet równolegle
- Commit: `424ba8a` + `bda83fa`

**Commit GitHub:**
```
commit 424ba8a
Tłumaczenie EN→PL: Faza 00-02 + Root docs + Glossary
122 files changed, 14273 insertions(+), 10029 deletions(-)
```

## Decisions Made This Session

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Nadpisuj `en.md` polską wersją | Łatwiejsze niż tworzenie `pl.md` obok | Brak oryginału EN w tym branchu |
| Branch `en` = oryginał | Dla sync z upstream | Oryginał zachowany |
| MiniMax Sonnet do tłumaczenia | Szybki, tani, dobry dla prostych tekstów | ~$0.01/lekcja |
| "Learning Objectives" → "Cele uczenia się" | Spójność w całym kursie | Standard Polish header |
| Kod po angielsku | Naturalne dla programistów | Czytelność kodu |

## Mental Context

**Approach:** Wielowarstwowy pipeline translate→verify→fix→loop. MiniMax Sonnet tłumaczy, weryfikator sprawdza 6 kategorii błędów, fix jeśli są.

**Key insight:** Agent glossary (translator-glossary) zapisuje output agenta zamiast tłumaczenia - trzeba było ręcznie poprawić. Sączy się też timeout na długich zadaniach (600s watchdog). Agent translator-01-1 mimo timeouta zdążył przetłumaczyć 4 lekcje.

**Watch out for:** Agent może zapisać output promptu zamiast tłumaczenia. Zawsze sprawdzać wynikowy plik po skończeniu agenta.

## Next Action
1. Powiedzieć "kontynuuj tłumaczenie AI Engineering Course"
2. Uruchomić 4-6 agentów dla Faz 03-04 (Deep Learning Core, Computer Vision)
3. Sprawdzać jakość próbek po każdym agencie

## Status Faz

| Faza | Status | Lekcji |
|------|--------|--------|
| 00 | ✅ GOTOWE | 12 |
| 01 | ✅ GOTOWE | 22 |
| 02 | ✅ GOTOWE | 18 |
| 03-19 | DO ZROBIENIA | ~371 |

## Pliki do przeczytania przy starcie
1. `projects/ai-engineering-course-pl/PHASES_STATUS.md`
2. `projects/ai-engineering-course-pl/SESSION_STATE.md` (ten plik)
3. `~/.claude/skills/translate-markdown-course/SKILL.md`
