#!/usr/bin/env python3
"""Translate and verify a single lesson EN→PL using MiniMax Sonnet."""

import re
import sys
from pathlib import Path

# Add execution dir to path
sys.path.insert(0, r'C:\VisualStudioCodeRepo\AIWorkspace\execution')

from minimax_utils import get_client, call_minimax, strip_think

TRANSLATE_SYSTEM = """Jestes translatorem kursu IT/AI. Tlumaczysz lekcje z EN→PL wiernie, zachowujac styl i terminologię techniczną. MINIMAL INTERVENTION - nie ulepszaj, nie skracaj, nie zmieniaj tonu.

ZASADY:
- Kod NIE tlumacz - zostaw dokladnie jak jest
- "Learning Objectives" → "Cele uczenia się"
- "The Problem" → "Problem"
- "The Concept" → "Koncepcja"
- Wszystkie sekcje Markdown przetlumacz na polski
- Bloki kodu (```python...) NIE tlumacz
- Linki URL zostaw bez zmian
- Obrazy NIE tlumacz

Wynik: pelny przetlumaczony plik Markdown w jezyku polskim."""

VERIFY_SYSTEM = """Jestes weryfikatorem tlumaczen. Sprawdzasz polskie tlumaczenia lekcji AI Engineering pod katem bledow. Raportuj dokladnie co jest zle i gdzie.

6 KATEGORII BLEDOW DO SPRAWDZENIA:
1. DIAKRYTYKI: zjqebany→zjebany, Huelałem→Hulałem, pamietam→pamiętam, itp.
2. NIEPOLSKIE ZNAKI: Cyrylica/rosyjskie/chińskie znaki
3. BRAK PRZECINKA: przed "ze", "bo", "zeby", "i" (dwa niezalezne zdania), "ktory", "a" (kontrast), itp.
4. ANGLICYZMY poza lista dozwolonych: API, GPU, Python, PyTorch, itp. sa OK
5. KOD: bloki ```python...``` NIE moga byc przetlumaczone
6. SEKCJE: "Learning Objectives" musi byc "Cele uczenia się", itp.

Format raportu gdy sa bledy:
BLEDY: N
1. [KATEGORIA] Linia X: [opis]
   Tresc: "[fragment z bledem]"
   Poprawic na: "[poprawna wersja]"

Format raportu gdy zero bledow:
ZERO ERRORS ✓"""


def translate_file(file_path: str, max_iter: int = 5) -> tuple[str, int]:
    source = Path(file_path).read_text(encoding='utf-8')
    translated = None

    for iteration in range(1, max_iter + 1):
        client = get_client()
        user = f"""Przetlumacz nastepujacy plik na jezyk polski. ZACHOWAJ wszystkie naglowki sekcji po angielsku jesli nie sa wymienione w zasadach powyzej.

PLIK ZRODLOWY:
{source}"""

        result = call_minimax(client, TRANSLATE_SYSTEM, user, temperature=0.1, model="MiniMax-M2.7")
        if not result:
            print(f"  [Iteration {iteration}] Translation call failed", file=sys.stderr)
            break
        translated = strip_think(result)

        verify_user = f"""Zweryfikuj to tlumaczenie. Sprawdz WSZYSTKIE 6 kategorii bledow.

TLUMACZENIE:
{translated}"""

        verify_result = call_minimax(client, VERIFY_SYSTEM, verify_user, temperature=0.1, model="MiniMax-M2.7")
        if not verify_result:
            print(f"  [Iteration {iteration}] Verify call failed", file=sys.stderr)
            break

        verify_text = strip_think(verify_result)

        if "ZERO ERRORS" in verify_text and "✓" in verify_text:
            print(f"  [Iteration {iteration}] ZERO ERRORS - OK")
            break

        error_match = re.search(r'BLEDY:\s*(\d+)', verify_text)
        if error_match:
            n_errors = int(error_match.group(1))
            print(f"  [Iteration {iteration}] {n_errors} errors found, fixing...")

            fix_user = f"""Popraw WSZYSTKIE bledy w tym tlumaczeniu. Lista bledow:

{verify_text}

TLUMACZENIE DO POPRAWY:
{translated}"""

            fix_result = call_minimax(client, TRANSLATE_SYSTEM, fix_user, temperature=0.1, model="MiniMax-M2.7")
            if not fix_result:
                print(f"  [Iteration {iteration}] Fix call failed", file=sys.stderr)
                break
            translated = strip_think(fix_result)
        else:
            print(f"  [Iteration {iteration}] Verification unclear, retrying translation...")
            source = translated

    return translated or source, iteration


def main():
    if len(sys.argv) < 2:
        print("Usage: python _translate_lesson.py <path_to_en.md>")
        sys.exit(1)

    file_path = sys.argv[1]
    p = Path(file_path)
    print(f"Translating: {file_path}")

    translated, iters = translate_file(str(p))
    p.write_text(translated, encoding='utf-8')
    print(f"Saved ({iters} iterations)")


if __name__ == '__main__':
    main()