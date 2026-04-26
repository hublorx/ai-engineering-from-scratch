#!/usr/bin/env python3
"""Translate AI Engineering Course lessons to Polish."""

import json
import subprocess
import sys
import os

def minimax_translate(text, system_prompt="""
Jestes translatorem kursu AI/ML z EN->PL. Minimal intervention, nie ulepszaj.
ZOSTAWIJ po angielsku: kod, sekcje code-related, Phase X, nazwy techniczne (API, GPU, SQL, etc.)
Tlumacz: naglowki "Learning Objectives" -> "Cele uczenia się", "The Problem" -> "Problem"
DOZWOLONE: API, GPU, CPU, RAM, SQL, Python, PyTorch, machine learning, neural network, etc.
NIE tlumacz blokow kodu.
Przecinki przed: ze, bo, zeby, i (dwa niezalezne zdania), ktory, a (kontrast), wrec, az, zanim, gdy, albo, lub.
Polish diacritics: pamietam -> pamietam, Cie -> Cie.
Output JSON: {"translation": "..."}
"""):
    result = subprocess.run(
        ["python", "execution/minimax_utils.py", "translate", text, system_prompt],
        capture_output=True,
        text=True,
        cwd="/c/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch"
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise Exception(f"Translation failed: {result.stderr}")
    try:
        data = json.loads(result.stdout.strip())
        return data.get("translation", data.get("text", ""))
    except:
        return result.stdout.strip()

def minimax_verify(text):
    """Verify Polish translation for errors."""
    prompt = """Sprawdz czy sa bledy w tym polskim tlumaczeniu lekcji. Raportuj dokladnie.

6 kategorii bledow:
1. DIAKRYTYKI: zjqebany, Huełałem, pisującego, przylapać, pamietam, Cie, Jerkałem
2. NIEPOLSKIE ZNAKI: Cyrylica, rosyjskie, chińskie znaki
3. BRAK PRZECINKA przed: ze, bo, zeby, i (dwa niezalezne zdania), ktory, a (kontrast)
4. ANGLICYZMY poza lista: API, GPU, CPU, RAM, SQL, Python, ML, DL, AI, NN, etc.
5. KOD w tlumaczeniu - NIE tlumaczmy kodu!
6. Angielskie sekcje co powinny byc polskie: "The Problem" -> "Problem", "Learning Objectives" -> "Cele uczenia się"

Jesli ZERO BLEDOW: napisz "ZERO ERRORS"
Jesli sa bledy: "BŁĘDY: N" + lista

Przeczytaj caly tekst, sprawdz kazda kategorie, podaj dokladne linie z bledami."""

    result = subprocess.run(
        ["python", "execution/minimax_utils.py", "translate", text, prompt],
        capture_output=True,
        text=True,
        cwd="/c/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch"
    )
    if result.returncode != 0:
        return f"Verify error: {result.stderr}"
    return result.stdout.strip()

def translate_lesson(lesson_path):
    """Translate a single lesson with verification loop (max 5 iterations)."""
    with open(lesson_path, 'r', encoding='utf-8') as f:
        source = f.read()

    print(f"  Translating {lesson_path}...")
    translated = minimax_translate(source)

    for iteration in range(1, 6):
        print(f"  Verification iteration {iteration}...")
        verification = minimax_verify(translated)

        if "ZERO ERRORS" in verification:
            print(f"  Verification passed (iteration {iteration})")
            break

        if "BŁĘDY:" in verification or "BLAD" in verification.upper():
            print(f"  Fixing errors (iteration {iteration})...")
            fix_prompt = f"""Popraw WSZYSTKIE wymienione bledy w tlumaczeniu. Nie zmieniaj nic poza poprawionymi bledami.

BLĘDY:
{verification}

TŁUMACZENIE DO POPRAWY:
{translated}
"""
            fixed = minimax_translate(translated,
                system_prompt="Fix all errors in this Polish translation. Preserve code blocks. Output JSON with 'translation' field.")
            translated = fixed
        else:
            print(f"  Unknown verification response: {verification[:200]}")
            break

    with open(lesson_path, 'w', encoding='utf-8') as f:
        f.write(translated)

    return translated

def main():
    lessons = [
        "phases/02-ml-fundamentals/01-what-is-machine-learning/docs/en.md",
        "phases/02-ml-fundamentals/02-linear-regression/docs/en.md",
        "phases/02-ml-fundamentals/03-logistic-regression/docs/en.md",
        "phases/02-ml-fundamentals/04-decision-trees/docs/en.md",
        "phases/02-ml-fundamentals/05-support-vector-machines/docs/en.md",
    ]

    base = "/c/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch"
    results = []

    for lesson in lessons:
        path = os.path.join(base, lesson)
        print(f"\nTranslating: {lesson}")
        try:
            translate_lesson(path)
            results.append(("OK", lesson))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(("FAIL", lesson))

    print("\n" + "="*60)
    print("FAZA 02 (01-05): GOTOWE")
    print("="*60)
    for status, lesson in results:
        print(f"  [{status}] {lesson}")

if __name__ == "__main__":
    main()