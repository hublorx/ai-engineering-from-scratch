#!/usr/bin/env python3
"""Translate AI Engineering Course Phase 02 lessons (01-05) to Polish."""

import sys
import os

# Add AIWorkspace to path
sys.path.insert(0, '/c/VisualStudioCodeRepo/AIWorkspace')
from execution.minimax_utils import get_client, call_minimax

SYSTEM_PROMPT = """Jestes translatorem kursu AI/ML z EN->PL. Minimal intervention, nie ulepszaj, nie skracaj.

ZOSTAWIJ po angielsku:
- Kod (bloki ```python ...```)
- Nazwy funkcji: train_model(), forward(), predict()
- Zmienne: learning_rate, batch_size, hidden_dim
- Frameworki: PyTorch, TensorFlow, scikit-learn, HuggingFace
- Metryki: accuracy, precision, recall, F1, AUC
- Skróty: ML, DL, AI, NN, CNN, RNN, LSTM, NLP
- Sekcje code-related: ## Code Examples, ## Implementation

TŁUMACZ:
- "Learning Objectives" -> "Cele uczenia się"
- "The Problem" -> "Problem"
- "The Concept" -> "Koncepcja"
- "Key Terms" -> "Kluczowe terminy"
- "Exercises" -> "Ćwiczenia"
- Naglowki sekcji (poza code-related)

Polish diacritics (popraw literowki):
- pamietam -> pamietam, Cie -> Cie, Huełałem -> Hulałem
- zjqebany -> zjebany, pisującego -> piszącego

Przecinki OBOWIAZKOWO przed: ze, bo, zeby, i (dwa niezalezne zdania), ktory, a (kontrast), wrec, az, zanim, gdy, albo, lub

Format output: JSON z polem "translation" zawierajacym przetlumaczony markdown."""

VERIFY_PROMPT = """Sprawdz czy sa bledy w tym polskim tlumaczeniu lekcji kursu AI Engineering. Raportuj dokladnie.

6 kategorii bledow do sprawdzenia:
1. DIAKRYTYKI: zjqebany->zjebany, Huełałem->Hulałem, pisującego->piszącego, przylapać->przyłapać, pamietam->pamiętam, Cie->Cię
2. NIEPOLSKIE ZNAKI: Cyrylica/rosyjskie, chińskie, "takже"->"także"
3. BRAK PRZECINKA przed: ze, bo, zeby, i (dwa niezalezne zdania), ktory, a (kontrast), wrec, az, zanim, gdy, albo, lub
4. ANGLICYZMY poza lista dozwolonych (API, GPU, CPU, RAM, SQL, Python, ML, DL, AI, NN, etc sa OK)
5. KOD w tlumaczeniu - BLOKI KODU NIE MOGA BYC PRZETLUMACZONE
6. Angielskie sekcje co powinny byc polskie: "The Problem"->"Problem", "Learning Objectives"->"Cele uczenia się"

Jesli ZERO BLEDOW: napisz "ZERO ERRORS"
Jesli sa bledy: "BŁĘDY: N" + dokladna lista z liniami i poprawkami"""


def translate(text):
    client = get_client()
    result = call_minimax(client, SYSTEM_PROMPT, text, temperature=0.1, model="MiniMax-M2.7")
    if result and "translation" in result.lower():
        import json, re
        try:
            data = json.loads(result)
            return data.get("translation", result)
        except:
            m = re.search(r'"translation"\s*:\s*"((?:[^"\\]|\\.)*)"', result, re.DOTALL)
            if m:
                return m.group(1)
    return result or ""


def verify(text):
    client = get_client()
    result = call_minimax(client, VERIFY_PROMPT, text, temperature=0.1, model="MiniMax-M2.7")
    return result or ""


def translate_lesson(path, max_iterations=5):
    """Translate a lesson file with verification loop."""
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()

    translated = translate(source)
    if not translated:
        print(f"    Translation failed")
        return False

    for iteration in range(1, max_iterations + 1):
        print(f"    Iteration {iteration}...")
        v = verify(translated)

        if "ZERO ERRORS" in v:
            print(f"    Verification passed!")
            break

        if "BŁĘDY:" in v or ("ERRORS" in v and "ZERO" not in v):
            print(f"    Fixing errors...")
            # Extract errors and create fix prompt
            fix_prompt = f"""Popraw WSZYSTKIE wymienione bledy w polskim tlumaczeniu.
NIE zmieniaj nic poza poprawionymi bledami. ZACHOWAJ wszystkie bloki kodu bez zmian.

BLĘDY:
{v}

TŁUMACZENIE:
{translated}
"""
            client = get_client()
            fixed = call_minimax(client, SYSTEM_PROMPT, fix_prompt, temperature=0.1, model="MiniMax-M2.7")
            if fixed:
                translated = fixed
        else:
            print(f"    Unknown verification response, continuing...")
            break

    with open(path, 'w', encoding='utf-8') as f:
        f.write(translated)

    return True


def main():
    base = "/c/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch"
    lessons = [
        "phases/02-ml-fundamentals/01-what-is-machine-learning/docs/en.md",
        "phases/02-ml-fundamentals/02-linear-regression/docs/en.md",
        "phases/02-ml-fundamentals/03-logistic-regression/docs/en.md",
        "phases/02-ml-fundamentals/04-decision-trees/docs/en.md",
        "phases/02-ml-fundamentals/05-support-vector-machines/docs/en.md",
    ]

    print("="*60)
    print("Translating Phase 02 Lessons (01-05) to Polish")
    print("="*60)

    results = []
    for lesson in lessons:
        path = os.path.join(base, lesson)
        print(f"\nProcessing: {lesson}")
        try:
            success = translate_lesson(path)
            results.append(("OK" if success else "FAIL", lesson))
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(("FAIL", lesson))

    print("\n" + "="*60)
    print("FAZA 02 (01-05): GOTOWE")
    print("="*60)
    for status, lesson in results:
        print(f"  [{status}] {lesson}")


if __name__ == "__main__":
    main()