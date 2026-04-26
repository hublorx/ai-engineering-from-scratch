#!/usr/bin/env python3
"""Translate AI Engineering Course lessons (Phase 01, lessons 11-15) EN→PL."""
import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

BASE_DIR = Path("C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch")

LESSONS = [
    ("11-singular-value-decomposition", "Rozklad wedlug wartosci singularnych"),
    ("12-tensor-operations", "Operacje na tensorach"),
    ("13-numerical-stability", "Stabilnosc numeryczna"),
    ("14-norms-and-distances", "Normy i odleglosci"),
    ("15-statistics-for-ml", "Statystyka dla uczenia maszynowego"),
]

TRANSLATE_SYSTEM = """RULES (MANDATORY):
1. MINIMAL INTERVENTION - tlumacz wiernie, NIE ulepszaj, NIE skracaj, NIE zmieniaj tonu
2. ZOSTAN po angielsku: API, GPU, CPU, RAM, SQL, Python, PyTorch, TensorFlow, NumPy, BLAS, itp.
3. ZOSTAN po angielsku: machine learning, deep learning, neural network, transformer, attention, embedding, vector, matrix, tensor, gradient, loss function, optimizer, hyperparameter
4. ZOSTAN po angielsku: forward, backward, reshape, transpose, broadcast, softmax, einsum, itp.
5. "Learning Objectives" -> "Cele uczenia sie"
6. "The Problem" -> "Problem"
7. "The Concept" -> "Koncepcja"
8. "Build It" -> "Zbuduj to"
9. "Use It" -> "Uzyj tego"
10. "Ship It" -> "Dostarcz to"
11. "Exercises" -> "Cwiczenia"
12. "Key Terms" -> "Kluczowe pojecia"
13. "Further Reading" -> "Dalsza lektura"
14. BLOKI KODU - NIE TLUMACZ (zostaw jak sa)
15. PRZECINKI przed: ze, bo, zeby, i (dwa niezalezne zdania), ktory/ktora/ktore, a (kontrast), wiec, az, zanim, gdy, albo, lub
16. POLSKIE DIAKRYTYKI: popraw wszystkie literowki w polskich znakach
17. Linki URL - zostaw bez zmian

Tlumacz ponizszy tekst z EN->PL. Zapisz TYLKO przetlumaczony tekst. Bez komentarzy."""

VERIFY_SYSTEM = """ROLA: Weryfikator tlumaczen. Sprawdzasz polskie tlumaczenia pod katem bledow.

6 KATEGORII BLEDOW:
1. DIAKRYTYKI (critical): pamietam->pamietam, Cie->Cie, zjqebany->zjebany, Hulałem, pisujacego->piszacego, przyłapać->przylapać
2. NIEPOLSKIE ZNAKI (critical): Cyrylica, rosyjskie, chinskie znaki
3. BRAK PRZECINKA (major): przed ze, bo, zeby, i (dwa niezalezne zdania), ktory/ktora/ktore, a (kontrast), wiec, az, zanim, gdy, albo, lub
4. ANGLICYZMY POZA LISTA (major): tylko dozwolone: API, GPU, CPU, RAM, SQL, Python, PyTorch, itp.
5. KOD W TLUMACZENIU (critical): bloki kodowe -> NIE TLUMACZONE
6. ANGielSKIE SEKCJE CO POWINNY BYC POLSKIE (minor): Learning Objectives->Cele uczenia się, The Problem->Problem, The Concept->Koncepcja

FORMAT RAPORTU:
Jesli bledow: "BLEDY: N" + lista bledow z liniami i poprawkami
Jesli 0 bledow: "ZERO ERRORS"

Sprawdz ponizszy tekst."""

FIX_SYSTEM = """ROLA: Ekspert translator. Poprawiasz bledy w tlumaczeniu.

Otrzymales liste bledow do poprawienia. Popraw WSZYSTKIE wymienione bledy w tlumaczeniu.
Zapisz calosc ponownie po polsku. Zachowaj styl oryginalu.
BLOKI KODU pozostaw nietlumaczone."""


def strip_think(text):
    if not text:
        return ""
    return re.sub(r'</think>.*?</think>', '', text, flags=re.DOTALL).strip()


def call_minimax(system, user, model="MiniMax-M2.7"):
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url="https://api.minimax.io/v1", max_retries=0, timeout=300)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=131072,
        temperature=0.3,
        timeout=300,
    )
    return strip_think(resp.choices[0].message.content)


def translate_lesson(slug, title_pl):
    doc_path = BASE_DIR / "phases/01-math-foundations" / slug / "docs/en.md"
    print(f"\n{'='*60}")
    print(f"TLUMACZENIE: {slug}")
    print(f"{'='*60}")

    text = doc_path.read_text(encoding="utf-8")

    # Translate
    print(f"[1] Translacja...")
    translated = call_minimax(TRANSLATE_SYSTEM, text, model="MiniMax-M2.7")
    if not translated:
        return {"slug": slug, "status": "ERROR", "iterations": 0}

    # Verify loop
    error_count = 0
    for i in range(1, 6):
        print(f"[2.{i}] Weryfikacja...")
        result = call_minimax(VERIFY_SYSTEM, translated, model="MiniMax-M2.7")
        if "ZERO ERRORS" in result or "BLEDY: 0" in result:
            print(f"  -> ZERO ERRORS")
            break
        # Extract errors
        errors = []
        error_count = 0
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith(('BLEDY:', 'BŁĘDY:', 'BLESY:', 'Bledy:')):
                try:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        error_count = int(parts[1].strip())
                except:
                    error_count = 1
                continue
            if error_count > 0 and line and not line.startswith('FORMAT'):
                errors.append(line)
        if errors:
            print(f"  -> Bledy: {errors[:3]}")
            # Fix
            print(f"[3.{i}] Poprawianie...")
            fix_prompt = "Lista bledow:\n" + "\n".join(errors) + "\n\nTekst do poprawienia:\n" + translated
            translated = call_minimax(FIX_SYSTEM, fix_prompt, model="MiniMax-M2.7")
        else:
            break

    # Save
    doc_path.write_text(translated, encoding="utf-8")
    print(f"  -> Zapisano: {doc_path}")
    return {"slug": slug, "status": "SUCCESS", "iterations": i}


def main():
    results = []
    for slug, title in LESSONS:
        try:
            r = translate_lesson(slug, title)
            results.append(r)
        except Exception as e:
            print(f.BLAD dla {slug}: {e}")
            results.append({"slug": slug, "status": "ERROR", "error": str(e)})

    print(f"\n{'='*60}")
    print("FAZA 01 (11-15): GOTOWE")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['slug']}: {r['status']}")

if __name__ == "__main__":
    main()
