#!/usr/bin/env python3
"""Translate and verify FORKING.md to Polish."""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')

from execution.minimax_utils import get_client, call_minimax

TRANSLATE_SYSTEM = '''Jestes translatorem kursu IT/AI. Tlumacz lekcje z EN->PL wiernie. MINIMAL INTERVENTION. ZOSTAWIJ po angielsku: API, GPU, CPU, RAM, SQL, Python, PyTorch, TensorFlow, HuggingFace, Phase 0, Phase 1, itp. KOD NIE TLUMACZONY. Przecinki przed: ze, bo, zeby, i (dwa niezalezne zdania), co, ktory, a (kontrast), wiec, az, zanim, gdy, albo, lub.'''

VERIFY_SYSTEM = '''Jestes weryfikatorem tlumaczen. Sprawdz polskie tlumaczenie pod katem:
1. DIAKRYTYKI (zjqebany, pamietam, Cie -> zjebany, pamietam, Cie)
2. NIEPOLSKIE ZNAKI (cyrylica, chinese)
3. BRAK PRZECINKA (przed: ze, bo, zeby, i, co, ktory, a, wiec, az, zanim, gdy, albo, lub)
4. ANGLICYZMY poza lista (nie: "sieć neuronowa" zamiast "neural network")
5. KOD wlacznie (nie tlumacz kodu)
6. ANGIELSKIE SEKCJE (### For Teams -> ### Dla zespolow, itp)

Jesli bledy: raportujdokladnie linie i popraw. Jesli zero: "ZERO ERRORS".'''

SOURCE = '''# Forking Guide

This course is MIT licensed. You're free to fork it and adapt it for your needs. Here's how to do it well.

## For Teams

Want to use this as internal training? Fork and customize:

1. Fork the repository
2. Remove phases your team doesn't need
3. Add company-specific examples and data
4. Add internal tool integrations to the outputs
5. Keep the attribution — it helps the community grow

## For Schools & Universities

Want to use this as course material?

1. Fork the repository
2. Map phases to your semester schedule
3. Add grading rubrics to exercises
4. Add your own assignments and exams
5. Consider contributing improvements back upstream

## For Bootcamps

Running a paid bootcamp? That's fine under MIT.

1. Fork and structure for your cohort timeline
2. Add video content, live sessions, mentorship
3. The code and docs are yours to build on
4. Consider sponsoring the project or contributing back

## For Other Languages

Want to teach this curriculum in a different programming language?

1. Fork the repository
2. Re-implement code examples in your language
3. Keep the lesson structure and documentation
4. Submit a PR to link your fork from the main README

## Keeping Your Fork Updated

```bash
git remote add upstream https://github.com/rohitg00/ai-engineering-from-scratch.git

git fetch upstream
git merge upstream/main
```

## Attribution

Not required by MIT, but appreciated:

```
Based on AI Engineering from Scratch
https://github.com/rohitg00/ai-engineering-from-scratch
```'''

client = get_client()

# Step 1: Translate
print("=== TRANSLATE ===")
result = call_minimax(client, TRANSLATE_SYSTEM, "Tlumacz na jezyk polski:\n\n" + SOURCE)
if not result:
    print("Translation failed")
    sys.exit(1)

# Save translation to temp file
trans_file = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/_tmp_forking_pl.md'
with open(trans_file, 'w', encoding='utf-8') as f:
    f.write(result)
print(f"Translation saved to {trans_file}")
print(result[:500])
print("...")

# Step 2: Verify
print("\n=== VERIFY ===")
verify_result = call_minimax(client, VERIFY_SYSTEM, f"Zweryfikuj ponizsze tlumaczenie:\n\n{result}")
print(verify_result)

# Check if errors found
if "ZERO ERRORS" in verify_result:
    print("\n=== COPY TO ORIGINAL ===")
    dest = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/FORKING.md'
    with open(trans_file, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Copied to {dest}")
    print("FORKING.md: GOTOWE")
else:
    print("\nErrors found - need to fix")
    # Fix and re-verify
    fix_result = call_minimax(client, VERIFY_SYSTEM, f"Popraw tlumaczenie, stosujac sie do Uwag:\n\n{verify_result}\n\nOryginal:\n\n{result}")
    print(fix_result)

    # Save fixed version
    fixed_file = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/_tmp_forking_fixed.md'
    with open(fixed_file, 'w', encoding='utf-8') as f:
        f.write(fix_result)

    # Verify again
    verify2 = call_minimax(client, VERIFY_SYSTEM, f"Zweryfikuj poprawione tlumaczenie:\n\n{fix_result}")
    print(verify2)

    if "ZERO ERRORS" in verify2:
        dest = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/FORKING.md'
        with open(fixed_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(dest, 'w', encoding='utf-8') as f:
            f.write(content)
        print("FORKING.md: GOTOWE (po korekcie)")
    else:
        print("Still has errors, needs more work")