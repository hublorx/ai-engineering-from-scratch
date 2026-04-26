#!/usr/bin/env python3
"""Translate and verify CODE_OF_CONDUCT.md to Polish."""
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

SOURCE = '''# Code of Conduct

## Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Our Standards

**Positive behavior:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior:**

- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project maintainer at ghumare64@gmail.com. All complaints will be reviewed and investigated.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org), version 2.1.'''

client = get_client()

print("=== TRANSLATE ===")
result = call_minimax(client, TRANSLATE_SYSTEM, "Tlumacz na jezyk polski:\n\n" + SOURCE)
if not result:
    print("Translation failed")
    sys.exit(1)

trans_file = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/_tmp_coc_pl.md'
with open(trans_file, 'w', encoding='utf-8') as f:
    f.write(result)
print(f"Translation saved")

print("\n=== VERIFY ===")
verify_result = call_minimax(client, VERIFY_SYSTEM, f"Zweryfikuj ponizsze tlumaczenie:\n\n{result}")
print(verify_result)

if "ZERO ERRORS" in verify_result:
    dest = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/CODE_OF_CONDUCT.md'
    with open(trans_file, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(content)
    print("CODE_OF_CONDUCT.md: GOTOWE")
else:
    fix_result = call_minimax(client, VERIFY_SYSTEM, f"Popraw tlumaczenie, stosujac sie do Uwag:\n\n{verify_result}\n\nOryginal:\n\n{result}")
    fixed_file = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/_tmp_coc_fixed.md'
    with open(fixed_file, 'w', encoding='utf-8') as f:
        f.write(fix_result)
    verify2 = call_minimax(client, VERIFY_SYSTEM, f"Zweryfikuj poprawione tlumaczenie:\n\n{fix_result}")
    print(verify2)
    if "ZERO ERRORS" in verify2:
        dest = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/CODE_OF_CONDUCT.md'
        with open(fixed_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(dest, 'w', encoding='utf-8') as f:
            f.write(content)
        print("CODE_OF_CONDUCT.md: GOTOWE (po korekcie)")
    else:
        print("Still has errors")