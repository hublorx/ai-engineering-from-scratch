#!/usr/bin/env python3
"""Verify glossary translations."""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')

from execution.minimax_utils import get_client, call_minimax
import codecs

VERIFY_SYSTEM = """Jestes weryfikatorem tlumaczen. Sprawdzasz polskie tlumaczenia pod wzgledem bledow. Raportuj dokladnie co jest zle i gdzie.

6 KATEGORII BLEDOW:

1. DIAKRYTYKI: zjqebany->zjebany, Huelałem->Hulałem, pamietam->pamiętam, Cie->Cię
2. NIEPOLSKIE ZNAKI: Cyrylica/rosyjskie/chińskie
3. BRAK PRZECINKA przed: ze, bo, zeby/zebym, i (dwa niezalezne zdania), co, ktory/ktora/ktore, a (kontrast), wiec, az, zanim, gdy, albo, lub
4. ANGLICYZMY poza lista: DOZWOLONE: API, GPU, CPU, RAM, SQL, REST, JSON, Python, PyTorch, itp.
5. KOD W TLUMACZENIU: bloki kodu NIE tlumaczone
6. ANGIELSKIE SEKCJE CO POWINNY BYC POLSKIE

FORMAT RAPORTU:
Gdy sa bledy: BLEDY: N i lista
Gdy zero bledow: ZERO ERRORS"""

client = get_client()

files = ['glossary/terms.md', 'glossary/myths.md']
for f in files:
    with codecs.open(f, 'r', 'utf-8') as fh:
        content = fh.read()
    print(f'=== WERYFIKACJA {f} ===')
    result = call_minimax(client, VERIFY_SYSTEM, 'Zweryfikuj tlumaczenie:\n\n' + content)
    if result:
        with codecs.open('_tmp_verify_output.txt', 'w', 'utf-8') as out:
            out.write(result)
        print(result[:2000])
    else:
        print('Brak odpowiedzi')
    print()
