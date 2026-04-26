#!/usr/bin/env python3
"""Fix glossary translation errors."""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')

from execution.minimax_utils import get_client, call_minimax
import codecs

FIX_SYSTEM = """Jestes korektorem tlumaczen. Popraw wszystkie bledy wymienione w raporcie.

ZACHOWAJ wszystko inne bez zmian - tylko popraw wymienione bledy.

ZASADY:
- Dodawaj przecinki przed: ze, bo, zeby, i (dwa niezalezne zdania), co, ktory/ktora/ktore, a (kontrast), wiec, az, zanim, gdy, albo, lub
- Zastepuj angielskie slowa poza lista dozwolonych
- Usuwaj znaki cyrylickie/czyranskie/chińskie"""

client = get_client()

# Fix terms.md
with codecs.open('glossary/terms.md', 'r', 'utf-8') as f:
    terms = f.read()

print('=== FIX terms.md ===')
fix_prompt = f"""Popraw nastepujace bledy w tlumaczeniu:

BLEDY:
1. BRAK PRZECINKA (Overfitting): "...dobrze radzi sobie z danymi treningowymi, ale słabo z niewidzianymi danymi." → Przed "ale" powinien byc przecinek.
2. BRAK PRZECINKA (GAN): "...generator używa, podczas gdy discriminator..." → Przed "podczas gdy" powinien byc przecinek.
3. BRAK PRZECINKA (Latent Space): "Skompresowana nauczona przestrzeń reprezentacji, gdzie podobne inputy..." → Przed "gdzie" powinien byc przecinek.
4. BRAK PRZECINKA (Decoder): "...każda pozycja może attendować tylko do wcześniejszych pozycji. GPT to decoder-only." → Między zdaniami srednik.
5. KROPKI ZAMIAST PRZECINKOW (Zero-Shot): "Na zadaniu, na którym nie był explicitnie trenowany, bez zadaniowo-specyficznych przykładów w prompcie." → Wszystkie trzy kropki powinny byc przecinkami.
6. NIEPOPRAWNA FORMA (Latent Space): "Skompresowana nauczona przestrzeń" → "Skompresowana, nauczona przestrzeń"
7. ANGLICYZM (Zero-Shot): "explicitly" → "jawnie"

Tresc:
{terms}"""

result = call_minimax(client, FIX_SYSTEM, fix_prompt)
if result:
    with codecs.open('glossary/terms.md', 'w', 'utf-8') as f:
        f.write(result)
    print('terms.md: FIXED')
else:
    print('FAILED terms.md', file=sys.stderr)

# Fix myths.md
with codecs.open('glossary/myths.md', 'r', 'utf-8') as f:
    myths = f.read()

print('=== FIX myths.md ===')
fix_prompt = f"""Popraw nastepujacy blad w tlumaczeniu:

BLEDY: Cyrillic character found - "проміжні checkpointy" (Ukrainian) - zastep to "pośrednie checkpointy"

Tresc:
{myths}"""

result = call_minimax(client, FIX_SYSTEM, fix_prompt)
if result:
    with codecs.open('glossary/myths.md', 'w', 'utf-8') as f:
        f.write(result)
    print('myths.md: FIXED')
else:
    print('FAILED myths.md', file=sys.stderr)

print('Done.')
