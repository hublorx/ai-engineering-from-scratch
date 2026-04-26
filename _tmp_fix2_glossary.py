#!/usr/bin/env python3
"""Fix remaining glossary errors."""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')

from execution.minimax_utils import get_client, call_minimax
import codecs

FIX_SYSTEM = """Jestes korektorem tlumaczen. Popraw wszystkie bledy.

1. W terms.md dodaj przecinek przed "— system prompts" (myślnik rozdzielający dwa niezależne zdania)
2. W myths.md zamien "guardrails" na "mechanizmy zabezpieczające" i "deployment" na "wdrożenie"

ZACHOWAJ wszystko inne bez zmian."""

client = get_client()

# Fix terms.md
with codecs.open('glossary/terms.md', 'r', 'utf-8') as f:
    terms = f.read()

print('=== FIX terms.md ===')
result = call_minimax(client, FIX_SYSTEM, f"Dodaj przecinek przed myślnikiem (—) gdy poprzedza on drugie zdanie w terms.md:\n\n{terms}")
if result:
    with codecs.open('glossary/terms.md', 'w', 'utf-8') as f:
        f.write(result)
    print('terms.md: FIXED')

# Fix myths.md
with codecs.open('glossary/myths.md', 'r', 'utf-8') as f:
    myths = f.read()

print('=== FIX myths.md ===')
result = call_minimax(client, FIX_SYSTEM, f"Zamien guardrails->mechanizmy zabezpieczające i deployment->wdrozenie:\n\n{myths}")
if result:
    with codecs.open('glossary/myths.md', 'w', 'utf-8') as f:
        f.write(result)
    print('myths.md: FIXED')

print('Done.')
