#!/usr/bin/env python3
"""Translate and verify LESSON_TEMPLATE.md to Polish."""
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

SOURCE = '''# Lesson Template

Use this template when creating a new lesson. Copy the folder structure and fill in the content.

## Folder Structure

```
NN-lesson-name/
├── code/
│   ├── main.py            (primary implementation)
│   ├── main.ts            (TypeScript version, if applicable)
│   ├── main.rs            (Rust version, if applicable)
│   └── main.jl            (Julia version, if applicable)
├── notebook/
│   └── lesson.ipynb       (Jupyter notebook for experimentation)
├── docs/
│   └── en.md              (lesson documentation)
└── outputs/
    ├── prompt-*.md         (prompts produced by this lesson)
    └── skill-*.md          (skills produced by this lesson)
```

## Documentation Format (docs/en.md)

```markdown
# [Lesson Title]

> [One-line motto — the core idea that sticks]

**Type:** Build | Learn
**Languages:** Python, TypeScript, Rust, Julia (list what's used)
**Prerequisites:** [List prior lessons needed]
**Time:** ~[estimated time] minutes

## The Problem

[2-3 paragraphs. What can't you do without this? Why should you care?
Make it concrete — show a scenario where not knowing this hurts.]

## The Concept

[Explain with diagrams and intuition. No code yet.
Use ASCII diagrams, tables, or link to visuals in the web app.
Build mental models before implementation.]

## Build It

[Step-by-step implementation from scratch.
Start with the simplest version, then add complexity.
Every code block should be runnable on its own.]

### Step 1: [Name]

[Explanation]

    [code block]

### Step 2: [Name]

[Explanation]

    [code block]

[...continue...]

## Use It

[Now show how frameworks/libraries do the same thing.
Compare your from-scratch version to the library version.
This proves the concept and introduces practical tools.]

## Ship It

[What reusable artifact does this lesson produce?
Could be a prompt, a skill, an agent, an MCP server, or a tool.
Include it here and save it in the outputs/ folder.]

## Exercises

1. [Easy — reinforce the core concept]
2. [Medium — apply it to a different problem]
3. [Hard — extend or combine with prior lessons]

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| [term] | [common misconception] | [actual definition] |

## Further Reading

- [Resource 1](url) — [why it's worth reading]
- [Resource 2](url) — [why it's worth reading]
```

## Code File Guidelines

- Code must run without errors
- No comments — code should be self-explanatory
- Use the language that fits best for the topic
- Include a `requirements.txt` or equivalent if there are dependencies
- Start simple, build up complexity
- Every function and class should have a clear purpose

## Output File Format

### Prompts

```markdown
---
name: prompt-name
description: What this prompt does
phase: [phase number]
lesson: [lesson number]
---

[Prompt content]
```

### Skills

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: [phase number]
lesson: [lesson number]
tags: [relevant, tags]
---

[Skill content]
```'''

client = get_client()

print("=== TRANSLATE ===")
result = call_minimax(client, TRANSLATE_SYSTEM, "Tlumacz na jezyk polski:\n\n" + SOURCE)
if not result:
    print("Translation failed")
    sys.exit(1)

trans_file = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/_tmp_lesson_pl.md'
with open(trans_file, 'w', encoding='utf-8') as f:
    f.write(result)
print(f"Translation saved")

print("\n=== VERIFY ===")
verify_result = call_minimax(client, VERIFY_SYSTEM, f"Zweryfikuj ponizsze tlumaczenie:\n\n{result}")
print(verify_result)

if "ZERO ERRORS" in verify_result:
    dest = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/LESSON_TEMPLATE.md'
    with open(trans_file, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(content)
    print("LESSON_TEMPLATE.md: GOTOWE")
else:
    fix_result = call_minimax(client, VERIFY_SYSTEM, f"Popraw tlumaczenie, stosujac sie do Uwag:\n\n{verify_result}\n\nOryginal:\n\n{result}")
    fixed_file = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/_tmp_lesson_fixed.md'
    with open(fixed_file, 'w', encoding='utf-8') as f:
        f.write(fix_result)
    verify2 = call_minimax(client, VERIFY_SYSTEM, f"Zweryfikuj poprawione tlumaczenie:\n\n{fix_result}")
    print(verify2)
    if "ZERO ERRORS" in verify2:
        dest = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/LESSON_TEMPLATE.md'
        with open(fixed_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(dest, 'w', encoding='utf-8') as f:
            f.write(content)
        print("LESSON_TEMPLATE.md: GOTOWE (po korekcie)")
    else:
        print("Still has errors")