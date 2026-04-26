#!/usr/bin/env python3
"""Translate and verify CHANGELOG.md to Polish."""
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

SOURCE = '''# Changelog

What's new in the curriculum. Most recent first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Each entry names the phase, lesson, and what changed, so learners can jump straight to the delta.

## [Unreleased]

### Added
- `scripts/scaffold-lesson.sh` — scaffolder that creates `phases/NN-phase/NN-lesson/` with the full folder structure and a `docs/en.md` skeleton prefilled from `LESSON_TEMPLATE.md`.
- `.github/PULL_REQUEST_TEMPLATE.md` — contributor checklist (code runs, no code comments, built-from-scratch-first, atomic per-lesson commit, markdown-link ROADMAP row).
- `.github/ISSUE_TEMPLATE/bug_report.md` and `new_lesson_proposal.md` — structured intake for bug reports and lesson pitches.
- This `CHANGELOG.md`.

## 2026-04 — Phase 4: Computer Vision complete

### Added
- All 28 Phase 4 lessons, covering image fundamentals through multi-modal vision (VLMs, 3D, video, self-supervised).
- Phase 4 rows in `ROADMAP.md` linked as markdown to the lesson folders, so the website surfaces them.

### Fixed
- Phase 4 precision pass across 15+ lessons:
  - `phase-4/02`: shape calculator specifies RF/stride handling for adaptive pool, flatten, and linear.
  - `phase-4/03`: backbone selector description lists all covered families; head guidance added for OCR, medical, industrial.
  - `phase-4/04`: classification diagnostics use quantitative thresholds per failure mode; `n/a` declared for undefined metrics; guard for fewer than 3 classes.
  - `phase-4/06`: detection metric reader uses `AP@0.5` (not `mAP@0.5`); per-class recall declared optional; anchor designer clarifies stride truncation and single-anchor-per-level path.
  - `phase-4/10`: sampler picker declares `unet_forward_ms` as an input; ControlNet guard promoted to rule 0.
  - `phase-4/14`: ViT inspector aligned with refusal rule — port attempts are audited, not endorsed.
  - `phase-4/24`: open-vocab stack picker has explicit rule precedence and license-filter semantics; concept designer resolves step-5/rule-80 conflict.
  - `phase-4/25`: VLM docs `_merge` raises descriptive `ValueError` on placeholder mismatch; CMER normalises internally.
  - `phase-4/27`: `synthetic_frames` clips GT boxes to frame H/W.
  - `phase-4/28`: `rope_3d` validates dim split; dropped unused `F` import from DiT block example.

## 2026-Q1 and earlier

### Added
- Phase 0 (Setup & Tooling): all 12 lessons.
- Phase 1 (Math Foundations): all 22 lessons.
- Phase 2 (ML Fundamentals): all 18 lessons.
- Phase 3 (Deep Learning Core): core lessons through perceptron, backprop, optimizers.
- Built-in Claude Code skills: `find-your-level` (placement quiz) and `check-understanding` (per-phase quiz).
- Website at `aiengineeringfromscratch.com`: catalog, per-lesson pages, roadmap, 277-term glossary.
- Initial scaffolding for all 20 phases (`phases/00-*` through `phases/19-*`).
- `LESSON_TEMPLATE.md`, `CONTRIBUTING.md`, `ROADMAP.md`, `README.md`.

[Unreleased]: https://github.com/rohitg00/ai-engineering-from-scratch/compare/HEAD...HEAD'''

client = get_client()

print("=== TRANSLATE ===")
result = call_minimax(client, TRANSLATE_SYSTEM, "Tlumacz na jezyk polski:\n\n" + SOURCE)
if not result:
    print("Translation failed")
    sys.exit(1)

trans_file = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/_tmp_changelog_pl.md'
with open(trans_file, 'w', encoding='utf-8') as f:
    f.write(result)
print(f"Translation saved")

print("\n=== VERIFY ===")
verify_result = call_minimax(client, VERIFY_SYSTEM, f"Zweryfikuj ponizsze tlumaczenie:\n\n{result}")
print(verify_result)

if "ZERO ERRORS" in verify_result:
    dest = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/CHANGELOG.md'
    with open(trans_file, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(dest, 'w', encoding='utf-8') as f:
        f.write(content)
    print("CHANGELOG.md: GOTOWE")
else:
    fix_result = call_minimax(client, VERIFY_SYSTEM, f"Popraw tlumaczenie, stosujac sie do Uwag:\n\n{verify_result}\n\nOryginal:\n\n{result}")
    fixed_file = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/_tmp_changelog_fixed.md'
    with open(fixed_file, 'w', encoding='utf-8') as f:
        f.write(fix_result)
    verify2 = call_minimax(client, VERIFY_SYSTEM, f"Zweryfikuj poprawione tlumaczenie:\n\n{fix_result}")
    print(verify2)
    if "ZERO ERRORS" in verify2:
        dest = 'C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/CHANGELOG.md'
        with open(fixed_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(dest, 'w', encoding='utf-8') as f:
            f.write(content)
        print("CHANGELOG.md: GOTOWE (po korekcie)")
    else:
        print("Still has errors")