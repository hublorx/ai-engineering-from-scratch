#!/usr/bin/env python3
"""Translate Phase 02 lessons 06-10 EN→PL using MiniMax Sonnet."""

import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace/execution')
from minimax_utils import get_client, call_minimax

translate_system = """Jestes translatorem kursu IT/AI. Tlumaczysz lekcje z EN→PL wiernie, zachowujac styl i terminologię techniczną. MINIMAL INTERVENTION - nie ulepszaj, nie skracaj, nie zmieniaj tonu.

ZOSTAŃ po angielsku (DOZWOLONE ANGLICYZMY): API, GPU, CPU, RAM, SQL, NoSQL, REST, JSON, XML, HTML, CSS, JavaScript, TypeScript, Python, Julia, Rust, C++, Java, Go, Ruby, PHP, Swift, Kotlin, machine learning, deep learning, neural network, neuron, layer, weight, bias, gradient, loss function, optimizer, hyperparameter, training, inference, overfitting, underfitting, token, embedding, vector, matrix, tensor, dimension, feature, transformer, attention, self-attention, multi-head attention, feed-forward, residual connection, LLM, GPT, BERT, language model, fine-tuning, RLHF, RAG, prompt engineering, agent, tool, function calling, chain, workflow, pipeline, Docker, Kubernetes, cloud, AWS, GCP, Azure, serverless, Git, CI/CD, deployment, testing, unit test, integration test, debugging, profiling, optimization, performance, latency, throughput, ROI, KPI, metrics, dashboard, analytics, A/B testing

NIE TLUMACZ - zostaw dokładnie jak jest: nazwy funkcji (train_model(), forward(), backward(), predict()), zmienne i stale (learning_rate, batch_size, hidden_dim), importy (import torch, from transformers import ...), pliki (model.py, train.py, config.json), frameworki (PyTorch, TensorFlow, HuggingFace, LangChain), metryki (accuracy, precision, recall, F1, AUC), skróty (ML, DL, AI, NN, CNN, RNN, LSTM, NLP, TTS, ASR)

TLUMACZ I ZAPAMIETAJ: learning objectives → cele uczenia się, prerequisites → wymagania wstępne, time estimate → szacowany czas, code → kod, notebook → notebook, outputs → wyniki/artefakty, lesson → lekcja, phase → faza, quiz → quiz, exercise → ćwiczenie, solution → rozwiązanie, example → przykład, tip → wskazówka, warning → ostrzeżenie, note → uwaga, summary → podsumowanie, introduction → wprowadzenie, conclusion → konkluzja/podsumowanie

BLOKI KODU NIE TLUMACZ - zostaw dokładnie jak są. Komentarze w kodzie po angielsku zostaw jako są.

Przecinki OBOWIĄZKOWO przed: że, bo, żeby/żebym, i (dwa niezależne zdania), co, który/która/które, a (kontrast), więc, aż, zanim, gdy, albo, lub

Sekcje markdown do tlumaczenia: ## Learning Objectives → ## Cele uczenia się, ## The Problem → ## Problem, ## The Concept → ## Koncepcja, ## Summary → ## Podsumowanie, ## Exercise → ## Ćwiczenie, ## Quiz → ## Quiz, ## Code → ## Kod, ## Key Terms → ## Kluczowe pojęcia, ## Further Reading → ## Dalsze czytanie, ## Use It → ## Użyj tego, ## Ship It → ## Wyślij to, ## Build It → ## Zbuduj to, ## Exercises → ## Ćwiczenia

ZOSTAŃ po angielsku: sekcje zwiazane z kodem (## Code Examples, ## Implementation, ## Walkthrough), nazwy własne (PyTorch, TensorFlow, HuggingFace), "Phase 0", "Phase 1" etc.

Anti-AI style: krótkie zdania (15-25 słów), strona czynna, konkretne liczby, bez zbędnych przymiotników, brak "należy", "ważne jest", "należy pamiętać"."""

verify_system = """Jesteś weryfikatorem tłumaczeń. Sprawdzasz polskie tłumaczenia lekcji AI Engineering pod kątem błędów. Raportuj dokładnie co jest źle i gdzie.

SPRAWDŹ 6 kategorii błędów:
1. DIAKRYTYKI: zjqebany→zjebany, Huełałem→Hulałem, pisującego→piszącego, przyłapać→przyłapać, pamietam→pamiętam, Cie→Cię, Jerkałem→Jęrkałem, k///→pomiń
2. NIEPOLSKIE ZNAKI: Chińskie, Rosyjskie/Cyrylica (np. "takже"→"także", "просто"→"po prostu")
3. BRAK PRZECINKA przed: że, bo, żeby, i (dwa zdania), co, który, a (kontrast), więc, aż, zanim, gdy, albo, lub
4. ANGLICYZMY poza listą dozwolonych
5. KOD w tłumaczeniu - bloki ```python ... ``` muszą być NIETŁUMACZONE
6. ANGIELSKIE SEKCJE co powinny być POLSKIE: Learning Objectives→Cele uczenia się, The Problem→Problem, The Concept→Koncepcja, Summary→Podsumowanie, Exercise→Ćwiczenie, Key Terms→Kluczowe pojęcia, Further Reading→Dalsze czytanie

Format raportu gdy są błędy:
BŁĘDY: N
1. [KATEGORIA] Linia X: [opis]
   Treść: "[fragment]"
   Poprawić na: "[poprawna]"

Gdy zero błędów: ZERO ERRORS ✓"""


def translate_and_verify(filepath, client, max_iter=5):
    """Translate one lesson file, verify, loop if needed, save."""
    # Read source
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Translate
    user_prompt = f"""Przeczytaj poniższą lekcję i przetłumacz ją NA POLSKI. Zapisz WYŁĄCZNIE przetłumaczony tekst markdown - bez komentarzy, bez zapowiedzi, bez meta-informacji. Tylko polski tekst lekcji.

Tekst do przetłumaczenia:
---
{content}
---"""

    result = call_minimax(client, translate_system, user_prompt, temperature=0.1, model="MiniMax-M2.7")
    if not result:
        print(f"BŁĄD: brak odpowiedzi dla {filepath}")
        return False

    translated = result

    # Verify loop
    for iteration in range(max_iter):
        verify_prompt = f"""Sprawdź poniższe tłumaczenie lekcji pod kątem błędów. Przeczytaj cały tekst, sprawdź wszystkie 6 kategorii, podaj raport.

Tekst do sprawdzenia:
---
{translated}
---"""

        v_result = call_minimax(client, verify_system, verify_prompt, temperature=0.1, model="MiniMax-M2.7")
        if not v_result:
            print(f"BŁĄD: weryfikacja nie powiodła się dla {filepath}")
            break

        v_lower = v_result.lower()
        if "zero errors" in v_lower or "zero errors ✓" in v_lower:
            print(f"  ZERO ERRORS (iteracja {iteration+1})")
            break

        # Fix errors
        fix_prompt = f"""Popraw WSZYSTKIE błędy w poniższym tłumaczeniu. Podaj WYŁĄCZNIE poprawiony tekst markdown - bez komentarzy, bez listy poprawek.

Błędy do naprawienia:
---
{v_result}
---

Tekst do poprawienia:
---
{translated}
---"""
        fix_result = call_minimax(client, translate_system, fix_prompt, temperature=0.1, model="MiniMax-M2.7")
        if fix_result:
            translated = fix_result
        else:
            print(f"BŁĄD: korekta nie powiodła się dla {filepath}")
            break

    # Save
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(translated)
    print(f"ZAPISANO: {filepath}")
    return True


def main():
    client = get_client()
    print("MiniMax client OK")

    lessons = [
        "phases/02-ml-fundamentals/06-knn-and-distances/docs/en.md",
        "phases/02-ml-fundamentals/07-unsupervised-learning/docs/en.md",
        "phases/02-ml-fundamentals/08-feature-engineering/docs/en.md",
        "phases/02-ml-fundamentals/09-model-evaluation/docs/en.md",
        "phases/02-ml-fundamentals/10-bias-variance/docs/en.md",
    ]

    for lesson in lessons:
        print(f"\n=== Tlumaczenie: {lesson} ===")
        translate_and_verify(lesson, client)
        print(f"=== Gotowe: {lesson} ===")

    print("\nFAZA 02 (06-10): GOTOWE")


if __name__ == "__main__":
    main()
