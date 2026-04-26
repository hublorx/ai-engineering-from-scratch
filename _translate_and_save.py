#!/usr/bin/env python3
"""Translate lesson using MiniMax and save to file."""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')

from execution.minimax_utils import get_client, call_minimax
import codecs

def translate_and_save(lesson_name, source_path, output_path):
    TRANSLATE_SYSTEM = '''Jestes translatorem kursu IT/AI. Tlumaczysz lekcje z EN->PL wiernie, zachowujac styl i terminologie techniczna. MINIMAL INTERVENTION - nie ulepszaj, nie skracaj, nie zmieniaj tonu.

ZOSTAWIJ po angielsku (DOZWOLONE ANGLICYZMY):
API, GPU, CPU, RAM, SQL, NoSQL, REST, JSON, XML, HTML, CSS, JavaScript, TypeScript, Python, Julia, Rust, C++, Java, Go, Ruby, PHP, Swift, Kotlin, machine learning, deep learning, neural network, neuron, layer, weight, bias, gradient, loss function, optimizer, hyperparameter, training, inference, overfitting, underfitting, token, embedding, vector, matrix, tensor, dimension, feature, transformer, attention, self-attention, multi-head attention, feed-forward, residual connection, LLM, GPT, BERT, language model, fine-tuning, RLHF, RAG, prompt engineering, agent, tool, function calling, chain, workflow, pipeline, Docker, Kubernetes, cloud, AWS, GCP, Azure, serverless, Git, CI/CD, deployment, testing, unit test, integration test, debugging, profiling, optimization, performance, latency, throughput, ROI, KPI, metrics, dashboard, analytics, A/B testing

NIE TLUMACZ - zostaw dokladnie jak jest:
- Nazwy funkcji: train_model(), forward(), backward(), predict()
- Zmienne: learning_rate, batch_size, hidden_dim
- Importy: import torch, from transformers import ...
- Frameworki: PyTorch, TensorFlow, HuggingFace, LangChain

Tlumacz sekcje markdown: ## Learning Objectives -> ## Cele uczenia się, ## The Problem -> ## Problem, ## The Concept -> ## Koncepcja, ## Prerequisites -> ## Wymagania wstepne, ## Time -> ## Czas, ## Summary -> ## Podsumowanie

KOD NIE TLUMACZONY - bloki kodu zostaja bez zmian.

Przecinki OBOWIAZKOWO przed: ze, bo, zeby/zebym, i (dwa niezalezne zdania), co, ktory/ktora/ktore, a (kontrast), wiec, az, zanim, gdy, albo, lub'''

    with codecs.open(source_path, 'r', 'utf-8') as f:
        source = f.read()

    client = get_client()
    result = call_minimax(client, TRANSLATE_SYSTEM, "Tlumacz nastepujacy tekst na jezyk polski. ZACHOWAJ wszystkie znaki specjalne markdown (#, ##, ```, |), bloki kodu, i linki bez zmian:\n\n" + source)

    if result:
        # Write as UTF-8
        with codecs.open(output_path, 'w', 'utf-8') as f:
            f.write(result)
        print(f"SUCCESS: {lesson_name}")
        print(f"Output: {output_path}")
        return True
    else:
        print(f"FAILED: {lesson_name}", file=sys.stderr)
        return False

if __name__ == "__main__":
    lesson = sys.argv[1]  # e.g., "09-data-management"
    source = sys.argv[2]  # e.g., "phases/00-setup-and-tooling/09-data-management/docs/en.md"
    output = sys.argv[3] # e.g., "phases/00-setup-and-tooling/09-data-management/docs/en.md"

    success = translate_and_save(lesson, source, output)
    sys.exit(0 if success else 1)
