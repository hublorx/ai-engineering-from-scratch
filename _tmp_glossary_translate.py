#!/usr/bin/env python3
"""Translate glossary files using MiniMax Sonnet."""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')

from execution.minimax_utils import get_client, call_minimax
import codecs

TRANSLATE_SYSTEM = """Jestes translatorem kursu IT/AI. Tlumaczysz lekcje z EN->PL wiernie, zachowujac styl i terminologie techniczna. MINIMAL INTERVENTION - nie ulepszaj, nie skracaj, nie zmieniaj tonu.

ZOSTAWIJ po angielsku (DOZWOLONE ANGLICYZMY):
API, GPU, CPU, RAM, SQL, NoSQL, REST, JSON, XML, HTML, CSS, JavaScript, TypeScript, Python, Julia, Rust, C++, Java, Go, Ruby, PHP, Swift, Kotlin, machine learning, deep learning, neural network, neuron, layer, weight, bias, gradient, loss function, optimizer, hyperparameter, training, inference, overfitting, underfitting, token, embedding, vector, matrix, tensor, dimension, feature, transformer, attention, self-attention, multi-head attention, feed-forward, residual connection, LLM, GPT, BERT, language model, fine-tuning, RLHF, RAG, prompt engineering, agent, tool, function calling, chain, workflow, pipeline, Docker, Kubernetes, cloud, AWS, GCP, Azure, serverless, Git, CI/CD, deployment, testing, unit test, integration test, debugging, profiling, optimization, performance, latency, throughput, ROI, KPI, metrics, dashboard, analytics, A/B testing

NIE TLUMACZ - zostaw dokladnie jak jest:
- Nazwy funkcji: train_model(), forward(), backward(), predict()
- Zmienne: learning_rate, batch_size, hidden_dim
- Importy: import torch, from transformers import ...
- Frameworki: PyTorch, TensorFlow, HuggingFace, LangChain

Tlumacz sekcje markdown: ## Learning Objectives -> ## Cele uczenia sie, ## The Problem -> ## Problem, ## The Concept -> ## Koncepcja, ## Prerequisites -> ## Wymagania wstepne, ## Time -> ## Czas, ## Summary -> ## Podsumowanie

KOD NIE TLUMACZONY - bloki kodu zostaja bez zmian.

Przecinki OBOWIAZKOWO przed: ze, bo, zeby/zebym, i (dwa niezalezne zdania), co, ktory/ktora/ktore, a (kontrast), wiec, az, zanim, gdy, albo, lub"""

def translate_file(source_path):
    with codecs.open(source_path, 'r', 'utf-8') as f:
        source = f.read()

    client = get_client()
    result = call_minimax(client, TRANSLATE_SYSTEM, "Tlumacz nastepujacy tekst na jezyk polski. ZACHOWAJ wszystkie znaki specjalne markdown (#, ##, ```, |), bloki kodu, i linki bez zmian:\n\n" + source)

    if result:
        with codecs.open(source_path, 'w', 'utf-8') as f:
            f.write(result)
        return True
    return False

if __name__ == "__main__":
    files = ['glossary/README.md', 'glossary/terms.md', 'glossary/myths.md']
    for f in files:
        if translate_file(f):
            print(f"{f}: GOTOWE")
        else:
            print(f"FAILED: {f}", file=sys.stderr)