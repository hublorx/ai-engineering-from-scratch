#!/usr/bin/env python3
"""Translate markdown lesson using MiniMax M2.7 API."""

import os
import sys
import json
import requests
from pathlib import Path
from openai import OpenAI

# Load .env from workspace root
for line in open("../.env"):
    if line.startswith("MINIMAX_API_KEY="):
        API_KEY = line.split("=", 1)[1].strip().strip('"')
        os.environ["MINIMAX_API_KEY"] = API_KEY
        break

client = OpenAI(api_key=API_KEY, base_url="https://api.minimax.io/v1")

TRANSLATE_SYSTEM = """Jesteś translatorem kursu IT/AI. Tłumaczysz lekcje z EN→PL wiernie, zachowując styl i terminologię techniczną. MINIMAL INTERVENTION - nie ulepszaj, nie skracaj, nie zmieniaj tonu.

## ZASADY TŁUMACZENIA

### ZOSTAŃ po angielsku (DOZWOLONE ANGLICYZMY)
API, GPU, CPU, RAM, SQL, NoSQL, REST, JSON, XML, HTML, CSS, JavaScript, TypeScript
Python, Julia, Rust, C++, Java, Go, Ruby, PHP, Swift, Kotlin
machine learning, deep learning, neural network, neuron, layer, weight, bias
gradient, loss function, optimizer, hyperparameter, training, inference, overfitting, underfitting
token, embedding, vector, matrix, tensor, dimension, feature
transformer, attention, self-attention, multi-head attention, feed-forward, residual connection
LLM, GPT, BERT, language model, fine-tuning, RLHF, RAG, prompt engineering
agent, tool, function calling, chain, workflow, pipeline
Docker, Kubernetes, cloud, AWS, GCP, Azure, serverless
Git, CI/CD, deployment, testing, unit test, integration test
debugging, profiling, optimization, performance, latency, throughput
ROI, KPI, metrics, dashboard, analytics, A/B testing

### NIE TŁUMACZ - zostaw dokładnie jak jest
- Nazwy funkcji: train_model(), forward(), backward(), predict()
- Zmienne i stałe: learning_rate, batch_size, hidden_dim
- Importy: import torch, from transformers import ...
- Pliki: model.py, train.py, config.json
- Frameworki i biblioteki: PyTorch, TensorFlow, HuggingFace, LangChain
- Metryki: accuracy, precision, recall, F1, AUC
- Skróty: ML, DL, AI, NN, CNN, RNN, LSTM, NLP, TTS, ASR

### TŁUMACZ I ZAPAMIĘTAJ
learning objectives → cele uczenia się
prerequisites → wymagania wstępne
time estimate → szacowany czas
code → kod
notebook → notebook
outputs → wyniki, artefakty
lesson → lekcja
phase → faza
quiz → quiz
exercise → ćwiczenie
solution → rozwiązanie
example → przykład
tip → wskazówka
warning → ostrzeżenie
note → uwaga
summary → podsumowanie
introduction → wprowadzenie
conclusion → konkluzja / podsumowanie

## POLSKIE DIAKRYTYKI - SPRAWDŹ i POPRAW
Huełałem → Hulałem
pisującego → piszącego
przylapać → przyłapać
pamietam → pamiętam
Cie → Cię
Jerkałem → Jęrkałem
zjqebany → zjebany

## USUŃ CAŁKOWICIE
- Wszystkie referencje do oryginalnego źródła
- Linki do zewnętrznych materiałów które nie są częścią kursu

## NIEPOLSKIE ZNAKI - WYKRYJ i ZAMIEŃ
Chińskie → polski odpowiednik
Rosyjskie/Cyrylica → polski odpowiednik

## PRZECINKI - OBOWIĄZKOWO DODAJ przed
- "że" (zdania z "że")
- "bo" (zdania z "bo")
- "żeby/żebym/żebyś"
- "i" (gdy łączy DWA NIEZALEŻNE zdania)
- "co" (nie po "to co")
- "który/która/które" (zdania względowe)
- "a" (gdy kontrast)
- "więc"
- "aż"
- "zanim"
- "gdy"
- "albo"
- "lub"

## SEKCJE MARKDOWN - CO TŁUMACZYĆ
# Tytuł lekcji → PRZETŁUMACZ
## Learning Objectives → ## Cele uczenia się
## Prerequisites → ## Wymagania wstępne
## Time → ## Czas
## Summary → ## Podsumowanie
## Exercise → ## Ćwiczenie
## Quiz → ## Quiz
## Code → ## Kod
## The Problem → ## Problem
## The Concept → ## Koncepcja

### ZOSTAŃ po angielsku (NIE TŁUMACZ sekcji!)
- Sekcje związane z kodem: ## Code Examples, ## Implementation, ## Walkthrough
- Tabele z danymi technicznymi - nie tłumacz wierszy z wartościami

## TŁUMACZENIE BLOKÓW KODU
BLOKI KODU NIE TŁUMACZ - zostaw dokładnie jak są

## PRACA (TYLKO 3 KROKI)
1. Przeczytaj źródłowy plik en.md
2. Tłumacz FAITHFULNIE - MINIMAL INTERVENTION
3. Zapisz przetłumaczoną wersję (nadpisz en.md)"""

VERIFY_SYSTEM = """Jesteś weryfikatorem tłumaczeń. Sprawdzasz polskie tłumaczenia lekcji AI Engineering pod kątem błędów.

## 6 KATEGORII BŁĘDÓW

### 1. DIAKRYTYKI (critical)
Znajdź i popraw:
- zjqebany → zjebany
- Huełałem → Hulałem
- pisującego → piszącego
- przylapać → przyłapać
- pamietam → pamiętam
- Cie → Cię
- Jerkałem → Jęrkałem

### 2. NIEPOLSKIE ZNAKI (critical)
Chińskie znaki → polski odpowiednik
Rosyjskie/Cyrylica → polski odpowiednik

### 3. BRAK PRZECINKA (major)
Sprawdź czy są przecinki PRZED:
- "że", "bo", "żeby", "i" (dwa niezależne zdania)
- "co", "który/która/które", "a" (kontrast)
- "więc", "aż", "zanim", "gdy", "albo", "lub"

### 4. ANGLICYZMY POZA LISTĄ (major)
DOZWOLONE: API, GPU, CPU, RAM, SQL, JSON, Python, PyTorch, itp.
NIE DOZWOLONE: "sieć neuronowa" zamiast "neural network"

### 5. KOD W TŁUMACZENIU (critical)
Bloki ```python ... ``` → NIE TŁUMACZONE

### 6. ANGIELSKIE SEKCJE CO POWINNY BYĆ POLSKIE (minor)
## Learning Objectives → ## Cele uczenia się
ALE: pozostaw "Phase 0", "Phase 1" etc.

## FORMAT RAPORTU
### Gdy są błędy:
BŁĘDY: N
1. [KATEGORIA] Linia X: [opis]
   Treść: "[fragment]"
   Poprawić na: "[poprawna]"

### Gdy zero błędów:
ZERO ERRORS ✓"""

def call_minimax(system, user_msg, max_tokens=8000):
    """Call MiniMax API."""
    resp = client.chat.completions.create(
        model="MiniMax-M2.7",
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ]
    )
    return resp.choices[0].message.content

def translate_file(file_path, max_iterations=5):
    """Translate a single lesson file."""
    file_path = Path(file_path)
    print(f"\n=== Translating: {file_path.name} ===")

    content = file_path.read_text(encoding="utf-8")
    print(f"Source length: {len(content)} chars")

    translated = call_minimax(TRANSLATE_SYSTEM, content)
    iteration = 1
    print(f"Translate done, iteration {iteration}")

    for i in range(max_iterations):
        verification = call_minimax(VERIFY_SYSTEM, translated)
        print(f"Verification iteration {i+1}: {verification[:100]}...")

        if "ZERO ERRORS" in verification:
            print(f"VERIFICATION PASSED at iteration {i+1}")
            break

        fix_prompt = f"""Popraw wszystkie błędy w poniższym tłumaczeniu:

{translated}

LISTA BŁĘDÓW:
{verification}

Popraw dokładnie te błędy."""
        translated = call_minimax(TRANSLATE_SYSTEM, fix_prompt)
        iteration += 1
        print(f"Fix iteration {iteration}")

    file_path.write_text(translated, encoding="utf-8")
    print(f"SAVED: {file_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python translate_lesson.py <file_path>")
        sys.exit(1)

    translate_file(sys.argv[1])