"""Translate lesson 22 Stochastic Processes EN->PL"""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')
from execution.minimax_utils import get_client, call_minimax

lesson_slug = "22-stochastic-processes"
source_file = "C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/phases/01-math-foundations/22-stochastic-processes/docs/en.md"
output_file = source_file

source_en_md = open(source_file, 'r', encoding='utf-8').read()

prompt = """Translate the following English markdown lesson to Polish. Follow these rules STRICTLY:

1. Translate FAITHFULLY, minimal intervention - do not improve, shorten, or change tone
2. Leave ALL code blocks EXACTLY as-is (```python, ```bash, ```mermaid, etc.)
3. Leave function names, variables, imports, metrics as-is: train_model(), learning_rate, batch_size, import torch, accuracy, precision, recall, F1
4. Keep these anglicisms in English: API, GPU, CPU, RAM, SQL, Python, PyTorch, TensorFlow, HuggingFace, LangChain, Docker, Kubernetes, Git, JSON, XML, HTML, CSS, JavaScript, TypeScript, LLM, GPT, BERT, MLOps, DevOps, CI/CD, REST, NoSQL, CUDA, Jupyter, JupyterLab, Colab, Pylance, Black, Ruff, Debugpy, Black Formatter
5. Polish diacritics: pamietam→pamiętam, pisującego→piszącego, przylapać→przyłapać, Cie→Cię, Huełałem→Hulałem, Jerkałem→Jęrkałem
6. Add commas before: że, bo, żeby, i (when joining two independent sentences), co, który/która/które, a (contrast), więc, aż, zanim, gdy, albo, lub
7. Section headers translate: Learning Objectives→Cele uczenia się, The Problem→Problem, The Concept→Koncepcja, Build It→Zbuduj to, Use It→Użyj tego, Ship It→Wdróż to, Exercises→Ćwiczenia, Key Terms→Kluczowe pojęcia, Further Reading→Dalsza lektura, Prerequisites→Wymagania wstępne, Time→Czas, Summary→Podsumowanie
8. Leave Phase 0, Phase 1, Lesson 01, Lesson 1 as-is (do not translate)
9. Remove any external references like "From the original course"
10. No [tlumaczenie] or [przyp. tłum.] annotations
11. Short sentences, active voice, no em dashes, no "it's important to note", no "however/therefore/essentially/basically"
12. Mermaid diagrams: keep the text content inside the diagram but do not translate node labels that are code or technical terms

Translate to Polish:

""" + source_en_md + """

Return ONLY the translated Polish markdown. No explanations, no comments, just the markdown."""

print(f"=== TRANSLATING {lesson_slug} ===")
client = get_client()
result = call_minimax(client, "", prompt, model="MiniMax-M2.7", max_retries=3)
if result:
    print(result[:500] + "...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"SAVED: {output_file}")
else:
    print("FAILED")
print("DONE")