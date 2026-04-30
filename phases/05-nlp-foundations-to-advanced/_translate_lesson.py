import os
import json
import requests

api_key = os.environ.get("MINIMAX_API_KEY", "")
if not api_key:
    print("MINIMAX_API_KEY not set")
    exit(1)

url = "https://api.minimax.io/anthropic/v1/messages"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "anthropic-version": "2025-01-01"
}

source_text = open(r"C:\VisualStudioCodeRepo\AIWorkspace\ai-engineering-from-scratch\phases\05-nlp-foundations-to-advanced\28-long-context-evaluation\docs\en.md", "r", encoding="utf-8").read()

prompt = f"""Translate to Polish. Keep code exactly as-is. Do not translate section headers that are code-related (like ## The Problem). Keep links and images as-is.

## Rules:
- Short sentences (15-20 words), active voice
- No: delve into, ensure, arguably, basically
- Keep English technical terms: API, GPU, CPU, SQL, JSON, Python, PyTorch, LLM, NLP, etc.
- Add commas before: że, bo, żeby, i (two independent sentences), który, a (contrast), więc, aż, zanim, gdy, albo, lub
- Verify Polish diacritics: ąęóśżźćńł

Source file to translate:

{source_text}"""

payload = {
    "model": "MiniMax-M2.7",
    "max_tokens": 16000,
    "messages": [{"role": "user", "content": prompt}]
}

response = requests.post(url, headers=headers, json=payload, timeout=120)
result = response.json()

if "content" in result:
    translated = result["content"][0]["text"]
    with open(r"C:\VisualStudioCodeRepo\AIWorkspace\ai-engineering-from-scratch\phases\05-nlp-foundations-to-advanced\28-long-context-evaluation\docs\en.md", "w", encoding="utf-8") as f:
        f.write(translated)
    print("SUCCESS: 28-long-context-evaluation")
    print(f"First 500 chars: {translated[:500]}")
else:
    print(f"ERROR: {result}")