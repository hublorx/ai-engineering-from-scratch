"""Verify translations for errors"""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')
from execution.minimax_utils import get_anthropic_client

def verify_translation(lesson_slug, translated_path):
    print(f"\n=== VERIFYING {lesson_slug} ===")

    with open(translated_path, "r", encoding="utf-8") as f:
        text = f.read()

    system_prompt = """Jestes weryfikatorem tlumaczen. Sprawdzasz polskie tlumaczenia lekcji AI Engineering pod katem bledow.

Raportuj dokladnie co jest zle i gdzie.

6 KATEGORII BLEDOW:

1. DIAKRYTYKI: zjqebany→zjebany, Huełałem→Hulałem, pisującego→piszącego, przylapać→przyłapać, pamietam→pamiętam, Cie→Cię, Jerkałem→Jęrkałem

2. NIEPOLSKIE ZNAKI: Cyrylica/rosyjskie/chińskie, "takже"→"także", "обично"→"zwykle"

3. BRAK PRZECINKA przed: że, bo, żeby, i (dwa niezależne zdania), co, który/która/które, a (kontrast), więc, aż, zanim, gdy, albo, lub

4. ANGLICYZMY poza lista dozwolonych: DOZWOLONE to API, GPU, CPU, RAM, SQL, Python, PyTorch, TensorFlow, HuggingFace, LangChain, Docker, Kubernetes, Git, JSON, XML, HTML, CSS, JavaScript, TypeScript, LLM, GPT, BERT, MLOps, DevOps, CI/CD, REST, NoSQL, CUDA, Jupyter, JupyterLab, Colab, Pylance, Black, Ruff, Debugpy, GitLens, SSH, VS Code, GPU, itp. NIE DOZWOLONE to "hardware", "softwar", "siec neuronowa" (zamiast neural network), itp.

5. KOD W TŁUMACZENIU: bloki kodu NIE MIALY byc tlumaczone. Sprawdz czy kod jest nie ruszony - ```python, ```bash, ```mermaid pozostaly bez zmian.

6. ANGIELSKIE SEKCJE CO POWINNY BYC POLSKIE: Learning Objectives→Cele uczenia się, The Problem→Problem, The Concept→Koncepcja, Build It→Zbuduj to, Use It→Użyj tego, Ship It→Wdróż to, Exercises→Ćwiczenia, Key Terms→Kluczowe pojęcia, Further Reading→Dalsza lektura, Prerequisites→Wymagania wstępne, Time→Czas

Jesli sa bledy: "BŁĘDY: N" + lista
Jesli zero bledow: "ZERO ERRORS"

Return in Polish."""

    user_prompt = f"""Zweryfikuj to tlumaczenie:

{text}

Raport:"""

    try:
        client = get_anthropic_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            thinking={"type": "disabled"},
        )
        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        result = "\n".join(text_parts)
        print(result)

        # Save report
        report_path = translated_path.replace("_translated.md", "_verification.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\nReport saved to: {report_path}")

    except Exception as e:
        print(f"Verification failed: {e}")

# Verify all 4
verify_translation("05-jupyter-notebooks", "C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/.tmp_05-jupyter-notebooks_translated.md")
verify_translation("06-python-environments", "C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/.tmp_06-python-environments_translated.md")
verify_translation("07-docker-for-ai", "C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/.tmp_07-docker-for-ai_translated.md")
verify_translation("08-editor-setup", "C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/.tmp_08-editor-setup_translated.md")
