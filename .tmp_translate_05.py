"""Translate lesson 05 Jupyter Notebooks EN->PL"""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')
from execution.minimax_utils import get_anthropic_client

lesson_slug = "05-jupyter-notebooks"

source_en_md = """# Jupyter Notebooks

> Notebooks are the lab bench of AI engineering. You prototype here, then move what works into production.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~30 minutes

## Learning Objectives

- Install and launch JupyterLab, Jupyter Notebook, or VS Code with the Jupyter extension
- Use magic commands (`%timeit`, `%%time`, `%matplotlib inline`) to benchmark and visualize inline
- Distinguish when to use notebooks vs scripts and apply the "explore in notebooks, ship in scripts" workflow
- Identify and avoid common notebook traps: out-of-order execution, hidden state, and memory leaks

## The Problem

Every AI paper, tutorial, and Kaggle competition uses Jupyter notebooks. They let you run code in pieces, see outputs inline, mix code with explanations, and iterate fast. If you try to learn AI without notebooks, you're doing math homework without scratch paper.

But notebooks have real traps. People use them for everything, including things they're terrible at. Knowing when to use a notebook and when to use a script will save you from debugging nightmares later.

## The Concept

A notebook is a list of cells. Each cell is either code or text.

```mermaid
graph TD
    A["**Markdown Cell**\\n# My Experiment\\nTesting learning rate 0.01"] --> B["**Code Cell** ► Run\\nmodel.fit(X, y, lr=0.01)\\n---\\nOutput: loss = 0.342"]
    B --> C["**Code Cell** ► Run\\nplt.plot(losses)\\n---\\nOutput: inline plot"]
```

The kernel is a Python process running in the background. When you run a cell, it sends the code to the kernel, which executes it and sends back the result. All cells share the same kernel, so variables persist between cells.

```mermaid
graph LR
    A[Notebook UI] <--> B[Kernel\\nPython process]
    B --> C[Keeps variables in memory]
    B --> D[Runs cells in whatever order you click]
    B --> E[Dies when you restart it]
```

That "whatever order you click" part is both the superpower and the foot-gun.

## Build It

### Step 1: Pick your interface

Three options, one format:

| Interface | Install | Best for |
|-----------|---------|----------|
| JupyterLab | `pip install jupyterlab` then `jupyter lab` | Full IDE experience, multiple tabs, file browser, terminal |
| Jupyter Notebook | `pip install notebook` then `jupyter notebook` | Simple, lightweight, one notebook at a time |
| VS Code | Install "Jupyter" extension | Already in your editor, git integration, debugging |

All three read and write the same `.ipynb` file. Pick whatever you like. JupyterLab is the most common in AI work.

```bash
pip install jupyterlab
jupyter lab
```

### Step 2: Keyboard shortcuts that matter

You operate in two modes. Press `Escape` for command mode (blue bar on the left), `Enter` for edit mode (green bar).

**Command mode (most used):**

| Key | Action |
|-----|--------|
| `Shift+Enter` | Run cell, move to next |
| `A` | Insert cell above |
| `B` | Insert cell below |
| `DD` | Delete cell |
| `M` | Convert to markdown |
| `Y` | Convert to code |
| `Z` | Undo cell operation |
| `Ctrl+Shift+H` | Show all shortcuts |

**Edit mode:**

| Key | Action |
|-----|--------|
| `Tab` | Autocomplete |
| `Shift+Tab` | Show function signature |
| `Ctrl+/` | Toggle comment |

`Shift+Enter` is the one you'll use a thousand times a day. Learn it first.

### Step 3: Cell types

**Code cells** run Python and show the output:

```python
import numpy as np
data = np.random.randn(1000)
data.mean(), data.std()
```

Output: `(0.0032, 0.9987)`

**Markdown cells** render formatted text. Use them to document what you're doing and why. Supports headers, bold, italic, LaTeX math (`$E = mc^2$`), tables, and images.

### Step 4: Magic commands

These aren't Python. They're Jupyter-specific commands that start with `%` (line magic) or `%%` (cell magic).

**Time your code:**

```python
%timeit np.random.randn(10000)
```

Output: `45.2 us +/- 1.3 us per loop`

```python
%%time
model.fit(X_train, y_train, epochs=10)
```

Output: `Wall time: 2.34 s`

`%timeit` runs the code many times and averages. `%%time` runs it once. Use `%timeit` for microbenchmarks, `%%time` for training runs.

**Enable inline plots:**

```python
%matplotlib inline
```

Every `plt.plot()` or `plt.show()` now renders directly in the notebook.

**Install packages without leaving the notebook:**

```python
!pip install scikit-learn
```

The `!` prefix runs any shell command.

**Check environment variables:**

```python
%env CUDA_VISIBLE_DEVICES
```

### Step 5: Display rich output inline

Notebooks auto-display the last expression in a cell. But you can control it:

```python
import pandas as pd

df = pd.DataFrame({
    "model": ["Linear", "Random Forest", "Neural Net"],
    "accuracy": [0.72, 0.89, 0.94],
    "training_time": [0.1, 2.3, 45.6]
})
df
```

This renders a formatted HTML table, not a text dump. Same with plots:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title("Inline Plot")
plt.show()
```

The plot appears right below the cell. This is why notebooks dominate AI work. You see the data, the plot, and the code together.

For images:

```python
from IPython.display import Image, display
display(Image(filename="architecture.png"))
```

### Step 6: Google Colab

Colab is a free Jupyter notebook in the cloud. It gives you a GPU, pre-installed libraries, and Google Drive integration. No setup required.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload any `.ipynb` file from this course
3. Runtime > Change runtime type > T4 GPU (free)

Colab differences from local Jupyter:
- Files don't persist between sessions (save to Drive or download)
- Pre-installed: numpy, pandas, matplotlib, torch, tensorflow, sklearn
- `from google.colab import files` to upload/download files
- `from google.colab import drive; drive.mount('/content/drive')` for persistent storage
- Sessions time out after 90 minutes of inactivity (free tier)

## Use It

### Notebooks vs Scripts: When to use which

| Use notebooks for | Use scripts for |
|-------------------|-----------------|
| Exploring a dataset | Training pipelines |
| Prototyping a model | Reusable utilities |
| Visualizing results | Anything with `if __name__` |
| Explaining your work | Code that runs on a schedule |
| Quick experiments | Production code |
| Course exercises | Packages and libraries |

The rule: **explore in notebooks, ship in scripts**.

A common workflow in AI:
1. Explore data in a notebook
2. Prototype your model in the notebook
3. Once it works, move the code to `.py` files
4. Import those `.py` files back into the notebook for further experiments

### Common traps

**Out-of-order execution.** You run cell 5, then cell 2, then cell 7. The notebook works on your machine but breaks when someone runs it top to bottom. Fix: Kernel > Restart & Run All before sharing.

**Hidden state.** You delete a cell but the variable it created is still in memory. The notebook looks clean but depends on a ghost cell. Fix: Restart the kernel regularly.

**Memory leaks.** Loading a 4GB dataset, training a model, loading another dataset. Nothing gets freed. Fix: `del variable_name` and `gc.collect()`, or restart the kernel.

## Ship It

This lesson produces:
- `outputs/prompt-notebook-helper.md` for debugging notebook issues

## Exercises

1. Open JupyterLab, create a notebook, and use `%timeit` to compare list comprehension vs numpy for creating an array of 100,000 random numbers
2. Create a notebook with both markdown and code cells that loads a CSV, displays a dataframe, and plots a chart. Then run Kernel > Restart & Run All to verify it works top to bottom
3. Take the code from `code/notebook_tips.py`, paste it into a Colab notebook, and run it with a free GPU

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Kernel | "The thing running my code" | A separate Python process that executes cells and keeps variables in memory |
| Cell | "A code block" | An independently runnable unit in a notebook, either code or markdown |
| Magic command | "Jupyter tricks" | Special commands prefixed with `%` or `%%` that control the notebook environment |
| `.ipynb` | "Notebook file" | A JSON file containing cells, outputs, and metadata. Stands for IPython Notebook |

## Further Reading

- [JupyterLab Docs](https://jupyterlab.readthedocs.io/) for the full feature set
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) for Colab-specific limits and features
- [28 Jupyter Notebook Tips](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/) for power-user shortcuts
"""

system_prompt = """Translate markdown lessons from English to Polish faithfully. Minimal intervention - do not improve, shorten, or change tone.

Rules:
1. Leave ALL code blocks EXACTLY as-is (```python, ```bash, ```mermaid, etc.)
2. Leave function names, variables, imports, metrics as-is: train_model(), learning_rate, batch_size, import torch, accuracy, precision, recall, F1
3. Keep these anglicisms in English: API, GPU, CPU, RAM, SQL, Python, PyTorch, TensorFlow, HuggingFace, LangChain, Docker, Kubernetes, Git, JSON, XML, HTML, CSS, JavaScript, TypeScript, LLM, GPT, BERT, MLOps, DevOps, CI/CD, REST, NoSQL, CUDA, Jupyter, JupyterLab, Colab, Pylance, Black, Ruff, Debugpy, Black Formatter, GitLens, Debugpy, Black Formatter, GPU, SSH
4. Polish diacritics: pamietam→pamiętam, pisującego→piszącego, przylapać→przyłapać, Cie→Cię, Huełałem→Hulałem, Jerkałem→Jęrkałem
5. Add commas before: że, bo, żeby, i (when joining two independent sentences), co, który/która/które, a (contrast), więc, aż, zanim, gdy, albo, lub
6. Section headers: Learning Objectives→Cele uczenia się, The Problem→Problem, The Concept→Koncepcja, Build It→Zbuduj to, Use It→Użyj tego, Ship It→Wdróż to, Exercises→Ćwiczenia, Key Terms→Kluczowe pojęcia, Further Reading→Dalsza lektura, Prerequisites→Wymagania wstępne, Time→Czas, Summary→Podsumowanie
7. Leave Phase 0, Phase 1, Lesson 01, Lesson 1 as-is
8. Remove any external references like "From the original course"
9. No [tlumaczenie] or [przyp. tłum.] annotations
10. Short sentences, active voice, no em dashes, no "it's important to note", no "however/therefore/essentially/basically"

Return ONLY the translated Polish markdown."""

user_prompt = f"""Translate this English markdown lesson to Polish. Return ONLY the Polish markdown, no explanations:

{source_en_md}"""

print(f"=== TRANSLATING {lesson_slug} ===")
client = get_anthropic_client()
try:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        thinking={"type": "disabled"},
    )
    text_parts = []
    for block in response.content:
        if hasattr(block, 'text'):
            text_parts.append(block.text)
    result = "\n".join(text_parts) if text_parts else None
except Exception as e:
    print(f"  Anthropic call failed: {e}", file=sys.stderr)
    result = None

if result:
    # Write to file (avoid Windows console encoding issues)
    out_path = f"C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/.tmp_{lesson_slug}_translated.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"SAVED: {out_path} ({len(result)} chars)")
else:
    print("TRANSLATION FAILED")
