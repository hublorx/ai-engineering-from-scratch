"""Translate lesson 08 Editor Setup EN->PL"""
import sys
sys.path.insert(0, 'C:/VisualStudioCodeRepo/AIWorkspace')
from execution.minimax_utils import get_anthropic_client

lesson_slug = "08-editor-setup"

source_en_md = """# Editor Setup

> Your editor is your co-pilot. Configure it once so it stays out of your way and starts pulling its weight.

**Type:** Build
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~20 minutes

## Learning Objectives

- Install VS Code with essential extensions for Python, Jupyter, linting, and remote SSH
- Configure format-on-save, type checking, and notebook output scrolling for AI workflows
- Set up Remote SSH to edit and debug code on remote GPU machines as if they were local
- Evaluate editor alternatives (Cursor, Windsurf, Neovim) and their tradeoffs for AI work

## The Problem

You'll spend thousands of hours inside your editor writing Python, running notebooks, debugging training loops, and SSH-ing into GPU boxes. A misconfigured editor turns every session into friction: no autocomplete, no type hints, no inline errors, manual formatting, and a clunky terminal workflow.

The right setup takes 20 minutes. Skipping it costs you 20 minutes every day.

## The Concept

An AI engineering editor setup needs five things:

```mermaid
graph TD
    L5["5. Remote Development<br/>SSH into GPU boxes, cloud VMs"] --> L4
    L4["4. Terminal Integration<br/>Run scripts, debug, monitor GPU"] --> L3
    L3["3. AI-Specific Settings<br/>Auto-format, type checking, rulers"] --> L2
    L2["2. Extensions<br/>Python, Jupyter, Pylance, GitLens"] --> L1
    L1["1. Base Editor<br/>VS Code — free, extensible, universal"]
```

## Build It

### Step 1: Install VS Code

VS Code is the recommended editor. It is free, runs on every OS, has first-class Jupyter notebook support, and the extension ecosystem covers everything you need for AI work.

Download it from [code.visualstudio.com](https://code.visualstudio.com/).

Verify from the terminal:

```bash
code --version
```

If `code` is not found on macOS, open VS Code, press `Cmd+Shift+P`, type "Shell Command", and select "Install 'code' command in PATH".

### Step 2: Install Essential Extensions

Open the integrated terminal in VS Code (`Ctrl+`` ` or `` Cmd+` ``) and install the extensions that matter for AI work:

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-python.debugpy
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
```

What each one does:

| Extension | Why |
|-----------|-----|
| Python | Language support, virtual env detection, run/debug |
| Pylance | Fast type checking, autocomplete, import resolution |
| Jupyter | Run notebooks inside VS Code, variable explorer |
| GitLens | See who changed what, inline git blame |
| Remote SSH | Open a folder on a remote GPU box as if it were local |
| Debugpy | Step-through debugging for Python |
| Black Formatter | Auto-format on save, consistent style |
| Ruff | Fast linting, catches common mistakes |

The file `code/.vscode/extensions.json` in this lesson contains the full recommendations list. When you open the project folder, VS Code will prompt you to install them.

### Step 3: Configure Settings

Copy the settings from `code/.vscode/settings.json` in this lesson, or apply them manually through `Settings > Open Settings (JSON)`.

The key settings for AI work:

```jsonc
{
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "notebook.output.scrolling": true,
    "files.autoSave": "afterDelay"
}
```

Why these matter:

- **Type checking on basic**: Catches wrong argument types before you run. Saves debugging time on tensor shape mismatches and wrong API parameters.
- **Format on save**: Never think about formatting again. Black handles it.
- **Rulers at 88 and 120**: Black wraps at 88. The 120 marker shows when docstrings and comments are getting too long.
- **Notebook output scrolling**: Training loops print thousands of lines. Without scrolling, the output panel explodes.
- **Auto-save**: You will forget to save. Your training script will run stale code. Auto-save prevents that.

### Step 4: Terminal Integration

VS Code's integrated terminal is where you run training scripts, monitor GPUs, and manage environments.

Set it up properly:

```jsonc
{
    "terminal.integrated.defaultProfile.osx": "zsh",
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.fontSize": 13,
    "terminal.integrated.scrollback": 10000
}
```

Useful shortcuts:

| Action | macOS | Linux/Windows |
|--------|-------|---------------|
| Toggle terminal | `` Ctrl+` `` | `` Ctrl+` `` |
| New terminal | `Ctrl+Shift+`` ` | `Ctrl+Shift+`` ` |
| Split terminal | `Cmd+\\` | `Ctrl+\\` |

Split terminals are useful: one for running your script, one for monitoring GPU with `nvidia-smi -l 1` or `watch -n 1 nvidia-smi`.

### Step 5: Remote Development (SSH into GPU Boxes)

This is the most important extension for AI work. You will run training on remote machines (cloud VMs, lab servers, Lambda, Vast.ai). Remote SSH lets you open the remote filesystem, edit files, run terminals, and debug as if everything were local.

Setup:

1. Install the Remote SSH extension (done in Step 2).
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`), type "Remote-SSH: Connect to Host".
3. Enter `user@your-gpu-box-ip`.
4. VS Code installs its server component on the remote machine automatically.

For passwordless access, set up SSH keys:

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
ssh-copy-id user@your-gpu-box-ip
```

Add the host to `~/.ssh/config` for convenience:

```
Host gpu-box
    HostName 203.0.113.50
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

Now `Remote-SSH: Connect to Host > gpu-box` connects instantly.

## Alternatives

### Cursor

[cursor.com](https://cursor.com) is a VS Code fork with built-in AI code generation. It uses the same extension ecosystem and settings format. If you use Cursor, everything in this lesson still applies. Import the same `settings.json` and `extensions.json`.

### Windsurf

[windsurf.com](https://windsurf.com) is another AI-first VS Code fork. Same story: same extensions, same settings format, same Remote SSH support.

### Vim/Neovim

If you already use Vim or Neovim and are productive in it, stay there. The minimum setup for AI Python work:

- **pyright** or **pylsp** for type checking (via Mason or manual install)
- **nvim-lspconfig** for language server integration
- **jupyter-vim** or **molten-nvim** for notebook-like execution
- **telescope.nvim** for file/symbol search
- **none-ls.nvim** with black and ruff for formatting/linting

If you do not already use Vim, do not start now. The learning curve will compete with learning AI engineering. Use VS Code.

## Use It

With this setup, your daily workflow looks like:

1. Open the project folder in VS Code (or connect via Remote SSH to a GPU box).
2. Write Python in the editor with autocomplete, type hints, and inline errors.
3. Run Jupyter notebooks inline with the Jupyter extension.
4. Use the integrated terminal for training scripts, `uv pip install`, and GPU monitoring.
5. Review changes with GitLens before committing.

## Exercises

1. Install VS Code and all extensions listed in Step 2
2. Copy the `settings.json` from this lesson into your VS Code config
3. Open a Python file and verify that Pylance shows type hints and Black formats on save
4. If you have access to a remote machine, set up Remote SSH and open a folder on it

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| LSP | "Autocomplete engine" | Language Server Protocol: a standard for editors to get type info, completions, and diagnostics from a language-specific server |
| Pylance | "The Python plugin" | Microsoft's Python language server using Pyright for type checking and IntelliSense |
| Remote SSH | "Working on the server" | VS Code extension that runs a lightweight server on a remote machine and streams the UI to your local editor |
| Format on save | "Auto-prettier" | The editor runs a formatter (Black, Ruff) every time you save, so code style is always consistent |
"""

system_prompt = """Translate markdown lessons from English to Polish faithfully. Minimal intervention - do not improve, shorten, or change tone.

Rules:
1. Leave ALL code blocks EXACTLY as-is (```bash, ```jsonc, etc.)
2. Leave function names, variables, settings, extensions as-is: VS Code, Pylance, Black, Ruff, Debugpy, GitLens, Jupyter, Remote SSH, nvidia-smi, pyright, pylsp, nvm-lspconfig, telescope.nvim, none-ls.nvim, jupyter-vim, molten-nvim, Cmd, Ctrl, etc.
3. Keep these anglicisms in English: API, GPU, CPU, RAM, SQL, Python, PyTorch, TensorFlow, HuggingFace, LangChain, Docker, Kubernetes, Git, JSON, XML, HTML, CSS, JavaScript, TypeScript, LLM, GPT, BERT, MLOps, DevOps, CI/CD, REST, NoSQL, CUDA, Jupyter, VS Code, SSH, LSP, IntelliSense, Colab, Cursor, Windsurf, Neovim, Vim, Zsh, Bash, macOS, Linux, Windows, WSL2, GPU
4. Polish diacritics: pamietam→pamiętam, pisującego→piszącego, przylapać→przyłapać, Cie→Cię, Huełałem→Hulałem, Jerkałem→Jęrkałem
5. Add commas before: że, bo, żeby, i (when joining two independent sentences), co, który/która/które, a (contrast), więc, aż, zanim, gdy, albo, lub
6. Section headers: Learning Objectives→Cele uczenia się, The Problem→Problem, The Concept→Koncepcja, Build It→Zbuduj to, Use It→Użyj tego, Alternatives→Alternatywy, Exercises→Ćwiczenia, Key Terms→Kluczowe pojęcia, Prerequisites→Wymagania wstępne, Time→Czas
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
    out_path = f"C:/VisualStudioCodeRepo/AIWorkspace/ai-engineering-from-scratch/.tmp_{lesson_slug}_translated.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"SAVED: {out_path} ({len(result)} chars)")
else:
    print("TRANSLATION FAILED")
