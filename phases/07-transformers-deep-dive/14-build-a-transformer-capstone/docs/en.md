# Zbuduj Transformer od Zera — Capstone

> Trzynaście lekcji. Jeden model. Bez skrótów.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 01 przez 13. Nie pomijaj.
**Czas:** ~120 minut

## Problem

Przeczytałeś każdy paper. Zaimplementowałeś attention, multi-head splits, positional encodings, bloki encoder i decoder, BERT i GPT losses, MoE, KV cache. Teraz spraw żeby działały razem na prawdziwym zadaniu.

Capstone: trenuj mały decoder-only transformer end-to-end na zadaniu character-level language modeling. Czyta Szekspira. Generuje nowego Szekspira. Jest wystarczająco mały żeby trenować na laptopie w mniej niż 10 minut. Jest wystarczająco poprawny żeby zamiana na większy dataset i dłuższy trening daje prawdziwy LM.

To jest "nanoGPT" kursu. Nie jest oryginalny — tutorial nanoGPT Karpathy'ego z 2023 to reference implementation które każdy student pisze przynajmniej raz. Podnosimy kształt i przerabiamy go wokół tego co pokryliśmy.

## Koncepcja

![Transformer-from-scratch block diagram](../assets/capstone.svg)

Architektura, z adnotacjami:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lekcja 04 (opcja RoPE)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lekcja 05
│  MultiHeadAttention (causal)      │  ◀── Lekcja 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lekcja 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lekcja 07
```

### Co wysyłamy

- `GPTConfig` — jedno miejsce do konfiguracji wszystkich hiperparametrów.
- `MultiHeadAttention` — causal, batched, z opcjonalną ścieżką Flash-style (PyTorch's `scaled_dot_product_attention`).
- `SwiGLUFFN` — nowoczesny FFN.
- `Block` — pre-norm, residual-wrapped attention + FFN.
- `GPT` — embeddings, stacked blocks, LM head, generate().
- Training loop z AdamW, cosine LR, gradient clipping.
- Char-level tokenizer na tekście Szekspira.

### Czego nie wysyłamy

- RoPE — zaimplementowany konceptualnie w Lekcji 04. Tutaj używamy learned positional embeddings dla prostoty. Ćwiczenia proszą cię żebyś zamienił na RoPE.
- KV cache podczas generacji — każdy krok generacji przelicza uwagę nad pełnym prefiksem. Wolniejsze ale prostsze. Ćwiczenia proszą żebyś dodał KV cache.
- Flash Attention — PyTorch 2.0+ auto-dispatches jeśli inputs pasują; używamy `F.scaled_dot_product_attention`.
- MoE — single FFN per block. Widziałeś MoE w Lekcji 11.

### Target metrics

Na Mac M2 laptopie, 4-layer, 4-head, d_model=128 GPT trenowany przez 2,000 steps na `tinyshakespeare.txt`:

- Training loss zbiega z ~4.2 (random) do ~1.5 w około 6 minut.
- Sampled output wygląda Szekspirowsko: archaiczne słowa, podziały linii, właściwe imiona jak "ROMEO:" emergują.
- Val loss (held-out final 10% of text) śledzi training loss closely; brak overfitting przy tym rozmiarze/budżecie.

## Zbuduj To

Ta lekcja używa PyTorch. Zainstaluj `torch` (CPU build jest ok). Zobacz `code/main.py`. Script obsługuje:

- Pobieranie `tinyshakespeare.txt` jeśli brakuje (lub czytanie local copy).
- Byte-level char tokenizer.
- Train/val split at 90/10.
- Training loop z bf16 autocast na supported hardware.
- Sampling po zakończeniu treningu.

### Krok 1: dane

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 unique characters. Tiny vocabulary. Fits a 4-byte vocab_size. No BPE, no tokenizer drama.

### Krok 2: model

Zobacz `code/main.py`. Block jest textbook z Lekcji 05 — pre-norm, RMSNorm, SwiGLU, causal MHA. Parameter count dla 4/4/128: ~800K.

### Krok 3: training loop

Weź random batch of length-256 token windows. Forward. Shift-by-one cross-entropy. Backward. AdamW step. Log. Repeat.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Krok 4: sample

Dany prompt, wielokrotnie forward, sample z top-p logits, append, i kontynuuj. Stop po 500 tokens.

### Krok 5: przeczytaj output

Po 2,000 steps:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Not Shakespeare. But Shakespeare-shaped. A clear win for ~800K parameters and 6 minutes on a laptop.

## Użyj To

Ten capstone to reference architecture. Trzy rozszerzenia żeby wysłać to do czegoś prawdziwego:

1. **Zamień tokenizer.** Użyj BPE (e.g. `tiktoken.get_encoding("cl100k_base")`). Vocab size skacze z 65 do ~50,000. Model capacity musi się przeskalować żeby zrekompensować.
2. **Trenuj na bigger corpus.** Użyj `OpenWebText` lub `fineweb-edu` (HuggingFace). 10B tokens na single A100 bierze ~24 godziny dla 125M-param GPT.
3. **Dodaj RoPE + KV cache + Flash Attention.** Ćwiczenia poniżej przechodzą cię przez każde.

To end up jako 125M-parameter GPT który generuje płynny English. Not a frontier model. But the same code path — just bigger — to co Karpathy, EleutherAI, i Allen Institute używają do trenowania research checkpoints w 2026.

## Wyślij To

Zobacz `outputs/skill-transformer-review.md`. Skill recenzuje transformer-from-scratch implementation pod kątem poprawności across all 13 prior lessons.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Zweryfikuj że final-step validation loss twojego trenowanego modelu jest poniżej 2.0. Zmień `max_steps` z 2,000 na 5,000 — czy val loss nadal się poprawia?
2. **Średnie.** Zastąp learned positional embeddings z RoPE. Apply the rotation to Q and K inside `MultiHeadAttention`. Trenuj i zweryfikuj że val loss jest przynajmniej tak niski.
3. **Średnie.** Zaimplementuj KV cache w pętli samplującej. Generuj 500 tokens z i bez cache. Wall-clock powinien się poprawić o 5–20× na laptopie.
4. **Trudne.** Dodaj drugą głowę do modelu która przewiduje next-plus-one token (MTP — Multi-Token Prediction from DeepSeek-V3). Trenuj joint. Czy to pomaga?
5. **Trudne.** Zastąp single FFN per block z 4-expert MoE. Router + top-2 routing. Zobacz jak val loss się zmienia przy matched active parameters.

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; każdy character-LM tutorial od 2015 go używa. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back w bf16, keep optimizer state w fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; zapobiega training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy na danych których model nigdy nie widział; overfit detector. |

## Dalsze Czytanie

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — classic annotated implementation.