# GPT — Maskowane Modelowanie Językowe

> BERT widzi obie strony. GPT widzi tylko przeszłość. Trójkątna maska to najważniejsza jedna linia kodu we współczesnym AI.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 02 (Self-Attention), Faza 7 · 05 (Full Transformer), Faza 7 · 06 (BERT)
**Czas:** ~75 minut

## Problem

Model językowy odpowiada na jedno pytanie: mając pierwsze `t-1` tokenów, jaki jest rozkład prawdopodobieństwa dla tokenu `t`? Trenuj na tym sygnale — predykcja następnego tokenu — a otrzymasz model, który może generować dowolny tekst jeden token na raz.

Aby trenować go end-to-end na całej sekwencji równolegle, potrzebujesz, aby predykcja każdej pozycji zależała tylko od wcześniejszych pozycji. W przeciwnym razie model oszukuje, patrząc na odpowiedź.

Maska causal (przyczynowa) to robi. To jest jedna macierz górnotrójkątna z wartościami `-inf` dodawana do wyników attention przed softmax. Po softmax te pozycje stają się 0. Każda pozycja może attending tylko do siebie i wcześniejszych pozycji. I ponieważ stosujesz to raz do całej sekwencji, otrzymujesz N równoległych predykcji następnego tokenu w jednym forward pass.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — to wszystko są dekodujące tylko causal transformers z tą samą główną pętlą. Po prostu większe, lepsze dane i lepszy RLHF.

## Koncepcja

![Maska causal tworzy trójkątną macierz attention](../assets/causal-attention.svg)

### Maska

Mając sekwencję długości `N`, budujesz macierz `N × N`:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Dodaj `M` do surowych wyników attention przed softmax. `exp(-inf) = 0`, więc zamaskowane pozycje wnoszą zero wagi. Każdy wiersz macierzy attention jest rozkładem prawdopodobieństwa tylko nad wcześniejszymi pozycjami.

Koszt implementacji: jedno wywołanie `torch.tril()`. Czas obliczenia: nanosekundy. Wpływ na dziedzinę: wszystko.

### Równoległe trening, sekwencyjna inferencja

Trening: forward-pass całą sekwencję `(N, d_model)` raz, oblicz N strat entropii krzyżowej (jedna na pozycję), zsumuj, backprop. Równolegle wzdłuż sekwencji. Dlatego trening GPT się skaluje — przetwarzasz 1M tokenów w batchu w jednym przejściu GPU.

Inferencja: generujesz token po tokenu. Podajesz `[t1, t2, t3]`, dostajesz `t4`. Podajesz `[t1, t2, t3, t4]`, dostajesz `t5`. Podajesz `[t1, t2, t3, t4, t5]`, dostajesz `t6`. KV cache (Lekcja 12) zapisuje ukryte stany `t1…tn`, żeby nie przeliczać ich każdorazowo. Ale sekwencyjna głębia przy inferencji = długość outputu. To jest autoregressive tax i dlatego dekodowanie jest bottleneckem latency każdego LLM.

### Strata — przesunięcie o jeden

Mając tokeny `[t1, t2, t3, t4]`:

- Input: `[t1, t2, t3]`
- Cele: `[t2, t3, t4]`

Dla każdej pozycji `i`, oblicz `-log P(target_i | inputs[:i+1])`. Zsumuj. To jest entropia krzyżowa dla całej sekwencji.

Każdy transformer LM, o którym słyszałeś, trenuje na tej stracie. Pre-training, fine-tuning, SFT — ta sama strata, inne dane.

### Strategie dekodowania

Po treningu, wybory samplingowe mają większe znaczenie niż ludzie myślą.

| Metoda | Co robi | Kiedy używać |
|--------|---------|--------------|
| Greedy | Argmax co krok | Zadania deterministyczne, uzupełnianie kodu |
| Temperature | Podziel logits przez T, sample | Zadania kreatywne, wyższe T = większa różnorodność |
| Top-k | Sample tylko z top-k tokenów | Redukuje niskoprawdopodobne ogony |
| Top-p (nucleus) | Sample z najmniejszego zbioru z prawdopodobieństwem skumulowanym ≥ p | Default od 2020; adaptuje się do kształtu rozkładu |
| Min-p | Zachowaj tokeny z `p > min_p * max_p` | Od 2024+; lepsze w odrzucaniu długich ogonów niż top-p |
| Speculative decoding | Model draft proponuje N tokenów, duży model weryfikuje | Redukcja latency 2–3× przy tej samej jakości |

W 2026, min-p + temperature 0.7 to rozsądny default dla open-weights models. Speculative decoding to table stakes dla każdego production inference stack.

### Co sprawiło, że "przepis GPT" zadziałał

1. **Decoder-only.** Brak overhead enkodera. Jeden pass attention + FFN na warstwę.
2. **Scaling.** 124M → 1.5B → 175B → biliony. Chinchilla scaling laws (Lekcja 13) mówią ci, jak wydać compute.
3. **In-context learning.** Wyłoniło się wokół 6B–13B. Model może follow few-shot examples bez fine-tuning.
4. **RLHF.** Post-training na ludzkich preferencjach zamienił surowy pretrained tekst w chat assistantów.
5. **Pre-norm + RoPE + SwiGLU.** Stabilny trening przy skali.

Główna architektura niewiele się zmieniła od GPT-2. Wszystko interesujące wydarzyło się w danych, skali i post-treningu.

## Zbuduj to

### Krok 1: causal mask

Zobacz `code/main.py`. One-liner:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Dodaj to do wyników attention przed softmax. To jest cały mechanizm.

### Krok 2: 2-warstwowy model GPT-ish

Złóż dwa bloki dekodera (masked self-attention + FFN, bez cross-attention). Dodaj token embedding, positional encoding i unembedding (powiązany z macierzą token embedding — standardowy trick od GPT-2).

### Krok 3: predykcja następnego tokenu, end-to-end

Na 20-tokenowym toy vocab, produkuj logits na każdej pozycji. Oblicz cross-entropy loss wobec przesuniętego o jeden celu. Bez gradientu — to jest forward-pass sanity check.

### Krok 4: sampling

Zaimplementuj greedy, temperature, top-k, top-p, min-p. Uruchom każdy na ustalonym prompt i porównaj outputs. Funkcja sampling to 10 linijek.

## Użyj tego

PyTorch, 2026 idiom:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Pod maską, `generate()` uruchamia forward pass, pobiera logits z ostatniej pozycji, sample następny token, appenduje go i powtarza. Każdy production LLM inference stack (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) implementuje tę samą pętlę z ciężką optymalizacją — batched prefill, continuous batching, KV cache paging, speculative decoding.

**GPT vs BERT, jedna linijka każdy:** GPT predicts `P(x_t | x_{<t})`. BERT predicts `P(x_masked | x_unmasked)`. Strata determinuje, czy model może generować.

## Wyślij to

Zobacz `outputs/skill-sampling-tuner.md`. Skill dobiera parametry sampling dla nowego zadania generacji i flaguje, gdy wymagane jest deterministyczne dekodowanie.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` i zweryfikuj, że macierz causal attention jest dolnotrójkątna po softmax. Spot-check: wiersz 3 powinien mieć wagi tylko w kolumnach 0–3.
2. **Średnie.** Zaimplementuj beam search o szerokości 4. Porównaj perplexity beam-4 vs greedy na 10 krótkich promptach. Czy beam zawsze wygrywa? (Hint: zwykle tak dla tłumaczenia, nie dla open-ended chat.)
3. **Trudne.** Zaimplementuj speculative decoding: użyj tiny 2-warstwowego modelu jako draft i 6-warstwowego modelu jako verifier. Zmierz wall-clock speedup na 100 uzupełnieniach o długości 64. Potwierdź, że outputs match greedy verifiera.

## Kluczowe Terminy

| Term | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|--------------------------|
| Causal mask | "Trójkąt" | Górnotrójkątna macierz `-inf` dodawana do wyników attention, żeby pozycja `i` widziała tylko pozycje `≤ i`. |
| Next-token prediction | "Strata" | Entropia krzyżowa rozkładu modelu wobec prawdziwego następnego tokenu na każdej pozycji. |
| Autoregressive | "Generuj jeden na raz" | Podawaj output z powrotem jako input; równoległość tylko podczas treningu, nie podczas generacji. |
| Logits | "Pre-softmax scores" | Surowy output LM head przed softmax; sampling odbywa się na nich. |
| Temperature | "Pokrętło kreatywności" | Podziel logits przez T; T→0 = greedy, T→∞ = uniform. |
| Top-p | "Nucleus sampling" | Obetnij rozkład do najmniejszego zbioru sumującego się do ≥p; sample z tego, co zostanie. |
| Min-p | "Lepsze niż top-p" | Zachowaj tokeny gdzie `p ≥ min_p × max_p`; adaptuje cutoff do ostrości rozkładu. |
| Speculative decoding | "Draft + verify" | Cheap model proponuje N tokenów; big model weryfikuje równolegle. |
| Teacher forcing | "Trick treningowy" | Podczas treningu podawaj prawdziwy poprzedni token, nie predykcję modelu. Standard dla każdego seq2seq LM. |

## Dalsze Czytanie

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 i in-context learning.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — paper o spec decoding.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — kanoniczny kod referencyjny causal-LM.