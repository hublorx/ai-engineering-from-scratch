# BERT — Masked Language Modeling

> GPT przewiduje następne słowo. BERT przewiduje brakujące słowo. Jedno zdanie różnicy — i pół dekady wszystkiego w kształcie embeddingów.

**Typ:** Buduj
**Języki:** Python
**Wymagania wstępne:** Faza 7 · 05 (Pełny Transformer), Faza 5 · 02 (Text Representation)
**Czas:** ~45 minut

## Problem

W 2018 każde NLP task — sentiment, NER, QA, entailment — trenowało własny model od zera na własnych oznaczonych danych. Nie było pre-trained "rozumiej angielski" checkpoint, który mogłeś fine-tune. ELMo (2018) pokazał, że można pre-train contextual embeddings z bidirectional LSTM; to pomagało, ale nie uogólniało.

BERT (Devlin et al. 2018) zapytał: co gdybyśmy wzięli transformer encoder, trenowali na każdym zdaniu w internecie i zmusili go do przewidywania brakujących słów z kontekstu z obu stron? Potem fine-tune jeden head na swoim downstream task. Parameter efficiency była objawieniem.

Wynik: w ciągu 18 miesięcy BERT i jego warianty (RoBERTa, ALBERT, ELECTRA) zdominowały każdy NLP leaderboard, który istniał. Do 2020 każda wyszukiwarka, pipeline moderacji treści i system semantic-search na ziemi miał BERT inside.

W 2026 encoder-only models wciąż są właściwym narzędziem dla classification, retrieval i structured extraction — działają 5–10× szybciej per token niż decodery, a ich embeddings są backbone każdego nowoczesnego retrieval stack. ModernBERT (Dec 2024) popchnął architekturę do 8K kontekstu z Flash Attention + RoPE + GeGLU.

## Koncepcja

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### Sygnał treningowy

Weź zdanie: `the quick brown fox jumps over the lazy dog`.

Zamaskuj 15% tokenów losowo:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Trenuj model, żeby przewidywał oryginalne tokeny na zamaskowanych pozycjach. Bo encoder jest bidirectional, przewidywanie `[MASK]` na pozycji 1 może użyć `brown fox jumps` na pozycjach 2+. To jest to, czego GPT nie może zrobić.

### Zasady maskowania BERT

Z 15% tokenów wybranych do przewidywania:

- 80% jest zastąpionych `[MASK]`.
- 10% jest zastąpionych losowym tokenem.
- 10% pozostaje niezmienionych.

Dlaczego nie zawsze `[MASK]`? Bo `[MASK]` nigdy nie pojawia się w czasie inferencji. Trenowanie modelu, żeby oczekiwał `[MASK]` na 100% zamaskowanych pozycji stworzyłoby distribution shift między pretraining a fine-tuning. 10% random + 10% unchanged utrzymuje model honest.

### Next Sentence Prediction (NSP) — i dlaczego została upuszczona

Oryginalny BERT trenował też na NSP: przy dwóch zdaniach A i B, przewiduj czy B następuje po A. RoBERTa (2019) to abladowała i pokazała, że NSP szkodzi, nie pomaga. Nowoczesne encodery to pomijają.

### Co się zmieniło w 2026: ModernBERT

Artykuł ModernBERT z 2024 przebudował block z 2026 primitives:

| Komponent | Oryginalny BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Pozycyjne | Learned absolute | RoPE |
| Aktywacja | GELU | GeGLU |
| Normalizacja | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Naprzemienne local (128) + global |
| Długość kontekstu | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

I w przeciwieństwie do stacka z 2018, jest Flash-Attention-native. Inferencja jest 2–3× szybsza przy długości sekwencji 8K niż DeBERTa-v3 z lepszymi GLUE scores.

### Przypadki użycia, które wciąż wybierają encoder w 2026

| Task | Dlaczego encoder bije decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings | Bidirectional context = better embedding quality per token |
| Classification (sentiment, intent, toxicity) | One forward pass; no generation overhead |
| NER / token labeling | Per-position output, natively bidirectional |
| Zero-shot entailment (NLI) | Classifier head on top of encoder |
| Reranker for RAG | Cross-encoder scoring, 10x faster than LLM rerankers |

## Zbuduj to

### Krok 1: logika maskowania

Zobacz `code/main.py`. Funkcja `create_mlm_batch` przyjmuje listę token IDs, rozmiar vocab i mask probability. Zwraca input IDs (z maskami applied) i labels (tylko na zamaskowanych pozycjach, -100 gdzie indziej — konwencja ignore index PyTorch).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Krok 2: uruchom MLM prediction na tiny corpus

Trenuj 2-warstwowy encoder + MLM head na vocabularzu 20 słów, 200 zdań. Bez gradientu — robimy forward-pass sanity checks. Pełny trening potrzebuje PyTorch.

### Krok 3: porównaj typy mask

Pokaż jak trój-drogowa zasada utrzymuje model usable bez `[MASK]`. Przewiduj na niezamaskowanym zdaniu i na zamaskowanym zdaniu. Oba powinny produkować rozsądne rozkłady tokenów, bo model widział oba patterny w treningu.

### Krok 4: fine-tune head

Zamień MLM head na classification head na toy sentiment dataset. Tylko head się trenuje; encoder jest zamrożony. To jest pattern, który każdy BERT application followuje.

## Użyj tego

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding models to fine-tuned BERT.** `sentence-transformers` models jak `all-MiniLM-L6-v2` to BERTs trenowane z contrastive loss. Encoder jest ten sam. Loss się zmienił.

**Cross-encoder rerankers to też fine-tuned BERT.** Pair-classification na `[CLS] query [SEP] doc [SEP]`. Bidirectional attention między query i doc jest dokładnie tym, co daje cross-encoderom ich jakościową przewagę nad biencoders.

**Kiedy nie wybierać BERT w 2026.** Cokolwiek generacyjnego. Encoder nie ma sensible sposobu na autoregressive produkowanie tokenów. Też: cokolwiek poniżej 1B params gdzie small decoder może match quality z większą flexibility (Phi-3-Mini, Qwen2-1.5B).

## Wyślij to

Zobacz `outputs/skill-bert-finetuner.md`. Skill scopinguje BERT fine-tune (wybór backbone, specyfikacja head, dane, eval, stopping) dla nowego classification lub extraction task.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py` i wydrukuj mask distribution przez 10,000 tokenów. Potwierdź ~15% jest wybranych, a z tych ~80% staje się `[MASK]`.
2. **Średnie.** Zaimplementuj whole-word masking: jeśli słowo jest tokenizowane na subwords, maskuj wszystkie subwords razem lub żadne. Zmierz, czy to poprawia MLM accuracy na 500-zdaniowym corpusie.
3. **Trudne.** Trenuj tiny (2-warstwowy, d=64) BERT na 10,000 zdań z public dataset. Fine-tune `[CLS]` token dla SST-2 sentiment. Porównaj przeciwko decoder-only baseline przy matched params — który wygrywa?

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-----------------------|
| MLM | "Masked language modeling" | Sygnał treningowy: losowo zastąp 15% tokenów `[MASK]`, przewiduj oryginalne. |
| Bidirectional | "Looks both ways" | Encoder attention nie ma causal mask — każda pozycja widzi każdą inną pozycję. |
| `[CLS]` | "The pooler token" | Specjalny token dodany do każdej sekwencji; jego final embedding jest używany jako sentence-level representation. |
| `[SEP]` | "Segment separator" | Separuje paired sequences (np. query/doc, zdanie A/B). |
| NSP | "Next sentence prediction" | Drugi BERT pretraining task; pokazano, że bezużyteczny w RoBERTa, upuszczony po 2019. |
| Fine-tuning | "Adapt to a task" | Trzymaj encoder mostly frozen; trenuj mały head na górze dla downstream task. |
| Cross-encoder | "A reranker" | BERT który bierze zarówno query jak i doc jako input, zwraca relevance score. |
| ModernBERT | "2024 refresh" | Encoder przebudowany z RoPE, RMSNorm, GeGLU, naprzemiennym local/global attention, 8K kontekst. |

## Dalsze Czytanie

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — oryginalny artykuł.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — jak trenować BERT poprawnie; zabija NSP.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — replaced-token detection bije MLM przy matched compute.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — artykuł ModernBERT.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — kanoniczny encoder reference.