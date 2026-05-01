# Word Embeddings — Word2Vec from Scratch

> Słowo poznaje się po towarzystwie, w jakim się znajduje. Wytrenuj płytką sieć na tej zasadzie, a geometria wyłoni się sama.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Szacowany czas:** ~75 minut

## Problem

TF-IDF wie, że `dog` i `puppy` to różne słowa. Nie wie jednak, że znaczą niemal to samo. Classifier wytrenowany na `dog` nie może się uogólnić na recenzję o `puppy`. Możesz to obejść, wymieniając synonimy, ale to zawodzi dla rzadkich terminów, żargonu domenowego i każdego języka, którego nie przewidziałeś.

Chcesz reprezentację, w której `dog` i `puppy` znajdują się blisko siebie w przestrzeni. Gdzie `king - man + woman` ląduje niedaleko `queen`. Gdzie model wytrenowany na `dog` przenosi część sygnału na `puppy` za darmo.

Word2Vec dał nam tę przestrzeń. Dwuwarstwowa sieć neuronowa, trylionowe przebiegi trenowania, opublikowane w 2013. Architektura jest niemal embarrassingly prosta. Wyniki przekształciły NLP na dekadę.

## Koncepcja

![Skip-gram window and embedding space](./assets/word2vec.svg)

**Hipoteza dystrybucyjna** (Firth, 1957): „Poznasz słowo po towarzystwie, w jakim się znajduje." Jeśli dwa słowa pojawiają się w podobnych kontekstach, prawdopodobnie znaczą podobne rzeczy.

Word2Vec występuje w dwóch wariantach, oba wykorzystują tę ideę.

- **Skip-gram.** Mając słowo centralne, przewiduj otaczające słowa. `cat -> (the, sat, on)` z rozmiarem okna 2.
- **CBOW (continuous bag of words).** Mając otaczające słowa, przewiduj centralne. `(the, sat, on) -> cat`.

Skip-gram trenuje się wolniej, ale radzi sobie lepiej z rzadkimi słowami. Stał się domyślnym wyborem.

Sieć ma jedną warstwę ukrytą bez nieliniowości. Input to wektor one-hot nad słownictwem. Output to softmax nad słownictwem. Po treningu wyrzucasz warstwę output. Wagi warstwy ukrytej to embeddings.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          to jest embedding
```

Tri: softmax nad 100k słów jest prohibicyjnie drogi. Word2Vec używa **negative sampling**, by zamienić to w zadanie klasyfikacji binarnej. Przewiduj „czy to słowo kontekstowe pojawiło się w pobliżu tego słowa centralnego, tak lub nie". Próbkuj garść negatywnych (nie-współwystępujących) słów na parę treningową zamiast obliczać softmax nad całym słownictwem.

## Zbuduj to

### Krok 1: pary treningowe z korpusu

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Każda para (center, context) w oknie to pozytywny przykład treningowy.

### Krok 2: tabele embeddingów

Dwie macierze. `W` to tabela embeddingów dla słów centralnych (ta, którą zachowujesz). `W'` to tabela embeddingów dla słów kontekstowych (często wyrzucana, czasem uśredniana z `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Mała losowa inicjalizacja. Rozmiar słownictwa 10k i dim 100 jest realistyczny; do nauki, 50 vocab x 16 dim wystarczy, by zobaczyć geometrię.

### Krok 3: cel negative sampling

Dla każdej pozytywnej pary `(center, context)`, próbkuj `k` losowych słów ze słownictwa jako negatywne. Trenuj model tak, żeby iloczyn skalarny `W[center] · W'[context]` był wysoki dla pozytywnych i niski dla negatywnych.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Magiczny wzór: logistyczna strata na pozytywnej parze (chcesz sigmoid blisko 1) plus logistyczna strata na parach negatywnych (chcesz sigmoid blisko 0). Gradienty płyną do obu tabel. Pełna derivacja jest w oryginalnej pracy; przejdź przez nią raz ołówkiem i papierem, jeśli chcesz, żeby utkwiła.

### Krok 4: trenuj na korpusie zabawkowym

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Po wystarczającej liczbie epoch na dużym korpusie, słowa, które dzielą konteksty, mają podobne centralne embeddingi. Na korpusie zabawkowym widać efekt słabo. Na miliardach tokenów, widać go dramatycznie.

### Krok 5: trick z analogiami

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Na wstępnie wytrenowanych wektorach Google News 300d:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Nie dlatego, że model wie, czym jest monarchia. Dlatego, że wektor `(king - man)` przechwytuje coś jak „królewski", a dodanie go do `woman` ląduje niedaleko regionu królewsko-żeńskiego.

## Użyj tego

Pisanie Word2Vec od zera to nauka. Produkcyjne NLP używa `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

W prawdziwej pracy prawie nigdy nie trenujesz Word2Vec sam. Ściągasz wstępnie wytrenowane wektory.

- **GloVe** — podejście Standfordu oparte na faktoryzacji macierzy współwystępowania. Checkpointy 50d, 100d, 200d, 300d. Dobry ogólny zasięg. Lekcja 04 omawia GloVe konkretnie.
- **fastText** — rozszerzenie Word2Vec od Facebooka, które osadza character n-gramy. Radzi sobie ze słowami out-of-vocabulary poprzez składanie subwordów. Lekcja 04.
- **Pretrained Word2Vec na Google News** — 300d, 3M słownictwo, opublikowane w 2013. Wciąż pobierane codziennie.

### Kiedy Word2Vec wciąż wygrywa w 2026

- Lekkie retrieval specyficzne dla domeny. Trenuj na streszczeniach medycznych w godzinę na laptopie, dostajesz wyspecjalizowane wektory, których żaden ogólny model nie przechwytuje.
- Feature engineering w stylu analogii. `gender_vector = mean(man - woman pairs)`. Odejmuj go od innych słów, by uzyskać oś neutralną płciowo. Wciąż używane w badaniach nad fairness.
- Interpretowalność. 100d jest na tyle małe, że można je wykreślić przez PCA lub t-SNE i faktycznie zobaczyć, jak formują się klastry.
- Wszędzie tam, gdzie inference musi działać on-device bez GPU. Word2Vec lookup to pojedyncze pobranie wiersza.

### Gdzie Word2Vec zawodzi

Ściana polisemii. `bank` ma jeden wektor. `river bank` i `financial bank` dzielą go. `table` (spreadsheet vs. meble) dzieli go. Classifier na downstream nie może rozróżnić znaczeń z samego wektora.

Contextual embeddings (ELMo, BERT, każdy transformer od tego czasu) rozwiązały to, produkując inny wektor dla każdego wystąpienia słowa na podstawie otaczającego kontekstu. To jest skok od Word2Vec do BERT: ze statycznego na kontekstowy. Phase 7 obejmuje połowę transformerów.

Problem out-of-vocabulary to druga porażka. Word2Vec nigdy nie widziało `Zoomer-approved`, jeśli nie było w danych treningowych. Brak fallbacku. fastText to naprawia subword composition (lekcja 04).

## Wyślij to

Zapisz jako `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Ćwiczenia

1. **Łatwe.** Uruchom pętlę treningową na małym korpusie (20 zdań o kotach i psach). Po 200 epoch, zweryfikuj, że `nearest(vocab, W, W[vocab["cat"]])` zwraca `dog` w top 3. Jeśli nie, zwiększ epoch lub słownictwo.
2. **Średnie.** Dodaj subsampling częstych słów. Słowa z częstością powyżej `10^-5` są usuwane z par treningowych z prawdopodobieństwem proporcjonalnym do ich częstości. Zmierz efekt na podobieństwo rzadkich słów.
3. **Trudne.** Trenuj model na korpusie 20 Newsgroups. Oblicz dwie osie bias: `he - she` i `doctor - nurse`. Projektuj słowa zawodów na obie osie. Zgłoś, które zawody mają największą lukę bias. To jest typ probe, którego używają badacze fairness.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Word embedding | Słowo jako wektor | Gęsta, niskowymiarowa (typowo 100-300) reprezentacja nauczona z kontekstu. |
| Skip-gram | Sztuczka Word2Vec | Przewiduj słowa kontekstowe ze słowa centralnego. Wolniejsze niż CBOW, lepsze dla rzadkich słów. |
| Negative sampling | Skrót treningowy | Zastąp softmax nad pełnym vocab klasyfikacją binarną przeciwko `k` losowym słowom. |
| Static embedding | Jeden wektor na słowo | Ten sam wektor niezależnie od kontekstu. Zawodzi na polisemii. |
| Contextual embedding | Wektor wrażliwy na kontekst | Inny wektor dla każdego wystąpienia na podstawie otaczających słów. Co produkują transformery. |
| OOV | Out of vocabulary | Słowo niewidziane w treningu. Word2Vec nie może wyprodukować wektora dla takich. |

## Dalsze czytanie

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — praca o negative sampling. Krótka i czytelna.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — najjasniejsza derivacja gradientów, jeśli matematyka oryginału wydaje się gęsta.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — ustawienia treningowe produkcyjne, które faktycznie działają.