# Topic Modeling — LDA i BERTopic

> LDA: dokumenty to mieszanki tematów, tematy to rozkłady nad słowami. BERTopic: dokumenty tworzą klastry w przestrzeni embeddingów, klastry to tematy. Ten sam cel, inne prymitywy.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 5 · 02 (BoW + TF-IDF), Faza 5 · 03 (Word2Vec)
**Szacowany czas:** ~45 minut

## Problem

Masz 10 000 zgłoszeń obsługi klienta, 50 000 artykułów informacyjnych lub 200 000 tweetów. Musisz wiedzieć, o czym jest cała kolekcja, bez jej czytania. Nie masz oznaczonych kategorii. Nie wiesz nawet, ile kategorii istnieje.

Topic modeling odpowiada na to bez nadzoru. Podajesz korpus, otrzymujesz niewielki zestaw spójnych tematów i dla każdego dokumentu rozkład prawdopodobieństwa nad tymi tematami.

Dwie rodziny algorytmów dominują. LDA (2003) traktuje każdy dokument jako mieszankę ukrytych tematów, a każdy temat jako rozkład nad słowami. Wnioskowanie jest bayesowskie. Wciąż działa w produkcji, gdzie potrzebujesz przypisań tematów z mieszaną przynależnością i wyjaśnialnych rozkładów prawdopodobieństwa na poziomie słów.

BERTopic (2020) koduje dokumenty za pomocą BERT, redukuje wymiarowość za pomocą UMAP, klastruje za pomocą HDBSCAN i wyodrębnia słowa tematyczne przez class-based TF-IDF. Wygrywa na krótkim tekście, mediach społecznościowych i wszystkim, gdzie podobieństwo semantyczne ma większe znaczenie niż nakładanie się słów. Jeden dokument otrzymuje jeden temat, co jest ograniczeniem dla treści długich.

Ta lekcja buduje intuicję dla obu podejść i wskazuje, które wybrać dla danego korpusu.

## Koncepcja

![Model mieszanki LDA a klastrowanie BERTopic](../assets/topic-modeling.svg)

**Generatywna historia LDA.** Każdy temat to rozkład nad słowami. Każdy dokument to mieszanka tematów. Aby wygenerować słowo w dokumencie, próbkuj temat z mieszanki dokumentu, następnie próbkuj słowo z rozkładu tego tematu. Wnioskowanie odwraca to: mając obserwowane słowa, wnioskuj rozkład tematów na dokument i rozkład słów na temat. Collapsed Gibbs sampling lub variational Bayes robi matematykę.

Kluczowe wyniki LDA:

- `doc_topic`: macierz `(n_docs, n_topics)`, każdy wiersz sumuje się do 1 (mieszanka tematów dokumentu).
- `topic_word`: macierz `(n_topics, vocab_size)`, każdy wiersz sumuje się do 1 (rozkład słów tematu).

**Pipeline BERTopic.**

1. Koduj każdy dokument za pomocą sentence transformera (np. `all-MiniLM-L6-v2`). Wektory 384-wymiarowe.
2. Redukuj wymiarowość za pomocą UMAP do ~5 wymiarów. Embeddingi BERT są zbyt wysokowymiarowe do klastrowania.
3. Klastruj za pomocą HDBSCAN. Oparte na gęstości, tworzy klastry o zmiennej wielkości i etykietę „outlier".
4. Dla każdego klastra oblicz class-based TF-IDF nad dokumentami klastra, aby wyodrębnić najważniejsze słowa.

Wynikiem jest jeden temat na dokument (plus etykieta -1 dla outlierów). Opcjonalnie, miękka przynależność przez wektor prawdopodobieństwa HDBSCAN.

## Zbuduj to

### Krok 1: LDA przez scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Zwróć uwagę: stopwords usunięte, min_df i max_df filtrują rzadkie i wszechobecne terminy, CountVectorizer (nie TfidfVectorizer), ponieważ LDA oczekuje surowych zliczeń.

### Krok 2: BERTopic (produkcja)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Filtr na `Topic != -1` usuwa bucket outlierów BERTopic (dokumenty, których HDBSCAN nie mógł sklastrować). `min_topic_size` kontroluje minimalny rozmiar klastra HDBSCAN; domyślna wartość biblioteki BERTopic to 10. W tym przykładzie ustawiono jawnie na 15 dla skali tej lekcji. Dla korpusów powyżej 10 000 dokumentów zwiększ do 50 lub 100.

### Krok 3: ewaluacja

Obie metody wyprowadzają słowa tematyczne. Pytanie brzmi, czy te słowa są spójne.

- **Topic coherence (c_v).** Łączy NPMI (normalized pointwise mutual information) par najważniejszych słów w kontekście sliding-window, agreguje wyniki w wektory tematyczne i porównuje te wektory przez cosine similarity. Wyższy jest lepszy. Użyj `gensim.models.CoherenceModel` z `coherence="c_v"`.
- **Topic diversity.** Ułamek unikalnych słów we wszystkich najważniejszych słowach tematów. Wyższy jest lepszy (tematy się nie nakładają).
- **Inspekcja jakościowa.** Przeczytaj najważniejsze słowa każdego tematu. Czy nazywają prawdziwą rzecz? Ludzki osąd jest wciąż ostatnią linią obrony.

## Kiedy wybrać które

| Sytuacja | Wybierz |
|----------|---------|
| Krótki tekst (tweety, recenzje, nagłówki) | BERTopic |
| Długie dokumenty z mieszankami tematów | LDA |
| Brak GPU / ograniczone zasoby obliczeniowe | LDA lub NMF |
| Potrzebujesz rozkładów wielotematycznych na poziomie dokumentu | LDA |
| Integracja z LLM do etykietowania tematów | BERTopic (bezpośrednie wsparcie) |
| Deployment na ograniczonym urządzeniu brzegowym | LDA |
| Maksymalna spójność semantyczna | BERTopic |

Najważniejsze praktyczne rozważanie to długość dokumentu. Embeddingi BERT obcinają; zliczenia LDA działają na dowolnej długości. Dla dokumentów dłuższych niż context window modelu embeddingów, albo chunkuj + agreguj, albo użyj LDA.

## Użyj tego

Stack 2026:

- **BERTopic.** Domyślny wybór dla krótkiego tekstu i wszystkiego, gdzie liczy się semantyka.
- **`gensim.models.LdaModel`.** Klasyczny LDA do produkcji, dojrzały, sprawdzony w boju.
- **`sklearn.decomposition.LatentDirichletAllocation`.** Łatwy LDA do eksperymentów.
- **NMF.** Non-negative matrix factorization. Szybka alternatywa dla LDA, porównywalna jakość na krótkim tekście.
- **Top2Vec.** Podobny design do BERTopic. Mniejsza społeczność, ale dobre wyniki na niektórych benchmarkach.
- **FASTopic.** Nowszy, szybszy niż BERTopic na bardzo dużych korpusach.
- **LLM-based labeling.** Uruchom dowolne klastrowanie, następnie zapromptuj model, aby nazwał każdy klaster.

## Wyślij to

Zapisz jako `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Ćwiczenia

1. **Łatwe.** Dopasuj LDA z 5 tematami na zbiorze 20 Newsgroups. Wydrukuj top 10 słów na temat. Oznacz każdy temat ręcznie. Czy algorytm znalazł prawdziwe kategorie?
2. **Średnie.** Dopasuj BERTopic na tym samym podzbiorze 20 Newsgroups. Porównaj liczbę znalezionych tematów, najważniejsze słowa i spójność jakościową w porównaniu z LDA. Który bardziej czysto ujawnia prawdziwe kategorie?
3. **Trudne.** Oblicz c_v coherence dla LDA i BERTopic na swoim korpusie. Uruchom każdy z 5, 10, 20, 50 tematami. Wykreśl coherence vs liczba tematów. Określ, która metoda jest bardziej stabilna wzdłuż liczby tematów.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| Topic | Rzecz, o której jest korpus | Rozkład prawdopodobieństwa nad słowami (LDA) lub klaster podobnych dokumentów (BERTopic). |
| Mixed membership | Dokument jest wieloma tematami | LDA przypisuje każdemu dokumentowi rozkład nad wszystkimi tematami. |
| UMAP | Redukcja wymiarowości | Uczenie rozmaitości zachowujące lokalną strukturę; używane w BERTopic. |
| HDBSCAN | Klastrowanie gęstości | Znajduje klastry o zmiennej wielkości; produkuje etykietę „szumu" (-1) dla outlierów. |
| c_v coherence | Metryka jakości tematu | Średni pointwise mutual information najważniejszych słów tematu w sliding windows. |

## Dalsza lektura

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — artykuł o LDA.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — artykuł o BERTopic.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — artykuł, który wprowadził c_v i pokrewne miary.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — referencja produkcyjna. Świetne przykłady.