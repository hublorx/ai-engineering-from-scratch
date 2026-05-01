# Dlaczego Transformers — Problemy z RNN

> RNN przetwarzają tokeny jeden po drugim. Transformers przetwarzają wszystkie tokeny na raz. Ten jeden architectural bet zmienił każdą krzywą skalowania w deep learning po 2017.

**Typ:** Nauka
**Języki:** Python
**Wymagania wstępne:** Faza 3 (Deep Learning Core), Faza 5 · 09 (Sequence-to-Sequence), Faza 5 · 10 (Attention Mechanism)
**Czas:** ~45 minut

## Problem

Przed 2017, każdy najlepszy model sekwencyjny na świecie — język, tłumaczenie, mowa — był siecią rekurencyjną. LSTM i GRU wygrały benchmarki tłumaczenia porównywalne z ImageNet przez pół dekady. Były jedynym narzędziem, jakie mieli ludzie.

Miały trzy śmiertelne słabości. Sekwencyjne obliczenia oznaczały, że nie można było zrównoleglić wzdłuż osi czasu: token `t+1` potrzebuje ukrytego stanu od tokena `t`. Sekwencja 1,024 tokenów oznaczała 1,024 kroków seryjnych na GPU, które może wykonać 1,000,000 operacji zmiennoprzecinkowych na cykl. Czas treningu rzeczywistego zegara skaliował się liniowo z długością sekwencji na sprzęcie zaprojektowanym dla równoległości.

Znikające gradienty oznaczały, że informacja 50 tokenów wstecz była już skompresowana przez 50 nieliniowości. Gated recurrent units (LSTM, GRU) złagodziły ten ucisk, ale nigdy go nie wyeliminowały. Zależności długiego zasięgu — "książka, którą przeczytałem ostatniego lata w samolocie do Kioto była…" — rutynowo zawodziły.

Ukryte stany o stałej szerokości oznaczały, że encoder ściskał całą sekwencję źródłową w jeden wektor, zanim decoder cokolwiek zobaczył. Nie ma znaczenia, czy źródło ma 5 tokenów czy 500; bottleneck ma ten sam kształt.

Artykuł z 2017 "Attention Is All You Need" zaproponował coś radykalnego: porzuć rekurencję całkowicie. Pozwól każdej pozycji uczestniczyć w każdej innej pozycji równolegle. Trenuj w jednej wielkiej macierzowej operacji mnożenia zamiast 1,024 sekwencyjnych.

Wynik dominuje każdą modalność do 2026. Język (GPT-5, Claude 4, Llama 4), wizja (ViT, DINOv2, SAM 3), audio (Whisper), biologia (AlphaFold 3), robotyka (RT-2). Ten sam block, różne wejścia.

## Koncepcja

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Rekurencja jako bottleneck.** RNN oblicza `h_t = f(h_{t-1}, x_t)`. Każdy krok zależy od poprzedniego. Nie możesz obliczyć `h_5` przed `h_4`. Na nowoczesnych GPU z 10,000+ równoległymi rdzeniami, to marnuje 99% krzemu na długiej sekwencji.

**Attention jako broadcast.** Self-attention oblicza `output_i = sum_j(a_ij * v_j)` dla każdej pary `(i, j)` jednocześnie. Cała macierz attention N×N wypełnia się w jednej wsadowej operacji macierzowej. Żaden krok nie zależy od drugiego. GPU to kochają.

**Przyspieszenie nie jest stałą.** To różnica między `O(N)` głębokością seryjną a `O(1)` głębokością seryjną. W praktyce, transformers trenują 5–10× szybciej na epokę na dopasowanym sprzęcie przy N=512, a luka rośnie z długością sekwencji, aż uderzysz w `O(N²)` memory wall attention (co Flash Attention później naprawił — patrz Lekcja 12).

**Co kosztują transformers.** Attention memory skaluje się jako `O(N²)`. Dla 2K kontekstu, w porządku. Dla 128K kontekstu, potrzebujesz sliding windows, RoPE extrapolation, Flash Attention tiling, lub linear attention variants. Rekurencja była `O(N)` zarówno w czasie jak i pamięci; transformers wymieniają czas na pamięć, a potem wygrywają czas z powrotem przez równoległość.

**Zmiana inductive bias.** RNN zakładają lokalność i rekurencję. Transformers niczego nie zakładają — każda para jest kandydatem na attention. Dlatego transformers potrzebują więcej danych do trenowania, ale skalują się dalej, gdy już je mają. Chinchilla (2022) sformalizowała to: przy wystarczającej liczbie tokenów, transformer zawsze pokonuje RNN o równej liczbie parametrów.

## Zbuduj to

Żadnej sieci neuronowej tutaj — symulujemy główny bottleneck numerycznie, żebyś poczuł lukę na swoim laptopie.

### Krok 1: zmierz głębokość seryjną

Zobacz `code/main.py`. Budujemy dwie funkcje. Jedna koduje sekwencję jako łańcuch dodawań (szeregowo, jak RNN). Jedna koduje ją jako parallel reduction (broadcast, jak attention). Ta sama matematyka, różny graf zależności.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Mierzymy czas obu na sekwencjach do 100,000 elementów. Wersja RNN jest O(N) i jednopipeline CPU. Nawet w czystym Pythonie, attention-style reduction bije ją przy długości ≥ 1,000, ponieważ Pythonowa `sum()` jest implementowana w C i iteruje bez overheadu interpretera na krok.

### Krok 2: policz teoretyczne operacje

Obie algorytmy wykonują N dodawań. Różnica to *dependency depth*: ile operacji musi nastąpić sekwencyjnie, zanim następna może się zacząć. RNN depth = N. Attention depth = log(N) z tree reduction, lub 1 z parallel scan. Depth, nie liczba op, decyduje o czasie GPU.

### Krok 3: empiryczne skalowanie na długich sekwencjach

Drukujemy tabelę czasową, która pokazuje O(N) lukę. Na laptopie Mac 2026, sekwencje poniżej 1,000 elementów są za szybkie, żeby mierzyć. Sekwencje 100,000 pokazują czysty linear scan. Skaluj to do transformera 16,384 tokenów z 12-warstwowym LSTM equivalent, a zobaczysz, dlaczego czas treningu rzeczywistego był blockerem w 2016.

## Użyj tego

Kiedy w 2026 nadal wybierać RNN:

| Sytuacja | Wybierz |
|-----------|---------|
| Streaming inference, jeden token na raz, stała pamięć | RNN lub state-space model (Mamba, RWKV) |
| Bardzo długie sekwencje (>1M tokenów) gdzie attention memory eksploduje | Linear attention, Mamba 2, Hyena |
| Edge device bez matmul accelerator | Depthwise-separable RNN nadal wygrywa na FLOPs/watt |
| Cokolwiek innego (trening, batched inference, kontekst do 128K) | Transformer |

State-space models (SSMs) jak Mamba są zasadniczo RNN ze strukturalną parametryzacją, która daje im najlepsze z obu: `O(N)` scan memory, parallel training via selective scan. Odtwarzają 90% jakości transformer z lepszym long-context skalowaniem. W 2026 większość frontier labs trenuje hybrydowe SSM+transformer models (np. Jamba, Samba) — rekurencja nie jest martwa, to component.

## Wyślij to

Zobacz `outputs/skill-architecture-picker.md`. Skill wybiera architekturę dla nowego problemu sekwencyjnego przy danej długości, przepustowości i ograniczeniach budżetu treningowego. Powinien zawsze odmawiać polecania czystego RNN dla treningów powyżej 1B tokenów bez stwierdzenia trade-off.

## Ćwiczenia

1. **Łatwe.** Weź `rnn_style` z `code/main.py` i zamień skalarny ukryty stan na wektor ukrytych stanów długości 64. Zmierz ponownie. Jak bardzo rośnie serial overhead z wymiarem ukrytego stanu?
2. **Średnie.** Zaimplementuj parallel prefix-sum (Hillis-Steele scan) w czystym Pythonie. Zweryfikuj, że produkuje tę samą numeryczną output co serial scan na długości 1024. Policz depth.
3. **Trudne.** Przenoś attention-style reduction do PyTorch na GPU. Mierz czas obu, gdy sweepujesz długość sekwencji od 64 do 65,536. Wykreśl i wyjaśnij kształt krzywej.

## Kluczowe Terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|--------------------------|
| Recurrence | "RNNs are sequential" | Obliczenia gdzie krok `t` zależy od kroku `t-1`, wymuszając szeregowe wykonanie wzdłuż osi czasu. |
| Serial depth | "How deep the graph is" | Najdłuższy łańcuch zależnych ops; ogranicza rzeczywisty zegar nawet na nieskończonym sprzęcie. |
| Attention | "Let tokens look at each other" | Ważona suma `sum_j a_ij v_j` gdzie `a_ij` pochodzi ze stopnia podobieństwa między pozycjami i i j. |
| Context window | "How much the model sees" | Liczba pozycji, którą warstwa attention może wziąć jako wejście; kwadratowy koszt pamięci skaluje się tutaj. |
| Inductive bias | "Assumptions baked into the architecture" | Priorytet o tym, jak dane wyglądają; CNN zakładają translation invariance, RNN zakładają rekurencję. |
| State-space model | "RNN with algebra behind it" | Rekurencja sparametryzowana dla parallel training przez strukturalne macierze state-space. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = `O(N²)` w długości sekwencji; Flash Attention ukrywa stałe, nie skalowanie. |

## Dalsze Czytanie

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — artykuł, który zabił rekurencję w mainstream NLP.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — gdzie attention się urodziło, doczepione do RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — oryginalny artykuł LSTM, dla rekordu.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — nowoczesna odpowiedź rekurencyjna na transformers.