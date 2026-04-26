# Słownik terminów AI Engineering

## A

### Agent
- **Co ludzie mówią:** "Autonomiczny AI, który samodzielnie myśli i działa"
- **Co to właściwie oznacza:** Pętla while, gdzie LLM decyduje, które narzędzie wywołać następne, wykonuje je, widzi wynik i powtarza
- **Dlaczego tak się nazywa:** Zapożyczone z filozofii - "agent" to cokolwiek, co może działać w świecie. W AI to po prostu "LLM + narzędzia + pętla"

### Attention
- **Co ludzie mówią:** "Jak AI skupia się na ważnych częściach"
- **Co to właściwie oznacza:** Mechanizm, gdzie każdy token oblicza ważoną sumę wartości wszystkich innych tokenów, z wagami określonymi przez to, jak są istotne (przez iloczyn skalarny wektorów query i key)
- **Dlaczego tak się nazywa:** Praca z 2017 "Attention Is All You Need" nazwała to przez analogię do ludzkiej selektywnej uwagi

### Alignment
- **Co ludzie mówią:** "Uczynienie AI bezpiecznym"
- **Co to właściwie oznacza:** Techniczne wyzwanie sprawienia, by zachowanie systemu AI odpowiadało ludzkim intencjom, wartościom i preferencjom, włącznie z przypadkami brzegowymi, których projektant nie przewidział

### Autoregressive
- **Co ludzie mówią:** "AI generuje jedno słowo na raz"
- **Co to właściwie oznacza:** Model, który przewiduje następny token warunkowany przez wszystkie poprzednie tokeny, a potem wkłada to przewidywanie z powrotem jako wejście do następnego kroku. GPT, LLaMA i Claude to wszystko autoregresywne.

### Activation Function (Funkcja aktywacji)
- **Co ludzie mówią:** "Nieliniowa rzecz między warstwami"
- **Co to właściwie oznacza:** Funkcja stosowana po każdej warstwie liniowej, która wprowadza nieliniowość. Bez niej składanie dowolnej liczby warstw liniowych redukuje się do jednej transformacji liniowej. ReLU, GELU i SiLU są najczęstsze. Wybór wpływa bezpośrednio na to, czy gradienty przepływają podczas treningu.

### Adam (Optimizer)
- **Co ludzie mówią:** "Domyślny optimizer"
- **Co to właściwie oznacza:** Adaptive Moment Estimation. Łączy momentum (pierwszy moment) z adaptacyjnymi stopami uczenia na parametr (drugi moment). Ma korekcję biasu dla wczesnych kroków. Działa dobrze w większości zadań bez dużego dostrajania.

### AdamW
- **Co ludzie mówią:** "Adam, ale lepszy"
- **Co to właściwie oznacza:** Adam ze sprzężonym spadkiem wag. W standardowym Adamie L2 regularization jest skalowany przez adaptacyjną stopę uczenia na parametr, co nie jest tym, czego chcesz. AdamW stosuje weight decay bezpośrednio do wag, niezależnie od statystyk gradientu. Domyślny optimizer do trenowania transformerów.

### Autograd
- **Co ludzie mówią:** "Automatyczne gradienty"
- **Co to właściwie oznacza:** System, który nagrywa operacje na tensorach i automatycznie oblicza gradienty przez tryb odwrotny różniczkowania. Autograd w PyTorch buduje graf obliczeniowy w locie (dynamic graph), podczas gdy JAX używa transformacji funkcjnych (grad). To jest to, co sprawia, że backpropagation jest praktyczna - piszesz forward pass, a framework oblicza wszystkie pochodne.

## B

### Batch Size
- **Co ludzie mówią:** "Ile przykładów na raz"
- **Co to właściwie oznacza:** Liczba przykładów treningowych przetwarzanych w jednym forward/backward pass przed aktualizacją wag. Większe batche dają stabilniejsze oszacowania gradientu, ale zużywają więcej pamięci. Typowe wartości: 32-512 do treningu, większe do wnioskowania. Batch size współdziała ze stopą uczenia - podwój batch, podwój LR (reguła liniowego skalowania).

### Backpropagation
- **Co ludzie mówią:** "Jak sieci neuronowe się uczą"
- **Co to właściwie oznacza:** Algorytm, który oblicza, ile każda waga przyczyniła się do błędu, stosując regułę łańcuchową wstecz przez sieć, a potem dostosowuje wagi proporcjonalnie
- **Dlaczego tak się nazywa:** Błędy propagują się wstecz od wyjścia do wejścia, warstwa po warstwie

## C

### Context Window (Okno kontekstowe)
- **Co ludzie mówią:** "Ile AI może zapamiętać"
- **Co to właściwie oznacza:** Maksymalna liczba tokenów (wejście + wyjście), które mieszczą się w jednym wywołaniu API. To nie pamięć - to bufor o stałym rozmiarze, który resetuje się przy każdym wywołaniu

### Chain of Thought (CoT)
- **Co ludzie mówią:** "Zmuszanie AI do myślenia krok po kroku"
- **Co to właściwie oznacza:** Technika promptingu, gdzie prosisz model, żeby pokazał swoje kroki rozumowania, co poprawia dokładność w problemach wielokrokowych, bo każdy krok warunkuje następny

### CNN (Convolutional Neural Network)
- **Co ludzie mówią:** "Obrazowe AI"
- **Co to właściwie oznacza:** Sieć neuronowa, która używa operacji splotu (ślizgające się filtry po wejściu) do wykrywania lokalnych wzorców. Składanie splotów wykrywa coraz bardziej złożone cechy: krawędzie, tekstury, obiekty.

### CUDA
- **Co ludzie mówią:** "Programowanie GPU"
- **Co to właściwie oznacza:** Równolegława platforma obliczeniowa NVIDIA. Pozwala uruchamiać operacje macierzowe na tysiącach rdzeni GPU jednocześnie. PyTorch i TensorFlow używają CUDA pod spodem.

### Chunking
- **Co ludzie mówią:** "Dzielnie dokumentów na kawałki"
- **Co to właściwie oznacza:** Rozbijanie tekstu na segmenty przed embeddingiem do retrieval. Rozmiar chunku determinuje granularność wyników wyszukiwania. Za mały: traci kontekst. Za duży: rozcieńcza relewancję. Typowe strategie: stały rozmiar z nakładką, oparty na zdaniach, lub semantyczny podział. Typowy rozmiar chunku: 256-512 tokenów z 10-20% nakładką.

### Contrastive Learning
- **Co ludzie mówią:** "Uczenie się przez porównywanie"
- **Co to właściwie oznacza:** Trening przez przyciąganie podobnych par bliżej i odpychanie niepodobnych par dalej w przestrzeni embeddingów. CLIP tego używa: dopasowywanie par obraz-tekst vs niedopasowane pary.

### Cosine Similarity (Podobieństwo cosinusowe)
- **Co ludzie mówią:** "Jak podobne są dwa wektory"
- **Co to właściwie oznacza:** Cosinus kąta między dwoma wektorami: dot(a, b) / (||a|| * ||b||). Zakres od -1 (przeciwne) do 1 (identyczny kierunek). Ignoruje magnitudę, liczy się tylko kierunek. Standardowa metryka podobieństwa dla embeddingów i wyszukiwania semantycznego.

### Cross-Entropy (Entropia krzyżowa)
- **Co ludzie mówią:** "Funkcja straty klasyfikacji"
- **Co to właściwie oznacza:** Mierzy różnicę między dwoma dystrybucjami prawdopodobieństwa. Dla klasyfikacji: -sum(y_true * log(y_pred)). Dla modeli językowych: ujemne log prawdopodobieństwo poprawnego następnego tokenu. Niższy jest lepszy. Perplexity to po prostu exp(cross-entropy).

## D

### Data Augmentation
- **Co ludzie mówią:** "Tworzenie więcej danych treningowych"
- **Co to właściwie oznacza:** Tworzenie zmodyfikowanych kopii istniejących danych (obracanie obrazów, dodawanie szumu, parafrazywanie tekstu), żeby zwiększyć różnorodność zbioru treningowego bez zbierania nowych danych. Redukuje overfitting.

### Decoder (Dekoder)
- **Co ludzie mówią:** "Część wyjściowa"
- **Co to właściwie oznacza:** W transformerach, dekoder używa causal (zamaskowanego) self-attention, więc każda pozycja może Attendować tylko do wcześniejszych pozycji. GPT to tylko dekodery. BERT to tylko enkoder. T5 to enkoder-dekoder.

### Diffusion Model
- **Co ludzie mówią:** "AI, które generuje obrazy z szumu"
- **Co to właściwie oznacza:** Model trenowany do odwracania procesu stopniowego zaszumiania - uczy się przewidywać i usuwać szum, a w czasie generowania startuje od czystego szumu i iteracyjnie odzyskuje

### DPO (Direct Preference Optimization)
- **Co ludzie mówią:** "Prostszy RLHF"
- **Co to właściwie oznacza:** Metoda treningowa, która pomija model nagrody całkowicie - optymalizuje bezpośrednio model językowy, żeby woleć lepszą odpowiedź w parach preferencji ludzkich

### Dropout
- **Co ludzie mówią:** "Losowe wyłączanie neuronów"
- **Co to właściwie oznacza:** Podczas treningu losowo ustaw część aktywacji na zero. Zmusza sieć do niepolegania na żadnym pojedynczym neuronie. Wyłączasz podczas wnioskowania. Prosty, ale skuteczny regularization.

## E

### Eigenvalue
- **Co ludzie mówią:** "Jakaś matematyczna rzecz dla PCA"
- **Co to właściwie oznacza:** Dla macierzy A, eigenvalue lambda spełnia Av = lambda*v dla jakiegoś wektora v. Mówi Ci, jak bardzo macierz skaluje wektory w tym kierunku. Duże eigenvalues = kierunki wysokiej wariancji w Twoich danych.

### Embedding
- **Co ludzie mówią:** "Jakaś AI magia, która zamienia słowa na liczby"
- **Co to właściwie oznacza:** Nauczone odwzorowanie z dyskretnych elementów (słowa, obrazy, użytkownicy) na gęste wektory w przestrzeni ciągłej, gdzie podobne elementy lądują blisko siebie
- **Dlaczego tak się nazywa:** Elementy są "osadzone" w przestrzeni geometrycznej, gdzie odległość ma znaczenie

### Encoder (Enkoder)
- **Co ludzie mówią:** "Część wejściowa"
- **Co to właściwie oznacza:** W transformerach, enkoder używa dwukierunkowego self-attention, więc każda pozycja może Attendować do wszystkich pozycji. BERT to tylko enkoder. Dobry do zadań rozumienia (klasyfikacja, NER), ale nie do generacji.

### Epoch
- **Co ludzie mówią:** "Jeden przebieg przez dane"
- **Co to właściwie oznacza:** Dokładnie to. Jeden kompletny przebieg przez każdy przykład w zbiorze treningowym. Wiele epoch = zobaczenie danych wiele razy. Więcej epoch może poprawić uczenie, ale ryzykuje overfitting.

## F

### Feature (Cecha)
- **Co ludzie mówią:** "Kolumna w Twoich danych"
- **Co to właściwie oznacza:** Indywidualna mierzalna właściwość danych. W klasycznym ML, projektujesz cechy ręcznie. W głębokim uczeniu, sieć uczy się cech automatycznie z surowych danych.

### Few-Shot
- **Co ludzie mówią:** "Daj AI kilka przykładów najpierw"
- **Co to właściwie oznacza:** Dołączenie małej liczby przykładów wejście-wyjście w prompcie przed poproszeniem modelu o wykonanie zadania. Typowo 3-5 przykładów. Model dopasowuje wzorce na tych przykładach, żeby zrozumieć pożądany format i zachowanie. Porównaj z zero-shot (bez przykładów) i fine-tuning (tysiące przykładów wbudowane w wagi).

### Fine-tuning
- **Co ludzie mówią:** "Trenowanie AI na Twoich danych"
- **Co to właściwie oznacza:** Wyruszanie od wag pre-trenowanego modelu i kontynuowanie treningu na mniejszym, specyficznym dla zadania zbiorze danych. Aktualizuje tylko istniejące wagi, nie dodaje nowej wiedzy od zera

### Function Calling
- **Co ludzie mówią:** "AI, które może używać narzędzi"
- **Co to właściwie oznacza:** Ustrukturyzowany sposób dla LLM do żądania wykonania zewnętrznych funkcji. Definiujesz narzędzia z opisami JSON Schema, model wyprowadza ustrukturyzowany obiekt JSON określający, którą funkcję wywołać z jakimi argumentami, Twój kod wykonuje to, a wynik wraca do modelu. To nie to samo co agenci - function calling to mechanizm, agenci to pętla.

## G

### Guardrails
- **Co ludzie mówią:** "Filtry bezpieczeństwa dla AI"
- **Co to właściwie oznacza:** Warstwy walidacji wejścia/wyjścia wokół LLM, które wykrywają i blokują szkodliwe treści, próby wstrzykiwania promptów, wycieki PII, lub odpowiedzi off-topic. Typowo pipeline: filtr wejścia -> LLM -> filtr wyjścia. Mogą być oparte na regułach (regex, listy słów kluczowych) lub modelowe (klasyfikator, który ocenia bezpieczeństwo).

### GPT
- **Co ludzie mówią:** "ChatGPT" lub "AI"
- **Co to właściwie oznacza:** Generative Pre-trained Transformer - konkretna architektura, która przewiduje następny token używając dekodera-only transformer trenowanego na dużych korpusach tekstowych
- **Dlaczego tak się nazywa:** Generative (produkuje tekst), Pre-trained (trenowany raz na dużych danych, potem adaptowany), Transformer (architektura)

### GAN (Generative Adversarial Network)
- **Co ludzie mówią:** "Dwa AI walczące ze sobą"
- **Co to właściwie oznacza:** Sieć generatora próbuje tworzyć realistyczne dane, podczas gdy sieć dyskryminatora próbuje odróżnić prawdziwe od fałszywych. Trenują razem: generator staje się lepszy w oszukiwaniu dyskryminatora, a dyskryminator staje się lepszy w wykrywaniu.

### Gradient
- **Co ludzie mówią:** "Nachylenie"
- **Co to właściwie oznacza:** Wektor pochodnych cząstkowych wskazujący kierunek najszybszego wzrostu. W ML idziesz przeciwnie do gradientu (gradient descent), żeby zminimalizować stratę.

### Gradient Descent
- **Co ludzie mówią:** "Jak AI się poprawia"
- **Co to właściwie oznacza:** Algorytm optymalizacyjny, który dostosowuje parametry w kierunku, który najbardziej redukuje funkcję straty, jak schodzenie w dół w krajobrazie wielowymiarowym

## H

### Hyperparameter
- **Co ludzie mówią:** "Ustawienia, które dostrajasz"
- **Co to właściwie oznacza:** Wartości ustawiane przed treningiem, które kontrolują sam proces treningu: learning rate, batch size, liczba warstw, dropout rate. W przeciwieństwie do parametrów modelu (wagi), te nie są uczone z danych.

### Hallucination (Halucynacja)
- **Co ludzie mówią:** "AI kłamie" lub "wymyśla rzeczy"
- **Co to właściwie oznacza:** Model generuje tekst brzmiący wiarygodnie, ale nie oparty na jego danych treningowych lub danym kontekście - to dopełnianie wzorców, nie pobieranie faktów

## I

### Inference
- **Co ludzie mówią:** "Uruchamianie AI"
- **Co to właściwie oznacza:** Używanie wytrenowanego modelu do robienia predykcji na nowych danych. Nie zachodzą żadne aktualizacje wag. To jest to, co robisz w produkcji: wysyłasz wejście, dostajesz wyjście.

### Inductive Bias
- **Co ludzie mówią:** Nigdy o tym nie słyszałem
- **Co to właściwie oznacza:** Założenia wbudowane w architekturę modelu. CNN zakładają, że lokalne wzorce mają znaczenie (splot). RNN zakładają, że kolejność ma znaczenie (przetwarzanie sekwencyjne). Transformery zakładają, że wszystko może się ze sobą wiązać (attention). Właściwy bias pomaga modelowi uczyć się szybciej z mniejszej ilości danych.

### JAX
- **Co ludzie mówią:** "Framework ML Google"
- **Co to właściwie oznacza:** Biblioteka kompatybilna z NumPy, która dodaje automatyczne różniczkowanie (grad), kompilację JIT (jit), automatyczną wektoryzację (vmap) i równoległość na wielu urządzeniach (pmap). W przeciwieństwie do stylu obiektowego PyTorch, JAX jest czysto funkcyjny - brak ukrytego stanu, brak in-place mutation. Używane przez Google DeepMind dla AlphaFold, Gemini i badań na dużą skalę.

## K

### KV Cache
- **Co ludzie mówią:** "Przyspiesza inference"
- **Co to właściwie oznacza:** Podczas autoregresywnej generacji, buforowanie macierzy key i value z poprzednich tokenów, żeby nie przeliczać ich przy każdym kroku. Trade-off: pamięć za szybkość. Kluczowe dla szybkiego inference LLM.

## L

### Latent Space (Przestrzeń latentna)
- **Co ludzie mówią:** "Ukryta reprezentacja"
- **Co to właściwie oznacza:** Skompresowana, nauczona przestrzeń reprezentacji, gdzie podobne wejścia mapują do pobliskich punktów. Autoenkodery, VAE i modele dyfuzyjne wszystkie działają w przestrzeni latentnej. Jest niższa wymiarowo niż wejście, ale przechwytuje ważną strukturę.

### Learning Rate
- **Co ludzie mówią:** "Jak szybko AI się uczy"
- **Co to właściwie oznacza:** Skalar, który kontroluje wielkość kroku podczas gradient descent. Za wysoka: przeskakuje minimum i dywerguje. Za niska: zbiega się za wolno lub blokuje. Najważniejszy hyperparameter.

### LLM (Large Language Model)
- **Co ludzie mówią:** "AI" lub "mózg"
- **Co to właściwie oznacza:** Sieć neuronowa oparta na transformerze trenowana do przewidywania następnego tokenu w sekwencji, z miliardami parametrów, trenowana na danych tekstowych w skali internetu

### LoRA (Low-Rank Adaptation)
- **Co ludzie mówią:** "Efektywny fine-tuning"
- **Co to właściwie oznacza:** Zamiast aktualizować wszystkie wagi, wstawiasz małe macierze niskiego rzędu obok oryginalnych wag. Tylko te małe macierze są trenowane, redukując pamięć o 10-100x

### Loss Function (Funkcja straty)
- **Co ludzie mówią:** "Jak bardzo AI się myli"
- **Co to właściwie oznacza:** Funkcja, która mierzy lukę między przewidywanym a rzeczywistym wyjściem. Trening minimalizuje tę funkcję. MSE dla regresji, cross-entropy dla klasyfikacji, contrastive loss dla embeddingów. Wybór funkcji straty definiuje, co dla modelu oznacza "dobre".

## M

### Mixed Precision (Mieszana precyzja)
- **Co ludzie mówią:** "Sztuczka treningowa dla szybkości"
- **Co to właściwie oznacza:** Używanie float16 dla forward pass i większości operacji (szybsze, mniej pamięci), ale trzymanie float32 dla akumulacji gradientów i aktualizacji wag (bardziej precyzyjne). Daje 2x przyspieszenie z pomijalną stratą dokładności.

### MoE (Mixture of Experts)
- **Co ludzie mówią:** "Tylko część modelu się uruchamia"
- **Co to właściwie oznacza:** Model z wieloma "eksperckimi" podsieciami, gdzie mechanizm routingu wysyła każde wejście tylko do kilku ekspertów. Pełny model jest ogromny, ale każdy forward pass jest tani, bo większość ekspertów jest pomijana. Mixtral i GPT-4 tego używają.

### MCP (Model Context Protocol)
- **Co ludzie mówią:** "Sposób dla AI na używanie narzędzi"
- **Co to właściwie oznacza:** Otwarty protokół (JSON-RPC over stdio/HTTP), który standaryzuje, jak aplikacje AI łączą się ze źródłami danych i narzędziami zewnętrznymi, z typowymi schematami dla narzędzi, zasobów i promptów

## N

### NaN (Not a Number)
- **Co ludzie mówią:** "Trening się zawiesił"
- **Co to właściwie oznacza:** Wartość zmiennoprzecinkowa wskazująca niezdefiniowane wyniki (0/0, inf-inf). W treningu, strata NaN zwykle oznacza: learning rate za wysoki, eksplodujące gradienty, log z zera, lub dzielenie przez zero. Zawsze pierwsza rzecz do sprawdzenia, gdy trening się nie udaje.

### Normalization (Normalizacja)
- **Co ludzie mówią:** "Skalowanie danych"
- **Co to właściwie oznacza:** Dostosowywanie wartości do standardowego zakresu. Batch normalization normalizuje przez batch. Layer normalization normalizuje przez cechy. Obie stabilizują trening i pozwalają na wyższe learning rates.

## O

### Overfitting
- **Co ludzie mówią:** "Model zapamiętał dane"
- **Co to właściwie oznacza:** Model dobrze działa na danych treningowych, ale słabo na niewidzianych danych. Uczył się szumu, nie sygnału. Napraw przez: więcej danych, regularizację (dropout, weight decay), wczesne zatrzymanie, data augmentation, prostszy model.

### Optimizer (Optymalizator)
- **Co ludzie mówią:** "Rzecz, która aktualizuje wagi"
- **Co to właściwie oznacza:** Algorytm, który używa gradientów do aktualizacji parametrów modelu. SGD to najprostszy. Adam to najczęstszy. Każdy optymalizator ma inne właściwości: szybkość zbieżności, użycie pamięci, wrażliwość na hyperparameters.

## P

### Parameter (Parametr)
- **Co ludzie mówią:** "Rozmiar modelu"
- **Co to właściwie oznacza:** Nauczalna wartość w modelu, typowo waga lub bias. "7B parametrów" oznacza 7 miliardów nauczalnych liczb. Każdy float32 zajmuje 4 bajty, więc 7B parametrów = 28GB pamięci tylko na wagi.

### Perplexity
- **Co ludzie mówią:** "Jak niepewny jest model"
- **Co to właściwie oznacza:** Eksponent średniej cross-entropy. Niższy jest lepszy. Perplexity 10 oznacza, że model jest tak niepewny, jakby wybierał równomiernie spośród 10 tokenów w każdym kroku.

### Precision & Recall
- **Co ludzie mówią:** "Metryki dokładności"
- **Co to właściwie oznacza:** Precision = z elementów, które zaznaczyłeś, ile było poprawnych. Recall = ze wszystkich poprawnych elementów, ile znalazłeś. Trade-off: łapanie każdego spam email (wysoki recall) oznacza więcej fałszywych alarmów (niski precision). F1 to ich średnia harmoniczna. Używaj precision, gdy fałszywe pozytywy są kosztowne, recall gdy fałszywe negatywy są kosztowne.

### Prompt Engineering
- **Co ludzie mówią:** "Rozmawianie z AI w dobry sposób"
- **Co to właściwie oznacza:** Projektowanie tekstu wejściowego, żeby wiarygodnie produkować pożądane wyjścia - włącznie z system prompts, few-shot examples, instrukcjami formatu i triggerami chain-of-thought

### Prompt Injection
- **Co ludzie mówią:** "Hackowanie AI słowami"
- **Co to właściwie oznacza:** Atak, gdzie złośliwy tekst w wejściu nadpisuje system prompt lub instrukcje. Direct injection: użytkownik pisze "Ignore previous instructions." Indirect injection: pobrany dokument zawiera ukryte instrukcje. Odpowiednik LLM dla SQL injection. Nie istnieje pełne rozwiązanie - obrona to wielowarstwowa walidacja wejścia, filtrowanie wyjścia i separacja przywilejów.

## Q

### QLoRA
- **Co ludzie mówią:** "LoRA, ale tańsze"
- **Co to właściwie oznacza:** Quantized LoRA. Trzyma zamrożone wagi modelu bazowego w precyzji 4-bitowej (format NF4), trenując LoRA adapters w 16-bitach. Redukuje pamięć o kolejne 3-4x w porównaniu do standardowego LoRA. Model 7B, który potrzebuje 14GB z LoRA, mieści się w 4-6GB z QLoRA. Jakość jest w 1% pełnego fine-tuningu na większości benchmarków.

## R

### RAG (Retrieval-Augmented Generation)
- **Co ludzie mówią:** "AI, które może wyszukiwać"
- **Co to właściwie oznacza:** Wzorzec, gdzie pobierasz relewantne dokumenty z bazy wiedzy (używając podobieństwa embeddingów), wkładasz je do promptu, i pozwalasz LLM odpowiedzieć na podstawie tego kontekstu
- **Dlaczego tak się nazywa:** Retrieval (znajdź dokumenty) + Augmented (dodaj do promptu) + Generation (LLM pisze odpowiedź)

### RLHF (Reinforcement Learning from Human Feedback)
- **Co ludzie mówią:** "Jak sprawiają, że AI jest pomocny"
- **Co to właściwie oznacza:** Pipeline treningowy: (1) zbierz preferencje ludzi na wyjściach modelu, (2) trenuj model nagrody na tych preferencjach, (3) użyj PPO do optymalizacji LLM, żeby produkował wyjścia o wyższej nagrodzie

### Quantization (Kwantyzacja)
- **Co ludzie mówią:** "Zmniejszanie modelu"
- **Co to właściwie oznacza:** Redukowanie precyzji wag modelu z float32 (4 bajty) do int8 (1 bajt) lub int4 (0.5 bajta). Trade-off: mała strata dokładności za 4-8x mniej pamięci i szybsze inference. GPTQ, AWQ i GGUF to popularne formaty.

### ReLU
- **Co ludzie mówią:** "Funkcja aktywacji"
- **Co to właściwie oznacza:** Rectified Linear Unit: f(x) = max(0, x). Najprostsza nieliniowa funkcja aktywacji. Szybka do obliczenia, nie nasyca się dla dodatnich wartości. Używana wszędzie, bo działa i jest tania. Warianty: LeakyReLU, GELU, SiLU.

### ROUGE
- **Co ludzie mówią:** "Metryka dla sumaryzacji"
- **Co to właściwie oznacza:** Recall-Oriented Understudy for Gisting Evaluation. Mierzy overlap między wygenerowanym tekstem a tekstem referencyjnym. ROUGE-1 liczy dopasowania unigramów, ROUGE-2 bigramów, ROUGE-L znajduje najdłuższy wspólny podciąg. Tania do obliczenia, ale mierzy tylko powierzchowne podobieństwo - dwa zdania o tym samym znaczeniu, ale innych słowach, dostają słaby wynik.

## S

### Semantic Search (Wyszukiwanie semantyczne)
- **Co ludzie mówią:** "Mądre wyszukiwanie, które rozumie znaczenie"
- **Co to właściwie oznacza:** Znajdowanie dokumentów przez znaczenie, nie dopasowanie słów kluczowych. Embeduj zapytanie i wszystkie dokumenty w tę samą przestrzeń wektorową, potem zwróć dokumenty, których embeddingi są najbliżej embeddingu zapytania. "payment failed" znajduje "transaction declined" nawet jeśli nie mają wspólnych słów. Napędzane przez modele embedding + wektorowe bazy danych.

### Streaming
- **Co ludzie mówią:** "Widzenie odpowiedzi pojawiającej się słowo po słowie"
- **Co to właściwie oznacza:** LLM wysyła tokeny w miarę generowania, zamiast czekać na pełną odpowiedź. Używa protokołów Server-Sent Events (SSE) lub WebSocket. Redukuje postrzeganą latencję z sekund do milisekund dla pierwszego tokenu. Kluczowe dla produkcyjnych interfejsów czatowych. Każdy chunk zawiera deltę (cząstkowy token lub słowo).

### Self-Attention
- **Co ludzie mówią:** "Jak model decyduje, na czym się skupić"
- **Co to właściwie oznacza:** Każdy token oblicza wektory query, key i value. Waga attention między dwoma tokenami = iloczyn skalarny ich query i key, skalowany i softmaxowany. Wyjście = ważona suma wektorów value. Pozwala każdemu tokenowi zobaczyć każdy inny token.

### SFT (Supervised Fine-Tuning)
- **Co ludzie mówią:** "Uczyć model, jak podążać za instrukcjami"
- **Co to właściwie oznacza:** Fine-tuning pre-trenowanego modelu na parach (instrukcja, odpowiedź). Model uczy się generować odpowiedź, gdy dostaje instrukcję. To jest to, co zmienia base model w chat model.

### Softmax
- **Co ludzie mówią:** "Zamienia liczby w prawdopodobieństwa"
- **Co to właściwie oznacza:** softmax(x_i) = exp(x_i) / sum(exp(x_j)). Transformuje wektor dowolnych liczb rzeczywistych w dystrybucję prawdopodobieństwa (wszystkie dodatnie, suma = 1). Używane w głowicach klasyfikacyjnych, wagach attention i wszędzie, gdzie potrzebujesz prawdopodobieństw.

### Swarm
- **Co ludzie mówią:** "Wiele AI agentów pracujących razem jak pszczoły"
- **Co to właściwie oznacza:** Wielu agentów dzielących stan i koordynujących się przez przekazywanie wiadomości, z emergentnym zachowaniem wynikającym z prostych indywidualnych reguł, nie z centralnej kontroli

## T

### System Prompt
- **Co ludzie mówią:** "Instrukcje AI"
- **Co to właściwie oznacza:** Specjalna wiadomość na początku rozmowy, która ustawia zachowanie modelu, personę i ograniczenia. Przetwarzana przed wiadomościami użytkownika. Niewidoczna dla użytkownika w większości interfejsów. Definiuje, co model powinien i nie powinien robić, jego ton, preferencje formatu i domenowy focus. Inne niż user prompts - system prompts są ustawiane przez developera.

### Tensor
- **Co ludzie mówią:** "Wielowymiarowa tablica"
- **Co to właściwie oznacza:** Podstawowa struktura danych w frameworkach głębokiego uczenia. Tensor 0D to skalar, 1D to wektor, 2D to macierz, 3D+ to tensor. W PyTorch i JAX tensory śledzą historię obliczeń dla automatycznego różniczkowania i mogą żyć na CPU lub GPU. Wszystkie wejścia, wyjścia, wagi i gradienty sieci neuronowych to tensory.

### Token
- **Co ludzie mówią:** "Słowo"
- **Co to właściwie oznacza:** Jednostka subword (typowo 3-4 znaki po angielsku) produkowana przez tokenizer jak BPE. "unbelievable" może być 3 tokenami: "un" + "believ" + "able"

### Temperature
- **Co ludzie mówią:** "Ustawienie kreatywności"
- **Co to właściwie oznacza:** Skalar, który dzieli logity przed softmax. Temperature=1 to domyślna. Wyższa = bardziej płaska dystrybucja = bardziej losowe wyjścia. Niższa = ostrzejsza dystrybucja = bardziej deterministyczne. Temperature=0 to argmax (zawsze wybiera najbardziej prawdopodobny token).

### Transfer Learning
- **Co ludzie mówią:** "Używanie pre-trenowanego modelu"
- **Co to właściwie oznacza:** Branie modelu trenowanego na jednym zadaniu i adaptowanie go do innego. Wczesne warstwy uczą się ogólnych cech (krawędzie, wzorce składniowe), które się transferują. Tylko późniejsze warstwy potrzebują treningu specyficznego dla zadania. Dlatego możesz fine-tune'ować BERT dla dowolnego zadania NLP.

### Transformer
- **Co ludzie mówią:** "Architektura za nowoczesnym AI"
- **Co to właściwie oznacza:** Architektura sieci neuronowej, która przetwarza sekwencje używając self-attention (pozwalając każdej pozycji Attendować do każdej innej pozycji) zamiast rekurencji, umożliwiając masową równoległość
- **Dlaczego tak się nazywa:** Przekształca reprezentacje wejściowe w reprezentacje wyjściowe przez warstwy attention

## U

### Underfitting
- **Co ludzie mówią:** "Model się nie uczy"
- **Co to właściwie oznacza:** Model jest zbyt prosty, żeby uchwycić wzorce w danych. Strata treningowa pozostaje wysoka. Napraw przez: więcej parametrów, więcej warstw, dłuższy trening, mniejszą regularizację, lepsze cechy.

## V

### VAE (Variational Autoencoder)
- **Co ludzie mówią:** "Generatywny model"
- **Co to właściwie oznacza:** Autoenkoder, który uczy się gładkiej przestrzeni latentnej, wymuszając, żeby wyjście enkodera podążało za rozkładem Gaussa. Możesz próbkować z tego rozkładu i dekodować, żeby generować nowe dane. Reparametryzacja trick sprawia, że jest trenowalny przez backpropagation.

### Vector Database (Wektorowa baza danych)
- **Co ludzie mówią:** "Specjalna baza danych dla AI"
- **Co to właściwie oznacza:** Baza danych zoptymalizowana do przechowywania wektorów (gęstych tablic floatów) i wykonywania szybkiego przybliżonego wyszukiwania najbliższego sąsiada. Podstawowa operacja w wyszukiwaniu podobieństw, RAG i systemach rekomendacji.

## W

### Weight (Waga)
- **Co ludzie mówią:** "To, czego model się nauczył"
- **Co to właściwie oznacza:** Pojedyncza liczba w macierzy parametrów modelu. Warstwa liniowa z rozmiarem wejścia 768 i wyjścia 3072 ma 768*3072 = 2,359,296 wag. Trening dostosowuje każdą wagę, żeby zminimalizować funkcję straty.

### Weight Decay
- **Co ludzie mówią:** "Regularizacja"
- **Co to właściwie oznacza:** Dodawanie kary proporcjonalnej do magnitudy wag do funkcji straty. Równoważne L2 regularizacji. Zapobiega wagom rosnącym zbyt dużo. Typowa wartość: 0.01-0.1.

## Z

### Zero-Shot
- **Co ludzie mówią:** "Bez treningu"
- **Co to właściwie oznacza:** Używanie modelu na zadaniu, na którym nie był explicitnie trenowany, bez żadnych przykładów specyficznych dla zadania w prompcie. Model generalizuje z pre-treningu. Działa, bo duże modele widziały wystarczająco dużo różnorodności, żeby radzić sobie z nowymi formatami zadań.
