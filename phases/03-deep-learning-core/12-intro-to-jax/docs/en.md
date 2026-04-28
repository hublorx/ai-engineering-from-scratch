# Wprowadzenie do JAX

> PyTorch mutuje tensory. TensorFlow buduje grafy. JAX kompiluje czyste funkcje. To ostatnie zmienia sposób myślenia o deep learningu.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Lekcje 01-10 z Fazy 03, podstawy NumPy
**Szacowany czas:** ~90 minut

## Cele uczenia się

- Pisać kod sieci neuronowych oparty na czystych funkcjach, używając funkcjonalnego API JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Wyjaśnić kluczową różnicę projektową między zachłannym mutowaniem w PyTorch a funkcyjnym modelem kompilacji w JAX
- Zastosować kompilację JIT i wektoryzację vmap w celu przyspieszenia pętli treningowych w porównaniu z naiwnym Pythonem
- Trenować prostą sieć w JAX i porównać jawną obsługę stanu z podejściem obiektowym PyTorcha

## Problem

Wiesz już, jak budować sieci neuronowe w PyTorch. Definiujesz `nn.Module`, wywołujesz `.backward()`, robisz krok optimizera. To działa. Miliony ludzi tego używają.

Ale PyTorch ma ograniczenie wbudowane w swoje DNA: śledzi operacje zachłannie, jedna po drugiej, w Pythonie. Każdy `tensor + tensor` to osobne uruchomienie kernela. Każdy krok treningowy ponownie interpretuje ten sam kod Pythona. To działa dobrze, dopóki nie musisz trenować modelu z 540 miliardami parametrów na 2048 TPU. Wtedy narzut cię zabija.

Google DeepMind trenuje Gemini na JAX. Anthropic trenowało Claude'a na JAX. To nie są małe operacje -- to największe treningi sieci neuronowych na Ziemi. Wybrali JAX, bo traktuje on twoją pętlę treningową jako program możliwy do skompilowania, nie sekwencję wywołań Pythona.

JAX to NumPy z trzema supermocami: automatyczna dyferencjacja, kompilacja JIT do XLA i automatyczna wektoryzacja. Piszesz funkcję, która przetwarza jeden przykład. JAX daje ci funkcję, która przetwarza batch, oblicza gradienty, kompiluje do kodu maszynowego i uruchamia na wielu urządzeniach. Wszystko bez zmiany oryginalnej funkcji.

## Koncepcja

### Filozofia JAX

JAX to framework funkcyjny. Bez klas, bez mutable state, bez metody `.backward()`. Zamiast tego:

| PyTorch | JAX |
|---------|-----|
| Klasa `nn.Module` ze stanem | Czysta funkcja: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | Kompilacja JIT przez XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-wektoryzacja |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-równoległość |
| Mutable `model.parameters()` | Niezmienny pytree tablic |

To nie jest preferencja stylistyczna. To ograniczenie kompilatora. Kompilacja JIT wymaga czystych funkcji -- te same dane wejściowe zawsze dają te same dane wyjściowe, bez efektów ubocznych. To ograniczenie jest tym, co umożliwia 100-krotne przyspieszenia.

### jax.numpy: Znana powierzchnia

JAX reimplementuje API NumPy na akceleratorach:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Te same nazwy funkcji. Te same reguły broadcastingu. Te same semantyki slice'owania. Ale tablice żyją na GPU/TPU, a każda operacja jest śledzalna przez kompilator.

Jedna krytyczna różnica: tablice JAX są niezmienne. Nie `a[0] = 5`. Zamiсто: `a = a.at[0].set(5)`. Przez tydzień wydaje się to niezgrabne, potem robi się jasne -- niezmienność jest tym, co czyni transformacje jak `grad`, `jit` i `vmap` komponowalnymi.

### jax.grad: Funkcyjna autodiff

PyTorch dołącza gradienty do tensorów (`.grad`). JAX dołącza gradienty do funkcji.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` pobiera funkcję i zwraca nową funkcję, która oblicza gradient. Nie ma wywołania `.backward()`. Nie ma grafu obliczeń przechowywanego na tensorach. Gradient to just kolejna funkcja, którą możesz wywołać, skomponować lub skompilować JIT.

To komponuje się arbitralnie:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Drugie pochodne. Trzecie pochodne. Jakobiany. Hesjany. Wszystko przez komponowanie `grad`. PyTorch też potrafi to zrobić (`torch.autograd.functional.hessian`), ale jest doklejone. W JAX to jest fundament.

Ograniczenie: `grad` działa tylko z czystymi funkcjami. Żadnych instrukcji print w środku (uruchamiają się podczas trace'owania, nie wykonania). Żadnej mutacji zewnętrznego stanu. Żadnej generacji liczb losowych bez jawnego zarządzania kluczami.

### jit: Kompilacja do XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Przy pierwszym wywołaniu JAX trace'uje funkcję -- rejestruje, które operacje zachodzą, bez ich wykonywania. Następnie przekazuje ten trace do XLA (Accelerated Linear Algebra), kompilatora Google'a dla TPU i GPU. XLA łączy operacje, eliminuje redundantne kopie pamięci i generuje zoptymalizowany kod maszynowy.

Kolejne wywołania omijają Pythona całkowicie. Skompilowany kod działa na akceleratorze z prędkością C++.

Kiedy JIT pomaga:

- Kroki treningowe (te same obliczenia powtarzane tysiące razy)
- Inferencja (ten sam model, różne dane wejściowe)
- Każda funkcja wywoływana więcej niż raz z podobnie kształtowymi danymi wejściowymi

Kiedy JIT przeszkadza:

- Funkcje z kontrolą przepływu Pythona, która zależy od wartości (`if x > 0` gdzie x jest trace'owaną tablicą)
- Obliczenia jednorazowe (narzut kompilacji przewyższa czas wykonania)
- Debugowanie (trace'owanie ukrywa rzeczywiste wykonanie)

Ograniczenie kontroli przepływu jest realne. `jax.lax.cond` zastępuje `if/else`. `jax.lax.scan` zastępuje pętle `for`. To nie są opcjonalne -- to jest cena kompilacji.

### vmap: Automatyczna wektoryzacja

Piszesz funkcję, która przetwarza jeden przykład:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` podnosi ją do przetwarzania batcha:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` oznacza: nie batchuj nad `params` (współdzielone), batchuj nad osią 0 `x`. Żaden manualny `for` loop. Żadne reshape'owanie. Żadne przeplatanie wymiaru batcha. JAX sam ustala wymiar batcha i wektoryzuje całe obliczenia.

To nie jest syntactic sugar. `vmap` generuje połączony wektoryzowany kod, który działa 10-100x szybciej niż pętla Pythona. I komponuje się z `jit` i `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Per-example gradients. Jedna linia. To jest prawie niemożliwe w PyTorch bez haków.

### pmap: Równoległość danych między urządzeniami

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` replikuje funkcję na wszystkich dostępnych urządzeniach (GPU/TPU) i dzieli batch. Wewnątrz funkcji `jax.lax.pmean` i `jax.lax.psum` synchronizują gradienty między urządzeniami.

Google trenuje Gemini na tysiącach chipów TPU v5e używając `pmap` (i jego następcę `shard_map`). Model programowania: napisz wersję dla jednego urządzenia, owiń z `pmap`, gotowe.

### Pytree: Uniwersalna struktura danych

JAX operuje na "pytree" -- zagnieżdżonych kombinacjach list, tuple'ów, dict'ów i tablic. Parametry twojego modelu to pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Każda transformacja JAX -- `grad`, `jit`, `vmap` -- wie, jak przechodzić przez pytree. `jax.tree.map(f, tree)` stosuje `f` do każdego liścia. Tak optimizery aktualizują wszystkie parametry na raz:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Nie ma metody `.parameters()`. Nie ma rejestracji parametrów. Struktura drzewa to model.

### Funkcyjny vs obiektowy

PyTorch przechowuje stan wewnątrz obiektów:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX używa czystych funkcji z jawnym stanem:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Params są przekazywane. Nic nie jest przechowywane. Nic nie jest mutowane. To czyni każdą funkcję testowalną, komponowalną i kompilowalną. To też oznacza, że sam zarządzasz paramsami -- albo używasz biblioteki jak Flax lub Equinox.

### Ekosystem JAX

JAX daje ci prymitywy. Biblioteki dają ergonomikę:

| Biblioteka | Rola | Styl |
|---------|------|-------|
| **Flax** (Google) | Warstwy sieci neuronowych | `nn.Module` z jawnym stanem |
| **Equinox** (Patrick Kidger) | Warstwy sieci neuronowych | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optymizery + harmonogramy LR | Komponowalne transformacje gradientu |
| **Orbax** (Google) | Checkpointing | Zapisywanie/przywracanie pytree |
| **CLU** (Google) | Metryki + logging | Narzędzia pętli treningowej |

Optax to standardowa biblioteka optymizacyjna. Oddziela transformację gradientu (Adam, SGD, clipping) od aktualizacji parametrów, co czyni komponowanie trywialnym:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Kiedy używać JAX vs PyTorch

| Czynnik | JAX | PyTorch |
|--------|-----|---------|
| Wsparcie TPU | Pierwszoklasowe (Google zbudował oba) | Utrzymywane przez społeczność (torch_xla) |
| Wsparcie GPU | Dobre (CUDA przez XLA) | Najlepsze w klasie (natywny CUDA) |
| Debugowanie | Trudne (trace'owanie + kompilacja) | Łatwe (eager, linia po linii) |
| Ekosystem | Skupiony na badaniach (Flax, Equinox) | Ogromny (HuggingFace, torchvision, itp.) |
| Rekrutacja | Niszowy (Google/DeepMind/Anthropic) | Główny nurt (wszędzie) |
| Trening na dużą skalę | Superior (XLA, pmap, mesh) | Dobry (FSDP, DeepSpeed) |
| Szybkość prototypowania | Wolniejsze (nakład funkcyjny) | Szybsze (mutuj i jedź) |
| Inferencja produkcyjna | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Kto tego używa | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

Szczera odpowiedź: używaj PyTorcha, chyba że masz konkretny powód, żeby używać JAX. Te powody to -- dostęp do TPU, potrzeba per-example gradients, trening na wielu urządzeniach na masową skalę, albo praca w Google/DeepMind/Anthropic.

### Liczby losowe w JAX

JAX nie ma globalnego stanu losowego. Każda operacja losowa wymaga jawnego klucza PRNG:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

To irytuje na początku. Ale gwarantuje odtwarzalność między urządzeniami i kompilacjami -- właściwość, której `torch.manual_seed` w PyTorch nie może zagwarantować w ustawieniach multi-GPU.

## Zbuduj to

### Krok 1: Setup i dane

Będziemy trenować 3-warstwowy MLP na MNIST używając JAX i Optax. 784 wejścia, dwie warstwy ukryte 256 i 128 neuronów, 10 klas wyjściowych.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Krok 2: Inicjalizacja parametrów

Nie ma klasy. Just funkcja, która zwraca pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He-initialization, zrobione manualnie. Trzy klucze PRNG split z jednego seeda. Każda waga to niezmienna tablica w zagnieżdżonym dicie.

### Krok 3: Forward pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Czyste funkcje. Params w, predykcja out. Żadnego `self`, żadnego przechowywanego stanu. `loss_fn` oblicza cross-entropy od zera -- softmax, log, ujemna średnia.

### Krok 4: Kroki treningowe skompilowane JIT

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` zwraca i wartość loss i gradienty w jednym przejściu. Dekorator `@jax.jit` kompiluje obie funkcje do XLA. Po pierwszym wywołaniu każdy krok treningowy działa bez dotykania Pythona.

### Krok 5: Pętla treningowa

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epok. ~97% test accuracy. Pierwsza epoka jest wolna (kompilacja JIT). Epoki 2-10 są szybkie.

Zauważ, czego brakuje: nie ma `.zero_grad()`, nie ma `.backward()`, nie ma `.step()`. Cała aktualizacja to jedno złożone wywołanie funkcji. Gradienty są obliczane, transformowane przez Adama i aplikowane do parametrów -- wszystko inside `train_step`.

## Użyj tego

### Flax: Standard Google'a

Flax to najczęstsza biblioteka sieci neuronowych JAX. Dodaje z powrotem `nn.Module`, ale z jawnym zarządzaniem stanem:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Ta sama struktura co PyTorch, ale `params` jest oddzielone od modelu. `model.init()` tworzy params. `model.apply(params, x)` uruchamia forward pass. Obiekt modelu nie ma stanu.

### Equinox: Pythonic Alternative

Equinox (autorstwa Patricka Kidgera) reprezentuje modele jako pytree:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Sam model jest pytree. Nie trzeba `.apply()`. Parametry to just liście modelu. To jest bliższe temu, jak JAX myśli.

### Optax: Komponowalne optymizery

Optax oddziela transformację gradientu od aktualizacji:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping, learning rate warmup, weight decay -- wszystko skomponowane jako łańcuch transformacji. Każda transformacja widzi gradienty, modyfikuje je i przekazuje do następnej. Żadnego monolitycznego obiektu optimizera.

## Wyślij to

**Instalacja:**

```bash
pip install jax jaxlib optax flax
```

Dla wsparcia GPU:

```bash
pip install jax[cuda12]
```

Dla TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Pułapki wydajnościowe:**

- Pierwsze wywołanie JIT jest wolne (kompilacja). Rozgrzej przed benchmarkowaniem.
- Unikaj pętli Pythona nad tablicami JAX inside JIT. Używaj `jax.lax.scan` lub `jax.lax.fori_loop`.
- `jax.debug.print()` działa inside JIT. Zwykły `print()` nie.
- Profiluj z `jax.profiler` lub TensorBoard. Kompilacja XLA może ukrywać bottlenecki.
- JAX domyślnie alokuje 75% pamięci GPU. Ustaw `XLA_PYTHON_CLIENT_PREALLOCATE=false` żeby wyłączyć.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**Ta lekcja wytwarza:**

- `outputs/prompt-jax-optimizer.md` -- prompt do wyboru właściwej konfiguracji optimizera JAX
- `outputs/skill-jax-patterns.md` -- skill obejmujący funkcyjne wzorce w JAX

## Ćwiczenia

1. Dodaj dropout do MLP. W JAX dropout wymaga klucza PRNG -- przeplataj klucz przez forward pass i splituj go dla każdej warstwy dropout. Porównaj test accuracy z i bez.

2. Użyj `jax.vmap` do obliczenia per-example gradients dla batcha 32 obrazków MNIST. Oblicz normę gradientu dla każdego przykładu. Które przykłady mają największe gradienty i dlaczego?

3. Zastąp manualną funkcję forward generyczną `mlp_forward(params, x)`, która działa dla dowolnej liczby warstw. Użyj `jax.tree.leaves` żeby automatycznie określić głębokość.

4. Zbenchmarkuj krok treningowy z i bez `@jax.jit`. Zmierz czas 100 kroków każdego. Jak duże jest przyspieszenie na twoim sprzęcie? Jaki jest narzut kompilacji na pierwszym wywołaniu?

5. Zaimplementuj gradient clipping przez skomponowanie `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Trenuj z i bez clippingu. Wykreśl normę gradientu podczas treningu, żeby zobaczyć efekt.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|----------------|----------------------|
| XLA | "Ta rzecz, która robi JAX szybkim" | Accelerated Linear Algebra -- kompilator, który łączy operacje i generuje zoptymalizowane kernele GPU/TPU z grafu obliczeń |
| JIT | "Just-in-time compilation" | JAX trace'uje funkcję przy pierwszym wywołaniu, kompiluje do XLA, potem uruchamia skompilowaną wersję przy kolejnych wywołaniach |
| Pure function | "Bez efektów ubocznych" | Funkcja, gdzie wyjście zależy tylko od wejść -- żaden globalny stan, żadna mutacja, żadna losowość bez jawnego klucza |
| vmap | "Auto-batching" | Transformuje funkcję, która przetwarza jeden przykład, w taką, która przetwarza batch, bez przepisywania |
| pmap | "Auto-równoległość" | Replikuje funkcję na wielu urządzeniach i dzieli wejściowy batch |
| Pytree | "Zagnieżdżony dict tablic" | Każda zagnieżdżona struktura list, tuple'ów, dict'ów i tablic, którą JAX może przechodzić i transformować |
| Tracing | "Nagrywanie obliczeń" | JAX wykonuje funkcję z abstrakcyjnymi wartościami, żeby zbudować graf obliczeń, bez obliczania realnych wyników |
| Functional autodiff | "grad funkcji" | Obliczanie pochodnych przez transformowanie funkcji, nie przez dołączanie magazynu gradientów do tensorów |
| Optax | "Biblioteka optymizera JAX" | Komponowalna biblioteka transformacji gradientu -- Adam, SGD, clipping, scheduling -- które łączą się w łańcuchy |
| Flax | "nn.Module JAX" | Biblioteka sieci neuronowych Google dla JAX, dodająca abstrakcje warstw przy zachowaniu jawnego stanu |

## Dalsza lektura

- Dokumentacja JAX: https://jax.readthedocs.io/ -- oficjalne docs, z doskonałymi tutorialami o grad, jit i vmap
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- oryginalny artykuł wyjaśniający filozofię projektową
- Dokumentacja Flax: https://flax.readthedocs.io/ -- biblioteka sieci neuronowych Google dla JAX
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- pythonic alternativa do Flax
- DeepMind, "Optax: composable gradient transformation and optimisation" -- standardowa biblioteka optymizacyjna
- "You Don't Know JAX" (Colin Raffel, 2020) -- praktyczny przewodnik po JAX gotchas i wzorcach, od jednego z autorów T5