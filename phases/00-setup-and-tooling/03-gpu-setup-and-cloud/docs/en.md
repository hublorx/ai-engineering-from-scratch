# Konfiguracja GPU i chmura

> Trenowanie na CPU jest OK dla nauki. Trenowanie na poważnie wymaga GPU.

**Typ:** Budowa
**Języki:** Python
**Wymagania wstępne:** Phase 0, Lesson 01
**Czas:** ~45 minut

## Cele uczenia się

- Zweryfikuj dostępność lokalnego GPU używając `nvidia-smi` i CUDA API PyTorcha
- Skonfiguruj Google Colab z GPU T4 dla darmowych eksperymentów w chmurze
- Porównaj mnożenie macierzy na CPU vs GPU i zmierz przyspieszenie
- Oszacuj największy model który zmieści się w VRAM używając reguły fp16

## Problem

Większość lekcji w fazach 1-3 działa na CPU. Ale gdy zaczynasz trenować CNN, transformery lub LLM (fazy 4+), potrzebujesz przyspieszenia GPU. Trening który na CPU trwa 8 godzin, na GPU trwa 10 minut.

Masz trzy opcje: lokalne GPU, chmurowe GPU lub Google Colab (darmowe).

## Koncepcja

```
Twoje opcje:

1. Lokalne GPU NVIDIA
   Koszt: $0 (już je masz)
   Konfiguracja: Zainstaluj CUDA + cuDNN
   Najlepsze dla: Regularnego użytku, dużych zbiorów danych

2. Google Colab (darmowy tier)
   Koszt: $0
   Konfiguracja: Żadna
   Najlepsze dla: Szybkich eksperymentów, brak GPU w domu

3. Chmurowe GPU (Lambda, RunPod, Vast.ai)
   Koszt: $0.20-2.00/godz
   Konfiguracja: SSH + instalacja
   Najlepsze dla: Poważnego treningu, dużych modeli
```

## Zbuduj to

### Opcja 1: Lokalne GPU NVIDIA

Sprawdź czy je masz:

```bash
nvidia-smi
```

Zainstaluj PyTorch z CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### Opcja 2: Google Colab

1. Przejdź do [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Uruchom `!nvidia-smi` żeby zweryfikować

Wgraj notatniki z tego kursu bezpośrednio do Colab.

### Opcja 3: Chmurowe GPU

Dla Lambda Labs, RunPod lub Vast.ai:

```bash
ssh user@twoja-gpu-instancja

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Bez GPU? To nie problem.

Większość lekcji działa na CPU. Te które wymagają GPU to powiedzą i dołączą linki do Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Zbuduj: Benchmark GPU vs CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Ćwiczenia

1. Uruchom benchmark powyżej i porównaj czasy CPU vs GPU
2. Jeśli nie masz GPU, uruchom w Google Colab i porównaj
3. Sprawdź ile masz pamięci GPU i oszacuj największy model który możesz załadować (reguła: 2 bajty na parametr dla fp16)

## Kluczowe pojęcia

| Termin | Co ludzie mówią | Co to naprawdę oznacza |
|--------|-----------------|----------------------|
| CUDA | "Programowanie na GPU" | Platforma obliczeń równoległych NVIDIA która pozwala uruchamiać kod na GPU |
| VRAM | "Pamięć GPU" | Pamięć wideo na GPU, oddzielna od RAM systemu. Ogranicza rozmiar modelu. |
| fp16 | "Pół precyzji" | 16-bitowy floating point, zużywa połowę pamięci fp32 przy minimalnej stracie dokładności |
| Tensor Core | "Szybki sprzęt macierzowy" | Wyspecjalizowane rdzenie GPU do mnożenia macierzy, 4-8x szybsze od zwykłych rdzeni |
