# Zarzadzanie danymi

> Dane sa paliwem. To, jak nimi zarzadzasz, okresla, jak szybko jedziesz.

**Typ:** Budowanie
**Jezyk:** Python
**Wymagania wstepne:** Faza 0, Lekcja 01
**Czas:** ~45 minut

## Cele uczenia sie

- Ladowanie, strumieniowanie i cacheowanie zbiorow danych przy uzyciu biblioteki `datasets` Hugging Face
- Konwersja miedzy formatami CSV, JSON, Parquet i Arrow oraz wyjasnienie ich kompromisow
- Tworzenie powtarzalnych podzialow train/validation/test z ustalonymi ziarnami losowymi
- Zarzadzanie duzymi plikami modeli i zbiorow danych przy uzyciu `.gitignore`, Git LFS lub DVC

## Problem

Kazdy projekt AI zaczyna sie od danych. Musisz znalezc zbiory danych, pobrac je, konwertowac miedzy formatami, dzielic je do treningu i ewaluacji oraz wersjonowac, zeby eksperymenty byly powtarzalne. Robienie tego recznie za kazdym razem jest wolne i podatne na bledy. Potrzebujesz powtarzalnego workflow.

## Koncepcja

Biblioteka `datasets` Hugging Face to standardowy sposob ladowania danych do pracy z AI. Obsluguje pobieranie, cacheowanie, konwersje formatow i strumieniowanie out of the box.

## Zbuduj to

### Krok 1: Zainstaluj biblioteke datasets

```bash
pip install datasets huggingface_hub
```

### Krok 2: Zaladuj zbior danych

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
print(dataset["train"][0])
```

To pobiera zbior danych recenzji filmowych IMDB. Po pierwszym pobraniu laduje z cache'a znajdujacego sie w `~/.cache/huggingface/datasets/`.

### Krok 3: Strumieniuj duze zbiory danych

Niektore zbiory danych sa zbyt duze, zeby zmiescic sie na dysku. Strumieniowanie laduje je wiersz po wierszu bez pobierania calosci.

```python
dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break
```

Strumieniowanie daje ci `IterableDataset`. Przetwarzasz wiersze w miare ich naplywania. Uzycie pamieci pozostaje stale niezaleznie od rozmiaru zbioru danych.

### Krok 4: Formaty zbiorow danych

Biblioteka `datasets` uzywa Apache Arrow pod spodem. Mozesz konwertowac do innych formatow w zaleznosci od tego, czego potrzebuje twoj pipeline.

```python
dataset = load_dataset("imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")
```

Porownanie formatow:

| Format | Rozmiar | Szybkosc odczytu | Najlepsze dla |
|--------|---------|------------------|---------------|
| CSV | Duzy | Wolny | Czytelnosc dla czlowieka, arkusze kalkulacyjne |
| JSON | Duzy | Wolny | API, dane zagniezdzone |
| Parquet | Maly | Szybki | Analityka, zapytania kolumnowe |
| Arrow | Maly | Najszybszy | Przetwarzanie w pamieci (czego `datasets` uzywa wnetrznie) |

Do pracy z AI, Parquet jest najlepszym formatem storage. Arrow jest tym, z czym pracujesz w pamieci. CSV i JSON sa do wymiany.

### Krok 5: Podzialy danych

KAZdy projekt ML potrzebuje trzech podzialow:

- **Train**: Z tego uczy sie model (typowo 80%)
- **Validation**: Tutaj sprawdzasz postepy podczas treningu (typowo 10%)
- **Test**: Ewaluacja koncowa po zakonczeniu treningu (typowo 10%)

Niektore zbiory danych sa juz podzielone. Gdy nie sa, podziel je sam:

```python
dataset = load_dataset("imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

Zawsze ustawiaj seed dla powtarzalnosci. Ten sam seed produkuje ten sam podzial za kazdym razem.

### Krok 6: Pobieranie i cacheowanie modeli

Modele to duze pliki. Biblioteka `huggingface_hub` obsluguje pobieranie i cacheowanie.

```python
from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Cache'owany w: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Pelny model w: {model_dir}")
```

Modele cache'uja sie do `~/.cache/huggingface/hub/`. Po pobraniu laduja sie natychmiast przy kolejnych uruchomieniach.

### Krok 7: Obsluga duzych plikow

Wagi modeli i duze zbiory danych nie powinny trafic do git. Trzy opcje:

**Opcja A: .gitignore (najprostsza)**

```
*.bin
*.safetensors
*.pt
*.onnx
data/*.parquet
data/*.csv
models/
```

**Opcja B: Git LFS (sledz duze pliki w git)**

```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

Git LFS przechowuje wskazniki w twoim repo, a prawdziwe pliki na osobnym serwerze. GitHub daje ci 1 GB za darmo.

**Opcja C: DVC (kontrola wersji danych)**

```bash
pip install dvc
dvc init
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"
```

DVC tworzy male pliki `.dvc`, ktore wskazuja na twoje dane. Same dane zyja w S3, GCS lub innym zdalnym storage backend.

| Podejscie | Zlozonosc | Najlepsze dla |
|-----------|-----------|---------------|
| .gitignore | Niska | Osobiste projekty, pobrane dane, ktore mozesz ponownie pobrac |
| Git LFS | Srednia | Zespoly dzielace wagi modeli przez git |
| DVC | Wysoka | Powtarzalne eksperymenty, duze zbiory danych, zespoly |

Dla tego kursu, `.gitignore` wystarczy. Uzywaj DVC, gdy potrzebujesz odtwarzac dokladnie te same eksperymenty na roznych maszynach.

### Krok 8: Wzorce przechowywania

**Local storage** sprawdza sie dla zbiorow danych do ~10 GB. HF cache obsluguje to automatycznie.

**Cloud storage** jest dla czegokolwiek wiekszego lub wspoldzielonego miedzy maszynami:

```python
import os

local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"
```

DVC integruje sie z S3 i GCS bezposrednio:

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

Dla tego kursu, local storage wystarczy. Cloud storage staje sie istotny, gdy fine-tunujesz na zdalnych instancjach GPU.

## Zbiory danych uzywane w tym kursie

| Zbior danych | Lekcje | Rozmiar | Czego sie uczysz |
|--------------|--------|---------|------------------|
| IMDB | Tokenizacja, klasyfikacja | 84 MB | Podstawy klasyfikacji tekstu |
| WikiText | Modelowanie jezyka | 181 MB | Predykcja nastepnego tokenu |
| SQuAD | Systemy QA | 35 MB | Question answering, zakresy |
| Common Crawl (podzbior) | Embeddings | Zmienny | Przetwarzanie tekstu w duzej skali |
| MNIST | Podstawy wizji | 21 MB | Podstawy klasyfikacji obrazow |
| COCO (podzbior) | Multimodal | Zmienny | Pary obraz-tekst |

Nie musisz pobierac wszystkich teraz. Kazda lekcja okresla, czego potrzebuje.

## Uzyj tego

Uruchom skrypt pomocniczy, zeby zweryfikowac, ze wszystko dziala:

```bash
python code/data_utils.py
```

To pobiera maly zbior danych, konwertuje go, dzieli i drukuje podsumowanie.

## Wyslij to

Ta lekcja tworzy:
- `code/data_utils.py` - narzedzie do wielokrotnego uzycia do ladowania i cacheowania danych
- `outputs/prompt-data-helper.md` - prompt do znajdowania wlasciwego zbioru danych dla zadania

## CWiczenia

1. Zaladuj zbior danych `glue` z konfiguracja `mrpc` i sprawdz pierwsze 5 przykladow
2. Strumieniuj zbior danych `c4` i policz, ile przykladow mozesz przetworzyc w 10 sekund
3. Konwertuj zbior danych do Parquet i porownaj rozmiar pliku z CSV
4. Utworz podzial train/val/test w proporcji 70/15/15 z ustalonym seed i zweryfikuj rozmiary

## Kluczowe terminy

| Termin | Co ludzie mowia | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| Dataset split | "Training data" | Nazwany podzbior (train/val/test) uzywany na roznych etapach cyklu zycia ML |
| Streaming | "Load it lazily" | Przetwarzanie danych wiersz po wierszu ze zrodla zdalnego bez pobierania calego zbioru |
| Parquet | "Skompresowany CSV" | Kolumnowy format plikow zoptymalizowany pod katem zapytan analitycznych i efektywnosci storage |
| Arrow | "Szybki dataframe" | Format kolumnowy w pamieci uzywany wewnetrznie przez biblioteke datasets do odczytow bez kopiowania |
| Git LFS | "Git dla duzych plikow" | Rozszerzenie, ktore przechowuje duze pliki poza repo git, zachowujac wskazniki w wersjonowaniu |
| DVC | "Git dla danych" | System kontroli wersji dla zbiorow danych i modeli, ktory integruje sie z cloud storage |
| Cache | "Already downloaded" | Lokalna kopia wczesniej pobranych danych, domyslnie przechowywana w ~/.cache/huggingface/ |
