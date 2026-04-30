# Generowanie muzyki — MusicGen, Stable Audio, Suno i trzęsienie ziemi w licencjonowaniu

> Generowanie muzyki w 2026: Suno v5 i Udio v4 dominują w komercyjnym zastosowaniu; MusicGen, Stable Audio Open i ACE-Step prowadzą w open-source. Problem techniczny jest w dużej mierze rozwiązany. Problem prawny (ugoda Warner Music za 500 mln $, ugoda z UMG) zmienił oblicze dziedziny w latach 2025-2026.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 6 · 02 (Spektrogramy), Faza 4 · 10 (Modele dyfuzyjne)
**Czas:** ~75 minut

## Problem

Tekst → klip muzyczny o długości od 30 sekund do 4 minut, z tekstami, wokalem i strukturą. Trzy pod-problemy:

1. **Generowanie instrumentalne.** Tekst jak „lo-fi hip-hop, bębny z ciepłymi klawiszami" → audio. MusicGen, Stable Audio, AudioLDM.
2. **Generowanie piosenek (z wokalem i tekstami).** „Piosenka country o deszczowych nocach w Teksasie" → pełna piosenka. Suno, Udio, YuE, ACE-Step.
3. **Warunkowe / kontrolowalne.** Przedłuż istniejący klip, wygeneruj ponownie most, zmień gatunek, oddziel ścieżki lub wykonaj inpainting. Inpainting Udio + separacja stemów to funkcja 2026 do której dążą inni.

## Koncepcja

![Generowanie muzyki: token-LM vs dyfuzja, mapa modeli 2026](../assets/music-generation.svg)

### Token LM nad tokenami neuralnego kodeka

**MusicGen** Meta (2023, MIT) i wiele pochodnych: warunkowanie tekstem/melodią, autoregresyjna predykcja tokenów EnCodec (32 kHz, 4 codebooki), dekodowanie przez EnCodec. 300M - 3,3B parametrów. Silny baseline; problemy z generowaniem powyżej 30 sekund.

**ACE-Step** (open-source, 4B XL wydany w kwietniu 2026) rozszerza to dla pełno-piosenkowej generacji z warunkowaniem tekstowym. Najbliższa rzecz Suno dla społeczności open-source.

### Dyfuzja nad mel lub latentami

**Stable Audio (2023)** i **Stable Audio Open (2024)**: latent diffusion na skompresowanym audio. Wyróżnia się w pętlach, sound designie, teksturach ambient. Nie najlepsze w ustrukturyzowanych pełnych piosenkach.

**AudioLDM / AudioLDM2**: text-to-audio przez T2I-style latent diffusion, uogólnione na muzykę, efekty dźwiękowe, mowę.

### Hybryda (produkcyjna) — Suno, Udio, Lyria

Zamknięte wagi. Prawdopodobnie AR codec LM + dyfuzyjny wokoder ze specjalizowanymi głowami voice/drum/melody. Suno v5 (2026) to lider jakości z ELO 1293. Udio v4 dodaje inpainting + separację stemów (bas, bębny, wokal jako osobne pliki do pobrania).

### Ewaluacja

- **FAD (Fréchet Audio Distance).** Odległość embeddingów między generowanym a prawdziwym rozkładem audio przy użyciu cech VGGish lub PANNs. Niższy wynik jest lepszy. MusicGen small: 4.5 FAD na MusicCaps; SOTA ~3.0.
- **Muzykalność (subiektywna).** Preferencje ludzkie. Suno v5 ELO 1293 prowadzi.
- **Wyrównanie tekst-audio.** Wynik CLAP między promptem a wynikiem.
- **Artefakty muzyczne.** Przejścia off-beat, dryf frazy wokalnej, utrata struktury po 30 s.

## Mapa modeli 2026

| Model | Parametry | Długość | Wokale | Licencja |
|-------|-----------|---------|--------|----------|
| MusicGen-large | 3,3B | 30 s | nie | MIT |
| Stable Audio Open | 1,2B | 47 s | nie | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | > 2 min | tak | Apache-2.0 |
| YuE | 7B | > 2 min | tak, wielojęzyczne | Apache-2.0 |
| Suno v5 (zamknięty) | ? | 4 min | tak, ELO 1293 | komercyjny |
| Udio v4 (zamknięty) | ? | 4 min | tak + stems | komercyjny |
| Google Lyria 3 (zamknięty) | ? | real-time | tak | komercyjny |
| MiniMax Music 2.5 | ? | 4 min | tak | komercyjny API |

## Krajobraz prawny (2025-2026)

- **Ugoda Warner Music vs Suno.** 500 mln $. WMG ma teraz nadzór nad podobieństwem do AI, prawami do muzyki i utworami generowanymi przez użytkowników na Suno. Podobna ugoda UMG z Udio.
- **EU AI Act** + **California SB 942**: Muzyka generowana przez AI musi być ujawniona.
- **Riffusion / MusicGen** na licencji MIT nie mają obciążeń compliance, ale też nie mają komercyjnego wokalu.

Bezpieczne do wdrożenia wzorce:

1. Generuj tylko instrumental (MusicGen, Stable Audio Open, wyjścia MIT/CC0).
2. Używaj komercyjnych API (Suno, Udio, ElevenLabs Music) z licencją per-generacja.
3. Trenuj na własnym lub licencjonowanym katalogu (większość przedsiębiorstw tu trafia).
4. Taguj generacje wodnymi znakami + metadanymi.

## Zbuduj to

### Krok 1: generowanie z MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Trzy rozmiary: `small` (300M, szybki), `medium` (1,5B), `large` (3,3B). Small wystarczy do sprawdzenia „czy pomysł się sprawdzi."

### Krok 2: warunkowanie melodią

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody przyjmuje chromagram i zachowuje melodię podczas zmiany barwy dźwięku. Przydatne do „daj mi tę melodię jako kwartet smyczkowy."

### Krok 3: ewaluacja FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Oblicza odległość embeddingów VGGish. Przydatne do testów regresji na poziomie gatunku; nie zastępuje słuchaczy ludzkich.

### Krok 4: dodanie do workflow LLM-music

Połącz z pomysłami z Lekcji 7-8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Użyj tego

| Cel | Stack |
|-----|-------|
| Sound design instrumentalny | Stable Audio Open |
| Muzyka do gier / adaptacyjna | Google Lyria RealTime (zamknięty) |
| Pełne piosenki z wokalem (komercyjne) | Suno v5 lub Udio v4 z jasną licencją |
| Pełne piosenki z wokalem (open) | ACE-Step XL lub YuE |
| Krótki jingiel reklamowy | MusicGen melody-conditioned na z humming reference |
| Tło do teledysku | MusicGen + Stable Video Diffusion |

## Pułapki które wciąż trafiają do produkcji w 2026

- **Prompty „prania copyrightu".** „Piosenka w stylu Taylor Swift" — komercyjne filtry Suno/Udio to teraz blokują, modele open nie. Dodaj własną listę filtrów.
- **Powtarzanie / dryf po 30 s.** Modele AR zapętlają. Krzyżuj wiele generacji, lub użyj ACE-Step dla spójności strukturalnej.
- **Dryf tempa.** Modele odchodzą od BPM. Używaj tagów BPM w prompcie i filtruj post-hoc z `librosa.beat_track`.
- **Zrozumiałość wokalu.** Suno jest doskonały; modele open często są „maziste" na słowach. Jeśli tekst ma znaczenie, użyj komercyjnego API lub dostrój.
- **Wyjście mono.** Modele open generują mono lub fałszywe stereo. Ulepsz przez proper stereo reconstruction (ezst, Cartesia's stereo diffusion).

## Wyślij to

Zapisz jako `outputs/skill-music-designer.md`. Wybierz model, strategię licencyjną, plan długości / struktury i metadane disclosure dla wdrożenia music-gen.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Produkuje „generatywny" progresja akordów + wzorzec bębnów jako symbole ASCII — kreskówka music-gen. Odtwórz przez dowolny renderer MIDI jeśli chcesz.
2. **Średnie.** Zainstaluj `audiocraft`, generuj 10-sekundowe klipy przez 4 prompty gatunkowe z MusicGen-small, zmierz FAD względem referencyjnego zbioru gatunków.
3. **Trudne.** Używając ACE-Step (lub MusicGen-melody), wygeneruj trzy wariacje tej samej melodii z różnymi promptami barwy dźwięku. Oblicz podobieństwo CLAP do prompta żeby zweryfikować wyrównanie.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|--------|-----------------|-------------------------|
| FAD | Audio FID | Odległość Frécheta między rozkładami embeddingów prawdziwego vs generowanego audio. |
| Chromagram | Melodia jako wysokości | Wektor 12-wymiarowy per-frame; wejście do warunkowania melodią. |
| Stems | Ścieżki instrumentów | Oddzielone bas / bębny / wokal / melodia jako WAV. |
| Inpainting | Prze-generuj sekcję | Zmaskuj okno czasowe; model regeneruje tylko to. |
| CLAP | Text-audio CLIP | Kontrastywny audio-text embedding; ewaluacja wyrównania tekst-audio. |
| EnCodec | Kodek muzyczny | Neuralny kodek Meta używany przez MusicGen; 32 kHz, 4 codebooki. |

## Dalsze czytanie

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — otwarty benchmark autoregresyjny.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — domyślny sound design.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — otwarty generator 4B pełnych piosenek, kwiecień 2026.
- [Dokumentacja platformy Suno v5](https://suno.com) — komercyjny lider jakości.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — latent diffusion dla muzyki + efektów dźwiękowych.
- [Ugoda WMG-Suno](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — precedens z listopada 2025.