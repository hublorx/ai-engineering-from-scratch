# Generacja wideo

> Obraz to tensor 2-D. Wideo to tensor 3-D. Teoria jest ta sama; obliczenia są 10-100x trudniejsze. Sora od OpenAI (luty 2024) udowodniła, że to możliwe. Do 2026 Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0 i WAN 2.2 dostarczają produkcyjne wideo z tekstu w 1080p — a stos otwartych wag (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) jest 12 miesięcy za nimi.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Czas:** ~45 minut

## Problem

10-sekundowe wideo 1080p przy 24fps to 240 klatek o rozmiarze 1920×1080×3 pikseli. To około 1,5 GB surowych danych na klip. Dyfuzja w przestrzeni pikseli jest niewykonalna. Potrzebujesz:

1. **Kompresji spatiotemporalnej.** VAE kodujący wideo, a nie poszczególne klatki, do sekwencji fragmentów przestrzenno-czasowych.
2. **Koherencji czasowej.** Klatki muszą dzielić treść, oświetlenie i tożsamość obiektów przez sekundy. Sieć musi modelować ruch.
3. **Budżetu obliczeniowego.** Trenowanie wideo jest 10-100x droższe niż obrazu przy tym samym rozmiarze modelu.
4. **Warunkowania.** Tekst, obraz (pierwsza klatka), audio lub inne wideo. Większość produkcyjnych modeli akceptuje wszystkie cztery.

Architekturą, która to rozwiązała, jest **Diffusion Transformer (DiT)** zastosowany do fragmentów spatiotemporalnych, trenowany na ogromnych zbiorach danych (prompt, podpis, wideo). Ta sama funkcja strat dyfuzyjnych co w Lekcji 06.

## Koncepcja

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchify

Koduj wideo za pomocą 3D VAE (nauczona kompresja spatiotemporalna). Latent ma kształt `[T_latent, H_latent, W_latent, C_latent]`. Podziel na fragmenty o rozmiarze `[t_p, h_p, w_p]`. W modelach stylu Sora `t_p = 1` (fragmenty na klatkę) lub `t_p = 2` (co dwie klatki). 10-sekundowe wideo 1080p kompresuje się do około 20 000-100 000 fragmentów.

### Spatiotemporal DiT

Transformer przetwarza spłaszczoną sekwencję fragmentów. Każdy fragment ma 3D positional embedding (czas + y + x). Uwaga jest zwykle rozkładana:

- **Uwaga przestrzenna** w ramach fragmentów każdej klatki.
- **Uwaga czasowa** między klatkami w tej samej lokalizacji przestrzennej.
- **Pełna uwaga 3D** jest 16-100x droższa; używana tylko przy niskiej rozdzielczości lub w badaniach.

### Warunkowanie tekstowe

Cross-attention z dużym enkoderem tekstowym (T5-XXL dla Sora, CogVideoX-5B używa T5-XXL). Długie prompty mają znaczenie — zbiór treningowy Sora zawierał gęste ponowne podpisy wygenerowane przez GPT, średnio 200 tokenów na klip.

### Trenowanie

Standardowa funkcja strat dyfuzyjnych (ε lub v prediction) na latentach spatiotemporalnych. Dane: wideo z internetu + ~100M wyselekcjonowanych klipów + syntetyczne podpisy tekstowe. Obliczenia: 10 000+ godzin GPU nawet dla małego eksperymentu badawczego; skala Sora to 100 000+.

## Krajobraz produkcyjny 2026

| Model | Data | Maks. czas | Maks. rozdz. | Otwarte wagi? | Uwagi |
|-------|------|------------|--------------|---------------|-------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | Nie | Pierwszy model pokazujący właściwości symulatora świata na skali |
| Sora Turbo | 2024-12 | 20s | 1080p | Nie | Produkcja Sora przy 5x szybszej inferencji |
| Veo 2 (Google) | 2024-12 | 8s | 4K | Nie | Najwyższa jakość + fizyka w 2025 |
| Veo 3 | 2025 Q3 | 15s | 4K | Nie | Natywny audio i silniejsza kontrola kamery |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | Nie | Najlepszy ruch ludzki w 2025 Q1 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | Nie | Profesjonalne narzędzia wideo na wierzchu |
| Pika 2.0 | 2024-10 | 5s | 1080p | Nie | Najsilniejsza spójność postaci |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Tak (2B, 5B) | Pierwszy otwarty wideo w skali 5B |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Tak (13B) | Otwarty SOTA późnego 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Tak (10B) | Najbardziej permisywnie licencjonowany |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Tak | Najsilniejszy otwarty model w połowie 2025 |

Otwarte wagi zamykają przepaść szybciej niż w przestrzeni obrazów: HunyuanVideo + LoRA WAN 2.2 już napędzają większość otwartych workflowów w połowie 2026.

## Zbuduj to

`code/main.py` symuluje podstawową koncepcję spatiotemporal DiT: patchify małego syntetycznego wideo, dodaj per-fragment position embedding i denoise całą sekwencję z transformer-style attention nad fragmentami. Bez numpy; czyste Python. Pokazujemy, że koherencja czasowa pojawia się nawet w 1-D, gdy sąsiednie fragmenty klatek dzielą denoiser i position embeddings.

### Krok 1: patchify syntetycznego 1-D "wideo"

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Krok 2: position embedding per klatka

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Krok 3: denoiser widzi całą sekwencję

Zamiast denoisować każdą klatkę niezależnie, nasza mała sieć konkatenuje wszystkie wartości klatek + ich position embeddings i przewiduje szum dla wszystkich klatek wspólnie.

### Krok 4: test koherencji czasowej

Po treningu, próbkuj wideo. Zmierz delta między klatkami. Jeśli model nauczył się struktury czasowej, delty pozostają mniejsze niż przy próbkowaniu każdej klatki niezależnie.

## Pułapki

- **Niezależne próbkowanie na klatkę = migotanie.** Jeśli uruchomisz dyfuzję obrazu na każdej klatce osobno, wyjście migocze, bo szum każdej klatki jest niezależny. Dyfuzja wideo to naprawia, sprzęgając klatki przez attention lub wspólny szum.
- **Naiwna uwaga 3D = OOM.** Pełna uwaga 3D na 10-sekundowym latent 1080p to setki miliardów operacji. Rozkładaj na przestrzenny + czasowy.
- **Podpisywanie danych ma większe znaczenie niż rozmiar.** Główna aktualizacja Sora w porównaniu do wcześniejszych prac było trening na ~10x bardziej szczegółowych podpisach (klipy przeetikietowane przez GPT-4). Techniczny raport OpenAI jest w tym względzie jednoznaczny.
- **Warunkowanie pierwszej klatki.** Większość produkcyjnych modeli akceptuje również obraz jako pierwszą klatkę. To tryb "image-to-video"; trening zawiera tę wariant.
- **Dryf fizyki.** Długie klipy (>10s) akumulują subtelne niespójności. Generacja sliding-window + keyframe anchoring pomaga.

## Użyj tego

| Przypadek użycia | Wybór 2026 |
|----------|-----------|
| Najwyższa jakość text-to-video, hostowana | Veo 3 lub Sora |
| Kinowa kontrola kamery | Runway Gen-3 z motion brushes |
| Spójność postaci między klipami | Pika 2.0 lub Kling 2.1 |
| Otwarte wagi, szybki fine-tune | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V, Kling 2.1 I2V lub Runway |
| Audio-to-video synchronizacja ust | Veo 3 (natywny audio) lub dedykowany model synchronizacji ust |
| Edycja wideo | Runway Act-Two, Kling Motion Brush, Flux-Kontext (still-frame) |

Koszt na sekundę wideo przy jakości parytetowej spadł 20x między 2024 a 2026.

## Wyślij to

Zapisz `outputs/skill-video-brief.md`. Skill przyjmuje brief wideo (czas trwania, proporcje ekranu, styl, plan kamery, spójność tematu, audio) i zwraca: model + hosting, scaffolding promptów (język kamery, opis tematu, deskryptory ruchu), seed + protokół reprodukowalności i checklistę QA na poziomie klatek.

## Ćwiczenia

1. **Łatwe.** W `code/main.py`, porównaj deltę między klatkami dla (a) niezależnego próbkowania na klatkę, (b) wspólnego próbkowania sekwencji. Raportuj średnią i wariancję delt.
2. **Średnie.** Dodaj warunek pierwszej klatki: przypnij klatkę 0 do danej wartości i próbkuj resztę. Zmierz, jak przypięta wartość się propaguje.
3. **Trudne.** Użyj HuggingFace diffusers do uruchomienia CogVideoX-2B na lokalnym GPU. Zmierz czas 20 kroków inferencji w 720p dla 6-sekundowego klipu. Profiluj spatiotemporal attention, aby zidentyfikować wąskie gardło.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| Video VAE | "3-D VAE" | Enkoder kompresujący `(T, H, W, C)` → spatiotemporal latent. |
| Patches | "The tokens" | Fragmenty o stałym rozmiarze z latenta; wejście do DiT. |
| Rozkładana uwaga | "Spatial + temporal" | Uruchom uwagę najpierw przestrzennie, potem czasowo; pomiń pełną uwagę 3D. |
| Image-to-video (I2V) | "Animuj to zdjęcie" | Model przyjmuje obraz + tekst, zwraca wideo zaczynające się od niego. |
| Keyframe conditioning | "Anchor frames" | Przypnij konkretne klatki, aby kontrolować łuk wideo. |
| Motion brush | "Directional hint" | Wejście UI gdzie użytkownik malował wektory ruchu na obrazie. |
| Re-captioning | "Dense captions" | Użycie LLM do przeetykietowania klipów treningowych szczegółowymi promptami. |
| Migotanie | "Temporal artifact" | Niespójność między klatkami; naprawione przez sprzężony denoising. |

## Uwaga produkcyjna: latenty wideo to problem przepustowości pamięci

10-sekundowy klip 1080p przy 24 fps to 240 klatek × 1920 × 1080 × 3 ≈ 1,5 GB surowych pikseli. Po kompresji 4× przez video VAE (`2 × przestrzenne × 2 × czasowe`) latent to około 100 MB na żądanie. Przepuść to przez spatiotemporal DiT przez 30 kroków przy batch 1 i przesyłasz około 3 GB/krok przez HBM — przepustowość pamięci, nie FLOPy, jest wąskim gardłem.

Trzy pokrętła produkcyjne, wszystkie prosto z rozdziału o inferencji w literaturze produkcyjnej:

- **TP przez DiT.** Modele text-to-video są rutynowo ≥10B parametrów. TP=4 na 4 H100 to standard; PP=2 × TP=2 dla modeli klasy 405B. Latencja na krok spada w przybliżeniu liniowo z TP aż do all-reduce wall.
- **Frame batching = continuous batching.** W czasie generacji, wideo jest koncepcyjnie batchem klatek połączonych przez uwagę. Continuous batching (in-flight scheduling) ma zastosowanie: zacznij renderować klatkę `t+1` podczas gdy klatka `t-1` jest zwracana, jeśli architektura modelu pozwala na sliding-window generation.
- **Clip-level prefill cache.** Dla image-to-video, warunek pierwszej klatki jest analogiczny do prompt prefill LLM: oblicz raz, używaj ponownie przez przejścia temporalnego dekodera. To efektywnie KV-cache dla wideo.

## Dalsza lektura

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — techniczny raport Sora.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — otwarty SOTA w połowie 2025.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — przełomowa praca o dyfuzji wideo.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — przodek Stable Video Diffusion.