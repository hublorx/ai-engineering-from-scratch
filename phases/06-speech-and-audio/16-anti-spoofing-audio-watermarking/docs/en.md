# Voice Anti-Spoofing i znakowanie wodne audio — ASVspoof 5, AudioSeal, WaveVerify

> Voice cloning dotarł do produkcji szybciej, niż mechanizmy obronne. Systemy głosowe w 2026 roku potrzebują dwóch rzeczy: detektora (AASIST, RawNet2), który klasyfikuje mowę prawdziwą vs syntetyczną, oraz znaku wodnego (AudioSeal), który przetrwa kompresję i edycję. Dostarcz oba lub nie dostarczaj voice clonów.

**Typ:** Budowanie
**Języki:** Python
**Wymagania wstępne:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Szacowany czas:** ~75 minut

## Problem

Trzy powiązane mechanizmy obronne:

1. **Anti-spoofing / detekcja deepfake.** Czy dany klip audio jest syntetyczny, czy prawdziwy? Benchmarki ASVspoof (ASVspoof 2019 → 2021 → 5) są standardem branżowym.
2. **Znakowanie wodne audio.** Osadź niesłyszalny sygnał w wygenerowanym audio, który detektor może później wykryć. AudioSeal (Meta) i WavMark to opcje open-source.
3. **Uwierzytelnione pochodzenie.** Kryptograficzne podpisywanie plików audio + metadanych. C2PA / Content Authenticity Initiative.

Detekcja obsługuje atakujących, którzy nie współpracują. Znakowanie wodne obsługuje compliance — audio wygenerowane przez AI powinno być identyfikowalne jako takie. Oba są wymagane w 2026 roku.

## Koncepcja

![Anti-spoofing vs znakowanie wodne vs pochodzenie — trzy warstwy obronne](../assets/spoofing-watermark.svg)

### ASVspoof 5 — benchmark 2024-2025

Największa zmiana w porównaniu z poprzednimi edycjami:

- **Dane crowdsourced** (nie studio clean) — realistyczne warunki.
- **~2000 speakerów** (vs ~100 przedtem).
- **32 algorytmy ataku.** TTS + voice conversion + adversarial perturbation.
- **Dwa tryby.** Countermeasure (CM) samodzielna detekcja; Spoofing-robust ASV (SASV) dla systemów biometrycznych.

State-of-the-art na ASVspoof 5: ~7.23% EER. Na starszym ASVspoof 2019 LA: 0.42% EER. Rzeczywiste wdrożenie produkcyjne: spodziewaj się 5-10% EER na klipach w warunkach rzeczywistych.

### AASIST i RawNet2 — rodziny modeli detekcyjnych

**AASIST** (2021, aktualizowany przez 2026). Graph-attention na cechach spektralnych. Obecny SOTA na ASVspoof 5, countermeasure task.

**RawNet2.** Konwolucyjny front-end na surowym waveformzie + TDNN backbone. Prostszy baseline; nadal konkurencyjny po fine-tuning.

**NeXt-TDNN + SSL features.** Wariant z 2025: architektura stylu ECAPA + cechy WavLM + focal loss. Osiąga 0.42% EER na ASVspoof 2019 LA.

### AudioSeal — domyślny watermark 2024

**AudioSeal** Meta (Jan 2024, v0.2 Dec 2024). Kluczowy design:

- **Zlocalizowany.** Wykrywa watermark per-frame przy rozdzielczości próbki 16 kHz (1/16000 s).
- **Generator + detector jointly trained.** Generator uczy się osadzać niesłyszalny sygnał; detector uczy się go znajdować przez augmentacje.
- **Odporny.** Przetrwa kompresję MP3 / AAC, EQ, speed-shift ±10%, noise mix +10 dB SNR.
- **Szybki.** Detector działa przy 485× realtime; 1000× szybciej niż WavMark.
- **Pojemność.** 16-bitowy payload (może kodować model ID, timestamp generacji, user ID) osadzalny w każdej wypowiedzi.

### WavMark

Open-source'owy baseline sprzed AudioSeal. Invertible neural network, 32 bity/sec. Problemy:

- Synchronizacja brute-force jest wolna.
- Może zostać usunięty przez szum Gaussowski lub kompresję MP3.
- Nieprzyjazny dla przetwarzania w czasie rzeczywistym.

### WaveVerify (lipiec 2025)

Adresuje słabości AudioSeal — konkretnie manipulacje temporalne (odwrócenie, speed). Używa FiLM-based generator + Mixture-of-Experts detector. Konkurencyjny z AudioSeal na standardowych atakach; obsługuje edycje temporalne.

### Luka, którą exploitują adversary

Z AudioMarkBench: "under pitch shift, all watermarks show Bit Recovery Accuracy below 0.6, indicating near-complete removal." **Pitch-shift to uniwersalny atak.** Żaden watermark z 2026 nie jest w pełni odporny na agresywną modyfikację pitch. Dlatego potrzebujesz detekcji (AASIST) obok znakowania wodnego.

### C2PA / Content Authenticity Initiative

Nie technika ML — format manifestu. Pliki audio noszą kryptograficznie podpisane metadane o narzędziu tworzenia, autorze, dacie. Audobox / Seamless go używają. Dobre dla provenance; nie robi nic, jeśli zły aktor ponownie zakoduje i stripuje metadane.

## Zbuduj to

### Krok 1: prosty detektor na cechach spektralnych (przykładowy)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Syntetyczna mowa często ma niezwykle płaską energię wysokich częstotliwości. Produkcjne detektory używają AASIST, nie tego. Ale intuicja się sprawdza.

### Krok 2: AudioSeal embed + detect

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: wartość w [0, 1] — prawdopodobieństwo obecności znaku wodnego
# decoded_payload: 16 bitów; porównaj z osadzonym payloadem
```

### Krok 3: ewaluacja — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Krok 4: integracja produkcyjna

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Każda generacja zawiera: (1) watermark, (2) podpisany manifest, (3) retention-policy-compliant audit log.

## Użyj tego

| Przypadek użycia | Obrona |
|----------|---------|
| Wysyłanie TTS / voice clone'ingu | AudioSeal embed na każdym wyjściu (obowiązkowe) |
| Biometryczny voice unlock | AASIST + ECAPA ensemble; wyzwanie żywotności |
| Wykrywanie oszustw w call-center | AASIST na 20% próbce przychodzących połączeń |
| Autentyczność podcastu | C2PA signing przy upload, AudioSeal jeśli AI-generated |
| Badania / trenowanie detektorów | Zbiory train/dev/eval ASVspoof 5 |

## Pułapki

- **Watermark bez uruchomionego detektora.** Bezsensu. Dostarcz detektor w CI.
- **Detekcja bez kalibracji.** AASIST trenowany na ASVspoof LA overfittuje; real-world accuracy spada. Kalibruj na swojej domenie.
- **Luka pitch-shift.** Agresywny pitch shift usuwa większość watermarków. Miej fallback detekcyjny.
- **Metadata strip-and-rehost.** C2PA jest trywialnie omijalne przez re-encoding. Zawsze dodawaj obronę kryptograficzną + percepcyjną (watermark) razem.
- **Żywotności jako detekcja.** Poproś użytkownika o powiedzenie losowej frazy. Zapobiega replay attacks ale nie real-time clone'ingu.

## Dostarcz to

Zapisz jako `outputs/skill-spoof-defender.md`. Wybierz model detekcyjny, watermark, manifest pochodzenia i operacyjny playbook dla voice-gen deploymentu.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Przykładowy detektor + przykładowy watermark embed/detect na syntetycznym audio.
2. **Średnie.** Zainstaluj `audioseal`, osadź 16-bitowy payload w wyjściu TTS, zdekoduj ponownie. Zepsuj audio szumem i zmierz Bit Recovery Accuracy.
3. **Trudne.** Fine-tune'uj RawNet2 lub AASIST na ASVspoof 2019 LA. Zmierz EER. Testuj na held-out zbiorze F5-TTS-generated clips — zobacz jak OOD detection się pogarsza.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|------|-----------------|-----------------------|
| ASVspoof | Benchmark | Biennial challenge; 2024 = ASVspoof 5. |
| CM (countermeasure) | Detektor | Klasyfikator: prawdziwa mowa vs syntetyczna / converted. |
| SASV | Połączenie ASV + CM | Standard dla systemów biometrycznych z obroną przed spoofingiem. |
| AudioSeal | Meta watermark | Zlokalizowany, 16-bit payload, 485× szybszy niż WavMark. |
| Bit Recovery Accuracy | Przetrwanie watermarka | Frakcja odzyskanych bitów payload po ataku. |
| C2PA | Manifest pochodzenia | Kryptograficzne metadane o tworzeniu / autorstwie. |
| AASIST | Rodzina detektorów | Graph-attention-based anti-spoofing SOTA. |

## Dalsze czytanie

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — current benchmark.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — domyślny watermark.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — MoE detector dla ataków temporalnych.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA detection backbone.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — ewaluacja odporności.
- [C2PA specification](https://c2pa.org/specifications/specifications/) — format manifestu provenance.