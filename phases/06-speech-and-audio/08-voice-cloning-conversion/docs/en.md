# Klonowanie głosu i konwersja głosu

> Voice cloning odczytuje Twój tekst głosem innej osoby. Voice conversion przepisuje Twój głos na głos innej osoby, zachowując to, co powiedziałeś. Oba opierają się na tym samym prymitywie: oddzieleniu tożsamości mówcy od treści.

**Typ:** Build
**Języki:** Python
**Wymagania wstępne:** Faza 6 · 06 (Speaker Recognition), Faza 6 · 07 (TTS)
**Szacowany czas:** ~75 minut

## Problem

W 2026 roku wystarczy 5-sekundowy klip audio, aby wyprodukować wysokiej jakości klon głosu dowolnej osoby na konsumenckim GPU. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox wszyscy oferują zero-shot lub few-shot cloning. Technologia jest błogosławieństwem (dostępność TTS, dubbing, głosy asystujące) i bronią (oszustwa telefoniczne, polityczne deepfakes, kradzież IP).

Dwa ściśle powiązane zadania:

- **Voice cloning (TTS-side):** tekst + 5-sekundowy głos referencyjny → audio w tym głosie.
- **Voice conversion (speech-side):** audio źródłowe (osoba A mówiąca X) + głos referencyjny osoby B → audio osoby B mówiącej X.

Oba rozkładają waveform na (treść, mówca, prozodia) i rekombinują treść z jednego źródła z mówcą z drugiego.

Kluczowe ograniczenie, z którym teraz wysyłasz w 2026 roku: **watermarking i bramki zgody są prawnie wymagane w UE (AI Act, obowiązuje od sierpnia 2026) i w Kalifornii (AB 2905, w mocy od 2025)**. Twój pipeline musi emitować niesłyszalny watermark i odmawiać niekonsensualnych klonów.

## Koncepcja

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning.** Przekaż 5-sekundowy klip do modelu wytrenowanego na tysiącach mówców. Speaker encoder mapuje klip na speaker embedding; dekoder TTS warunkuje na tym embeddingu plus tekście.

Używane przez: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Few-shot fine-tuning.** Nagraj 5-30 minut głosu docelowego. Dostrój LoRA-base model przez godzinę. Jakość skacze z "okey" do "nierozróżnialnego". Coqui i ElevenLabs oba wspierają ten wzorzec; społeczność używa go z F5-TTS.

**Voice conversion (VC).** Dwie rodziny:

- **Recognition-synthesis.** Uruchom model typu ASR, aby wyekstrahować reprezentację treści (np. soft phoneme posteriors, PPGs), a następnie resyntetyzuj z target speaker embedding. Odporny na język i akcent. Używane przez KNN-VC (2023), Diff-HierVC (2023).
- **Disentanglement.** Trenuj autoencoder, który rozdziela treść, mówcę i prozodię w przestrzeni latent na bottleneck. Zamień speaker embedding podczas inferencji. Niższa jakość, ale szybszy. Używane przez AutoVC (2019), warianty VITS-VC.

**Neural codec-based cloning (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — traktują audio jako dyskretne tokeny z SoundStream / EnCodec, trenują duży autoregresywny model lub flow-matching nad tokenami codec. Jakość porównywalna z ElevenLabs na krótkich promptach.

### Część o etyce, nie dodatek

**Watermarking.** PerTh (Perth) i SilentCipher (2024) osadzają ~16-32 bitowy ID niezauważalnie w audio. Przetrwa re-enkodowanie, streaming i typowe edycje. Gotowe do produkcji open source.

**Consent gates.** Musisz sparować każdy sklonowany wynik z weryfikowalnym recordem zgody. "Ja, Rohit, w dniu 2026-04-22, autoryzuję ten głos do celu X." Przechowuj w logu odpornym na manipulację.

**Detection.** AASIST, RawNet2 i Wav2Vec2-AASIST działają jako detektory. Wyzwanie ASVspoof 2025 opublikowało EERs 0.8–2.3% dla detektorów SOTA przeciwko ElevenLabs, VALL-E 2 i wyjściom Bark.

### Liczby (2026)

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 jest generalnie nierozróżnialny od celu dla większości słuchaczy.

## Zbuduj to

### Krok 1: dekompozycja z recognition-synthesis (demo tylko kod w main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Koncepcyjnie proste; masa implementacji jest w `tts_model` i speaker encoderze.

### Krok 2: zero-shot clone z F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Referencyjny transkrypt musi dokładnie odpowiadać audio; mismatch łamie alignment.

### Krok 3: voice conversion z KNN-VC

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC uruchamia WavLM, aby wyekstrahować per-frame embeddings dla source i target pool, a następnie zastępuje każdy source frame jego najbliższym sąsiadem w pool. Non-parametric, działa z minutą mowy docelowej.

### Krok 4: osadź watermark

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32 bity payload, wykrywalne po re-enkodzie MP3 i lekkim szumie.

### Krok 5: consent gate

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Użyj tego

Stack 2026:

| Sytuacja | Wybierz |
|-----------|------|
| 5-sec zero-shot clone, open-source | F5-TTS lub OpenVoice v2 |
| Komercyjna produkcja klonowania | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion (przepisywanie) | KNN-VC lub Diff-HierVC |
| Many-speaker fine-tune | StyleTTS 2 + speaker adapter |
| Cross-lingual cloning | XTTS v2 lub VALL-E X |
| Wykrywanie deepfake | Wav2Vec2-AASIST |

## Pułapki

- **Misaligned reference transcript.** F5-TTS i podobne wymagają, aby referencyjny tekst dokładnie odpowiadał referencyjnemu audio, łącznie ze znakami interpunkcyjnymi.
- **Reverberant reference.** Echo zabija klona. Nagrywaj sucho, bliskim mikrofonem.
- **Emotional mismatch.** Trening reference "cheerful" produkuje cheerful klony wszystkiego. Dopasuj referencyjną emocję do docelowego użycia.
- **Language leakage.** Klonowanie angielskiego mówcy, a następnie proszenie modelu o mówienie po francusku często niesie akcent; użyj cross-lingual models (XTTS, VALL-E X).
- **Brak watermarka.** Prawnie niewysyłalne w UE od sierpnia 2026.

## Wyślij to

Zapisz jako `outputs/skill-voice-cloner.md`. Zaprojektuj pipeline klonowania lub konwersji z bramką zgody + watermark + celem jakości.

## Ćwiczenia

1. **Łatwe.** Uruchom `code/main.py`. Demonstrates zamianę speaker-embedding przez obliczanie cosinusa między dwoma "mówcami" przed i po zamianie.
2. **Średnie.** Użyj OpenVoice v2, aby sklonować swój własny głos. Zmierz SECS między reference i clone. Zmierz CER przez Whisper.
3. **Trudne.** Zastosuj SilentCipher watermark do 20 klonów, przepuść je przez 128 kbps MP3 encode+decode, wykryj payload. Zgłoś bit-accuracy.

## Kluczowe terminy

| Termin | Co ludzie mówią | Co to faktycznie oznacza |
|-------|-----------------|-----------------------|
| Zero-shot clone | 5 sekund wystarczy | Model pretrained + speaker embedding; bez treningu. |
| PPG | Phonetic posteriorgram | Per-frame ASR posteriors używane jako language-agnostic content rep. |
| KNN-VC | Nearest-neighbor conversion | Zastąp każdy source frame najbliższym target-pool frame. |
| Neural codec TTS | Styl VALL-E | Model AR nad tokenami EnCodec/SoundStream. |
| Watermark | Niesłyszalny podpis | Bity osadzone w audio, przetrwają re-encode. |
| SECS | cloning fidelity | Cosine między target i clone speaker embeddings. |
| AASIST | Deepfake detector | Model anti-spoofing; wykrywa syntetyzowaną mowę. |

## Dalsze czytanie

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — open-source SOTA zero-shot cloning.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) i [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — disentanglement-based voice conversion.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — retrieval-based VC.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — produkcyjny 32-bit audio watermark.
- [ASVspoof 2025 results](https://www.asvspoof.org/) — detector vs synthesizer arms race, updated 2026.