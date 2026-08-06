# Derization — Speaker Diarization & Voice Separation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5%20cu121-EE4C2C.svg?logo=pytorch&logoColor=white) ![CUDA](https://img.shields.io/badge/GPU-CUDA%2012.1-76B900.svg?logo=nvidia&logoColor=white) ![Platform](https://img.shields.io/badge/Platform-Windows-0078D6.svg?logo=windows&logoColor=white) ![Status](https://img.shields.io/badge/status-active-success.svg)

Separate the voices of **2–3 speakers** in an audio recording. Derization runs a
full **GPU** pipeline — **vocal isolation → speaker diarization → true overlap
separation → cross-talk gate → per-speaker studio enhancement → export
(+ optional transcription)** — figuring out **who spoke when** and writing a
**separate, cleaned-up audio track for each speaker** (in WAV/FLAC/OGG/MP3/AAC),
plus a timeline and an optional **speaker-labelled transcript with subtitles**.

It works best on conversational audio where people mostly take turns (interviews,
meetings, podcasts, phone calls). It is primarily *diarization-based* separation
(each speaker's track holds their turns, silence elsewhere), hardened against
bleed by two extra stages:

- **True overlap separation** (SepFormer) un-mixes *simultaneous* speech into
  each speaker's track and deletes slices it can't confidently tell apart.
- **Cross-talk gate** re-checks each speaker's own segments against the
  voiceprints and silences a brief *foreign* sound that leaked in — a laugh or
  interjection that wasn't simultaneous speech, so the overlap stage never saw
  it. The rule throughout: **when in doubt, delete rather than guess**, so
  nothing bleeds between tracks.

Un-mixing real, noisy recordings is imperfect — where it isn't confident, those
moments are removed rather than guessed.

---

## What you get

For an input like `interview.mp3`, the output folder contains:

| File | Description |
|------|-------------|
| `SPEAKER_1.<ext>`, `SPEAKER_2.<ext>`, … | Full-length, denoised + enhanced track per speaker (their voice, silence elsewhere — stays in sync with the original). Extension follows your chosen format. |
| `SPEAKER_k_compact.<ext>` *(optional)* | Only that speaker's turns, concatenated. |
| `diarization.txt` | Human-readable "who spoke when" timeline + speaking-time totals. |
| `diarization.csv` | `start, end, duration, speaker` for spreadsheets/scripts. |
| `diarization.rttm` | Standard NIST RTTM (compatible with diarization scoring tools). |
| `transcript.txt` *(with `--transcribe`)* | Speaker-labelled transcript grouped into turns (`[00:00:03] SPEAKER_1: …`). |
| `transcript.srt`, `transcript.vtt` *(with `--transcribe`)* | Subtitles carrying the speaker label on each cue. |
| `SPEAKER_k.txt` *(with `--transcribe`)* | Each speaker's words as plain text. |

---

## Install

You need Python 3.9+ (you have 3.11). From this folder:

```bash
pip install -r requirements.txt
```

That gives a fully working tool using the built-in fallback backends
(energy-based voice detection + MFCC voiceprints).

**For best accuracy**, also install the neural backends (a ~80 MB speaker model
downloads automatically on first use).

If you have an **NVIDIA GPU** (recommended — much faster):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install speechbrain silero-vad
```

CPU only:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install speechbrain silero-vad
```

> MP3, M4A and AAC input/output work out of the box — the `imageio-ffmpeg`
> dependency bundles a private ffmpeg binary (inside `.venv`), so nothing needs
> to be installed system-wide. WAV/FLAC/OGG use libsndfile directly.

### GPU acceleration

The tool **automatically uses your NVIDIA GPU** when a CUDA build of PyTorch is
installed — the ECAPA voiceprints are batched onto the GPU, which is the main
speedup. Control it with `--device auto|cuda|cpu` (CLI) or the **Device**
dropdown (GUI); the GUI's status line shows the detected GPU. By default the
batch size is **auto** — the tool reads the GPU's free VRAM and sizes the batch
to fill it (leaving headroom) so work stays on the GPU rather than spilling to
the CPU; pass `--batch-size N` to force a size. The CUDA wheels bundle their own
runtime, so you only need a recent NVIDIA driver.

---

## Use it — Desktop GUI

**Easiest (Windows): double-click `start.bat`.** On first run it builds its own
isolated environment (`.venv`), auto-installs everything it needs (choosing the
GPU or CPU PyTorch build automatically), and launches the app. Later runs just
launch it. No manual setup required — only a system Python 3.10+ is needed (it
tells you where to get it if missing).

> **Self-contained folder.** Everything the app needs — all Python
> dependencies *and* the bundled ffmpeg — is installed into `.venv` right next
> to `start.bat`, not system-wide. The project folder is the whole app; nothing
> else on the machine is touched.

If you already have the dependencies installed, you can also run:

```bash
python gui.py
```

or double-click **`run_gui.bat`**. Pick a file, choose the number of speakers
(`auto`, `2`, or `3`), press **Separate speakers**, and watch the verbose log
stream as it works. When it finishes, click **Open output folder**.

## Use it — Command line

```bash
# Auto-detect 2 vs 3 speakers, verbose output
python cli.py interview.mp3 -v

# Force exactly 2 speakers, custom output folder
python cli.py meeting.wav --speakers 2 --outdir ./out

# Also export compact (concatenated) tracks
python cli.py podcast.wav --speakers auto --compact
```

If you omit `--format`, the CLI **prompts** you for the export format and
confirms the destination directory (spec-driven). Pass `-y` to skip prompts.

Key options: `--speakers auto|2|3`, `--outdir`, `--format wav|flac|ogg|mp3|aac`,
`--overlap off|light|medium|strong|extreme` (how aggressively to cut cross-talk /
overlap), `--overlap-mode separate|delete`, `--transcribe`,
`--whisper-model tiny|base|small|medium|large-v3`, `--engine builtin|pyannote`,
`--hf-token`, `--bitrate 192k`, `--no-denoise`, `--no-enhance`,
`--denoise-backend rvc_hp5|auto|noisereduce|spectral|none`,
`--vad auto|silero|energy`, `--embeddings auto|ecapa|mfcc`,
`--device auto|cuda|cpu`, `--allow-cpu`, `--batch-size N`, `--compact`,
`--no-tracks`, `-y`, `-v/--verbose`, `-q/--quiet`. Run `python cli.py --help`
for the full list.

Example forcing the GPU and MP3 output, with a speaker-labelled transcript:

```bash
python cli.py meeting.wav --speakers 2 --device cuda --format mp3 --transcribe -y -v
```

---

## Extras

### Transcription (`--transcribe`)

Adds speech-to-text with **speaker attribution**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2) transcribes the cleaned audio on the GPU with word-level
timestamps, and each word is attributed to a speaker via the diarization
timeline. You get `transcript.txt` (turns), per-speaker `.txt`, and `.srt`/`.vtt`
subtitles. Pick the model with `--whisper-model` (`small` is a good balance on a
6 GB card; `large-v3` is most accurate). The model downloads once into
`models/whisper` and then loads offline.

### State-of-the-art diarization engine (`--engine pyannote`)

Optional [pyannote.audio](https://github.com/pyannote/pyannote-audio) 3.1
backend — more accurate speaker turns and native overlap detection. It is
**opt-in** and falls back to the built-in engine if anything is missing, so the
app always runs. To enable it:

1. `pip install pyannote.audio`
2. Accept the licences (free) for
   [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1)
   and [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0).
3. Create a Hugging Face access token and set it: `set HF_TOKEN=hf_…`
   (or pass `--hf-token`).
4. Run with `--engine pyannote` (CLI) or pick **pyannote** in the GUI's
   **Engine** dropdown.

pyannote only produces the timeline; speaker voiceprints for the overlap and
cross-talk stages are still computed with the same ECAPA embedder, so the rest
of the pipeline (separation, cross-talk gate, enhancement, transcription) is
identical.

## Use it — from Python

```python
from diarizer import diarize_file

result, files = diarize_file("interview.wav", outdir="out", num_speakers=2)
print(result.num_speakers, "speakers")
for spk, secs in result.speaker_time().items():
    print(result.label(spk), f"{secs:.1f}s")
```

---

## Try it without any audio

**Real speech (best — exercises the neural GPU path).** Stitches a conversation
from different real LibriSpeech speakers (small clips download once):

```bash
python make_real_sample.py real_sample.wav --speakers 2
python cli.py real_sample.wav --speakers 2 --device cuda -v
```

**Synthetic tones (offline, no download).** These are *not* real speech, so the
neural backends (Silero VAD, ECAPA) won't treat them as voices — use the MFCC +
energy backends for this file:

```bash
python make_sample.py sample.wav --speakers 2 --seconds 30
python cli.py sample.wav --speakers 2 --vad energy --embeddings mfcc -v
```

---

## How it works

```
audio ─▶ [isolate vocals] RVC/UVR HP5 ─▶ [diarize] who spoke when
      ─▶ [separate] un-mix overlaps ─▶ [cross-talk gate] kill foreign bleed
      ─▶ [enhance] per speaker ─▶ per-speaker tracks + timeline
      ─▶ [transcribe] speaker-labelled transcript + subtitles (optional)
```

0. **Vocal isolation** removes music/background so only voices remain
   (RVC/UVR **HP5** model via `audio-separator`, running on the GPU; other
   backends available via `--denoise-backend`).
1. **Diarization** — either the built-in engine or pyannote (`--engine`):
   Silero neural VAD trims silence; overlapping ~1.5 s windows become speaker
   "voiceprints" (SpeechBrain **ECAPA-TDNN** on the GPU, batched to fit VRAM);
   agglomerative complete-linkage cosine clustering groups them (auto 2-vs-3 by
   silhouette); the timeline is rasterised, smoothed and merged into segments.
2. **Overlap separation** un-mixes *simultaneous* speech with **SepFormer**,
   routes each source to its speaker by ECAPA centroid, and deletes slices it
   can't confidently split.
3. **Cross-talk gate** re-checks each speaker's segments at fine resolution and
   silences windows that clearly belong to a different speaker (leaked laughs /
   interjections the overlap stage never saw).
4. **Enhancement** (per speaker): high-pass, presence EQ, gentle compression,
   and loudness normalization (−16 LUFS) so every speaker sits at a consistent,
   clear level.
5. **Export** writes each speaker's track in your chosen format (the CLI prompts
   for format + destination; the GUI has dropdowns).
6. **Transcription** (optional) transcribes on the GPU with Whisper and
   attributes each word to a speaker.

Everything runs **GPU-only by default** (no silent CPU fallback); pass
`--allow-cpu` to relax that. Models download once into the project's `models/`
folder, so the app stays self-contained. Tunable parameters live in
`diarizer/config.py`.

---

## Project layout

```
derization/
├─ cli.py                 # command-line entry point (verbose)
├─ gui.py                 # Tkinter desktop GUI with log console
├─ make_sample.py         # synthetic multi-speaker test generator (offline)
├─ make_real_sample.py    # real-speech multi-speaker test generator (neural path)
├─ run_gui.bat            # Windows launcher
├─ requirements.txt
└─ diarizer/              # the engine (shared by CLI + GUI)
   ├─ config.py           # all tunable parameters
   ├─ audioio.py          # load / resample / save audio
   ├─ formats.py          # encode to wav/flac/ogg/mp3/aac
   ├─ vad.py              # voice activity detection (silero + energy)
   ├─ embeddings.py       # speaker voiceprints (ecapa + mfcc), GPU-batched
   ├─ hardware.py         # GPU/CPU device selection + VRAM-aware batching
   ├─ cluster.py          # clustering + auto speaker-count
   ├─ pipeline.py         # built-in orchestration -> segments
   ├─ pyannote_backend.py # optional SOTA diarization engine (pyannote.audio)
   ├─ rvc_denoise.py      # RVC/UVR HP5 vocal isolation on the GPU
   ├─ separate.py         # SepFormer overlap un-mixing
   ├─ crosstalk.py        # residual cross-talk gate (kill leaked foreign voice)
   ├─ enhance.py          # per-speaker studio enhancement
   ├─ transcribe.py       # Whisper transcription + speaker attribution
   ├─ export.py           # per-speaker audio + timeline files
   └─ logutil.py          # shared verbose logging
```

## Limitations

- Overlap un-mixing on real, similar-sounding voices is hard: where SepFormer
  can't confidently split simultaneous speech, those moments are **deleted**
  (silent in both tracks) rather than guessed — clean, but not recovered.
- Accuracy depends on audio quality; noisy/cross-talk-heavy audio is harder.
  The built-in engine is strong on turn-taking conversation; for the toughest
  audio, enable the pyannote engine (`--engine pyannote`).
- Auto speaker-count is a best guess — pass `--speakers 2`/`3` if you know it.
- GPU-only by default: a CUDA GPU is expected. Pass `--allow-cpu` to run the
  neural stages on the CPU (much slower).
