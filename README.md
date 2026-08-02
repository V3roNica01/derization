# Derization — Speaker Diarization & Voice Separation

Separate the voices of **2–3 speakers** in an audio recording. Derization runs a
full pipeline — **noise reduction → speaker diarization → per-speaker studio
enhancement → export** — figuring out **who spoke when** and writing a
**separate, cleaned-up audio track for each speaker** (in WAV/FLAC/OGG/MP3/AAC),
plus a timeline you can read or feed into other tools.

It works best on conversational audio where people mostly take turns (interviews,
meetings, podcasts, phone calls). It is *diarization-based* separation: each
speaker's track contains their turns with silence elsewhere (it does not un-mix
two people talking at the exact same instant).

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
dropdown (GUI); the GUI's status line shows the detected GPU. On a 6 GB card the
default batch size of 32 is comfortable; raise `--batch-size` for more speed or
lower it if you ever hit out-of-memory (the tool also auto-recovers from OOM by
shrinking the batch). The CUDA wheels bundle their own runtime, so you only need
a recent NVIDIA driver.

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
`--overlap off|light|medium|strong` (how hard to delete cross-talk from both
speakers), `--bitrate 192k`, `--no-denoise`, `--no-enhance`, `--denoise-backend`,
`--vad auto|silero|energy`, `--embeddings auto|ecapa|mfcc`,
`--device auto|cuda|cpu`, `--batch-size N`, `--compact`, `--no-tracks`, `-y`,
`-v/--verbose`, `-q/--quiet`. Run `python cli.py --help` for the full list.

Example forcing the GPU and MP3 output, no prompts:

```bash
python cli.py meeting.wav --speakers 2 --device cuda --format mp3 -y -v
```

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
audio ─▶ [denoise] clean vocals ─▶ [VAD] speech regions
      ─▶ [window + embed] voiceprints ─▶ [cluster] speaker per window
      ─▶ [smooth/merge] segments ─▶ [enhance] per speaker
      ─▶ per-speaker tracks (chosen format) + timeline files
```

0. **Noise reduction** removes ambient/background noise so only voices remain
   (noisereduce spectral gating — GPU-accelerated — or a numpy spectral gate).
1. **Voice activity detection** trims silence (Silero neural VAD, or an adaptive
   energy gate).
2. **Embedding** turns overlapping ~1.5 s windows into speaker "voiceprints"
   (SpeechBrain ECAPA-TDNN on the GPU, or MFCC statistics). Very short windows
   are centre-extended so they don't become clustering outliers.
3. **Clustering** groups windows by speaker (agglomerative, complete linkage,
   cosine distance). With `auto`, it picks 2 vs 3 by silhouette score.
4. **Post-processing** rasterises, smooths short flickers, and merges runs into
   clean segments.
5. **Enhancement** (per speaker): high-pass, presence EQ, gentle compression,
   and loudness normalization (−16 LUFS) so every speaker sits at a consistent,
   clear level.
6. **Export** writes each speaker's track in your chosen format (the CLI prompts
   for format + destination; the GUI has dropdowns).

Tunable parameters live in `diarizer/config.py`.

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
   ├─ vad.py              # voice activity detection (silero + energy)
   ├─ embeddings.py       # speaker voiceprints (ecapa + mfcc), GPU-batched
   ├─ hardware.py         # GPU/CPU device selection + reporting
   ├─ cluster.py          # clustering + auto speaker-count
   ├─ pipeline.py         # orchestration -> segments
   ├─ export.py           # per-speaker audio + timeline files
   └─ logutil.py          # shared verbose logging
```

## Limitations

- Turn-taking assumption: fully overlapping speech isn't un-mixed.
- Accuracy depends on audio quality; noisy/cross-talk-heavy audio is harder.
- Auto speaker-count is a best guess — pass `--speakers 2`/`3` if you know it.
