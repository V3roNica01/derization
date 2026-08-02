#!/usr/bin/env python3
"""Build a REAL-speech multi-speaker test file to exercise the neural path.

Unlike ``make_sample.py`` (synthetic tones, only good for the MFCC backend),
this stitches together clips from 2-3 *different real speakers* (LibriSpeech
excerpts downloaded once via librosa's example data) into an alternating
conversation. Use it to try the Silero-VAD + ECAPA GPU pipeline:

    python make_real_sample.py real_sample.wav --speakers 2
    python cli.py real_sample.wav --speakers 2 --device cuda -v

The first run downloads a few small clips from librosa.org and caches them.
"""
from __future__ import annotations

import argparse
import numpy as np
import soundfile as sf

SR = 16000
# Three different real speakers from LibriSpeech (via librosa example data).
SPEAKER_EXAMPLES = ["libri1", "libri2", "libri3"]


def load_speaker(name: str) -> np.ndarray:
    import librosa
    path = librosa.example(name)                 # downloads + caches
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return (0.9 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def build(speakers: int, turns: int, turn_sec: float, gap_sec: float) -> np.ndarray:
    speakers = max(2, min(3, speakers))
    clips = [load_speaker(SPEAKER_EXAMPLES[i]) for i in range(speakers)]
    turn_len = int(turn_sec * SR)
    gap = np.zeros(int(gap_sec * SR), dtype=np.float32)
    pos = [0] * speakers
    out = []
    for i in range(turns):
        spk = i % speakers
        c = clips[spk]
        if pos[spk] + turn_len > len(c):
            pos[spk] = 0
        out.append(c[pos[spk]:pos[spk] + turn_len])
        pos[spk] += turn_len
        out.append(gap)
    return np.concatenate(out).astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description="Create a real-speech multi-speaker test WAV.")
    p.add_argument("output", nargs="?", default="real_sample.wav")
    p.add_argument("--speakers", type=int, default=2, help="2 or 3")
    p.add_argument("--turns", type=int, default=8, help="number of alternating turns")
    p.add_argument("--turn-sec", type=float, default=4.0)
    p.add_argument("--gap-sec", type=float, default=0.4)
    args = p.parse_args()

    audio = build(args.speakers, args.turns, args.turn_sec, args.gap_sec)
    sf.write(args.output, audio, SR, subtype="PCM_16")
    print(f"Wrote {args.output}: {len(audio)/SR:.1f}s, {min(3, max(2, args.speakers))} "
          f"real speakers alternating every {args.turn_sec:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
