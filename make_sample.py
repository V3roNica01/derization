#!/usr/bin/env python3
"""Generate a synthetic multi-speaker WAV to test the pipeline end-to-end.

This does NOT produce real speech - it makes 2-3 spectrally distinct
"voice-like" signals (different pitch + formant structure, speech-rate
amplitude modulation) that alternate in turns with short pauses. It's enough to
verify diarization runs and clusters speakers correctly (use the MFCC backend,
which keys on spectral shape).

    python make_sample.py sample.wav --speakers 2 --seconds 30
"""
from __future__ import annotations

import argparse
import numpy as np
import soundfile as sf

SR = 16000

# Each "voice" = fundamental frequency (Hz) + three formant centres (Hz).
VOICES = [
    (110.0, (700, 1220, 2600)),   # low-pitched
    (190.0, (900, 1600, 2900)),   # mid-pitched
    (260.0, (650, 1900, 3200)),   # high-pitched
]


def _formant_env(freqs, formants, bandwidth=140.0):
    """Weight harmonic amplitudes by proximity to formant peaks."""
    env = np.zeros_like(freqs)
    for fc in formants:
        env += np.exp(-0.5 * ((freqs - fc) / bandwidth) ** 2)
    return env


def synth_voice(duration: float, f0: float, formants) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n, dtype=np.float64)

    # Sum harmonics up to Nyquist, shaped by the formant envelope.
    harmonics = np.arange(1, int((SR / 2) / f0))
    freqs = harmonics * f0
    weights = _formant_env(freqs, formants)
    weights /= (weights.max() + 1e-9)
    for h, w in zip(harmonics, weights):
        if w < 0.02:
            continue
        sig += w * np.sin(2 * np.pi * f0 * h * t) / h

    # Syllable-rate amplitude modulation (~4 Hz) so it reads as speech.
    mod = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t + np.random.uniform(0, 6.28)))
    mod = 0.3 + 0.7 * mod
    sig *= mod
    sig /= (np.abs(sig).max() + 1e-9)
    return 0.6 * sig


def build(duration: float, speakers: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    speakers = max(2, min(3, speakers))
    out = []
    t = 0.0
    turn = 0
    while t < duration:
        spk = turn % speakers
        f0, formants = VOICES[spk]
        seg_len = float(rng.uniform(1.5, 3.5))
        out.append(synth_voice(seg_len, f0, formants))
        # short pause between turns
        pause = float(rng.uniform(0.2, 0.5))
        out.append(np.zeros(int(pause * SR)))
        t += seg_len + pause
        turn += 1
    audio = np.concatenate(out).astype(np.float32)
    # light noise floor
    audio += 0.002 * rng.standard_normal(audio.shape).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def main() -> int:
    p = argparse.ArgumentParser(description="Create a synthetic multi-speaker test WAV.")
    p.add_argument("output", nargs="?", default="sample.wav")
    p.add_argument("--speakers", type=int, default=2)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    audio = build(args.seconds, args.speakers, args.seed)
    sf.write(args.output, audio, SR, subtype="PCM_16")
    print(f"Wrote {args.output}: {len(audio)/SR:.1f}s, {args.speakers} synthetic voices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
