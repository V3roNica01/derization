"""Audio loading, resampling and saving.

Design goals:
  * Load common formats robustly (WAV/FLAC/OGG always; MP3/M4A when the
    installed backends support them).
  * Keep the *original* full-resolution audio for export, while producing a
    16 kHz mono copy for analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import TARGET_SR
from .logutil import get_logger

log = get_logger()


@dataclass
class LoadedAudio:
    """Container for a decoded file.

    ``samples`` is float32 in ``[-1, 1]`` with shape ``(n,)`` for mono or
    ``(n, channels)`` for multi-channel, at ``sr`` Hz. ``mono16k`` is the
    analysis copy: float32 mono at 16 kHz.
    """

    samples: np.ndarray
    sr: int
    mono16k: np.ndarray
    path: str

    @property
    def duration_sec(self) -> float:
        return self.samples.shape[0] / float(self.sr)

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]


def load_audio(path: str | Path) -> LoadedAudio:
    """Decode ``path`` and build the 16 kHz mono analysis copy."""
    path = str(path)
    samples, sr = _decode(path)
    samples = samples.astype(np.float32, copy=False)

    mono = _to_mono(samples)
    mono16k = resample(mono, sr, TARGET_SR)

    log.info("Loaded '%s': %.2fs, %d Hz, %d channel(s)",
             Path(path).name, samples.shape[0] / sr, sr, 1 if samples.ndim == 1 else samples.shape[1])
    return LoadedAudio(samples=samples, sr=sr, mono16k=mono16k, path=path)


# Compressed formats we prefer to decode with ffmpeg: broader codec support
# and it avoids libsndfile/mpg123 spewing "id3" tag warnings to the console.
_LOSSY_EXTS = {".mp3", ".m4a", ".aac", ".wma", ".mp4"}


def _decode(path: str) -> Tuple[np.ndarray, int]:
    """Decode ``path`` to (samples, sr). Uses ffmpeg for compressed formats,
    libsndfile (soundfile) for WAV/FLAC/OGG, with sensible fallbacks."""
    ext = Path(path).suffix.lower()
    last_err: Exception | None = None

    if ext in _LOSSY_EXTS:
        try:
            return _decode_ffmpeg(path)
        except Exception as exc:
            last_err = exc
            log.debug("ffmpeg decode failed for '%s' (%s); trying soundfile", path, exc)

    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        return data, int(sr)
    except Exception as exc:  # unsupported codec, missing lib, etc.
        last_err = exc
        log.debug("soundfile could not read '%s' (%s); trying librosa", path, exc)

    try:
        import librosa
        # sr=None keeps the native sample rate; mono=False preserves channels.
        data, sr = librosa.load(path, sr=None, mono=False)
        # librosa returns (channels, n) for multi-channel -> transpose to (n, channels).
        if data.ndim == 2:
            data = data.T
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception as exc:
        last_err = exc
        log.debug("librosa could not read '%s' (%s); trying ffmpeg", path, exc)

    try:  # last resort for anything else ffmpeg understands
        return _decode_ffmpeg(path)
    except Exception:
        raise RuntimeError(
            f"Could not decode audio file '{path}'. Convert it to WAV and try "
            f"again. Underlying error: {last_err}"
        ) from last_err


def _decode_ffmpeg(path: str) -> Tuple[np.ndarray, int]:
    """Decode via ffmpeg (system, or the bundled imageio-ffmpeg binary)."""
    import os
    import subprocess
    import tempfile

    import soundfile as sf

    from .formats import find_ffmpeg

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not available")
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "decoded.wav")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", path, wav]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(wav):
            raise RuntimeError((proc.stderr or "ffmpeg failed").strip()[:300])
        data, sr = sf.read(wav, dtype="float32", always_2d=False)
    return data, int(sr)


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1).astype(np.float32)


def resample(mono: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a 1-D signal. Uses librosa/soxr if available, else linear interp."""
    if sr_in == sr_out:
        return mono.astype(np.float32, copy=False)

    try:
        import librosa
        return librosa.resample(mono.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)
    except Exception:
        pass

    # Fallback: simple linear interpolation (adequate for VAD/embedding input).
    duration = mono.shape[0] / float(sr_in)
    n_out = int(round(duration * sr_out))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, duration, num=mono.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, mono).astype(np.float32)


def save_wav(path: str | Path, samples: np.ndarray, sr: int) -> None:
    """Write a WAV file (16-bit PCM). Accepts mono (n,) or multi-channel (n, ch)."""
    import soundfile as sf

    path = str(path)
    data = np.clip(samples, -1.0, 1.0)
    sf.write(path, data, sr, subtype="PCM_16")
    log.debug("Wrote %s", path)
