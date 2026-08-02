"""Save audio in the user's chosen format.

  * WAV / FLAC / OGG  -> written directly by libsndfile (soundfile).
  * MP3 / AAC (.m4a)  -> encoded with ffmpeg. Uses a system ffmpeg if present,
    otherwise the static binary bundled by the ``imageio-ffmpeg`` package.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .logutil import get_logger

log = get_logger()

SNDFILE_FORMATS = {"wav", "flac", "ogg"}
FFMPEG_FORMATS = {"mp3", "aac"}
_EXT = {"wav": ".wav", "flac": ".flac", "ogg": ".ogg", "mp3": ".mp3", "aac": ".m4a"}


def normalize_format(fmt: str) -> str:
    fmt = (fmt or "wav").lower().lstrip(".")
    if fmt in ("m4a", "aac"):
        return "aac"
    return fmt


def ext_for(fmt: str) -> str:
    return _EXT.get(normalize_format(fmt), ".wav")


def find_ffmpeg() -> str | None:
    """Locate an ffmpeg executable (system PATH, else imageio-ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ffmpeg_maybe_available() -> bool:
    """True if ffmpeg is (or can be) available, without forcing a download."""
    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg  # noqa: F401
        return True
    except Exception:
        return False


def supported_formats() -> list[str]:
    fmts = ["wav", "flac", "ogg"]
    if _ffmpeg_maybe_available():
        fmts += ["mp3", "aac"]
    return fmts


def save_audio(base_path: str | Path, samples: np.ndarray, sr: int, cfg) -> str:
    """Write ``samples`` next to ``base_path`` using ``cfg.export_format``.

    ``base_path`` has no extension; the correct one is appended. Returns the
    path actually written.
    """
    import soundfile as sf

    fmt = normalize_format(cfg.export_format)
    base = Path(base_path)
    data = np.clip(samples, -1.0, 1.0)

    if fmt in SNDFILE_FORMATS:
        out = base.with_suffix(_EXT[fmt])
        subtype = "PCM_16" if fmt == "wav" else None
        sf.write(str(out), data, sr, subtype=subtype)
        return str(out)

    if fmt in FFMPEG_FORMATS:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            log.warning("ffmpeg not available for %s export; writing WAV instead", fmt)
            out = base.with_suffix(".wav")
            sf.write(str(out), data, sr, subtype="PCM_16")
            return str(out)
        return _encode_ffmpeg(ffmpeg, base, data, sr, fmt, cfg)

    # Unknown format -> WAV.
    out = base.with_suffix(".wav")
    sf.write(str(out), data, sr, subtype="PCM_16")
    return str(out)


def _encode_ffmpeg(ffmpeg: str, base: Path, data: np.ndarray, sr: int,
                   fmt: str, cfg) -> str:
    import soundfile as sf

    out = base.with_suffix(_EXT[fmt])
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "src.wav"
        sf.write(str(wav), data, sr, subtype="PCM_16")
        bitrate = getattr(cfg, "mp3_bitrate", "192k")
        codec = ["-c:a", "libmp3lame"] if fmt == "mp3" else ["-c:a", "aac"]
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav),
               *codec, "-b:a", bitrate, str(out)]
        log.debug("ffmpeg encode: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"ffmpeg failed to encode {fmt}: {proc.stderr.strip()[:400]}")
    return str(out)
