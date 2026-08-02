"""RVC/UVR HP5-style vocal isolation for noise removal, on the GPU.

Runs a UVR VR-architecture model (the HP family — the same kind RVC ships as
"HP5" for isolating main vocals) via the ``audio-separator`` package. VR models
run on PyTorch, so they use CUDA whenever a CUDA build of torch is installed.
The isolated *Vocals* stem (human voices, background/music/noise removed) is
returned and fed to diarization.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from .logutil import get_logger

log = get_logger()

from .config import MODELS_DIR

_SEPARATOR = None
_LOADED_MODEL = None
_OUT_DIR = None
_MODEL_DIR = os.path.join(MODELS_DIR, "uvr")


def available() -> bool:
    try:
        import audio_separator  # noqa: F401
        return True
    except Exception:
        return False


def free() -> None:
    """Release the RVC/UVR model and free its VRAM (call after denoising, so the
    separation stage has GPU memory to work with)."""
    global _SEPARATOR, _LOADED_MODEL
    _SEPARATOR = None
    _LOADED_MODEL = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _ensure_out_dir() -> str:
    """A stable scratch dir for stems (reused across chunks; not deleted mid-run)."""
    global _OUT_DIR
    if _OUT_DIR is None or not os.path.isdir(_OUT_DIR):
        _OUT_DIR = tempfile.mkdtemp(prefix="derization_rvc_")
    return _OUT_DIR


def _get_separator(model_file: str):
    """Return a cached Separator with ``model_file`` loaded, writing to a stable dir."""
    global _SEPARATOR, _LOADED_MODEL
    from audio_separator.separator import Separator

    os.makedirs(_MODEL_DIR, exist_ok=True)
    if _SEPARATOR is None:
        _SEPARATOR = Separator(output_dir=_ensure_out_dir(), model_file_dir=_MODEL_DIR,
                               output_format="WAV", log_level=logging.WARNING)
    if _LOADED_MODEL != model_file:
        log.info("Loading UVR/RVC HP5 model '%s' (first time downloads it)...", model_file)
        _SEPARATOR.load_model(model_filename=model_file)
        _LOADED_MODEL = model_file
    return _SEPARATOR


def isolate_vocals(samples: np.ndarray, sr: int, cfg, device: str = "cuda",
                   progress=None) -> np.ndarray:
    """Return a vocals-only version of ``samples`` (same sr; channels matched).

    Long files are processed in chunks (with a short crossfade) so audio isn't
    loaded entirely into RAM and progress is logged per chunk. In GPU-only mode
    this raises if CUDA torch is not present (rather than letting
    audio-separator quietly run on the CPU).
    """
    if getattr(cfg, "gpu_only", False):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU-only mode: RVC HP5 needs a CUDA GPU, none available.")

    n = samples.shape[0]
    chunk_sec = float(getattr(cfg, "rvc_chunk_sec", 0) or 0)
    chunk = int(chunk_sec * sr)

    # Short enough (or chunking disabled) -> one pass.
    if chunk <= 0 or n <= int(chunk * 1.25):
        if progress is not None:
            progress("Isolating vocals (RVC HP5)...", 0.05)
        out = _separate_array(samples, sr, cfg)
        if progress is not None:
            progress("Vocal isolation complete.", 1.0)
        return out

    xf = max(1, int(0.3 * sr))          # 300 ms crossfade between chunks
    hop = max(1, chunk - xf)
    starts = list(range(0, n, hop))
    total = len(starts)
    log.info("RVC HP5: %.0f min file -> %d chunk(s) of ~%.0fs (GPU, crossfaded)",
             n / sr / 60.0, total, chunk_sec)

    result = np.zeros(samples.shape, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    for i, s0 in enumerate(starts):
        s1 = min(n, s0 + chunk)
        log.info("  RVC HP5 chunk %d/%d  (%.0f-%.0fs)...", i + 1, total, s0 / sr, s1 / sr)
        if progress is not None:
            progress(f"Isolating vocals (RVC HP5) - chunk {i + 1}/{total}",
                     i / max(1, total))
        piece = _separate_array(samples[s0:s1], sr, cfg)
        length = min(piece.shape[0], s1 - s0)
        w = np.ones(length, dtype=np.float32)
        if s0 > 0:
            r = min(xf, length)
            w[:r] *= np.linspace(0.0, 1.0, r, dtype=np.float32)
        if s1 < n:
            r = min(xf, length)
            w[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
        if samples.ndim == 1:
            result[s0:s0 + length] += piece[:length] * w
        else:
            result[s0:s0 + length] += piece[:length] * w[:, None]
        weight[s0:s0 + length] += w
        if s1 >= n:
            break

    weight[weight == 0] = 1.0
    if samples.ndim == 1:
        result /= weight
    else:
        result /= weight[:, None]
    return result


def _separate_array(samples: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Run the UVR model on one array; return the vocals stem matched to input."""
    import contextlib
    import glob
    import uuid

    import soundfile as sf

    out_dir = _ensure_out_dir()
    sep = _get_separator(cfg.rvc_hp5_model)
    stem = "seg_" + uuid.uuid4().hex[:10]
    in_path = os.path.join(out_dir, stem + ".wav")
    sf.write(in_path, np.clip(samples, -1.0, 1.0), sr, subtype="PCM_16")
    try:
        # Silence audio-separator's tqdm progress bars (they go to stderr);
        # our own per-chunk log line is the progress indicator.
        with open(os.devnull, "w") as _null, contextlib.redirect_stderr(_null):
            outputs = sep.separate(in_path)
        vocal_path = None
        for name in outputs:
            p = name if os.path.isabs(name) else os.path.join(out_dir, name)
            if "vocal" in os.path.basename(p).lower() and os.path.exists(p):
                vocal_path = p
                break
        if vocal_path is None:  # fall back to scanning the dir for this segment
            for p in glob.glob(os.path.join(out_dir, stem + "*")):
                if "vocal" in os.path.basename(p).lower():
                    vocal_path = p
                    break
        if vocal_path is None or not os.path.exists(vocal_path):
            raise RuntimeError("RVC HP5 did not produce a Vocals stem")
        voc, vsr = sf.read(vocal_path, dtype="float32", always_2d=False)
    finally:
        for p in glob.glob(os.path.join(out_dir, stem + "*")):
            try:
                os.remove(p)
            except OSError:
                pass

    voc = _match_channels(voc, samples)
    if vsr != sr:
        voc = _resample(voc, vsr, sr)
    # Guard against off-by-a-few-samples so the caller's indexing stays aligned.
    target = samples.shape[0]
    if voc.shape[0] < target:
        pad = target - voc.shape[0]
        voc = (np.pad(voc, (0, pad)) if voc.ndim == 1
               else np.pad(voc, ((0, pad), (0, 0))))
    elif voc.shape[0] > target:
        voc = voc[:target]
    return voc.astype(np.float32, copy=False)


def _match_channels(voc: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref_mono = ref.ndim == 1
    if ref_mono and voc.ndim == 2:
        return voc.mean(axis=1)
    if not ref_mono and voc.ndim == 1:
        return np.column_stack([voc, voc])
    return voc


def _resample(voc: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from .audioio import resample
    if voc.ndim == 1:
        return resample(voc, sr_in, sr_out)
    return np.stack([resample(voc[:, c], sr_in, sr_out) for c in range(voc.shape[1])],
                    axis=1)
