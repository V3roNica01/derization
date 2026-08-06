"""Optional state-of-the-art diarization backend (pyannote.audio 3.1).

pyannote's pretrained pipeline is more accurate than the built-in engine and
detects overlapping speech natively. It is **opt-in** because it needs:

  1. ``pip install pyannote.audio``
  2. Accepting the licences for ``pyannote/speaker-diarization-3.1`` and
     ``pyannote/segmentation-3.0`` on Hugging Face (free).
  3. A Hugging Face access token (set ``HF_TOKEN`` or ``cfg.hf_token``).

Every entry point is guarded: if the package, licence, or token is missing,
:func:`diarize_pyannote` returns ``None`` and the caller falls back to the
self-contained built-in engine, so the app always runs.

pyannote only provides the *timeline* (who spoke when + overlaps). Speaker
voiceprint centroids - needed by the overlap-separation and cross-talk stages -
are computed here with the same ECAPA embedder the built-in engine uses, so the
downstream pipeline is identical regardless of which backend produced the
segments.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .config import TARGET_SR, DiarizationConfig
from .embeddings import SpeakerEmbedder, build_windows, _l2_normalize
from .logutil import get_logger
from .pipeline import DiarizationResult, Segment, _canonicalize_speaker_ids
from .vad import SpeechSegment

log = get_logger()

_PIPELINE = None
_PIPE_KEY = None


def available() -> bool:
    try:
        import pyannote.audio  # noqa: F401
        return True
    except Exception:
        return False


def get_token(cfg: DiarizationConfig) -> Optional[str]:
    """Resolve the Hugging Face token from the config or the environment."""
    tok = getattr(cfg, "hf_token", None)
    if tok:
        return tok
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    return None


def should_use(cfg: DiarizationConfig) -> bool:
    """Whether the pyannote backend is selected *and* usable right now."""
    backend = getattr(cfg, "diarization_backend", "builtin")
    if backend == "builtin":
        return False
    if not available():
        if backend == "pyannote":
            log.warning("diarization_backend='pyannote' but pyannote.audio is not "
                        "installed; using the built-in engine.")
        return False
    if get_token(cfg) is None:
        if backend == "pyannote":
            log.warning("diarization_backend='pyannote' but no Hugging Face token "
                        "was found (set HF_TOKEN); using the built-in engine.")
        return False
    return True


def _load_pipeline(cfg: DiarizationConfig, device: str):
    global _PIPELINE, _PIPE_KEY
    key = (cfg.pyannote_model, device)
    if _PIPELINE is not None and _PIPE_KEY == key:
        return _PIPELINE

    import torch
    from pyannote.audio import Pipeline

    log.info("Loading pyannote pipeline '%s' on %s (first time downloads it)...",
             cfg.pyannote_model, device.upper())
    pipe = Pipeline.from_pretrained(cfg.pyannote_model, use_auth_token=get_token(cfg))
    if pipe is None:
        # from_pretrained returns None when the licence hasn't been accepted.
        raise RuntimeError(
            "pyannote returned no pipeline - accept the model licences at "
            f"https://hf.co/{cfg.pyannote_model} and check your token.")
    pipe.to(torch.device("cuda" if device == "cuda" else "cpu"))
    _PIPELINE, _PIPE_KEY = pipe, key
    return pipe


def diarize_pyannote(audio, cfg: DiarizationConfig, device: str,
                     progress=None) -> Optional[DiarizationResult]:
    """Diarize with pyannote. Returns ``None`` if it can't run (caller falls back)."""
    if not should_use(cfg):
        return None
    try:
        return _run(audio, cfg, device, progress)
    except Exception as exc:                       # never break the pipeline
        log.warning("pyannote diarization failed (%s); falling back to the "
                    "built-in engine.", exc)
        return None


def _run(audio, cfg, device, progress):
    import torch

    def report(msg, frac):
        log.info(msg)
        if progress is not None:
            progress(msg, frac)

    report("Diarizing with pyannote (state-of-the-art)...", 0.15)
    pipe = _load_pipeline(cfg, device)

    mono16k = audio.mono16k.astype(np.float32)
    duration = mono16k.shape[0] / TARGET_SR
    waveform = torch.from_numpy(mono16k)[None, :]          # (1, samples)
    args = {"waveform": waveform, "sample_rate": TARGET_SR}

    kw = {}
    if cfg.num_speakers:
        kw["num_speakers"] = int(cfg.num_speakers)
    else:
        kw["min_speakers"] = int(cfg.min_speakers)
        kw["max_speakers"] = int(cfg.max_speakers)

    report("Running pyannote diarization...", 0.30)
    ann = pipe(args, **kw)

    # Overlap timeline (regions where >=2 speakers talk at once).
    overlap_spans = [(float(s.start), float(s.end)) for s in ann.get_overlap()]

    # Raw turns -> provisional integer speaker ids (encounter order).
    label_ids: dict = {}
    raw: List[Segment] = []
    for turn, _track, label in ann.itertracks(yield_label=True):
        sid = label_ids.setdefault(label, len(label_ids))
        raw.append(Segment(float(turn.start), float(turn.end), sid))
    if not raw:
        return DiarizationResult([], 0, duration, audio.sr, audio.channels,
                                 "ecapa", "pyannote")

    # Remove overlap regions from each speaker's segments so simultaneous speech
    # is deleted (or re-injected by the separation stage) rather than bleeding
    # into both tracks - the same rule the built-in engine follows.
    segs = _subtract_overlaps(raw, overlap_spans)
    segs = [s for s in segs if s.duration > 1e-3]
    segs.sort(key=lambda s: s.start)
    segs, mapping = _canonicalize_speaker_ids(segs)

    report("Extracting speaker voiceprints...", 0.80)
    centroids = _centroids(mono16k, segs, cfg)

    result = DiarizationResult(
        segments=segs,
        num_speakers=len({s.speaker for s in segs}),
        duration=duration,
        source_sr=audio.sr,
        source_channels=audio.channels,
        embedding_backend="ecapa",
        vad_backend="pyannote",
        overlap_spans=overlap_spans,
        centroids=centroids,
    )
    ov = sum(e - s for s, e in overlap_spans)
    log.info("pyannote: %d speaker(s), %d segment(s), %.1fs overlap",
             result.num_speakers, len(segs), ov)
    report("Diarization complete.", 1.0)
    return result


def _subtract_overlaps(segments: List[Segment],
                       overlaps: List[tuple]) -> List[Segment]:
    """Cut every overlap span out of every segment, returning the pieces."""
    if not overlaps:
        return list(segments)
    ov = sorted(overlaps)
    out: List[Segment] = []
    for seg in segments:
        pieces = [(seg.start, seg.end)]
        for os_, oe in ov:
            nxt = []
            for ps, pe in pieces:
                if oe <= ps or os_ >= pe:            # no intersection
                    nxt.append((ps, pe))
                    continue
                if ps < os_:
                    nxt.append((ps, os_))
                if oe < pe:
                    nxt.append((oe, pe))
            pieces = nxt
        for ps, pe in pieces:
            if pe - ps > 1e-3:
                out.append(Segment(ps, pe, seg.speaker))
    return out


def _centroids(mono16k: np.ndarray, segments: List[Segment],
               cfg: DiarizationConfig) -> Optional[np.ndarray]:
    """Mean ECAPA voiceprint per (canonical) speaker, L2-normalised."""
    ids = sorted({s.speaker for s in segments})
    if not ids:
        return None
    embedder = SpeakerEmbedder(cfg)
    vecs = []
    for spk in ids:
        spk_segs = [SpeechSegment(s.start, s.end) for s in segments if s.speaker == spk]
        raw = build_windows(mono16k, spk_segs, cfg)
        wins = embedder.embed_windows(raw) if raw else []
        if wins:
            m = np.mean([w.embedding for w in wins], axis=0)
        else:                                        # speaker with no usable window
            m = np.zeros(192, dtype=np.float32)      # ECAPA-TDNN embedding dim
        vecs.append(m.astype(np.float32))
    return _l2_normalize(np.vstack(vecs))
