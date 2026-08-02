"""True source separation of overlapping speech (SepFormer, GPU).

For each overlap span found by diarization, un-mix the two simultaneous voices
with SepFormer, then use ECAPA speaker embeddings to route each separated
source to the correct speaker (matched to that speaker's voiceprint centroid).
Spans where the two sources cannot be confidently told apart are left deleted
(silent in both tracks) - the "delete if unsure" fallback.

SepFormer is a 2-speaker, 8 kHz model; audio is resampled around it. It is
memory-light on the GPU (a few hundred MB), so spans are processed in short
sub-chunks to bound VRAM.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .audioio import LoadedAudio, resample
from .config import MODELS_DIR, DiarizationConfig
from .embeddings import SpeakerEmbedder
from .hardware import resolve_device
from .logutil import get_logger

log = get_logger()

SEP_SR = 8000
_SEPFORMER = None
_SEP_DEVICE = None


def available() -> bool:
    try:
        import speechbrain  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _load_sepformer(cfg: DiarizationConfig, device: str):
    global _SEPFORMER, _SEP_DEVICE
    if _SEPFORMER is not None and _SEP_DEVICE == device:
        return _SEPFORMER

    from .embeddings import _patch_speechbrain_lazy_imports
    _patch_speechbrain_lazy_imports()
    from speechbrain.inference.separation import SepformerSeparation

    extra = {}
    try:
        from speechbrain.utils.fetching import LocalStrategy
        extra["local_strategy"] = LocalStrategy.COPY   # Windows: no symlinks
    except Exception:
        pass
    sb_device = "cuda:0" if device == "cuda" else device
    savedir = os.path.join(MODELS_DIR, "sepformer")
    log.info("Loading SepFormer separation model '%s' on %s (first time downloads it)...",
             cfg.sepformer_model, sb_device.upper())
    _SEPFORMER = SepformerSeparation.from_hparams(
        source=cfg.sepformer_model, savedir=savedir,
        run_opts={"device": sb_device}, **extra)
    _SEP_DEVICE = device
    return _SEPFORMER


def separate_overlaps(audio: LoadedAudio, result, cfg: DiarizationConfig,
                      device: Optional[str] = None, progress=None):
    """Un-mix overlap spans into per-speaker audio.

    Returns ``(injections, separated_sec, deleted_sec)`` where ``injections`` is
    ``{speaker_id: [(start_sec, end_sec, audio_at_source_sr)]}`` (audio channels
    match the source recording).
    """
    spans = getattr(result, "overlap_spans", None)
    centroids = getattr(result, "centroids", None)
    if not spans or centroids is None or centroids.shape[0] < 2:
        return {}, 0.0, 0.0

    device = device or resolve_device(cfg.device, cfg.gpu_only)
    import torch

    model = _load_sepformer(cfg, device)
    embedder = SpeakerEmbedder(cfg)                      # cached ECAPA
    cents = centroids.astype(np.float32)
    sr = audio.sr
    src_mono = audio.samples if audio.samples.ndim == 1 else audio.samples.mean(axis=1)
    stereo = audio.samples.ndim == 2

    injections: dict[int, list] = {}
    sep_sec = 0.0
    del_sec = 0.0
    reasons = {"same": 0, "low": 0, "short": 0}   # why chunks were deleted
    best_margins = []
    sub = max(1.0, float(cfg.sep_chunk_sec))
    total = len(spans)

    for si, (t0, t1) in enumerate(spans):
        if progress is not None:
            progress(f"Un-mixing overlap {si + 1}/{total}", si / max(1, total))
        u = t0
        while u < t1 - 1e-3:
            v = min(t1, u + sub)
            payload, reason, mm = _separate_chunk(model, embedder, src_mono, sr,
                                                  u, v, cents, cfg, device, torch)
            dur = v - u
            if mm is not None:
                best_margins.append(mm)
            if reason == "ok":
                spk0, sig0, spk1, sig1 = payload
                for spk, sig in ((spk0, sig0), (spk1, sig1)):
                    sig = _to_channels(sig, stereo)
                    injections.setdefault(spk, []).append((u, v, sig))
                sep_sec += dur
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
                del_sec += dur
            u = v

    mm_txt = ""
    if best_margins:
        arr = np.array(best_margins)
        mm_txt = (f" | source-assignment margins: median {np.median(arr):.2f}, "
                  f"max {arr.max():.2f} (need >= {cfg.overlap_assign_margin:.2f})")
    log.info("Overlap separation: %.1fs un-mixed, %.1fs deleted "
             "[same-speaker=%d, low-confidence=%d, too-short=%d]%s",
             sep_sec, del_sec, reasons["same"], reasons["low"], reasons["short"], mm_txt)
    return injections, sep_sec, del_sec


def _separate_chunk(model, embedder, src_mono, sr, t0, t1, cents, cfg, device, torch):
    """Separate one sub-chunk. Returns (payload, reason, best_margin) where
    reason is 'ok' / 'same' / 'low' / 'short'."""
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(src_mono.shape[0], int(round(t1 * sr)))
    if i1 - i0 < int(0.2 * sr):
        return None, "short", None
    mix = src_mono[i0:i1].astype(np.float32)
    mix8 = resample(mix, sr, SEP_SR)
    if mix8.shape[0] < int(0.2 * SEP_SR):
        return None, "short", None

    with torch.no_grad():
        est = model.separate_batch(
            torch.from_numpy(np.ascontiguousarray(mix8))[None].to(
                "cuda:0" if device == "cuda" else device))
    est = est[0].detach().cpu().numpy()                 # (T, 2)
    s0, s1 = est[:, 0], est[:, 1]

    e0 = _embed(embedder, s0)
    e1 = _embed(embedder, s1)
    if e0 is None or e1 is None:
        return None, "short", None
    sims0 = cents @ e0
    sims1 = cents @ e1
    a0, m0 = _best(sims0)
    a1, m1 = _best(sims1)
    mm = float(min(m0, m1))
    margin = float(cfg.overlap_assign_margin)
    if a0 == a1:
        return None, "same", mm                          # both map to one speaker
    if m0 < margin or m1 < margin:
        return None, "low", mm                           # too fuzzy to be sure

    # Scale each isolated voice to the mixture level, then back to source sr.
    mix_rms = float(np.sqrt(np.mean(mix8 ** 2)) + 1e-9)
    sig0 = _scale(s0, mix_rms)
    sig1 = _scale(s1, mix_rms)
    sig0 = resample(sig0, SEP_SR, sr)
    sig1 = resample(sig1, SEP_SR, sr)
    return (int(a0), sig0, int(a1), sig1), "ok", mm


def _embed(embedder, x8):
    x16 = resample(x8.astype(np.float32), SEP_SR, 16000)
    if x16.shape[0] < int(0.2 * 16000):
        return None
    w = embedder.embed_windows([(0.0, x16.shape[0] / 16000.0, x16)])
    if not w:
        return None
    return w[0].embedding.astype(np.float32)


def _best(sims):
    order = np.argsort(sims)
    top = int(order[-1])
    margin = float(sims[order[-1]] - sims[order[-2]]) if sims.shape[0] > 1 else float(sims[top])
    return top, margin


def _scale(x, target_rms):
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    g = min(8.0, target_rms / rms)
    return (x * g).astype(np.float32)


def _to_channels(sig, stereo):
    if stereo and sig.ndim == 1:
        return np.column_stack([sig, sig])
    return sig
