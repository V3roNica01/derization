"""Residual cross-talk gate.

Segment-boundary masking keeps a speaker's whole segment audible, but a brief
foreign sound *inside* that segment - a laugh, a short interjection that isn't
simultaneous speech - is never flagged as overlap (the overlap detector only
looks for two voices at once) and so leaks into the track.

This pass re-checks each speaker's own segments at fine resolution against the
speaker voiceprints and returns the intervals that clearly belong to a
*different* speaker, so the exporter can silence them - the same "delete if
we're sure it's someone else" philosophy applied to non-overlapping bleed.

Runs on the GPU via the cached ECAPA embedder; windows are embedded in one
batch, so even long recordings add only a few seconds.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import TARGET_SR
from .embeddings import SpeakerEmbedder
from .logutil import get_logger

log = get_logger()


def gate_crosstalk(mono16k: np.ndarray, result, cfg, device=None,
                   progress=None) -> Dict[int, List[Tuple[float, float]]]:
    """Return ``{speaker_id: [(start_sec, end_sec), ...]}`` intervals to silence
    because they embed as a *different* speaker than the segment they sit in."""
    if not getattr(cfg, "crosstalk_gate", False):
        return {}
    cents = getattr(result, "centroids", None)
    if cents is None or cents.shape[0] < 2 or not result.segments:
        return {}

    win = float(cfg.crosstalk_win)
    hop = float(cfg.crosstalk_hop)
    margin = float(cfg.crosstalk_margin)
    min_rms = float(cfg.crosstalk_min_rms)
    dur_total = mono16k.shape[0] / TARGET_SR
    cents = cents.astype(np.float32)

    # Build fine windows, each tagged with the speaker of the segment it sits in.
    tagged: List[Tuple[float, float, int]] = []          # (kill_start, kill_end, spk)
    audio_wins: List[Tuple[float, float, np.ndarray]] = []
    for seg in result.segments:
        t = seg.start
        while t < seg.end - 1e-3:
            c = min(seg.end, t + win * 0.5)              # window centre
            a0 = max(0.0, c - win / 2)
            a1 = min(dur_total, a0 + win)
            a0 = max(0.0, a1 - win)
            a = _slice(mono16k, a0, a1)
            if a.size > 0 and float(np.sqrt(np.mean(a ** 2))) >= min_rms:
                # Kill only the central hop-width slice: window edges that straddle
                # a real->foreign transition are ambiguous and left to the speaker.
                k0 = max(seg.start, c - hop / 2)
                k1 = min(seg.end, c + hop / 2)
                if k1 > k0:
                    tagged.append((k0, k1, seg.speaker))
                    audio_wins.append((a0, a1, a))
            t += hop

    if not audio_wins:
        return {}
    if progress is not None:
        progress("Cross-talk gate: checking segments", 0.0)

    embedder = SpeakerEmbedder(cfg)
    wins = embedder.embed_windows(audio_wins)

    kill: Dict[int, List[Tuple[float, float]]] = {}
    gated = 0.0
    for (k0, k1, spk), w in zip(tagged, wins):
        e = w.embedding.astype(np.float32)
        sims = cents @ e
        best = int(np.argmax(sims))
        if best != spk and float(sims[best] - sims[spk]) >= margin:
            kill.setdefault(spk, []).append((k0, k1))
            gated += (k1 - k0)

    merged = {spk: _merge(iv) for spk, iv in kill.items()}
    if gated > 0:
        log.info("Cross-talk gate: silenced %.1fs of foreign voice across %d "
                 "speaker track(s)", gated, len(merged))
    return merged


def _slice(mono16k: np.ndarray, start: float, end: float) -> np.ndarray:
    i0 = max(0, int(round(start * TARGET_SR)))
    i1 = min(mono16k.shape[0], int(round(end * TARGET_SR)))
    return mono16k[i0:i1]


def _merge(intervals: List[Tuple[float, float]],
           gap: float = 0.06) -> List[Tuple[float, float]]:
    """Merge intervals that touch or are within ``gap`` seconds of each other."""
    out: List[Tuple[float, float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out
