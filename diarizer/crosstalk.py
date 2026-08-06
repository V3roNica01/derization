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
    keep_margin = float(cfg.crosstalk_keep_margin)
    floor_ratio = float(cfg.crosstalk_self_floor)
    min_rms = float(cfg.crosstalk_min_rms)
    embed_win = max(win, 0.9)                             # >= 0.9s keeps ECAPA stable
    dur_total = mono16k.shape[0] / TARGET_SR
    cents = cents.astype(np.float32)

    # Build fine windows, each tagged with the speaker of the segment it sits in.
    tagged: List[Tuple[float, float, int]] = []          # (kill_start, kill_end, spk)
    audio_wins: List[Tuple[float, float, np.ndarray]] = []
    for seg in result.segments:
        t = seg.start
        while t < seg.end - 1e-3:
            c = min(seg.end, t + hop * 0.5)              # kill-slice centre
            a0 = max(0.0, c - embed_win / 2)             # embedder gets a wider,
            a1 = min(dur_total, a0 + embed_win)          # centre-extended window
            a0 = max(0.0, a1 - embed_win)
            a = _slice(mono16k, a0, a1)
            if a.size > 0 and float(np.sqrt(np.mean(a ** 2))) >= min_rms:
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

    # Cosine similarity of every window to every speaker centroid.
    emb = np.vstack([w.embedding for w in wins]).astype(np.float32)   # (N, D)
    sims = emb @ cents.T                                              # (N, K)
    spks = np.array([t[2] for t in tagged])
    idx = np.arange(len(spks))
    sim_self = sims[idx, spks]                                        # own speaker
    other = sims.copy()
    other[idx, spks] = -1e9
    sim_other = other.max(axis=1)                                    # best other

    # Per-speaker self-similarity floor: a fraction of that speaker's *median*
    # window match (robust to the contaminating windows we're trying to remove).
    floor = np.zeros(len(spks), dtype=np.float32)
    for spk in np.unique(spks):
        m = spks == spk
        base = float(np.median(sim_self[m])) if m.any() else 0.0
        floor[m] = floor_ratio * max(base, 0.0)

    # Silence a window if its own speaker doesn't clearly win (foreign voice) OR
    # it barely matches its own speaker at all (laughter / non-speech).
    foreign = (sim_self - sim_other) < keep_margin
    weak = sim_self < floor
    drop = foreign | weak

    # Optional second-speaker pass (max-aggression): a simultaneous backchannel
    # leaves a small "residual" lean toward the other speaker even when the loud
    # owner dominates the window. Remove the owner's linear component, then flag
    # windows whose residual toward the best-other centroid is a robust outlier
    # vs that speaker's own baseline.
    second = np.zeros(len(spks), dtype=bool)
    if getattr(cfg, "crosstalk_second_speaker", False):
        cc = cents @ cents.T                                          # (K, K)
        other_idx = other.argmax(axis=1)
        cos_so = cc[spks, other_idx]
        res_other = sim_other - sim_self * cos_so                     # residual lean
        factor = float(cfg.crosstalk_second_factor)
        for spk in np.unique(spks):
            m = spks == spk
            v = res_other[m]
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med))) + 1e-6
            second[m] = v > (med + factor * 1.4826 * mad)
        drop = drop | second

    kill: Dict[int, List[Tuple[float, float]]] = {}
    gated = 0.0
    for i in np.nonzero(drop)[0]:
        k0, k1, spk = tagged[i]
        kill.setdefault(int(spk), []).append((k0, k1))
        gated += (k1 - k0)

    merged = {spk: _merge(iv) for spk, iv in kill.items()}
    if gated > 0:
        log.info("Cross-talk gate: silenced %.1fs (%d foreign, %d weak-match, "
                 "%d second-speaker) across %d speaker track(s)", gated,
                 int(foreign.sum()), int((weak & ~foreign).sum()),
                 int((second & ~foreign & ~weak).sum()), len(merged))
    return merged


def deterministic_kills(result, cfg) -> Dict[int, List[Tuple[float, float]]]:
    """Time-based anti-bleed that doesn't rely on hearing the quiet voice:
    dilate detected overlaps into neighbouring segments, and guard the edges of
    every segment that borders a DIFFERENT speaker (the turn-change collision
    zone). Returns per-speaker intervals to silence."""
    dilate = float(getattr(cfg, "overlap_dilate_sec", 0.0) or 0.0)
    guard = float(getattr(cfg, "boundary_guard_sec", 0.0) or 0.0)
    kill: Dict[int, List[Tuple[float, float]]] = {}
    if dilate <= 0.0 and guard <= 0.0:
        return kill
    speakers = sorted({s.speaker for s in result.segments})

    # Overlap dilation: widen each deleted overlap span into every track.
    if dilate > 0.0:
        for s, e in getattr(result, "overlap_spans", None) or []:
            for spk in speakers:
                kill.setdefault(spk, []).append((s - dilate, e + dilate))

    # Boundary guards: silence the start/end of a segment when the adjacent
    # speech belongs to another speaker.
    if guard > 0.0:
        segs = sorted(result.segments, key=lambda s: s.start)
        for i, seg in enumerate(segs):
            prev = segs[i - 1] if i > 0 else None
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            if prev is not None and prev.speaker != seg.speaker:
                kill.setdefault(seg.speaker, []).append((seg.start, seg.start + guard))
            if nxt is not None and nxt.speaker != seg.speaker:
                kill.setdefault(seg.speaker, []).append((seg.end - guard, seg.end))
    return {spk: _merge(iv) for spk, iv in kill.items()}


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
