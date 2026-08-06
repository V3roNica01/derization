"""End-to-end diarization pipeline.

Stages:
  1. Voice activity detection    -> speech regions
  2. Windowing + embedding       -> voiceprints for short windows
  3. Clustering                  -> a speaker label per window
  4. Rasterise + smooth + merge  -> clean per-speaker segments

The result feeds ``export.py`` to produce per-speaker audio and timelines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .audioio import LoadedAudio
from .config import TARGET_SR, DiarizationConfig
from .cluster import OVERLAP_LABEL, cluster_windows
from .embeddings import SpeakerEmbedder, Window, build_windows
from .logutil import get_logger
from .vad import detect_speech

log = get_logger()

# Frame resolution used to rasterise window labels onto the timeline.
FRAME_SEC = 0.01  # 10 ms

ProgressCB = Callable[[str, float], None]


@dataclass
class Segment:
    """A contiguous stretch of one speaker's audio (seconds)."""
    start: float
    end: float
    speaker: int  # 0-based index

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    segments: List[Segment]
    num_speakers: int
    duration: float
    source_sr: int
    source_channels: int
    embedding_backend: str
    vad_backend: str
    # Overlap (simultaneous-speech) time spans in seconds.
    overlap_spans: List[tuple] = field(default_factory=list)
    # Per-speaker (canonical id) mean voiceprint, L2-normalised: (num_speakers, dim).
    centroids: Optional[np.ndarray] = None
    # Filled in later (separation stage): speaker id -> [(start, end, audio)].
    overlap_injections: Optional[dict] = None
    # Filled in later (cross-talk gate): speaker id -> [(start, end)] to silence.
    crosstalk_kill: Optional[dict] = None

    def speaker_time(self) -> dict[int, float]:
        totals: dict[int, float] = {}
        for s in self.segments:
            totals[s.speaker] = totals.get(s.speaker, 0.0) + s.duration
        return dict(sorted(totals.items()))

    def label(self, speaker: int) -> str:
        return f"SPEAKER_{speaker + 1}"


def diarize(audio: LoadedAudio, cfg: DiarizationConfig,
            progress: Optional[ProgressCB] = None) -> DiarizationResult:
    """Run the full pipeline on a :class:`LoadedAudio` object."""
    def report(msg: str, frac: float) -> None:
        log.info(msg)
        if progress is not None:
            progress(msg, frac)

    duration = audio.mono16k.shape[0] / TARGET_SR

    # 1. VAD -----------------------------------------------------------------
    report("Detecting speech regions...", 0.05)
    speech = detect_speech(audio.mono16k, cfg)
    if not speech:
        log.warning("No speech detected in the file.")
        return DiarizationResult([], 0, duration, audio.sr, audio.channels,
                                 "n/a", cfg.vad_backend)

    # 2. Windowing + embeddings ---------------------------------------------
    report("Extracting speaker voiceprints...", 0.20)
    raw_windows = build_windows(audio.mono16k, speech, cfg)
    log.info("Prepared %d analysis window(s)", len(raw_windows))
    embedder = SpeakerEmbedder(cfg)
    windows = embedder.embed_windows(raw_windows)
    report(f"Embedded {len(windows)} window(s) [{embedder.backend}]", 0.65)

    # 3. Clustering ----------------------------------------------------------
    report("Grouping windows by speaker...", 0.75)
    labels, k = cluster_windows(windows, cfg)

    # 4. Rasterise -> smooth -> merge ---------------------------------------
    report("Building diarized timeline...", 0.90)
    frame_labels = _rasterize(windows, labels, k, duration)
    overlap_spans = _spans_from_frames(frame_labels, OVERLAP_LABEL)
    overlap_sec = sum(e - s for s, e in overlap_spans)
    if overlap_sec > 0:
        log.info("Overlapping speech detected: %.1fs across %d span(s)",
                 overlap_sec, len(overlap_spans))
    segments = _labels_to_segments(frame_labels, k, cfg)
    segments, mapping = _canonicalize_speaker_ids(segments)
    centroids = _speaker_centroids(windows, labels, mapping)

    result = DiarizationResult(
        segments=segments,
        num_speakers=len({s.speaker for s in segments}) if segments else 0,
        duration=duration,
        source_sr=audio.sr,
        source_channels=audio.channels,
        embedding_backend=embedder.backend,
        vad_backend=cfg.vad_backend,
        overlap_spans=overlap_spans,
        centroids=centroids,
    )
    report("Diarization complete.", 1.0)
    _log_summary(result)
    return result


# --------------------------------------------------------------------------- #
# Window labels -> frame labels -> segments
# --------------------------------------------------------------------------- #
def _rasterize(windows: List[Window], labels: np.ndarray, k: int,
               duration: float) -> np.ndarray:
    """Vote each 10 ms frame's speaker from all windows covering it.

    Non-speech frames stay -1.
    """
    n_frames = max(1, int(np.ceil(duration / FRAME_SEC)))
    votes = np.zeros((n_frames, max(1, k)), dtype=np.float32)
    overlap_votes = np.zeros(n_frames, dtype=np.float32)

    for w, lab in zip(windows, labels):
        f0 = max(0, int(np.floor(w.start / FRAME_SEC)))
        f1 = min(n_frames, int(np.ceil(w.end / FRAME_SEC)))
        if f1 <= f0:
            continue
        if lab == OVERLAP_LABEL:
            overlap_votes[f0:f1] += 1.0
        elif 0 <= lab < k:
            votes[f0:f1, lab] += 1.0

    spk_max = votes.max(axis=1)
    covered = (votes.sum(axis=1) > 0) | (overlap_votes > 0)
    frame_labels = np.full(n_frames, -1, dtype=int)

    # A frame is overlap when overlap windows dominate it -> excluded from all
    # speakers (deleted). Otherwise it goes to the winning speaker.
    is_overlap = covered & (overlap_votes > 0) & (overlap_votes >= spk_max)
    frame_labels[is_overlap] = OVERLAP_LABEL
    spk_frames = covered & ~is_overlap
    frame_labels[spk_frames] = np.argmax(votes[spk_frames], axis=1)
    return frame_labels


def _labels_to_segments(frame_labels: np.ndarray, k: int,
                        cfg: DiarizationConfig) -> List[Segment]:
    # Smooth out speech runs shorter than min_segment by absorbing them into
    # the neighbouring speaker (reduces flicker at overlap boundaries).
    frame_labels = _smooth_short_runs(frame_labels, cfg)

    segments: List[Segment] = []
    n = len(frame_labels)
    i = 0
    while i < n:
        lab = frame_labels[i]
        j = i
        while j < n and frame_labels[j] == lab:
            j += 1
        if lab >= 0:
            segments.append(Segment(i * FRAME_SEC, j * FRAME_SEC, int(lab)))
        i = j
    return segments


def _smooth_short_runs(frame_labels: np.ndarray, cfg: DiarizationConfig) -> np.ndarray:
    labels = frame_labels.copy()
    min_frames = max(1, int(round(cfg.min_segment_sec / FRAME_SEC)))
    n = len(labels)

    # Identify runs.
    runs = []  # (start, end, label)
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append([i, j, labels[i]])
        i = j

    # Relabel short *speech* runs using the longer neighbour's speaker label.
    for idx, (s, e, lab) in enumerate(runs):
        if lab < 0 or (e - s) >= min_frames:
            continue
        left = runs[idx - 1] if idx > 0 else None
        right = runs[idx + 1] if idx + 1 < len(runs) else None
        cand = [r for r in (left, right) if r is not None and r[2] >= 0]
        if not cand:
            continue
        best = max(cand, key=lambda r: r[1] - r[0])
        labels[s:e] = best[2]
    return labels


def _canonicalize_speaker_ids(segments: List[Segment]):
    """Renumber speakers by first appearance. Returns (segments, mapping) where
    mapping maps the original cluster id -> canonical speaker id."""
    mapping: dict[int, int] = {}
    for s in segments:
        if s.speaker not in mapping:
            mapping[s.speaker] = len(mapping)
    for s in segments:
        s.speaker = mapping[s.speaker]
    return segments, mapping


def _spans_from_frames(frame_labels: np.ndarray, label: int) -> List[tuple]:
    """Return (start_sec, end_sec) spans of contiguous frames equal to ``label``."""
    spans: List[tuple] = []
    n = len(frame_labels)
    i = 0
    while i < n:
        if frame_labels[i] == label:
            j = i
            while j < n and frame_labels[j] == label:
                j += 1
            spans.append((i * FRAME_SEC, j * FRAME_SEC))
            i = j
        else:
            i += 1
    return spans


def _speaker_centroids(windows, labels, mapping) -> Optional[np.ndarray]:
    """Mean L2-normalised voiceprint per canonical speaker (for source assignment)."""
    if not windows or not mapping:
        return None
    dim = windows[0].embedding.shape[0]
    ncanon = max(mapping.values()) + 1
    cents = np.zeros((ncanon, dim), dtype=np.float64)
    counts = np.zeros(ncanon, dtype=np.int64)
    for w, lab in zip(windows, labels):
        c = mapping.get(int(lab))
        if c is not None:
            cents[c] += w.embedding
            counts[c] += 1
    for c in range(ncanon):
        if counts[c] > 0:
            v = cents[c] / counts[c]
            nrm = np.linalg.norm(v)
            cents[c] = v / nrm if nrm > 0 else v
    return cents.astype(np.float32)


def _log_summary(result: DiarizationResult) -> None:
    log.info("=" * 52)
    log.info("Result: %d speaker(s), %d segment(s), %.1fs audio",
             result.num_speakers, len(result.segments), result.duration)
    for spk, secs in result.speaker_time().items():
        pct = 100.0 * secs / result.duration if result.duration else 0.0
        log.info("  %-11s %6.1fs  (%4.1f%% of file)", result.label(spk), secs, pct)
    log.info("=" * 52)
