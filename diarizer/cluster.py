"""Cluster speaker embeddings into N speakers.

If the speaker count is fixed we cluster directly. If it is "auto" we try each
candidate count and pick the one with the best silhouette score (cohesion vs.
separation) using cosine distance, which suits L2-normalised voiceprints.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import DiarizationConfig
from .embeddings import Window
from .logutil import get_logger

log = get_logger()


def cluster_windows(windows: List[Window], cfg: DiarizationConfig) -> Tuple[np.ndarray, int]:
    """Assign a speaker label to each window.

    Returns ``(labels, num_speakers)`` where ``labels[i]`` is the 0-based
    speaker index for ``windows[i]``.
    """
    if not windows:
        return np.array([], dtype=int), 0

    X = np.vstack([w.embedding for w in windows]).astype(np.float64)
    n = X.shape[0]

    if n == 1:
        return np.zeros(1, dtype=int), 1

    if cfg.num_speakers is not None:
        k = max(1, min(int(cfg.num_speakers), n))
        labels = _agglomerative(X, k)
        log.info("Clustering: forced %d speaker(s)", k)
        return _mark_overlaps(X, labels, k, cfg), k

    # Auto: evaluate each candidate count.
    candidates = [k for k in cfg.resolved_speaker_range() if k <= n]
    if not candidates:
        candidates = [1]

    best_labels = None
    best_k = candidates[0]
    best_score = -np.inf
    log.info("Clustering: auto-detecting speaker count among %s", list(candidates))

    for k in candidates:
        if k == 1:
            score = _single_cluster_score(X)
            labels = np.zeros(n, dtype=int)
        else:
            labels = _agglomerative(X, k)
            score = _silhouette(X, labels)
        log.debug("  k=%d -> score %.4f", k, score)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    log.info("Clustering: selected %d speaker(s) (score %.3f)", best_k, best_score)
    return _mark_overlaps(X, best_labels, best_k, cfg), best_k


# Windows that sit between two speakers' voiceprints (simultaneous speech) get
# this label; the pipeline excludes them from every speaker track.
OVERLAP_LABEL = -2


def _mark_overlaps(X: np.ndarray, labels: np.ndarray, k: int,
                   cfg: DiarizationConfig) -> np.ndarray:
    """Flag windows whose voiceprint is nearly equidistant to the two closest
    speaker centroids (i.e. two people talking at once) as OVERLAP_LABEL."""
    if k < 2 or not getattr(cfg, "remove_overlap", False):
        return labels
    labels = labels.copy()
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    centroids = np.zeros((k, X.shape[1]), dtype=X.dtype)
    for c in range(k):
        m = labels == c
        if np.any(m):
            v = Xn[m].mean(axis=0)
            centroids[c] = v / (np.linalg.norm(v) + 1e-9)

    sims = Xn @ centroids.T            # cosine similarity to each speaker
    srt = np.sort(sims, axis=1)
    top1, top2 = srt[:, -1], srt[:, -2]
    overlap = (top1 - top2) < float(cfg.overlap_margin)
    n_over = int(overlap.sum())
    if n_over:
        labels[overlap] = OVERLAP_LABEL
        log.info("Overlap: %d/%d window(s) look like simultaneous speech "
                 "-> removed from both speakers", n_over, len(labels))
    return labels


def _agglomerative(X: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative clustering with cosine distance, version-tolerant.

    Uses *complete* linkage: unlike average/single linkage it does not peel a
    lone outlier window off as its own cluster (which would lump two real
    speakers together), so it separates well-formed speaker clusters reliably.
    """
    from sklearn.cluster import AgglomerativeClustering

    try:  # sklearn >= 1.2
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="complete")
        return model.fit_predict(X)
    except TypeError:  # older sklearn used ``affinity``
        model = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="complete")
        return model.fit_predict(X)


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    if len(set(labels.tolist())) < 2:
        return -1.0
    try:
        return float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        return -1.0


def _single_cluster_score(X: np.ndarray) -> float:
    """Heuristic score for the "everyone is one speaker" hypothesis.

    High mean pairwise cosine similarity => genuinely one speaker => high score.
    This lets auto-mode avoid inventing a second speaker from a monologue.
    """
    # cosine similarity of L2-normalised vectors is just the dot product.
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    n = sim.shape[0]
    off = (sim.sum() - np.trace(sim)) / max(1, (n * n - n))
    # Map similarity (typically 0..1) to a silhouette-comparable scale.
    return float(off) * 0.5
