"""Speaker embedding extraction.

Each short analysis window is turned into a fixed-length vector that captures
*who* is speaking (a "voiceprint"). Windows from the same speaker land close
together, which lets clustering group them.

Backends:
  * ``ecapa`` - SpeechBrain ECAPA-TDNN (192-dim, high quality). Downloads a
    ~80 MB model from Hugging Face on first use (not gated).
  * ``mfcc``  - librosa MFCC statistics (dependency-light fallback).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import MODELS_DIR, TARGET_SR, DiarizationConfig
from .hardware import auto_batch_size, empty_cache, log_device_info, resolve_device
from .logutil import get_logger
from .vad import SpeechSegment

log = get_logger()


@dataclass
class Window:
    """An embedded analysis window on the analysis timeline (seconds)."""
    start: float
    end: float
    embedding: np.ndarray


def build_windows(mono16k: np.ndarray, segments: List[SpeechSegment],
                  cfg: DiarizationConfig) -> List[Tuple[float, float, np.ndarray]]:
    """Slice speech ``segments`` into overlapping windows of raw audio.

    Returns a list of ``(start_sec, end_sec, audio)`` tuples.
    """
    win = cfg.window_sec
    hop = cfg.hop_sec
    min_win = min(cfg.min_window_sec, win)
    duration = mono16k.shape[0] / TARGET_SR
    out: List[Tuple[float, float, np.ndarray]] = []

    for seg in segments:
        seg_len = seg.duration
        if seg_len <= win:
            # Short utterance -> one window. Centre-extend the *audio* fed to the
            # embedder to at least ``min_win`` (grabbing neighbouring samples) so
            # it isn't an outlier, but keep the reported time span = the true
            # speech region so it only votes for its own frames.
            need = max(min_win, seg_len)
            center = 0.5 * (seg.start + seg.end)
            a0 = max(0.0, center - need / 2)
            a1 = min(duration, a0 + need)
            a0 = max(0.0, a1 - need)
            a = _slice(mono16k, a0, a1)
            if a.size > 0:
                out.append((seg.start, seg.end, a))
            continue

        t = seg.start
        while t + win <= seg.end + 1e-6:
            a = _slice(mono16k, t, t + win)
            if a.size > 0:
                out.append((t, t + win, a))
            t += hop
        # Ensure the tail of the segment is covered.
        if t < seg.end - 1e-3:
            a = _slice(mono16k, seg.end - win, seg.end)
            if a.size > 0:
                out.append((seg.end - win, seg.end, a))
    return out


def _slice(mono16k: np.ndarray, start: float, end: float) -> np.ndarray:
    i0 = max(0, int(round(start * TARGET_SR)))
    i1 = min(mono16k.shape[0], int(round(end * TARGET_SR)))
    return mono16k[i0:i1]


class SpeakerEmbedder:
    """Wraps whichever embedding backend is active."""

    def __init__(self, cfg: DiarizationConfig) -> None:
        self.cfg = cfg
        self.backend = cfg.embedding_backend
        self._model = None
        self.device = "cpu"
        self.batch = int(cfg.embed_batch_size) if cfg.embed_batch_size else 0
        if self.backend == "auto":
            self.backend = "ecapa" if _ecapa_available() else "mfcc"

        # MFCC is a CPU backend; GPU-only mode must use neural ECAPA on the GPU.
        if cfg.gpu_only and self.backend != "ecapa":
            log.info("GPU-only mode: forcing ECAPA embeddings on the GPU.")
            self.backend = "ecapa"

        if self.backend == "ecapa":
            try:
                self.device = resolve_device(cfg.device, cfg.gpu_only)
                self._model = _load_ecapa(self.device)
                log_device_info(self.device)
                # Auto-size the batch to the GPU's free VRAM (unless forced).
                if self.batch <= 0:
                    self.batch = (auto_batch_size(cfg.vram_reserve_gb)
                                  if self.device == "cuda" else 16)
                log.info("Embeddings: SpeechBrain ECAPA-TDNN (192-dim) on %s, batch=%d",
                         self.device.upper(), self.batch)
            except Exception as exc:
                if cfg.gpu_only:
                    raise RuntimeError(
                        f"GPU-only mode: ECAPA embeddings could not initialize on "
                        f"the GPU: {exc}") from exc
                log.warning("Could not load ECAPA model (%s); using MFCC embeddings", exc)
                self.backend = "mfcc"
        if self.backend == "mfcc":
            log.info("Embeddings: MFCC statistics (fallback backend, CPU)")

    def embed_windows(self, windows: List[Tuple[float, float, np.ndarray]]) -> List[Window]:
        if not windows:
            return []
        audios = [w[2] for w in windows]
        if self.backend == "ecapa":
            vecs = self._embed_ecapa(audios)
        else:
            vecs = self._embed_mfcc(audios)
        vecs = _l2_normalize(vecs)
        return [Window(w[0], w[1], v) for w, v in zip(windows, vecs)]

    # --- ECAPA ----------------------------------------------------------
    def _embed_ecapa(self, audios: List[np.ndarray]) -> np.ndarray:
        """Batch windows through ECAPA on the GPU (or CPU).

        Batching is where the GPU speedup comes from. Handles CUDA
        out-of-memory by halving the batch size, and as a last resort reloads
        the model on the CPU so the run still completes.
        """
        model = self._model
        device = self.device
        min_len = int(0.4 * TARGET_SR)      # pad very short clips for the CNN
        bs = max(1, int(self.batch))
        results: List[np.ndarray] = []

        i = 0
        while i < len(audios):
            batch = audios[i:i + bs]
            try:
                results.append(self._encode_batch(model, batch, device, min_len))
                i += bs
            except RuntimeError as exc:
                if device == "cuda" and "out of memory" in str(exc).lower():
                    empty_cache()
                    if bs > 1:
                        bs = max(1, bs // 2)
                        log.warning("CUDA out of memory - reducing batch size to %d", bs)
                        continue
                    if self.cfg.gpu_only:
                        raise RuntimeError(
                            "GPU-only mode: ran out of VRAM even at batch size 1. "
                            "Lower --batch-size or free GPU memory.") from exc
                    log.warning("CUDA OOM at batch size 1 - reloading ECAPA on CPU")
                    self.device = device = "cpu"
                    model = self._model = _load_ecapa("cpu")
                    continue
                raise
        return np.vstack(results)

    @staticmethod
    def _encode_batch(model, batch: List[np.ndarray], device: str,
                      min_len: int) -> np.ndarray:
        """Zero-pad a batch to equal length and encode it in one GPU call.

        ``wav_lens`` (relative lengths) tells ECAPA to ignore the padding when
        it pools statistics, so padding does not corrupt the embeddings.
        """
        import torch

        lengths = [max(int(a.shape[0]), 1) for a in batch]
        maxlen = max(max(lengths), min_len)
        arr = np.zeros((len(batch), maxlen), dtype=np.float32)
        for j, a in enumerate(batch):
            arr[j, :a.shape[0]] = a
        wavs = torch.from_numpy(arr).to(device)
        wav_lens = torch.tensor([l / maxlen for l in lengths],
                                dtype=torch.float32, device=device)
        with torch.no_grad():
            emb = model.encode_batch(wavs, wav_lens)   # (B, 1, D)
        return emb.squeeze(1).detach().cpu().numpy().astype(np.float32)

    # --- MFCC -----------------------------------------------------------
    def _embed_mfcc(self, audios: List[np.ndarray]) -> np.ndarray:
        import librosa

        out = []
        for a in audios:
            if a.shape[0] < int(0.1 * TARGET_SR):
                a = np.pad(a, (0, int(0.1 * TARGET_SR) - a.shape[0]))
            mfcc = librosa.feature.mfcc(y=a, sr=TARGET_SR, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            feat = np.concatenate([
                mfcc.mean(axis=1), mfcc.std(axis=1),
                delta.mean(axis=1), delta.std(axis=1),
            ])
            out.append(feat.astype(np.float32))
        return np.vstack(out)


# --------------------------------------------------------------------------- #
# Backend availability / loading
# --------------------------------------------------------------------------- #
def _ecapa_available() -> bool:
    try:
        import torch  # noqa: F401
        import speechbrain  # noqa: F401
        return True
    except Exception:
        return False


_ECAPA_CACHE: dict = {}


def _patch_speechbrain_lazy_imports() -> None:
    """Work around a SpeechBrain-on-Windows crash.

    SpeechBrain exposes optional integrations (e.g. ``k2``) as lazy modules
    whose ``__getattr__`` raises ``ImportError`` when the backing package is
    missing. Python's ``inspect`` walks ``sys.modules`` and probes ``__file__``
    on every module (during hparams loading), which forces those lazy imports
    and crashes on machines without ``k2``. We make dunder probes raise
    ``AttributeError`` instead, which is what introspection expects.
    """
    try:
        from speechbrain.utils import importutils
        LazyModule = importutils.LazyModule
        if getattr(LazyModule, "_diarizer_patched", False):
            return
        _orig_getattr = LazyModule.__getattr__

        def _safe_getattr(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _orig_getattr(self, name)

        LazyModule.__getattr__ = _safe_getattr
        LazyModule._diarizer_patched = True
    except Exception as exc:  # never let the workaround itself break loading
        log.debug("Could not patch SpeechBrain lazy imports: %s", exc)


def _load_ecapa(device: str = "cpu"):
    """Load ECAPA-TDNN on ``device`` once (cached per device). Returns the model."""
    if device in _ECAPA_CACHE:
        return _ECAPA_CACHE[device]

    import os

    _patch_speechbrain_lazy_imports()

    # SpeechBrain moved the inference API across versions; support both.
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

    # Copy model files instead of symlinking them: Windows symlinks need
    # elevated privileges, which breaks the default SYMLINK fetch strategy.
    extra = {}
    try:
        from speechbrain.utils.fetching import LocalStrategy
        extra["local_strategy"] = LocalStrategy.COPY
    except Exception:
        pass

    # SpeechBrain wants an explicit device index (e.g. "cuda:0"); a bare
    # "cuda" triggers a "Could not parse CUDA device string" warning.
    sb_device = "cuda:0" if device == "cuda" else device
    savedir = os.path.join(MODELS_DIR, "ecapa")
    log.debug("Loading ECAPA model (device=%s, cache=%s)", sb_device, savedir)
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": sb_device},
        **extra,
    )
    _ECAPA_CACHE[device] = model
    return model


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
