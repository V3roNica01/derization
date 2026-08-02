"""Voice activity detection (VAD).

Splits the 16 kHz mono analysis signal into speech regions so that silence and
noise are excluded from speaker analysis and export.

Two backends:
  * ``silero``  - a small neural VAD (best quality) via the ``silero-vad`` pkg.
  * ``energy``  - a dependency-free adaptive energy gate (always available).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import TARGET_SR, DiarizationConfig
from .hardware import resolve_device
from .logutil import get_logger

log = get_logger()


@dataclass
class SpeechSegment:
    """A region of speech, in seconds, on the analysis timeline."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_speech(mono16k: np.ndarray, cfg: DiarizationConfig) -> List[SpeechSegment]:
    """Return merged speech segments for the given 16 kHz mono signal.

    Robustness: if the chosen detector reports (almost) no speech, fall back to
    the more permissive energy detector; and if there is clearly audio present
    but still nothing is detected, treat the whole file as speech. This ensures
    a recording that actually contains audio always yields speaker tracks
    instead of silently producing only timeline files.
    """
    backend = cfg.vad_backend
    if backend == "auto":
        backend = "silero" if _silero_available() else "energy"

    # GPU-only mode must use the neural (GPU) VAD, not the CPU energy gate.
    if cfg.gpu_only and backend != "silero":
        log.info("GPU-only mode: forcing Silero VAD on the GPU.")
        backend = "silero"

    segs = _merge_and_filter(_run_vad(backend, mono16k, cfg), cfg)

    # The neural VAD can return an empty result (no exception) on quiet, noisy
    # or heavily-denoised audio -> retry with the energy gate (CPU-allowed only).
    if _too_little(segs) and backend != "energy" and not cfg.gpu_only:
        log.warning("%s VAD found little/no speech; retrying with energy VAD", backend)
        segs = _merge_and_filter(_energy_vad(mono16k, cfg), cfg)

    # Last resort: there is audio but no detected speech -> use the whole file.
    if _too_little(segs) and _has_audio(mono16k):
        dur = mono16k.shape[0] / TARGET_SR
        log.warning("No speech regions detected; treating the entire file as "
                    "speech so speaker tracks are still produced.")
        segs = [SpeechSegment(0.0, dur)]

    total = sum(s.duration for s in segs)
    log.info("VAD: %d speech segment(s) after cleanup, %.2fs of speech total",
             len(segs), total)
    return segs


def _run_vad(backend: str, mono16k: np.ndarray, cfg: DiarizationConfig) -> List[SpeechSegment]:
    """Run the requested backend, falling back to energy VAD if silero errors
    (unless GPU-only mode is on, in which case a GPU failure is raised)."""
    if backend == "silero":
        device = resolve_device(cfg.device, cfg.gpu_only)
        try:
            segs = _silero_vad(mono16k, device, strict=cfg.gpu_only)
            log.info("VAD (silero): %d raw speech region(s)", len(segs))
            return segs
        except Exception as exc:
            if cfg.gpu_only:
                raise RuntimeError(
                    f"GPU-only mode: Silero VAD failed on the GPU: {exc}") from exc
            log.warning("Silero VAD failed (%s); falling back to energy VAD", exc)
    segs = _energy_vad(mono16k, cfg)
    log.info("VAD (energy): %d raw speech region(s)", len(segs))
    return segs


def _too_little(segs: List[SpeechSegment], min_total: float = 0.5) -> bool:
    return sum(s.duration for s in segs) < min_total


def _has_audio(mono16k: np.ndarray, thresh: float = 1e-3) -> bool:
    if mono16k.size == 0:
        return False
    return float(np.sqrt(np.mean(mono16k ** 2))) > thresh


# --------------------------------------------------------------------------- #
# Silero backend
# --------------------------------------------------------------------------- #
def _silero_available() -> bool:
    try:
        import torch  # noqa: F401
        import silero_vad  # noqa: F401
        return True
    except Exception:
        return False


_SILERO_MODEL = None


def _silero_vad(mono16k: np.ndarray, device: str = "cpu",
                strict: bool = False) -> List[SpeechSegment]:
    global _SILERO_MODEL
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    if _SILERO_MODEL is None:
        log.debug("Loading Silero VAD model...")
        _SILERO_MODEL = load_silero_vad()

    wav = torch.from_numpy(np.ascontiguousarray(mono16k, dtype=np.float32))
    model = _SILERO_MODEL
    if device == "cuda":
        # In strict (GPU-only) mode let errors propagate; otherwise fall back.
        if strict:
            model = model.to("cuda")
            wav = wav.to("cuda")
        else:
            try:
                model = model.to("cuda")
                wav = wav.to("cuda")
            except Exception as exc:
                log.debug("Silero on GPU unavailable (%s); using CPU", exc)
                model = _SILERO_MODEL.to("cpu")
                wav = wav.cpu()

    try:
        ts = get_speech_timestamps(
            wav, model, sampling_rate=TARGET_SR, return_seconds=True
        )
    except Exception:
        if device == "cuda" and not strict:
            log.debug("Silero GPU inference failed; retrying on CPU")
            ts = get_speech_timestamps(
                wav.cpu(), _SILERO_MODEL.to("cpu"),
                sampling_rate=TARGET_SR, return_seconds=True
            )
        else:
            raise
    return [SpeechSegment(float(t["start"]), float(t["end"])) for t in ts]


# --------------------------------------------------------------------------- #
# Energy backend (no external model)
# --------------------------------------------------------------------------- #
def _energy_vad(mono16k: np.ndarray, cfg: DiarizationConfig) -> List[SpeechSegment]:
    """Adaptive short-time energy gate with hysteresis.

    Frames whose energy is within ``energy_vad_threshold_db`` of the loud
    portion of the file are treated as speech.
    """
    frame = int(0.030 * TARGET_SR)   # 30 ms
    hop = int(0.010 * TARGET_SR)     # 10 ms
    if mono16k.shape[0] < frame:
        return []

    # Frame the signal and compute RMS energy in dB.
    n_frames = 1 + (mono16k.shape[0] - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = mono16k[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    db = 20.0 * np.log10(rms + 1e-10)

    # Reference level = 95th percentile (robust "loud speech" estimate).
    ref = np.percentile(db, 95)
    threshold = ref - cfg.energy_vad_threshold_db
    voiced = db > threshold

    # Convert the boolean frame mask into time segments.
    segs: List[SpeechSegment] = []
    in_speech = False
    start_f = 0
    for i, v in enumerate(voiced):
        if v and not in_speech:
            in_speech = True
            start_f = i
        elif not v and in_speech:
            in_speech = False
            segs.append(_frames_to_seg(start_f, i, hop))
    if in_speech:
        segs.append(_frames_to_seg(start_f, len(voiced), hop))
    return segs


def _frames_to_seg(start_f: int, end_f: int, hop: int) -> SpeechSegment:
    return SpeechSegment(start_f * hop / TARGET_SR, end_f * hop / TARGET_SR)


# --------------------------------------------------------------------------- #
# Shared post-processing
# --------------------------------------------------------------------------- #
def _merge_and_filter(segs: List[SpeechSegment], cfg: DiarizationConfig) -> List[SpeechSegment]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s.start)

    # Bridge short silences between consecutive speech regions.
    merged: List[SpeechSegment] = [segs[0]]
    for s in segs[1:]:
        gap = s.start - merged[-1].end
        if gap <= cfg.vad_min_silence_sec:
            merged[-1] = SpeechSegment(merged[-1].start, max(merged[-1].end, s.end))
        else:
            merged.append(s)

    # Drop tiny speech blips.
    return [s for s in merged if s.duration >= cfg.vad_min_speech_sec]
