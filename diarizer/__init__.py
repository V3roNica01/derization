"""Derization - speaker diarization & voice separation for 2-3 speakers.

High-level usage::

    from diarizer import diarize_file
    result, files = diarize_file("meeting.wav", outdir="out", num_speakers=2)

Lower-level building blocks live in the submodules (``audioio``, ``vad``,
``embeddings``, ``cluster``, ``pipeline``, ``export``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .config import DiarizationConfig, TARGET_SR
from .pipeline import DiarizationResult, Segment, diarize
from .audioio import LoadedAudio, load_audio
from .logutil import get_logger

log = get_logger()
__version__ = "1.0.0"

__all__ = [
    "DiarizationConfig",
    "DiarizationResult",
    "Segment",
    "LoadedAudio",
    "load_audio",
    "diarize",
    "process_file",
    "diarize_file",
    "TARGET_SR",
    "__version__",
]


def process_file(input_path: str | Path,
                 outdir: str | Path = "diarization_output",
                 config: Optional[DiarizationConfig] = None,
                 num_speakers: Optional[int] = None,
                 progress=None) -> Tuple[DiarizationResult, List[str]]:
    """Run the full pipeline on ``input_path`` and write outputs to ``outdir``.

    Stages: load -> (noise reduction) -> diarize -> (per-speaker enhancement)
    -> export in the configured format. Returns ``(result, written_files)``.
    """
    import numpy as np
    from .audioio import resample
    from .crosstalk import gate_crosstalk
    from .enhance import denoise
    from .export import export_all
    from .hardware import log_device_info, resolve_device
    from .logutil import quiet_dependency_warnings

    quiet_dependency_warnings()
    cfg = config or DiarizationConfig()
    if num_speakers is not None:
        cfg.num_speakers = num_speakers

    def report(msg: str, frac: float) -> None:
        if progress is not None:
            progress(msg, frac)

    audio = load_audio(input_path)

    if cfg.denoise and cfg.denoise_backend != "none":
        # Denoise occupies the first ~18% of the progress bar; its own 0..1
        # progress (e.g. per RVC chunk) is remapped into that band.
        def dn_prog(msg: str, frac: float) -> None:
            report(msg, 0.02 + 0.16 * max(0.0, min(1.0, frac)))

        dn_prog("Reducing background noise / isolating voices...", 0.0)
        device = resolve_device(cfg.device, cfg.gpu_only)
        log_device_info(device)
        cleaned = denoise(audio.samples, audio.sr, cfg, device, progress=dn_prog)
        mono = cleaned if cleaned.ndim == 1 else cleaned.mean(axis=1)
        mono16k = resample(mono.astype(np.float32), audio.sr, TARGET_SR)
        audio = LoadedAudio(samples=cleaned.astype(np.float32), sr=audio.sr,
                            mono16k=mono16k, path=audio.path)
        # Release the RVC model so the GPU has room for embeddings/separation.
        from .rvc_denoise import free as free_rvc
        free_rvc()

    # Diarization drives 20%..88% of the bar; overlap separation 88%..99%.
    def di_prog(msg: str, frac: float) -> None:
        report(msg, 0.20 + 0.68 * max(0.0, min(1.0, frac)))

    result = diarize(audio, cfg, progress=di_prog)

    # True separation of overlapping speech (un-mix + assign; delete if unsure).
    if (cfg.remove_overlap and getattr(cfg, "overlap_mode", "delete") == "separate"
            and result.overlap_spans):
        from .separate import available as sep_available, separate_overlaps
        if sep_available():
            def sep_prog(msg: str, frac: float) -> None:
                report(msg, 0.88 + 0.11 * max(0.0, min(1.0, frac)))

            try:
                device = resolve_device(cfg.device, cfg.gpu_only)
                injections, _, _ = separate_overlaps(audio, result, cfg, device,
                                                     progress=sep_prog)
                result.overlap_injections = injections
            except Exception as exc:
                # An optional stage must never lose the diarization; on failure
                # overlaps simply stay deleted (silent in both tracks).
                log.warning("Overlap separation failed (%s); overlaps will be "
                            "deleted instead.", exc)
        else:
            report("SepFormer unavailable - overlaps will be deleted.", 0.9)

    # Residual cross-talk gate: silence foreign voice (laughs/interjections)
    # that leaked inside a speaker's own segment but was never overlap.
    if getattr(cfg, "crosstalk_gate", False) and result.num_speakers > 1:
        try:
            device = resolve_device(cfg.device, cfg.gpu_only)
            report("Removing residual cross-talk...", 0.985)
            result.crosstalk_kill = gate_crosstalk(audio.mono16k, result, cfg,
                                                    device, progress=None)
        except Exception as exc:
            log.warning("Cross-talk gate failed (%s); skipping", exc)

    files = export_all(audio, result, outdir, cfg)

    # Optional: transcribe + attribute words to speakers (Whisper).
    if getattr(cfg, "transcribe", False) and result.num_speakers > 0:
        from .transcribe import (assign_speakers, available as tr_available,
                                 transcribe_words, write_transcripts)
        if tr_available():
            def tr_prog(msg: str, frac: float) -> None:
                report(msg, 0.85 + 0.15 * max(0.0, min(1.0, frac)))

            try:
                device = resolve_device(cfg.device, cfg.gpu_only)
                words = transcribe_words(audio.mono16k, cfg, device, progress=tr_prog)
                labeled = assign_speakers(words, result)
                files += write_transcripts(labeled, result, outdir)
            except Exception as exc:
                log.warning("Transcription failed (%s); skipping", exc)
        else:
            log.warning("transcribe is on but faster-whisper is not installed; skipping")

    return result, files


def diarize_file(input_path: str | Path,
                 outdir: str | Path = "diarization_output",
                 num_speakers: Optional[int] = None,
                 config: Optional[DiarizationConfig] = None,
                 progress=None) -> Tuple[DiarizationResult, List[str]]:
    """Backwards-compatible alias for :func:`process_file`."""
    return process_file(input_path, outdir, config=config,
                        num_speakers=num_speakers, progress=progress)
