"""Write diarization outputs to disk.

Produces (audio extension follows ``cfg.export_format``: wav/flac/ogg/mp3/aac):
  * ``SPEAKER_k.<ext>``   - full-length track, that speaker audible, silence
                            elsewhere (keeps original timing for easy syncing).
  * ``SPEAKER_k_compact.<ext>`` (optional) - only that speaker's segments,
                            concatenated back-to-back.
Each speaker track is optionally denoised (whole recording) and studio-enhanced.
  * ``diarization.txt``   - human-readable "who spoke when" timeline.
  * ``diarization.csv``   - start,end,duration,speaker for spreadsheets.
  * ``diarization.rttm``  - standard NIST RTTM (works with scoring tools).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .audioio import LoadedAudio, save_wav
from .logutil import get_logger
from .pipeline import DiarizationResult, Segment

log = get_logger()


def export_all(audio: LoadedAudio, result: DiarizationResult, outdir: str | Path,
               cfg) -> List[str]:
    """Write every output file. Returns the list of files created."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    written += _write_timeline_txt(result, outdir)
    written += _write_csv(result, outdir)
    written += _write_rttm(result, outdir, Path(audio.path).stem)

    if result.num_speakers == 0:
        log.warning("No speakers found - only empty timeline files were written.")
        return written

    if cfg.export_per_speaker:
        written += _write_speaker_tracks(audio, result, outdir, cfg)
    if cfg.export_compact:
        written += _write_compact_tracks(audio, result, outdir, cfg)

    log.info("Wrote %d file(s) to %s", len(written), outdir)
    return written


# --------------------------------------------------------------------------- #
# Audio tracks
# --------------------------------------------------------------------------- #
def _write_speaker_tracks(audio: LoadedAudio, result: DiarizationResult,
                          outdir: Path, cfg) -> List[str]:
    from .enhance import enhance_track
    from .formats import save_audio

    written = []
    sr = audio.sr
    n = audio.samples.shape[0]
    fade = max(1, int(round(cfg.boundary_fade_ms / 1000.0 * sr)))
    if cfg.enhance:
        log.info("Enhancing each speaker track (HP/EQ/compress/loudness)...")

    injections = getattr(result, "overlap_injections", None) or {}
    for spk in sorted(result.speaker_time().keys()):
        mask = _speaker_mask(result.segments, spk, n, sr, fade)
        track = _apply_mask(audio.samples, mask)
        _inject_overlap(track, injections.get(spk, []), sr)
        if cfg.enhance:
            try:
                track = enhance_track(track, sr, cfg)
            except Exception as exc:  # never let enhancement lose the track
                log.warning("Enhancement failed for %s (%s); exporting raw audio",
                            result.label(spk), exc)
        path = _save_track(outdir / result.label(spk), track, sr, cfg)
        written.append(path)
        log.info("  -> %s (%.1fs of audio)", Path(path).name, result.speaker_time()[spk])
    return written


def _save_track(base: Path, track: np.ndarray, sr: int, cfg) -> str:
    """Save a track, falling back to WAV if the chosen encoder fails, so a
    speaker track is always written."""
    from .audioio import save_wav
    from .formats import save_audio

    try:
        return save_audio(base, track, sr, cfg)
    except Exception as exc:
        log.warning("Export as %s failed (%s); writing WAV instead",
                    getattr(cfg, "export_format", "wav"), exc)
        out = base.with_suffix(".wav")
        save_wav(out, track, sr)
        return str(out)


def _write_compact_tracks(audio: LoadedAudio, result: DiarizationResult,
                          outdir: Path, cfg) -> List[str]:
    from .enhance import enhance_track

    written = []
    sr = audio.sr
    by_spk: dict[int, List[Segment]] = {}
    for s in result.segments:
        by_spk.setdefault(s.speaker, []).append(s)

    for spk, segs in sorted(by_spk.items()):
        chunks = []
        for s in segs:
            i0 = max(0, int(round(s.start * sr)))
            i1 = min(audio.samples.shape[0], int(round(s.end * sr)))
            if i1 > i0:
                chunks.append(audio.samples[i0:i1])
        if not chunks:
            continue
        compact = np.concatenate(chunks, axis=0)
        if cfg.enhance:
            try:
                compact = enhance_track(compact, sr, cfg)
            except Exception as exc:
                log.warning("Enhancement failed for %s (%s); exporting raw audio",
                            result.label(spk), exc)
        # Underscore (not ".compact") so the format extension isn't clobbered.
        path = _save_track(outdir / f"{result.label(spk)}_compact", compact, sr, cfg)
        written.append(path)
    return written


def _speaker_mask(segments: List[Segment], speaker: int, n: int, sr: int,
                  fade: int) -> np.ndarray:
    """A float mask in [0,1] that is 1 during ``speaker`` segments, with short
    linear fades at the edges to avoid audible clicks."""
    mask = np.zeros(n, dtype=np.float32)
    for s in segments:
        if s.speaker != speaker:
            continue
        i0 = max(0, int(round(s.start * sr)))
        i1 = min(n, int(round(s.end * sr)))
        if i1 > i0:
            mask[i0:i1] = 1.0

    if fade > 1:
        # Smooth 0<->1 transitions with a short moving average (linear ramp).
        kernel = np.ones(fade, dtype=np.float32) / fade
        mask = np.convolve(mask, kernel, mode="same")
        np.clip(mask, 0.0, 1.0, out=mask)
    return mask


def _apply_mask(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples * mask
    return samples * mask[:, None]


def _inject_overlap(track: np.ndarray, injections, sr: int) -> None:
    """Add separated overlap audio into a speaker track in place. The overlap
    regions are silent in the masked track, so this fills them with the
    un-mixed voice assigned to this speaker."""
    n = track.shape[0]
    for t0, t1, sig in injections:
        i0 = max(0, int(round(t0 * sr)))
        length = min(sig.shape[0], n - i0)
        if length <= 0:
            continue
        if track.ndim == 1:
            add = sig[:length] if sig.ndim == 1 else sig[:length].mean(axis=1)
        else:
            add = (np.column_stack([sig[:length], sig[:length]])
                   if sig.ndim == 1 else sig[:length])
        track[i0:i0 + length] += add.astype(track.dtype, copy=False)


# --------------------------------------------------------------------------- #
# Timeline files
# --------------------------------------------------------------------------- #
def _write_timeline_txt(result: DiarizationResult, outdir: Path) -> List[str]:
    path = outdir / "diarization.txt"
    lines = ["# Diarization timeline (who spoke when)",
             f"# {result.num_speakers} speaker(s), {result.duration:.2f}s total",
             ""]
    for s in result.segments:
        lines.append(f"{_hms(s.start)} --> {_hms(s.end)}  [{s.duration:5.2f}s]  {result.label(s.speaker)}")
    lines.append("")
    lines.append("# Speaking time per speaker")
    for spk, secs in result.speaker_time().items():
        lines.append(f"#   {result.label(spk)}: {secs:.2f}s")
    path.write_text("\n".join(lines), encoding="utf-8")
    return [str(path)]


def _write_csv(result: DiarizationResult, outdir: Path) -> List[str]:
    import csv
    path = outdir / "diarization.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec", "duration_sec", "speaker"])
        for s in result.segments:
            writer.writerow([f"{s.start:.3f}", f"{s.end:.3f}", f"{s.duration:.3f}",
                             result.label(s.speaker)])
    return [str(path)]


def _write_rttm(result: DiarizationResult, outdir: Path, uri: str) -> List[str]:
    path = outdir / "diarization.rttm"
    with path.open("w", encoding="utf-8") as f:
        for s in result.segments:
            # SPEAKER <uri> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>
            f.write(f"SPEAKER {uri} 1 {s.start:.3f} {s.duration:.3f} "
                    f"<NA> <NA> {result.label(s.speaker)} <NA> <NA>\n")
    return [str(path)]


def _hms(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
