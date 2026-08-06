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

    if cfg.export_per_speaker or cfg.export_compact:
        written += _write_speaker_tracks(audio, result, outdir, cfg)

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
    kill = getattr(result, "crosstalk_kill", None) or {}
    for spk in sorted(result.speaker_time().keys()):
        mask = _speaker_mask(result.segments, spk, n, sr, fade, kill.get(spk))
        track = _apply_mask(audio.samples, mask)
        _inject_overlap(track, injections.get(spk, []), sr)
        if cfg.enhance:
            try:
                track = enhance_track(track, sr, cfg)
            except Exception as exc:  # never let enhancement lose the track
                log.warning("Enhancement failed for %s (%s); exporting raw audio",
                            result.label(spk), exc)
        if cfg.export_per_speaker:
            path = _save_track(outdir / result.label(spk), track, sr, cfg)
            written.append(path)
            log.info("  -> %s (%.1fs of audio)", Path(path).name,
                     result.speaker_time()[spk])
        if cfg.export_compact:
            # Compact = the cleaned track with silent gaps removed, so it keeps
            # the overlap deletion + cross-talk gate (slicing raw segments would
            # reintroduce the bleed those stages removed).
            compact = _strip_silence(track, sr)
            path = _save_track(outdir / f"{result.label(spk)}_compact", compact, sr, cfg)
            written.append(path)
            log.info("  -> %s (%.1fs audible)", Path(path).name, compact.shape[0] / sr)
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


def _strip_silence(track: np.ndarray, sr: int, thresh: float = 0.004,
                   win: float = 0.040, pad: float = 0.060,
                   merge_gap: float = 0.120, xfade: float = 0.005) -> np.ndarray:
    """Return ``track`` with silent gaps removed - the audible regions
    concatenated with short crossfades so there are no clicks. Detection is
    energy-based on the (already cleaned) track, so it keeps exactly what is
    audible after masking + overlap deletion + the cross-talk gate."""
    n = track.shape[0]
    mono = track if track.ndim == 1 else track.mean(axis=1)
    w = max(1, int(win * sr))
    nf = n // w
    if nf == 0:
        return track
    rms = np.sqrt(np.mean(mono[:nf * w].reshape(nf, w).astype(np.float64) ** 2, axis=1))
    hot = rms > thresh
    # frames -> sample intervals, then pad + merge nearby ones
    ivs = []
    i = 0
    while i < nf:
        if hot[i]:
            j = i
            while j < nf and hot[j]:
                j += 1
            ivs.append([i * w, j * w])
            i = j
        else:
            i += 1
    if not ivs:
        return track[:0]
    p = int(pad * sr); gap = int(merge_gap * sr)
    merged: List[List[int]] = []
    for s, e in ([max(0, s - p), min(n, e + p)] for s, e in ivs):
        if merged and s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    xf = max(1, int(xfade * sr))
    ramp = np.linspace(0.0, 1.0, xf).astype(track.dtype)
    if track.ndim == 2:
        ramp = ramp[:, None]
    pieces: List[np.ndarray] = []
    for s, e in merged:
        seg = track[s:e].copy()
        if pieces and seg.shape[0] > xf and pieces[-1].shape[0] > xf:
            pieces[-1][-xf:] = pieces[-1][-xf:] * (1.0 - ramp)
            seg[:xf] = seg[:xf] * ramp
        pieces.append(seg)
    return np.concatenate(pieces, axis=0)


def _speaker_mask(segments: List[Segment], speaker: int, n: int, sr: int,
                  fade: int, kill=None) -> np.ndarray:
    """A float mask in [0,1] that is 1 during ``speaker`` segments, with short
    linear fades at the edges to avoid audible clicks. ``kill`` is an optional
    list of ``(start_sec, end_sec)`` intervals inside this speaker's segments
    that the cross-talk gate found to be a different voice - they are forced
    back to 0 so foreign bleed is removed."""
    mask = np.zeros(n, dtype=np.float32)
    for s in segments:
        if s.speaker != speaker:
            continue
        i0 = max(0, int(round(s.start * sr)))
        i1 = min(n, int(round(s.end * sr)))
        if i1 > i0:
            mask[i0:i1] = 1.0

    for t0, t1 in (kill or []):
        i0 = max(0, int(round(t0 * sr)))
        i1 = min(n, int(round(t1 * sr)))
        if i1 > i0:
            mask[i0:i1] = 0.0

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
