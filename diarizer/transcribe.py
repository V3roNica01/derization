"""Speech-to-text with speaker attribution (Whisper, GPU).

Transcribes the cleaned audio with word-level timestamps, assigns each word to a
speaker using the diarization timeline, and writes:
  * transcript.txt   - speaker-labelled, grouped into turns ("SPEAKER_1: ...")
  * transcript.srt / .vtt - subtitles with speaker labels
  * SPEAKER_k.txt    - each speaker's words as plain text

Uses faster-whisper (CTranslate2) on the GPU, falling back to CPU if the GPU
runtime isn't available (transcription on CPU is slower but still works).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import MODELS_DIR, TARGET_SR
from .logutil import get_logger

log = get_logger()

_MODEL = None
_MODEL_KEY = None


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _load(cfg, device: str):
    global _MODEL, _MODEL_KEY
    from faster_whisper import WhisperModel

    key = (cfg.whisper_model, device)
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL, device

    dl = os.path.join(MODELS_DIR, "whisper")
    os.makedirs(dl, exist_ok=True)
    # If the model is already downloaded, load it offline (no network HEAD
    # requests that stall when there's no internet).
    cached = os.path.isdir(os.path.join(dl, f"models--Systran--faster-whisper-{cfg.whisper_model}"))

    def make(dev):
        ct = "float16" if dev == "cuda" else "int8"
        return WhisperModel(cfg.whisper_model, device=dev, compute_type=ct,
                            download_root=dl, local_files_only=cached)

    try:
        _MODEL = make("cuda" if device == "cuda" else "cpu")
        _MODEL_KEY = key
        return _MODEL, ("cuda" if device == "cuda" else "cpu")
    except Exception as exc:
        if device == "cuda":
            log.warning("Whisper GPU runtime unavailable (%s); transcribing on CPU", exc)
            _MODEL = make("cpu")
            _MODEL_KEY = (cfg.whisper_model, "cpu")
            return _MODEL, "cpu"
        raise


def transcribe_words(audio_mono16k: np.ndarray, cfg, device: str,
                     progress=None) -> List[Tuple[float, float, str]]:
    """Return [(start, end, word)] for the whole recording."""
    model, dev = _load(cfg, device)
    log.info("Transcribing with Whisper '%s' on %s...", cfg.whisper_model, dev.upper())
    total = max(1e-6, audio_mono16k.shape[0] / TARGET_SR)
    segs, info = model.transcribe(
        np.ascontiguousarray(audio_mono16k, dtype=np.float32),
        word_timestamps=True, language=cfg.whisper_language, beam_size=5,
        vad_filter=False)

    words: List[Tuple[float, float, str]] = []
    for s in segs:
        if s.words:
            for w in s.words:
                words.append((float(w.start), float(w.end), w.word))
        elif s.text:
            words.append((float(s.start), float(s.end), " " + s.text.strip()))
        if progress is not None:
            progress(f"Transcribing... {int(100 * min(1.0, s.end / total))}%",
                     min(1.0, s.end / total))
    log.info("Transcribed %d words (detected language: %s)",
             len(words), getattr(info, "language", "?"))
    return words


def assign_speakers(words, result) -> List[Tuple[float, float, str, int]]:
    """Attach a speaker id to each word from the diarization segments."""
    segs = sorted(result.segments, key=lambda s: s.start)
    out = []
    for ws, we, text in words:
        out.append((ws, we, text, _speaker_at(segs, 0.5 * (ws + we))))
    return out


def _speaker_at(segs, t: float) -> int:
    best, bestd = -1, 1e18
    for s in segs:
        if s.start <= t <= s.end:
            return s.speaker
        d = min(abs(t - s.start), abs(t - s.end))
        if d < bestd:
            bestd, best = d, s.speaker
    return best


def write_transcripts(labeled, result, outdir) -> List[str]:
    outdir = Path(outdir)
    cues = _build_cues(labeled)
    turns = _turns_from_cues(cues)
    files: List[str] = []

    # Speaker-labelled transcript
    lines = ["# Transcript (speaker-attributed)", ""]
    for start, _end, spk, text in turns:
        lab = result.label(spk) if spk >= 0 else "UNKNOWN"
        lines.append(f"[{_hms(start)}] {lab}: {text}")
    p = outdir / "transcript.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.append(str(p))

    # Per-speaker plain text
    per: dict = {}
    for _s, _e, text, spk in labeled:
        if spk >= 0:
            per.setdefault(spk, []).append(text)
    for spk, chunks in sorted(per.items()):
        p = outdir / f"{result.label(spk)}.txt"
        p.write_text("".join(chunks).strip() + "\n", encoding="utf-8")
        files.append(str(p))

    # Subtitles
    files.append(_write_srt(cues, result, outdir / "transcript.srt"))
    files.append(_write_vtt(cues, result, outdir / "transcript.vtt"))
    log.info("Wrote transcript + subtitles (%d turns)", len(turns))
    return files


def _build_cues(labeled, max_dur=6.0, max_words=14):
    cues = []
    cur = None
    for ws, we, text, spk in labeled:
        if cur is None:
            cur = [ws, we, spk, [text]]
        elif spk != cur[2] or (we - cur[0]) > max_dur or len(cur[3]) >= max_words:
            cues.append(cur)
            cur = [ws, we, spk, [text]]
        else:
            cur[1] = we
            cur[3].append(text)
    if cur:
        cues.append(cur)
    return [(c[0], c[1], c[2], "".join(c[3]).strip()) for c in cues if "".join(c[3]).strip()]


def _turns_from_cues(cues):
    turns = []
    for s, e, spk, text in cues:
        if turns and turns[-1][2] == spk:
            turns[-1] = (turns[-1][0], e, spk, turns[-1][3] + " " + text)
        else:
            turns.append((s, e, spk, text))
    return turns


def _write_srt(cues, result, path) -> str:
    out = []
    for i, (s, e, spk, text) in enumerate(cues, 1):
        lab = result.label(spk) if spk >= 0 else "UNKNOWN"
        out.append(str(i))
        out.append(f"{_srt_ts(s)} --> {_srt_ts(e)}")
        out.append(f"[{lab}] {text}")
        out.append("")
    Path(path).write_text("\n".join(out), encoding="utf-8")
    return str(path)


def _write_vtt(cues, result, path) -> str:
    out = ["WEBVTT", ""]
    for s, e, spk, text in cues:
        lab = result.label(spk) if spk >= 0 else "UNKNOWN"
        out.append(f"{_vtt_ts(s)} --> {_vtt_ts(e)}")
        out.append(f"<v {lab}>{text}")
        out.append("")
    Path(path).write_text("\n".join(out), encoding="utf-8")
    return str(path)


def _hms(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _srt_ts(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
