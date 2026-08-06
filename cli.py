#!/usr/bin/env python3
"""Command-line interface for Derization (speaker diarization / voice split).

Examples
--------
    python cli.py meeting.wav
    python cli.py interview.mp3 --speakers 2 --outdir ./out -v
    python cli.py podcast.wav --speakers auto --compact
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from diarizer.config import DiarizationConfig
from diarizer.logutil import configure_console_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="derization",
        description="Separate 2-3 speakers in an audio file: detect who spoke "
                    "when and export one isolated track per speaker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="path to the audio file (WAV/FLAC/OGG; MP3/M4A with ffmpeg)")
    p.add_argument("-o", "--outdir", default="diarization_output",
                   help="directory for the output files")
    p.add_argument("-s", "--speakers", default="auto",
                   help="number of speakers: 'auto', or an integer like 2 or 3")
    p.add_argument("--min-speakers", type=int, default=2,
                   help="lower bound when --speakers auto")
    p.add_argument("--max-speakers", type=int, default=3,
                   help="upper bound when --speakers auto")
    p.add_argument("--vad", choices=["auto", "silero", "energy"], default="auto",
                   help="voice-activity-detection backend")
    p.add_argument("--embeddings", choices=["auto", "ecapa", "mfcc"], default="auto",
                   help="speaker-embedding backend")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="compute device for neural backends (uses the GPU if available)")
    p.add_argument("--batch-size", type=int, default=0,
                   help="embedding windows per GPU batch (0 = auto-size from free VRAM)")
    p.add_argument("-f", "--format", choices=["wav", "flac", "ogg", "mp3", "aac"],
                   default=None, help="export format (you'll be prompted if omitted)")
    p.add_argument("--bitrate", default="192k", help="bitrate for mp3/aac, e.g. 192k")
    p.add_argument("--no-denoise", action="store_true",
                   help="skip the noise-reduction / vocal-isolation stage")
    p.add_argument("--no-enhance", action="store_true",
                   help="skip per-speaker studio enhancement")
    p.add_argument("--overlap", choices=["off", "light", "medium", "strong", "extreme"],
                   default="medium",
                   help="how aggressively to detect overlapping/cross-talk speech")
    p.add_argument("--overlap-mode", choices=["separate", "delete"], default="separate",
                   help="separate = un-mix simultaneous voices into each track; "
                        "delete = silence overlap in both")
    p.add_argument("--transcribe", action="store_true",
                   help="also transcribe: speaker-labeled transcript + SRT/VTT subtitles (Whisper)")
    p.add_argument("--whisper-model", choices=["tiny", "base", "small", "medium", "large-v3"],
                   default="small", help="Whisper model size for --transcribe")
    p.add_argument("--denoise-backend",
                   choices=["rvc_hp5", "auto", "noisereduce", "spectral", "none"],
                   default="rvc_hp5",
                   help="noise reduction: rvc_hp5 = UVR/RVC HP5 vocal isolation on GPU")
    p.add_argument("--allow-cpu", action="store_true",
                   help="allow CPU fallback (disables strict GPU-only mode)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="don't prompt for format/destination; use flags/defaults")
    p.add_argument("--compact", action="store_true",
                   help="also export concatenated-only track per speaker")
    p.add_argument("--no-tracks", action="store_true",
                   help="only write the timeline files, skip per-speaker audio")
    p.add_argument("-q", "--quiet", action="store_true", help="less console output")
    p.add_argument("-v", "--verbose", action="store_true", help="debug-level console output")
    return p


def parse_speakers(value: str) -> int | None:
    value = value.strip().lower()
    if value in ("auto", "0", ""):
        return None
    try:
        n = int(value)
    except ValueError:
        raise SystemExit(f"--speakers must be 'auto' or an integer, got '{value}'")
    if n < 1:
        raise SystemExit("--speakers must be >= 1")
    return n


def make_progress_printer(quiet: bool):
    if quiet:
        return None
    bar_width = 28

    def progress(msg: str, frac: float) -> None:
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        end = "\n" if frac >= 1.0 else "\r"
        print(f"  [{bar}] {frac*100:5.1f}%  {msg:<40}", end=end, flush=True, file=sys.stdout)

    return progress


def prompt_format(default: str = "wav") -> str:
    """Ask the user for an export format (spec step 4). Falls back to default
    when not running interactively."""
    from diarizer.formats import supported_formats
    opts = supported_formats()
    if not sys.stdin.isatty():
        return default
    print(f"\nExport formats available: {', '.join(opts)}")
    ans = input(f"  Select export format [{default}]: ").strip().lower().lstrip(".")
    if ans in opts:
        return ans
    if ans:
        print(f"  '{ans}' not available; using {default}.")
    return default


def prompt_outdir(default: str) -> str:
    """Ask the user to confirm/override the destination directory (spec step 4)."""
    if not sys.stdin.isatty():
        return default
    ans = input(f"  Output folder [{default}]: ").strip().strip('"')
    return ans or default


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.verbose and args.quiet:
        raise SystemExit("Choose either --verbose or --quiet, not both.")

    configure_console_logging(verbose=args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 2

    # Export format + destination: prompt unless given or --yes (spec step 4).
    fmt = args.format
    outdir = args.outdir
    if not args.yes:
        if fmt is None:
            fmt = prompt_format()
        outdir = prompt_outdir(outdir)
    fmt = fmt or "wav"

    _ov = {"off": (False, 0.15), "light": (True, 0.08), "medium": (True, 0.15),
           "strong": (True, 0.25), "extreme": (True, 0.35)}[args.overlap]
    cfg = DiarizationConfig(
        num_speakers=parse_speakers(args.speakers),
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        remove_overlap=_ov[0],
        overlap_margin=_ov[1],
        overlap_mode=args.overlap_mode,
        transcribe=args.transcribe,
        whisper_model=args.whisper_model,
        vad_backend=args.vad,
        embedding_backend=args.embeddings,
        device=args.device,
        gpu_only=not args.allow_cpu,
        embed_batch_size=args.batch_size,
        denoise=not args.no_denoise,
        denoise_backend=args.denoise_backend,
        enhance=not args.no_enhance,
        export_format=fmt,
        mp3_bitrate=args.bitrate,
        export_per_speaker=not args.no_tracks,
        export_compact=args.compact,
    )

    # Import here so --help stays instant even before heavy deps are installed.
    from diarizer import process_file

    print(f"\nDerization - processing '{input_path.name}'  "
          f"(denoise={'on' if cfg.denoise else 'off'}, "
          f"enhance={'on' if cfg.enhance else 'off'}, format={fmt})\n")
    t0 = time.time()
    try:
        result, files = process_file(input_path, outdir, config=cfg,
                                     progress=make_progress_printer(args.quiet))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # user-facing error, keep it readable
        print(f"\nError: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1
    elapsed = time.time() - t0

    _print_summary(result, files, outdir, elapsed)
    return 0


def _print_summary(result, files, outdir, elapsed) -> None:
    print("\n" + "=" * 60)
    print(f"  Done in {elapsed:.1f}s - {result.num_speakers} speaker(s) found")
    print("=" * 60)
    for spk, secs in result.speaker_time().items():
        pct = 100.0 * secs / result.duration if result.duration else 0.0
        print(f"  {result.label(spk):<12} {secs:6.1f}s  ({pct:4.1f}%)")
    print(f"\n  Output folder: {Path(outdir).resolve()}")
    for f in files:
        print(f"    - {Path(f).name}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
