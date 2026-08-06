"""Configuration objects for the diarization pipeline.

All tunable parameters live here so the CLI and GUI share one source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


# The speaker-embedding models we use expect 16 kHz mono audio.
TARGET_SR = 16000

# All downloaded models live inside the project folder (self-contained).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
# Keep the Hugging Face download cache in the project too, so nothing is
# written under the user's home directory.
os.environ.setdefault("HF_HOME", os.path.join(MODELS_DIR, "hf"))


@dataclass
class DiarizationConfig:
    """Parameters controlling how audio is analysed and split.

    Every value has a sensible default tuned for 2-3 speaker conversations
    (interviews, meetings, podcasts). Override any of them from the CLI/GUI.
    """

    # --- Speaker count ---------------------------------------------------
    # None / "auto" -> estimate automatically within [min_speakers, max_speakers].
    # An int (2 or 3) -> force exactly that many speakers.
    num_speakers: Optional[int] = None
    min_speakers: int = 2
    max_speakers: int = 3

    # --- Diarization engine ---------------------------------------------
    # "builtin" -> the self-contained Silero VAD + ECAPA + clustering engine.
    # "pyannote" -> pyannote.audio 3.1 (state-of-the-art, native overlap
    #               detection). Requires `pip install pyannote.audio`, accepting
    #               the model licences on Hugging Face, and an access token (set
    #               HF_TOKEN, or ``hf_token`` below).
    # "auto" -> pyannote if it's installed and a token is available, else builtin.
    diarization_backend: str = "builtin"
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    # Hugging Face access token for the gated pyannote models. None -> read the
    # HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variable instead.
    hf_token: Optional[str] = None

    # --- Voice activity detection ---------------------------------------
    # "silero" (deep model, best), "energy" (pure-numpy fallback), or
    # "auto" (silero if available, else energy).
    vad_backend: str = "auto"
    # Ignore speech chunks shorter than this many seconds (removes clicks).
    vad_min_speech_sec: float = 0.25
    # Bridge silence gaps shorter than this so a single utterance stays whole.
    vad_min_silence_sec: float = 0.20
    # Energy-VAD only: sensitivity in dB below the running peak.
    energy_vad_threshold_db: float = 30.0

    # --- Compute device (GPU acceleration) ------------------------------
    # "auto" (use CUDA GPU if available), "cuda" (force GPU) or "cpu".
    # The neural backends (ECAPA embeddings, Silero VAD) honour this.
    device: str = "cuda"
    # GPU-only: refuse to run neural stages on the CPU. If CUDA is unavailable,
    # or a neural stage cannot run on the GPU, raise an error instead of
    # silently falling back to a CPU backend.
    gpu_only: bool = True
    # Windows embedded per GPU batch. 0 = auto: read the GPU's free VRAM and
    # size the batch to fill it (leaving headroom) so work stays on the GPU
    # rather than spilling to the CPU. Set a positive number to force a size.
    embed_batch_size: int = 0
    # VRAM headroom (GB) to leave free when auto-sizing the batch.
    vram_reserve_gb: float = 1.2

    # --- Overlapping speech ---------------------------------------------
    # Detect regions where two speakers talk at once. What happens to them is
    # set by ``overlap_mode``.
    remove_overlap: bool = True
    # A window is "overlap" when its similarity to the two nearest speaker
    # voiceprints differs by less than this. HIGHER = more aggressive: more
    # cross-talk / ambiguous audio is flagged. ~0.08 light, 0.15 medium,
    # 0.25 strong, 0.35 extreme.
    overlap_margin: float = 0.15
    # "delete"   -> silence overlap in BOTH tracks (guaranteed no bleed).
    # "separate" -> un-mix the two voices with SepFormer and route each to its
    #               speaker; delete only where they can't be told apart.
    overlap_mode: str = "separate"
    # SepFormer separation model (via SpeechBrain, GPU). 2-speaker. The "whamr"
    # variant is trained with real noise + reverb, so it generalises to real
    # recordings far better than the clean "wsj02mix" model.
    sepformer_model: str = "speechbrain/sepformer-whamr"
    # Overlap is separated in sub-chunks of this many seconds (bounds VRAM).
    sep_chunk_sec: float = 8.0
    # Min confidence margin (own-speaker minus other-speaker cosine similarity)
    # for a separated source; below this the slice is deleted instead. Lower =
    # un-mix more (keeps more overlap, but risks some bleed on fuzzy splits).
    overlap_assign_margin: float = 0.10

    # --- Residual cross-talk gate ---------------------------------------
    # Segment-boundary masking keeps a speaker's whole segment, but a brief
    # foreign sound *inside* that segment - a laugh or a short interjection
    # that isn't simultaneous speech - is never flagged as overlap and leaks
    # into the track. This pass re-checks each speaker's own segments at fine
    # resolution against the voiceprints and SILENCES windows that clearly
    # belong to a different speaker ("delete if it's someone else").
    crosstalk_gate: bool = True
    crosstalk_win: float = 0.8       # fine analysis window (seconds)
    crosstalk_hop: float = 0.4       # step between fine windows
    # Silence a window only when another speaker's voiceprint beats this
    # speaker's by at least this cosine margin. LOWER = more aggressive.
    # ~0.10 light, 0.07 medium, 0.05 strong, 0.035 extreme.
    crosstalk_margin: float = 0.07
    crosstalk_min_rms: float = 0.005  # skip near-silent windows (already inaudible)

    # --- Embedding windows ----------------------------------------------
    # "ecapa" (SpeechBrain deep embeddings, best), "mfcc" (librosa fallback),
    # or "auto" (ecapa if available, else mfcc).
    embedding_backend: str = "auto"
    window_sec: float = 1.5          # length of each analysis window
    hop_sec: float = 0.75            # step between windows (50% overlap)
    # Minimum audio (seconds) fed to the embedder per window. Short speech
    # regions are centre-extended to this length so they don't produce
    # outlier voiceprints that break clustering.
    min_window_sec: float = 1.0

    # --- Segment post-processing ----------------------------------------
    min_segment_sec: float = 0.40    # merge/relabel speech runs shorter than this
    boundary_fade_ms: float = 8.0    # fade at segment edges to avoid clicks

    # --- Noise reduction / vocal cleanup (runs before diarization) ------
    denoise: bool = True
    # "rvc_hp5" (UVR/RVC HP5 vocal isolation on GPU - best, isolates human
    # voices from music/noise), "noisereduce", "spectral", "none", or "auto"
    # (rvc_hp5 if available, else noisereduce).
    denoise_backend: str = "rvc_hp5"
    # UVR VR-arch model used for HP5-style vocal isolation (via audio-separator,
    # runs on the GPU through PyTorch). "4_HP-Vocal-UVR.pth" isolates human
    # vocals (the RVC-HP5 function); other HP models are selectable here.
    rvc_hp5_model: str = "4_HP-Vocal-UVR.pth"
    # Process RVC vocal isolation in chunks of this many seconds (with a short
    # crossfade) so long files don't load entirely into RAM and you get
    # per-chunk progress. 0 = whole file at once.
    rvc_chunk_sec: float = 300.0
    denoise_strength: float = 0.85   # 0..1, how aggressively to cut noise (non-RVC)
    highpass_hz: float = 80.0        # remove sub-bass rumble/hum

    # --- Per-speaker studio enhancement (runs on each exported track) ---
    enhance: bool = True
    presence_gain_db: float = 3.0    # peaking-EQ clarity boost...
    presence_hz: float = 4000.0      # ...centred here
    air_gain_db: float = 1.5         # gentle high-shelf "air"
    compress: bool = True
    comp_threshold_db: float = -22.0
    comp_ratio: float = 3.0
    target_lufs: float = -16.0       # loudness-normalize target (speech standard)
    peak_ceiling_db: float = -1.0    # brickwall ceiling after normalization

    # --- Transcription (optional; Whisper on GPU) -----------------------
    # Transcribe the audio and attribute each word to a speaker, producing a
    # speaker-labelled transcript + SRT/VTT subtitles + per-speaker text.
    transcribe: bool = False
    # Whisper model size: tiny, base, small, medium, large-v3 (bigger = better
    # + slower + more VRAM). "small" is a good balance on a 6 GB card.
    whisper_model: str = "small"
    # Force a language code (e.g. "en") or None to auto-detect.
    whisper_language: Optional[str] = None

    # --- Output ----------------------------------------------------------
    # Export format: wav, flac, ogg (via libsndfile) or mp3, aac (via ffmpeg).
    export_format: str = "wav"
    mp3_bitrate: str = "192k"        # bitrate for lossy formats (mp3/aac)
    # Export one full-length track per speaker (their voice, silence elsewhere).
    export_per_speaker: bool = True
    # Also export a compact track per speaker with only their segments concatenated.
    export_compact: bool = False

    def resolved_speaker_range(self) -> range:
        """The candidate speaker counts to try when auto-detecting."""
        lo = max(1, int(self.min_speakers))
        hi = max(lo, int(self.max_speakers))
        return range(lo, hi + 1)

    def as_dict(self) -> dict:
        return asdict(self)
