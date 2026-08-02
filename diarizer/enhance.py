"""Noise reduction and per-speaker studio enhancement (DSP stages).

Two entry points:
  * ``denoise(samples, sr, cfg, device)`` - runs on the whole recording BEFORE
    diarization: removes ambient/background noise so only the voices remain.
  * ``enhance_track(track, sr, cfg)`` - runs on each isolated speaker track:
    high-pass, presence EQ, gentle compression, loudness normalization.

Everything is float32, mono ``(n,)`` or multi-channel ``(n, ch)`` and filters
are applied along axis 0 so both shapes work.
"""
from __future__ import annotations

import numpy as np

from .config import DiarizationConfig
from .logutil import get_logger

log = get_logger()


# ======================================================================= #
# Noise reduction
# ======================================================================= #
def denoise(samples: np.ndarray, sr: int, cfg: DiarizationConfig,
            device: str = "cpu", progress=None) -> np.ndarray:
    """Reduce background noise / isolate vocals on the full recording.

    ``progress`` (optional) is called as ``progress(message, fraction)`` with
    fraction in 0..1 so the caller (GUI) can show per-chunk RVC progress.
    """
    if not cfg.denoise or cfg.denoise_backend == "none":
        return samples

    backend = cfg.denoise_backend
    if backend == "auto":
        backend = "rvc_hp5" if _rvc_available() else "noisereduce"

    if backend == "rvc_hp5":
        try:
            from .rvc_denoise import available as rvc_available, isolate_vocals
            if not rvc_available():
                raise RuntimeError("the 'audio-separator' package is not installed")
            out = isolate_vocals(samples, sr, cfg, device, progress=progress)
            log.info("Noise reduction: RVC/UVR HP5 vocal isolation (%s) on %s",
                     cfg.rvc_hp5_model, device.upper())
            out = _highpass(out, sr, cfg.highpass_hz)
            return out.astype(np.float32, copy=False)
        except Exception as exc:
            if cfg.gpu_only:
                raise RuntimeError(
                    f"GPU-only mode: RVC HP5 vocal isolation failed: {exc}") from exc
            log.warning("RVC HP5 denoise failed (%s); falling back to noisereduce", exc)
            backend = "noisereduce"

    try:
        if backend == "noisereduce":
            out = _denoise_noisereduce(samples, sr, cfg, device)
            log.info("Noise reduction: noisereduce (%s)",
                     "GPU" if device == "cuda" else "CPU")
        else:
            out = _denoise_spectral(samples, sr, cfg)
            log.info("Noise reduction: spectral gating (fallback)")
    except Exception as exc:
        if cfg.gpu_only:
            raise
        log.warning("Noise reduction failed (%s); using spectral fallback", exc)
        out = _denoise_spectral(samples, sr, cfg)

    # A high-pass removes sub-bass rumble/hum that gating leaves behind.
    out = _highpass(out, sr, cfg.highpass_hz)
    return out.astype(np.float32, copy=False)


def _rvc_available() -> bool:
    try:
        import audio_separator  # noqa: F401
        return True
    except Exception:
        return False


def _noisereduce_available() -> bool:
    try:
        import noisereduce  # noqa: F401
        return True
    except Exception:
        return False


def _denoise_noisereduce(samples, sr, cfg, device):
    import noisereduce as nr
    prop = float(np.clip(cfg.denoise_strength, 0.0, 1.0))
    use_gpu = device == "cuda"

    def run(ch):
        if use_gpu:
            try:
                return nr.reduce_noise(y=ch, sr=sr, stationary=False,
                                       prop_decrease=prop, use_torch=True,
                                       device="cuda")
            except Exception as exc:
                log.debug("GPU denoise unavailable (%s); using CPU", exc)
        return nr.reduce_noise(y=ch, sr=sr, stationary=False, prop_decrease=prop)

    if samples.ndim == 1:
        return run(samples)
    return np.stack([run(samples[:, c]) for c in range(samples.shape[1])], axis=1)


def _denoise_spectral(samples, sr, cfg):
    """Dependency-free spectral-subtraction noise gate."""
    from scipy.signal import stft, istft
    prop = float(np.clip(cfg.denoise_strength, 0.0, 1.0))

    def run(ch):
        f, t, Z = stft(ch, fs=sr, nperseg=1024, noverlap=768)
        mag, phase = np.abs(Z), np.angle(Z)
        noise = np.percentile(mag, 10, axis=1, keepdims=True)  # per-freq floor
        gain = np.clip((mag - prop * 1.5 * noise) / (mag + 1e-9), 0.0, 1.0)
        _, y = istft(gain * mag * np.exp(1j * phase), fs=sr, nperseg=1024, noverlap=768)
        return y[:len(ch)]

    if samples.ndim == 1:
        return run(samples)
    return np.stack([run(samples[:, c]) for c in range(samples.shape[1])], axis=1)


# ======================================================================= #
# Per-speaker enhancement
# ======================================================================= #
def enhance_track(track: np.ndarray, sr: int, cfg: DiarizationConfig) -> np.ndarray:
    """Apply the studio-enhancement chain to one speaker track.

    Streams the whole chain (high-pass, presence EQ, air shelf, compressor,
    loudness + soft-clip) in fixed 60-second blocks with the IIR filter state
    carried across blocks. Memory stays ~constant regardless of length, so an
    hour-long track no longer allocates multi-GB buffers (which previously
    thrashed RAM and could take hours). Output equals the whole-file chain
    except for negligible block-edge effects in the compressor envelope.
    """
    if not cfg.enhance:
        return track
    from scipy.signal import sosfilt

    x = np.array(track, dtype=np.float32)          # own copy; edited in place
    n = x.shape[0]
    ceil = np.float32(10 ** (cfg.peak_ceiling_db / 20.0))

    # Global loudness gain, measured once on a cheap ~16 kHz mono proxy.
    loud = _measure_loudness(x, sr)
    if loud is not None and np.isfinite(loud) and loud > -70:
        gain = np.float32(10.0 ** ((cfg.target_lufs - loud) / 20.0))
    else:
        peak = float(np.abs(x).max()) + 1e-9
        gain = np.float32(ceil / peak)

    sos = _enhance_sos(sr, cfg)                     # combined HP + presence + air
    nch = 1 if x.ndim == 1 else x.shape[1]
    zi = (np.zeros((sos.shape[0], 2)) if x.ndim == 1
          else np.zeros((sos.shape[0], 2, nch)))
    # No compressor makeup gain: the loudness normalization below sets the final
    # level, so makeup would only double-count and overshoot the target.
    makeup = np.float32(1.0)

    block = max(1, int(60 * sr))
    for s0 in range(0, n, block):
        s1 = min(n, s0 + block)
        blk = np.asarray(x[s0:s1], dtype=np.float64)
        blk, zi = sosfilt(sos, blk, axis=0, zi=zi)
        blk = blk.astype(np.float32)
        if cfg.compress:
            blk = _compress_block(blk, sr, cfg.comp_threshold_db, cfg.comp_ratio, makeup)
        blk *= gain
        np.divide(blk, ceil, out=blk)              # soft-clip limiter, in place
        np.tanh(blk, out=blk)
        blk *= ceil
        x[s0:s1] = blk
    return x


def _enhance_sos(sr, cfg):
    """Combined second-order-section cascade for the EQ chain."""
    from scipy.signal import butter, tf2sos
    secs = []
    if cfg.highpass_hz > 0:
        secs.append(butter(4, min(cfg.highpass_hz / (sr / 2), 0.99),
                           btype="high", output="sos"))
    if cfg.presence_gain_db:
        b, a = _peaking_coeffs(sr, cfg.presence_hz, cfg.presence_gain_db, 1.0)
        secs.append(tf2sos(b, a))
    if cfg.air_gain_db:
        b, a = _high_shelf_coeffs(sr, 8000.0, cfg.air_gain_db)
        secs.append(tf2sos(b, a))
    if not secs:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    return np.vstack(secs)


def _peaking_coeffs(sr, f0, gain_db, q=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    cos = np.cos(w0)
    b = [1 + alpha * A, -2 * cos, 1 - alpha * A]
    a = [1 + alpha / A, -2 * cos, 1 - alpha / A]
    return [c / a[0] for c in b], [c / a[0] for c in a]


def _high_shelf_coeffs(sr, f0, gain_db, s=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cos, sin = np.cos(w0), np.sin(w0)
    alpha = sin / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    tsa = 2 * np.sqrt(A) * alpha
    b = [A * ((A + 1) + (A - 1) * cos + tsa),
         -2 * A * ((A - 1) + (A + 1) * cos),
         A * ((A + 1) + (A - 1) * cos - tsa)]
    a = [(A + 1) - (A - 1) * cos + tsa,
         2 * ((A - 1) - (A + 1) * cos),
         (A + 1) - (A - 1) * cos - tsa]
    return [c / a[0] for c in b], [c / a[0] for c in a]


def _compress_block(x, sr, threshold_db, ratio, makeup):
    from scipy.ndimage import uniform_filter1d
    eps = np.float32(1e-9)
    det = np.abs(x) if x.ndim == 1 else np.max(np.abs(x), axis=1)
    level_db = (20.0 * np.log10(det + eps)).astype(np.float32)
    over = np.maximum(level_db - np.float32(threshold_db), np.float32(0.0))
    gain_db = over * np.float32(-(1.0 - 1.0 / max(ratio, 1.0)))
    win = max(1, int(round(25.0 * 1e-3 * sr)))
    gain_db = uniform_filter1d(gain_db, size=win, mode="nearest").astype(np.float32)
    g = np.power(np.float32(10.0), gain_db / np.float32(20.0)).astype(np.float32)
    x = x * (g if x.ndim == 1 else g[:, None])
    x *= makeup
    return x


# --- filters ----------------------------------------------------------- #
def _highpass(x, sr, hz):
    if hz <= 0:
        return x
    from scipy.signal import butter, sosfilt
    sos = butter(4, min(hz / (sr / 2), 0.99), btype="high", output="sos").astype(np.float32)
    return sosfilt(sos, np.asarray(x, dtype=np.float32), axis=0).astype(np.float32, copy=False)


def _biquad(x, b, a):
    # Apply as a second-order section in float32 (sosfilt keeps the input dtype,
    # unlike lfilter which upcasts to float64 and doubles memory on long files).
    from scipy.signal import sosfilt, tf2sos
    sos = tf2sos(np.asarray(b, dtype=np.float64),
                 np.asarray(a, dtype=np.float64)).astype(np.float32)
    return sosfilt(sos, np.asarray(x, dtype=np.float32), axis=0).astype(np.float32, copy=False)


def _peaking_eq(x, sr, f0, gain_db, q=1.0):
    """RBJ peaking-EQ biquad."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    cos = np.cos(w0)
    b = [1 + alpha * A, -2 * cos, 1 - alpha * A]
    a = [1 + alpha / A, -2 * cos, 1 - alpha / A]
    b = [c / a[0] for c in b]
    a = [c / a[0] for c in a]
    return _biquad(x, b, a)


def _high_shelf(x, sr, f0, gain_db, s=1.0):
    """RBJ high-shelf biquad."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cos, sin = np.cos(w0), np.sin(w0)
    alpha = sin / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    two_sqrtA_alpha = 2 * np.sqrt(A) * alpha
    b = [A * ((A + 1) + (A - 1) * cos + two_sqrtA_alpha),
         -2 * A * ((A - 1) + (A + 1) * cos),
         A * ((A + 1) + (A - 1) * cos - two_sqrtA_alpha)]
    a = [(A + 1) - (A - 1) * cos + two_sqrtA_alpha,
         2 * ((A - 1) - (A + 1) * cos),
         (A + 1) - (A - 1) * cos - two_sqrtA_alpha]
    b = [c / a[0] for c in b]
    a = [c / a[0] for c in a]
    return _biquad(x, b, a)


def _compress(x, sr, threshold_db, ratio, tau_ms=25.0):
    """Simple downward compressor with a smoothed gain envelope.

    Kept entirely in float32 and applied in-place so it does not allocate
    float64 copies (which blow up RAM on hour-long tracks).
    """
    x = np.asarray(x, dtype=np.float32)
    eps = np.float32(1e-9)
    det = np.abs(x) if x.ndim == 1 else np.max(np.abs(x), axis=1)
    level_db = (20.0 * np.log10(det + eps)).astype(np.float32)
    del det
    over = np.maximum(level_db - np.float32(threshold_db), np.float32(0.0))
    del level_db
    gain_db = over * np.float32(-(1.0 - 1.0 / max(ratio, 1.0)))
    del over

    # Smooth the gain envelope (moving average ~ one-pole), float32, low memory.
    from scipy.ndimage import uniform_filter1d
    win = max(1, int(round(tau_ms * 1e-3 * sr)))
    gain_db = uniform_filter1d(gain_db, size=win, mode="nearest").astype(np.float32)
    gain = np.power(np.float32(10.0), gain_db / np.float32(20.0)).astype(np.float32)
    del gain_db

    makeup = np.float32(10.0 ** ((-threshold_db * 0.4) / 20.0))
    if x.ndim == 1:
        x *= gain
    else:
        x *= gain[:, None]
    x *= makeup
    return x


def _normalize_loudness(x, sr, target_lufs, peak_db):
    """Loudness-normalize to ``target_lufs`` (EBU R128), then soft-limit peaks.

    Peaks are tamed with a per-sample soft clip rather than by rescaling the
    whole track: rescaling would push peaky speakers quieter than others and
    undo the loudness *matching* between speakers.
    """
    x = np.asarray(x, dtype=np.float32)
    ceil = np.float32(10 ** (peak_db / 20.0))
    n = x.shape[0]
    dur = (n / sr) if n else 0
    gain = None

    if dur >= 0.5:
        loud = _measure_loudness(x, sr)
        if loud is not None and np.isfinite(loud) and loud > -70:
            gain = np.float32(10.0 ** ((target_lufs - loud) / 20.0))

    if gain is None:  # peak-normalize toward the ceiling
        peak = float(np.abs(x).max()) + 1e-9
        gain = np.float32(ceil / peak)

    x *= gain  # apply loudness gain in-place (no big float64 copy)

    # Soft-clip limiter: tanh gently bends only near-ceiling peaks toward the
    # ceiling, leaving the bulk (and thus the matched loudness) intact. Done
    # in-place to avoid extra full-length allocations.
    np.divide(x, ceil, out=x)
    np.tanh(x, out=x)
    x *= ceil
    return x


def _measure_loudness(x, sr):
    """Integrated loudness (LUFS). Measured on a ~16 kHz mono proxy so an
    hour-long, high-sample-rate track doesn't need multi-GB float64 buffers."""
    try:
        import pyloudnorm as pyln
    except Exception:
        return None
    mono = x if x.ndim == 1 else x.mean(axis=1)
    step = max(1, int(round(sr / 16000)))
    proxy = np.ascontiguousarray(mono[::step], dtype=np.float32)
    meter_sr = max(1, int(round(sr / step)))
    try:
        return float(pyln.Meter(meter_sr).integrated_loudness(proxy))
    except Exception as exc:
        log.debug("Loudness measurement failed (%s)", exc)
        return None
