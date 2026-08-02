"""GPU / device selection and reporting.

Central place that decides whether to run the neural backends (ECAPA
embeddings, Silero VAD) on CUDA or the CPU, and logs what was chosen so the
user can see the GPU is actually being used.
"""
from __future__ import annotations

from .logutil import get_logger

log = get_logger()

_LOGGED_ONCE = False


def resolve_device(pref: str = "auto", gpu_only: bool = False) -> str:
    """Return the concrete device string ("cuda" or "cpu").

    ``pref`` is one of "auto" (use CUDA if available), "cuda" (force GPU) or
    "cpu". When ``gpu_only`` is True the GPU is mandatory: if CUDA is not
    available this raises instead of silently falling back to the CPU.
    """
    pref = (pref or "auto").lower()

    try:
        import torch
    except Exception as exc:
        if gpu_only or pref == "cuda":
            raise RuntimeError(
                "GPU-only mode requires a CUDA build of PyTorch, which is not "
                "installed. Install it with:\n  pip install torch torchaudio "
                "--index-url https://download.pytorch.org/whl/cu121") from exc
        return "cpu"

    cuda_ok = bool(torch.cuda.is_available())

    if gpu_only:
        if not cuda_ok:
            raise RuntimeError(
                "GPU-only mode is on but no CUDA GPU is available. Ensure a CUDA "
                "build of PyTorch is installed and your NVIDIA driver works, or "
                "set gpu_only=False (device='cpu') to allow CPU.")
        return "cuda"

    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not cuda_ok:
            log.warning("CUDA requested but not available (no GPU torch build / "
                        "driver). Falling back to CPU.")
            return "cpu"
        return "cuda"
    # auto
    return "cuda" if cuda_ok else "cpu"


def log_device_info(device: str, force: bool = False) -> None:
    """Log a one-line summary of the compute device (GPU name + VRAM if CUDA)."""
    global _LOGGED_ONCE
    if _LOGGED_ONCE and not force:
        return
    _LOGGED_ONCE = True

    if device == "cuda":
        try:
            import torch
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            log.info("Compute device: GPU - %s (%.1f GB VRAM, CUDA %s)",
                     name, total, torch.version.cuda)
        except Exception:
            log.info("Compute device: GPU (CUDA)")
    else:
        log.info("Compute device: CPU")


def gpu_free_total_gb() -> tuple[float, float]:
    """Return (free_gb, total_gb) of the current CUDA device, or (0, 0)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        free, total = torch.cuda.mem_get_info()
        return free / 1024 ** 3, total / 1024 ** 3
    except Exception:
        return 0.0, 0.0


def auto_batch_size(reserve_gb: float = 1.2, per_window_mb: float = 14.0,
                    min_bs: int = 16, max_bs: int = 256) -> int:
    """Choose an embedding batch size that fills the GPU's *dedicated* VRAM.

    Reads the currently free VRAM, leaves ``reserve_gb`` headroom, and sizes the
    batch to use the rest — so the work runs on the GPU instead of spilling to
    the CPU. Clamped to [min_bs, max_bs].
    """
    free_gb, total_gb = gpu_free_total_gb()
    if total_gb <= 0:
        return min_bs
    usable_mb = max(0.0, free_gb * 1024 - reserve_gb * 1024)
    bs = int(usable_mb / max(1.0, per_window_mb))
    bs = max(min_bs, min(max_bs, bs))
    log.info("VRAM: %.1f GB free of %.1f GB dedicated -> embedding batch size %d",
             free_gb, total_gb, bs)
    return bs


def cuda_mem_summary() -> str:
    """Short free/total VRAM string for logging around heavy ops."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        free, total = torch.cuda.mem_get_info()
        return f"{(total - free) / 1024**3:.2f}/{total / 1024**3:.2f} GB VRAM used"
    except Exception:
        return ""


def empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
