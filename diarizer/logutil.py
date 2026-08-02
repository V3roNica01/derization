"""Verbose logging helpers shared by the CLI and GUI.

The pipeline emits everything through a standard ``logging.Logger`` so that:
  * the CLI can print a rich, timestamped trace to the terminal, and
  * the GUI can attach a handler that streams the same lines into its console.
"""
from __future__ import annotations

import logging
import sys
from typing import Callable, Optional


LOGGER_NAME = "diarizer"


class _ColorFormatter(logging.Formatter):
    """A compact formatter with optional ANSI colour for terminals."""

    COLORS = {
        logging.DEBUG: "\033[37m",     # grey
        logging.INFO: "\033[36m",      # cyan
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool) -> None:
        super().__init__("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelno, "")
            if color:
                text = f"{color}{text}{self.RESET}"
        return text


def get_logger() -> logging.Logger:
    """Return the package logger (does not add handlers)."""
    return logging.getLogger(LOGGER_NAME)


def configure_console_logging(verbose: bool = True, use_color: Optional[bool] = None) -> logging.Logger:
    """Set up terminal logging for CLI usage.

    ``verbose`` toggles DEBUG vs INFO. Returns the configured logger.
    """
    logger = get_logger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    # Remove any stream handlers we previously attached (avoid duplicates).
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "_diarizer_console", False):
            logger.removeHandler(handler)

    if use_color is None:
        use_color = _supports_color()

    handler = logging.StreamHandler(sys.stderr)
    handler._diarizer_console = True  # type: ignore[attr-defined]
    handler.setFormatter(_ColorFormatter(use_color))
    logger.addHandler(handler)
    return logger


class CallbackLogHandler(logging.Handler):
    """A logging handler that forwards formatted lines to a callback.

    The GUI uses this to pipe log records into its text console. The callback
    receives ``(levelno, message)`` and must be thread-safe (the GUI wraps it
    with a queue).
    """

    def __init__(self, callback: Callable[[int, str], None]) -> None:
        super().__init__()
        self.callback = callback
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.callback(record.levelno, msg)
        except Exception:  # never let logging crash the app
            self.handleError(record)


def quiet_dependency_warnings() -> None:
    """Silence noisy (but harmless) warnings from third-party libraries so the
    console/log stays readable. Only targets known messages; our own
    ``diarizer`` logger is untouched."""
    import warnings

    for msg in (r".*weights_only=False.*", r".*padding='same'.*",
                r".*Using padding='same'.*", r".*n_fft=.*",
                r".*[Cc]lipped samples.*"):
        warnings.filterwarnings("ignore", message=msg)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"noisereduce.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pyloudnorm.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"speechbrain.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch.*")
    # SpeechBrain prints device/info chatter through its own logger.
    logging.getLogger("speechbrain").setLevel(logging.ERROR)


def _supports_color() -> bool:
    # Windows 10+ terminals and most *nix terminals support ANSI.
    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    return True
