#!/usr/bin/env python3
"""Desktop GUI for Derization (Tkinter, ships with Python).

A single window that lets you pick an audio file, choose the number of
speakers, run diarization on a background thread, and watch a verbose,
terminal-style log stream while it works. When finished it lists the output
files and can open the output folder.

Run with:  python gui.py
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from diarizer.config import DiarizationConfig
from diarizer.logutil import CallbackLogHandler, get_logger


# --- Midnight Galaxy theme (from the theme-factory) ---------------------
THEME = {
    "bg": "#2b1e3e",        # deep purple - rich dark base
    "panel": "#372a4f",     # slightly lifted panel
    "accent": "#4a4e8f",    # cosmic blue
    "accent2": "#a490c2",   # lavender accent
    "text": "#e6e6fa",      # silver text
    "subtle": "#b9a9d6",    # muted lavender
    "console_bg": "#1a1526",
    "ok": "#8fe3b0",
}

# Colours for the "terminal" log console (tuned for the dark purple base).
LOG_COLORS = {
    logging.DEBUG: "#8a7fa6",
    logging.INFO: "#e6e6fa",
    logging.WARNING: "#e5c07b",
    logging.ERROR: "#ff7b8a",
    logging.CRITICAL: "#ff5555",
}


def apply_theme(root: tk.Tk) -> None:
    """Apply the Midnight Galaxy palette to all ttk widgets."""
    t = THEME
    style = ttk.Style(root)
    style.theme_use("clam")  # clam allows full colour control
    root.configure(bg=t["bg"])

    body = ("Segoe UI", 10)
    style.configure(".", background=t["bg"], foreground=t["text"],
                    fieldbackground=t["panel"], font=body, bordercolor=t["accent"])
    style.configure("TFrame", background=t["bg"])
    style.configure("TLabel", background=t["bg"], foreground=t["text"])
    style.configure("Header.TLabel", background=t["bg"], foreground=t["accent2"],
                    font=("Segoe UI Semibold", 16))
    style.configure("Sub.TLabel", background=t["bg"], foreground=t["subtle"],
                    font=("Segoe UI", 9))
    style.configure("TLabelframe", background=t["bg"], bordercolor=t["accent"],
                    relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=t["bg"], foreground=t["accent2"],
                    font=("Segoe UI Semibold", 10))
    style.configure("TButton", background=t["accent"], foreground=t["text"],
                    borderwidth=0, focusthickness=0, padding=7,
                    font=("Segoe UI Semibold", 10))
    style.map("TButton",
              background=[("active", t["accent2"]), ("disabled", "#3a3350")],
              foreground=[("disabled", "#8a83a0")])
    style.configure("Accent.TButton", background=t["accent2"], foreground="#241a33",
                    padding=9, font=("Segoe UI Semibold", 11))
    style.map("Accent.TButton", background=[("active", "#c3b4dd"),
                                            ("disabled", "#4a4066")])
    style.configure("TCheckbutton", background=t["bg"], foreground=t["text"])
    style.map("TCheckbutton", background=[("active", t["bg"])],
              indicatorcolor=[("selected", t["accent2"]), ("!selected", t["panel"])])
    style.configure("TEntry", fieldbackground=t["panel"], foreground=t["text"],
                    bordercolor=t["accent"], insertcolor=t["text"])
    style.configure("TCombobox", fieldbackground=t["panel"], background=t["accent"],
                    foreground=t["text"], arrowcolor=t["text"], bordercolor=t["accent"])
    style.map("TCombobox", fieldbackground=[("readonly", t["panel"])],
              foreground=[("readonly", t["text"])],
              selectbackground=[("readonly", t["panel"])])
    style.configure("TProgressbar", background=t["accent2"], troughcolor=t["panel"],
                    bordercolor=t["bg"], lightcolor=t["accent2"], darkcolor=t["accent"])
    style.configure("Vertical.TScrollbar", background=t["panel"], troughcolor=t["bg"],
                    bordercolor=t["bg"], arrowcolor=t["text"])


class DiarizationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Derization - Speaker Separation")
        self.root.geometry("860x640")
        self.root.minsize(720, 520)

        self.queue: "queue.Queue" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_outdir: str | None = None

        self._build_widgets()
        self.root.after(80, self._drain_queue)
        # Detect the GPU without blocking the UI (torch import can be slow).
        threading.Thread(target=self._probe_gpu, daemon=True).start()

    # ------------------------------------------------------------------ UI
    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # --- Header banner ---
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=12, pady=(12, 2))
        ttk.Label(header, text="✦ Derization", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="  GPU speaker separation",
                  style="Sub.TLabel").pack(side="left", padx=(6, 0), pady=(8, 0))

        # --- Input file row ---
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Audio file:").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(top, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Browse...", command=self._pick_input).grid(row=0, column=2)

        ttk.Label(top, text="Output folder:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.outdir_var = tk.StringVar(value=str(Path.cwd() / "diarization_output"))
        self.outdir_entry = ttk.Entry(top, textvariable=self.outdir_var)
        self.outdir_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(top, text="Browse...", command=self._pick_outdir).grid(row=1, column=2, pady=(6, 0))
        top.columnconfigure(1, weight=1)

        # --- Options row ---
        opts = ttk.LabelFrame(self.root, text="Options")
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="Speakers:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.speakers_var = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.speakers_var, width=8, state="readonly",
                     values=["auto", "2", "3"]).grid(row=0, column=1, sticky="w")

        ttk.Label(opts, text="VAD:").grid(row=0, column=2, sticky="w", padx=(16, 6))
        self.vad_var = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.vad_var, width=8, state="readonly",
                     values=["auto", "silero", "energy"]).grid(row=0, column=3, sticky="w")

        ttk.Label(opts, text="Embeddings:").grid(row=0, column=4, sticky="w", padx=(16, 6))
        self.emb_var = tk.StringVar(value="auto")
        ttk.Combobox(opts, textvariable=self.emb_var, width=8, state="readonly",
                     values=["auto", "ecapa", "mfcc"]).grid(row=0, column=5, sticky="w")

        ttk.Label(opts, text="Device:").grid(row=0, column=6, sticky="w", padx=(16, 6))
        self.device_var = tk.StringVar(value="cuda")
        ttk.Combobox(opts, textvariable=self.device_var, width=7, state="readonly",
                     values=["cuda", "auto", "cpu"]).grid(row=0, column=7, sticky="w", padx=(0, 6))

        # Live GPU status (probed in the background so startup stays instant).
        self.gpu_var = tk.StringVar(value="GPU: checking...")
        ttk.Label(opts, textvariable=self.gpu_var, foreground=THEME["ok"],
                  background=THEME["bg"]).grid(
            row=1, column=0, columnspan=8, sticky="w", padx=6, pady=(2, 2))

        ttk.Label(opts, text="Export format:").grid(row=2, column=0, sticky="w", padx=6)
        self.format_var = tk.StringVar(value="wav")
        ttk.Combobox(opts, textvariable=self.format_var, width=7, state="readonly",
                     values=["wav", "flac", "ogg", "mp3", "aac"]).grid(row=2, column=1, sticky="w")
        self.denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Reduce noise", variable=self.denoise_var).grid(
            row=2, column=2, columnspan=2, sticky="w", padx=(16, 6))
        self.enhance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Enhance voices", variable=self.enhance_var).grid(
            row=2, column=4, columnspan=2, sticky="w", padx=6)

        ttk.Label(opts, text="Overlap cut:").grid(row=2, column=6, sticky="w", padx=(16, 6))
        self.overlap_var = tk.StringVar(value="medium")
        ttk.Combobox(opts, textvariable=self.overlap_var, width=8, state="readonly",
                     values=["off", "light", "medium", "strong", "extreme"]).grid(
                         row=2, column=7, sticky="w")

        ttk.Label(opts, text="Overlap:").grid(row=3, column=0, sticky="w", padx=6)
        self.overlap_mode_var = tk.StringVar(value="un-mix")
        ttk.Combobox(opts, textvariable=self.overlap_mode_var, width=8, state="readonly",
                     values=["un-mix", "delete"]).grid(row=3, column=1, sticky="w")
        ttk.Label(opts, text="(un-mix = separate simultaneous voices; "
                             "delete = silence them)", style="Sub.TLabel").grid(
            row=3, column=2, columnspan=6, sticky="w", padx=(8, 0))

        self.compact_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Also export compact tracks",
                        variable=self.compact_var).grid(row=4, column=0, columnspan=4,
                                                         sticky="w", padx=6, pady=(2, 6))
        self.verbose_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Verbose log",
                        variable=self.verbose_var).grid(row=4, column=4, columnspan=3,
                                                        sticky="w", padx=6, pady=(2, 6))

        self.transcribe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Transcribe (speaker-labeled + subtitles)",
                        variable=self.transcribe_var).grid(row=5, column=0, columnspan=4,
                                                           sticky="w", padx=6, pady=(0, 6))
        ttk.Label(opts, text="Whisper:").grid(row=5, column=4, sticky="e", padx=(6, 4))
        self.whisper_var = tk.StringVar(value="small")
        ttk.Combobox(opts, textvariable=self.whisper_var, width=9, state="readonly",
                     values=["tiny", "base", "small", "medium", "large-v3"]).grid(
                         row=5, column=5, columnspan=2, sticky="w", padx=(0, 6))

        ttk.Label(opts, text="Engine:").grid(row=6, column=0, sticky="w", padx=6)
        self.engine_var = tk.StringVar(value="builtin")
        ttk.Combobox(opts, textvariable=self.engine_var, width=9, state="readonly",
                     values=["builtin", "pyannote"]).grid(row=6, column=1, sticky="w")
        ttk.Label(opts, text="(pyannote = state-of-the-art; needs setup + HF_TOKEN, "
                             "else falls back to builtin)", style="Sub.TLabel").grid(
            row=6, column=2, columnspan=6, sticky="w", padx=(8, 0), pady=(0, 6))

        # --- Action row ---
        actions = ttk.Frame(self.root)
        actions.pack(fill="x", **pad)
        self.run_btn = ttk.Button(actions, text="✦  Separate speakers",
                                   style="Accent.TButton", command=self._start)
        self.run_btn.pack(side="left")
        self.open_btn = ttk.Button(actions, text="Open output folder",
                                   command=self._open_output, state="disabled")
        self.open_btn.pack(side="left", padx=8)
        ttk.Button(actions, text="Clear log", command=self._clear_log).pack(side="left")

        self.progress = ttk.Progressbar(actions, mode="determinate", maximum=1.0)
        self.progress.pack(side="right", fill="x", expand=True, padx=(12, 0))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10)

        # --- Log console ---
        console_frame = ttk.Frame(self.root)
        console_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.console = tk.Text(console_frame, wrap="none", bg=THEME["console_bg"],
                               fg=THEME["text"], insertbackground=THEME["accent2"],
                               font=("Consolas", 10), state="disabled", relief="flat",
                               highlightthickness=1, highlightbackground=THEME["accent"])
        yscroll = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        self.console.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")
        self.console.pack(side="left", fill="both", expand=True)
        for level, color in LOG_COLORS.items():
            self.console.tag_configure(f"lvl{level}", foreground=color)

    # --------------------------------------------------------------- events
    def _pick_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[("Audio", "*.wav *.flac *.ogg *.mp3 *.m4a *.aac *.wma"),
                       ("All files", "*.*")],
        )
        if path:
            self.input_var.set(path)
            # Default the output folder next to the chosen file.
            self.outdir_var.set(str(Path(path).with_suffix("").parent /
                                    (Path(path).stem + "_speakers")))

    def _pick_outdir(self) -> None:
        path = filedialog.askdirectory(title="Choose an output folder")
        if path:
            self.outdir_var.set(path)

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        input_path = self.input_var.get().strip()
        if not input_path or not Path(input_path).exists():
            messagebox.showerror("Missing file", "Please choose a valid audio file.")
            return

        cfg = self._build_config()
        outdir = self.outdir_var.get().strip() or "diarization_output"
        self.last_outdir = outdir

        self.run_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.progress.configure(value=0.0)
        self._log_line(logging.INFO, f"Starting: {input_path}")
        self.status_var.set("Working...")

        self.worker = threading.Thread(
            target=self._worker, args=(input_path, outdir, cfg), daemon=True)
        self.worker.start()

    def _build_config(self) -> DiarizationConfig:
        spk = self.speakers_var.get()
        num = None if spk == "auto" else int(spk)
        # (remove_overlap, overlap_margin, crosstalk_gate, keep_margin, self_floor)
        ov_map = {"off":     (False, 0.15, False, 0.03, 0.55),
                  "light":   (True,  0.08, True,  0.00, 0.45),
                  "medium":  (True,  0.15, True,  0.03, 0.55),
                  "strong":  (True,  0.25, True,  0.07, 0.62),
                  "extreme": (True,  0.35, True,  0.12, 0.70)}
        remove_overlap, overlap_margin, ct_gate, ct_keep, ct_floor = ov_map.get(
            self.overlap_var.get(), (True, 0.15, True, 0.03, 0.55))
        overlap_mode = "separate" if self.overlap_mode_var.get() == "un-mix" else "delete"
        return DiarizationConfig(
            num_speakers=num,
            diarization_backend=self.engine_var.get(),
            vad_backend=self.vad_var.get(),
            embedding_backend=self.emb_var.get(),
            device=self.device_var.get(),
            denoise=self.denoise_var.get(),
            enhance=self.enhance_var.get(),
            export_format=self.format_var.get(),
            export_compact=self.compact_var.get(),
            remove_overlap=remove_overlap,
            overlap_margin=overlap_margin,
            crosstalk_gate=ct_gate,
            crosstalk_keep_margin=ct_keep,
            crosstalk_self_floor=ct_floor,
            overlap_mode=overlap_mode,
            transcribe=self.transcribe_var.get(),
            whisper_model=self.whisper_var.get(),
        )

    # ----------------------------------------------------------------- gpu
    def _probe_gpu(self) -> None:
        """Report GPU availability into the status label (runs off-thread)."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                text = f"GPU: {name}  ({gb:.1f} GB, CUDA {torch.version.cuda}) - ready"
            else:
                text = "GPU: PyTorch has no CUDA support - install a CUDA build for GPU speed"
        except Exception:
            text = "GPU: PyTorch not installed - CPU fallback active (install torch for GPU)"
        self.queue.put(("gpustatus", text))

    # --------------------------------------------------------------- worker
    def _worker(self, input_path: str, outdir: str, cfg: DiarizationConfig) -> None:
        """Runs off the UI thread; communicates only via the queue."""
        logger = get_logger()
        logger.setLevel(logging.DEBUG if self.verbose_var.get() else logging.INFO)
        handler = CallbackLogHandler(lambda lvl, msg: self.queue.put(("log", lvl, msg)))
        logger.addHandler(handler)
        try:
            from diarizer import process_file

            t0 = time.time()
            result, files = process_file(
                input_path, outdir, config=cfg,
                progress=lambda m, f: self.queue.put(("progress", f, m)))
            self.queue.put(("done", result, files, time.time() - t0))
        except Exception as exc:  # surface to the UI thread
            import traceback
            self.queue.put(("error", str(exc), traceback.format_exc()))
        finally:
            logger.removeHandler(handler)

    # --------------------------------------------------------------- queue
    def _drain_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._log_line(item[1], item[2])
                elif kind == "progress":
                    self.progress.configure(value=item[1])
                    self.status_var.set(item[2])
                elif kind == "gpustatus":
                    self.gpu_var.set(item[1])
                elif kind == "done":
                    self._on_done(item[1], item[2], item[3])
                elif kind == "error":
                    self._on_error(item[1], item[2])
        except queue.Empty:
            pass
        self.root.after(80, self._drain_queue)

    def _on_done(self, result, files, elapsed) -> None:
        self.progress.configure(value=1.0)
        self.run_btn.configure(state="normal")
        self.open_btn.configure(state="normal")
        self.status_var.set(
            f"Done in {elapsed:.1f}s - {result.num_speakers} speaker(s), "
            f"{len(files)} file(s).")
        self._log_line(logging.INFO, "-" * 50)
        self._log_line(logging.INFO, f"Finished: {result.num_speakers} speaker(s) found.")
        for f in files:
            self._log_line(logging.INFO, f"  wrote {Path(f).name}")

        # Make it obvious if no per-speaker AUDIO was produced.
        audio = [f for f in files if Path(f).suffix.lower() in
                 (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac")]
        if not audio:
            self._log_line(logging.WARNING, "No speaker audio was written "
                           "(no speech detected in the file).")
            messagebox.showwarning(
                "No speaker tracks",
                "Only the timeline files were written because no speech was "
                "detected in this recording.\n\nTry a clearer recording, turn "
                "off 'Reduce noise', or set Speakers to 2/3 explicitly.")
        else:
            messagebox.showinfo(
                "Done",
                f"{result.num_speakers} speaker(s) separated.\n"
                f"{len(audio)} audio track(s) written to:\n{self.last_outdir}")

    def _on_error(self, msg: str, tb: str) -> None:
        self.run_btn.configure(state="normal")
        self.status_var.set("Error.")
        self._log_line(logging.ERROR, f"ERROR: {msg}")
        for line in tb.strip().splitlines():
            self._log_line(logging.DEBUG, line)
        messagebox.showerror("Diarization failed", msg)

    # --------------------------------------------------------------- log UI
    def _log_line(self, level: int, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n", f"lvl{level}")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _clear_log(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def _open_output(self) -> None:
        if not self.last_outdir:
            return
        path = Path(self.last_outdir)
        if not path.exists():
            messagebox.showinfo("Not found", "The output folder does not exist yet.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception as exc:
            messagebox.showwarning("Could not open", str(exc))


def main() -> int:
    root = tk.Tk()
    try:
        apply_theme(root)
    except Exception:
        pass
    DiarizationApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
