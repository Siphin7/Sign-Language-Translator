#!/usr/bin/env python3
# =============================================================
#  app.py  ──  STEP 3: Main Sign Language Translator Application
# =============================================================
#
#  Prerequisites
#  -------------
#  1. python collect_data.py   → build training data
#  2. python train_model.py    → train ML model
#  3. python app.py            → run this app  ← YOU ARE HERE
#
#  Keyboard Shortcuts
#  ------------------
#  Enter        → Speak sentence aloud
#  Ctrl+Z       → Undo last committed sign
#  Ctrl+Space   → Add a space character
#  Escape       → Clear sentence
#  Ctrl+C       → Copy sentence to clipboard
#  F1           → Toggle Mirror mode
#
#  Features
#  --------
#  ✓  Real-time Sign → Text  (webcam + trained ML model)
#  ✓  Sign → Voice           (gTTS online / pyttsx3 offline)
#  ✓  Voice → Text           (microphone → SpeechRecognition)
#  ✓  Sentence builder with stability gate (15 stable frames)
#  ✓  Sentence history (last 20 spoken sentences)
#  ✓  Undo / Clear / Copy controls
#  ✓  Confidence threshold slider
#  ✓  Camera mirror & index selector
#  ✓  TTS rate & volume sliders
#  ✓  Auto-speak toggle
#  ✓  Keyboard shortcuts
# =============================================================

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# PIL must be imported at module level (not inside the loop)
from PIL import Image, ImageTk

import cv2
import numpy as np

# ── Ensure src/ is importable from any working directory ─────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.gesture_recognizer import GestureRecognizer
from src.sentence_builder   import SentenceBuilder
from src.tts_engine         import TTSEngine
from src.stt_engine         import STTEngine

# ── Paths ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_model.pkl")

# ── Design tokens (dark-mode tech palette) ────────────────────────────
BG_DARK   = "#0d1117"
BG_CARD   = "#161b22"
BG_PANEL  = "#1c2128"
ACCENT    = "#00e5a0"    # teal – primary action / active sign
ACCENT2   = "#4d9eff"    # blue – secondary / building text
WARN      = "#ff6b6b"    # red  – destructive actions
TEXT_MAIN = "#e6edf3"
TEXT_DIM  = "#8b949e"
BORDER    = "#30363d"

FONT_SANS = "Segoe UI"    if sys.platform == "win32" else "Helvetica Neue"
FONT_MONO = "Courier New" if sys.platform == "win32" else "Courier"


# ═══════════════════════════════════════════════════════════════════════
#  Utility: create a consistent styled button
# ═══════════════════════════════════════════════════════════════════════
def make_btn(parent, text: str, colour: str, command,
             width: int = 0, font_size: int = 9):
    fg = "#000000" if colour in (ACCENT, "#ffffff") else TEXT_MAIN
    kw = dict(
        text=text, command=command,
        bg=colour, fg=fg,
        activebackground=colour, activeforeground=fg,
        font=(FONT_SANS, font_size, "bold"),
        relief="flat", bd=0, highlightthickness=0,
        padx=8, pady=6, cursor="hand2",
    )
    if width:
        kw["width"] = width
    return tk.Button(parent, **kw)


# ═══════════════════════════════════════════════════════════════════════
#  Splash / no-model warning banner
# ═══════════════════════════════════════════════════════════════════════
class NoModelBanner(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#2d1b00")
        tk.Label(self,
                 text="⚠  No trained model found — sign recognition is disabled. "
                      "Run  collect_data.py  then  train_model.py  first.",
                 bg="#2d1b00", fg="#ffb347",
                 font=(FONT_SANS, 9), padx=14, pady=6).pack(side="left")
        make_btn(self, "✕", "#2d1b00", self.destroy, font_size=9
                 ).pack(side="right", padx=6)


# ═══════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════
class SignLanguageApp:

    CAM_W = 640
    CAM_H = 480
    FPS   = 30

    def __init__(self, root: tk.Tk):
        self.root = root
        self._setup_window()

        # ── Core modules ──────────────────────────────────────────────
        self.recognizer = GestureRecognizer(MODEL_PATH)
        self.builder    = SentenceBuilder()
        self.tts        = TTSEngine(prefer_online=True)
        self.stt        = STTEngine(callback=self._on_speech_recognised)

        # ── Camera state ──────────────────────────────────────────────
        self.cap            = None
        self.cam_thread     = None
        self._cam_running   = False
        self._cam_index     = 0
        self._flip_cam      = True
        self._frame_lock    = threading.Lock()

        # ── Shared frame data (written by cam thread, read by UI) ─────
        self._latest_frame    : np.ndarray | None = None
        self._latest_label    : str   = ""
        self._latest_conf     : float = 0.0
        self._latest_progress : float = 0.0
        self._latest_building : str   = ""

        # ── App state ─────────────────────────────────────────────────
        self._mode           = "SIGN"    # "SIGN" | "VOICE"
        self._conf_threshold = 0.70
        self._sign_count     = 0
        self._fps_counter    : list[float] = []
        self._last_committed = ""
        self._sentence_history: list[str] = []   # spoken sentences log
        self._prev_auto_sentence = ""            # guard for auto-speak

        # ── Build UI then start camera ─────────────────────────────────
        self._build_ui()
        self._bind_shortcuts()

        if not os.path.exists(MODEL_PATH):
            NoModelBanner(self.root).pack(fill="x", before=self._main_frame)

        self._start_camera()
        self._update_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────
    #  Window setup
    # ──────────────────────────────────────────────────────────────────
    def _setup_window(self):
        self.root.title("🤟  Sign Language Translator")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        ww, wh = min(1260, sw - 40), min(780, sh - 60)
        self.root.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")
        self.root.minsize(900, 620)

    # ──────────────────────────────────────────────────────────────────
    #  UI: top header
    # ──────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG_DARK)
        hdr.pack(fill="x", padx=16, pady=(10, 0))

        tk.Label(hdr, text="🤟  Sign Language Translator",
                 bg=BG_DARK, fg=ACCENT,
                 font=(FONT_SANS, 19, "bold")).pack(side="left")

        right = tk.Frame(hdr, bg=BG_DARK)
        right.pack(side="right")

        # Mode radio buttons
        self._mode_var = tk.StringVar(value="SIGN")
        for label, val in [("✋ Sign → Text", "SIGN"), ("🎤 Voice → Text", "VOICE")]:
            tk.Radiobutton(
                right, text=label, variable=self._mode_var, value=val,
                bg=BG_DARK, fg=TEXT_MAIN, selectcolor=BG_CARD,
                activebackground=BG_DARK, activeforeground=ACCENT,
                font=(FONT_SANS, 10), command=self._on_mode_change,
            ).pack(side="left", padx=8)

    # ──────────────────────────────────────────────────────────────────
    #  UI: main two-column layout
    # ──────────────────────────────────────────────────────────────────
    def _build_main(self):
        self._main_frame = tk.Frame(self.root, bg=BG_DARK)
        self._main_frame.pack(fill="both", expand=True, padx=14, pady=8)
        self._main_frame.columnconfigure(0, weight=55)
        self._main_frame.columnconfigure(1, weight=45)
        self._main_frame.rowconfigure(0, weight=1)

        self._build_camera_panel()
        self._build_control_panel()

    # ──────────────────────────────────────────────────────────────────
    #  UI: left camera panel
    # ──────────────────────────────────────────────────────────────────
    def _build_camera_panel(self):
        left = tk.Frame(self._main_frame, bg=BG_CARD)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        # Camera feed label
        self.cam_label = tk.Label(left, bg="#000000",
                                  text="Loading camera…",
                                  fg=TEXT_DIM, font=(FONT_SANS, 12))
        self.cam_label.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # ── Camera toolbar ────────────────────────────────────────────
        bar = tk.Frame(left, bg=BG_PANEL)
        bar.grid(row=1, column=0, sticky="ew")

        # Mirror checkbox
        self._flip_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Mirror", variable=self._flip_var,
                       bg=BG_PANEL, fg=TEXT_DIM, selectcolor=BG_CARD,
                       activebackground=BG_PANEL, font=(FONT_SANS, 9),
                       command=lambda: setattr(self, "_flip_cam",
                                               self._flip_var.get())
                       ).pack(side="left", padx=(10, 4), pady=5)

        # Camera index selector
        tk.Label(bar, text="Cam:", bg=BG_PANEL,
                 fg=TEXT_DIM, font=(FONT_SANS, 9)).pack(side="left")
        self._cam_idx_var = tk.IntVar(value=0)
        for idx in range(4):
            tk.Radiobutton(bar, text=str(idx), variable=self._cam_idx_var,
                           value=idx, bg=BG_PANEL, fg=TEXT_DIM,
                           selectcolor=BG_CARD, activebackground=BG_PANEL,
                           font=(FONT_SANS, 8),
                           command=self._change_camera).pack(side="left")

        tk.Frame(bar, bg=BORDER, width=1).pack(side="left",
                                                fill="y", padx=8, pady=4)

        # Confidence threshold
        tk.Label(bar, text="Min Confidence:", bg=BG_PANEL,
                 fg=TEXT_DIM, font=(FONT_SANS, 9)).pack(side="left")

        self._conf_dv = tk.DoubleVar(value=0.70)
        self._conf_dv.trace_add("write", self._on_conf_change)
        conf_sl = ttk.Scale(bar, from_=0.40, to=0.99,
                            variable=self._conf_dv,
                            orient="horizontal", length=110)
        conf_sl.pack(side="left", padx=4)

        self._conf_lbl = tk.Label(bar, text="70%", width=4,
                                  bg=BG_PANEL, fg=ACCENT,
                                  font=(FONT_SANS, 9, "bold"))
        self._conf_lbl.pack(side="left")

        # FPS counter
        self._fps_lbl = tk.Label(bar, text="FPS: –", bg=BG_PANEL,
                                 fg=TEXT_DIM, font=(FONT_MONO, 9))
        self._fps_lbl.pack(side="right", padx=12)

    # ──────────────────────────────────────────────────────────────────
    #  UI: right control panel (Notebook tabs)
    # ──────────────────────────────────────────────────────────────────
    def _build_control_panel(self):
        right = tk.Frame(self._main_frame, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ── Current sign display ──────────────────────────────────────
        sign_outer = tk.Frame(right, bg=BG_CARD, height=88)
        sign_outer.pack(fill="x", pady=(0, 6))
        sign_outer.pack_propagate(False)

        lbl_row = tk.Frame(sign_outer, bg=BG_CARD)
        lbl_row.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(lbl_row, text="CURRENT SIGN", bg=BG_CARD,
                 fg=TEXT_DIM, font=(FONT_SANS, 7, "bold")).pack(side="left")
        self._conf_badge = tk.Label(lbl_row, text="", bg=BG_CARD,
                                    fg=TEXT_DIM, font=(FONT_SANS, 8))
        self._conf_badge.pack(side="right")

        self._sign_lbl = tk.Label(sign_outer, text="–", bg=BG_CARD,
                                  fg=ACCENT, font=(FONT_SANS, 42, "bold"))
        self._sign_lbl.pack(expand=True)

        # Stability progress bar
        stab_row = tk.Frame(right, bg=BG_DARK)
        stab_row.pack(fill="x", pady=(0, 4))
        tk.Label(stab_row, text="STABILITY", bg=BG_DARK,
                 fg=TEXT_DIM, font=(FONT_SANS, 7)).pack(side="left")
        self._building_lbl = tk.Label(stab_row, text="", bg=BG_DARK,
                                      fg=ACCENT2, font=(FONT_MONO, 9, "bold"))
        self._building_lbl.pack(side="right")

        self._prog_var = tk.DoubleVar(value=0)
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("G.Horizontal.TProgressbar",
                    troughcolor=BG_CARD, background=ACCENT,
                    bordercolor=BG_DARK, lightcolor=ACCENT, darkcolor=ACCENT)
        ttk.Progressbar(right, variable=self._prog_var, maximum=1.0,
                        style="G.Horizontal.TProgressbar"
                        ).pack(fill="x", pady=(0, 8))

        # ── Notebook ─────────────────────────────────────────────────
        nb_style = ttk.Style()
        nb_style.configure("Dark.TNotebook",
                           background=BG_DARK, borderwidth=0)
        nb_style.configure("Dark.TNotebook.Tab",
                           background=BG_PANEL, foreground=TEXT_DIM,
                           padding=[10, 4])
        nb_style.map("Dark.TNotebook.Tab",
                     background=[("selected", BG_CARD)],
                     foreground=[("selected", TEXT_MAIN)])

        nb = ttk.Notebook(right, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True)

        # Tab 1: Sentence
        self._build_sentence_tab(nb)

        # Tab 2: History
        self._build_history_tab(nb)

        # Tab 3: Settings
        self._build_settings_tab(nb)

        # Tab 4: Help
        self._build_help_tab(nb)

    # ──────────────────────────────────────────────────────────────────
    #  Tab 1: Sentence
    # ──────────────────────────────────────────────────────────────────
    def _build_sentence_tab(self, nb):
        tab = tk.Frame(nb, bg=BG_DARK)
        nb.add(tab, text=" Sentence ")

        # Sentence text area
        tk.Label(tab, text="SENTENCE", bg=BG_DARK,
                 fg=TEXT_DIM, font=(FONT_SANS, 7, "bold")
                 ).pack(anchor="w", padx=4, pady=(8, 2))

        sent_card = tk.Frame(tab, bg=BG_CARD)
        sent_card.pack(fill="x", padx=0, pady=(0, 6))

        self._sentence_box = tk.Text(
            sent_card, height=4,
            bg=BG_CARD, fg=TEXT_MAIN,
            font=(FONT_SANS, 13),
            wrap="word", relief="flat",
            insertbackground=ACCENT,
            padx=10, pady=8,
            state="disabled",
            cursor="arrow",
        )
        self._sentence_box.pack(fill="both", expand=True)

        self._last_sign_lbl = tk.Label(tab, text="", bg=BG_DARK,
                                       fg=TEXT_DIM,
                                       font=(FONT_SANS, 8, "italic"))
        self._last_sign_lbl.pack(anchor="w", padx=2, pady=(0, 4))

        # Action buttons — row 1
        r1 = tk.Frame(tab, bg=BG_DARK)
        r1.pack(fill="x", pady=2)
        make_btn(r1, "🔊 Speak (Enter)",  ACCENT,  self._speak_sentence
                 ).pack(side="left", expand=True, fill="x", padx=(0, 3))
        make_btn(r1, "↩ Undo  (Ctrl+Z)",  ACCENT2, self._undo
                 ).pack(side="left", expand=True, fill="x", padx=3)
        make_btn(r1, "✕ Clear  (Esc)",     WARN,    self._clear
                 ).pack(side="left", expand=True, fill="x", padx=(3, 0))

        # Action buttons — row 2
        r2 = tk.Frame(tab, bg=BG_DARK)
        r2.pack(fill="x", pady=2)
        make_btn(r2, "＋ Space",   BG_PANEL, self._add_space
                 ).pack(side="left", expand=True, fill="x", padx=(0, 3))
        make_btn(r2, "⌫ Delete",  BG_PANEL, self._delete_char
                 ).pack(side="left", expand=True, fill="x", padx=3)
        make_btn(r2, "📋 Copy",   BG_PANEL, self._copy_sentence
                 ).pack(side="left", expand=True, fill="x", padx=(3, 0))

        # Auto-speak toggle
        auto_row = tk.Frame(tab, bg=BG_DARK)
        auto_row.pack(fill="x", pady=(6, 0))
        self._auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(auto_row, text="Auto-speak when sentence ends  (. ? !)",
                       variable=self._auto_var,
                       bg=BG_DARK, fg=TEXT_DIM,
                       selectcolor=BG_CARD, activebackground=BG_DARK,
                       font=(FONT_SANS, 9)
                       ).pack(side="left", padx=2)

        # Voice mode panel (hidden unless Voice mode selected)
        self._voice_frame = tk.Frame(tab, bg=BG_DARK)

        tk.Frame(self._voice_frame, bg=BORDER, height=1
                 ).pack(fill="x", pady=(8, 6))
        tk.Label(self._voice_frame, text="VOICE INPUT",
                 bg=BG_DARK, fg=TEXT_DIM,
                 font=(FONT_SANS, 7, "bold")).pack(anchor="w")

        self._mic_btn = make_btn(self._voice_frame,
                                 "🎤 Start Listening", ACCENT,
                                 self._toggle_mic)
        self._mic_btn.pack(fill="x", pady=(4, 2))

        self._mic_status = tk.Label(self._voice_frame,
                                    text="Microphone: off",
                                    bg=BG_DARK, fg=TEXT_DIM,
                                    font=(FONT_SANS, 9))
        self._mic_status.pack(anchor="w")

    # ──────────────────────────────────────────────────────────────────
    #  Tab 2: History
    # ──────────────────────────────────────────────────────────────────
    def _build_history_tab(self, nb):
        tab = tk.Frame(nb, bg=BG_DARK)
        nb.add(tab, text=" History ")

        tk.Label(tab, text="Past spoken sentences (newest first)",
                 bg=BG_DARK, fg=TEXT_DIM,
                 font=(FONT_SANS, 8)).pack(anchor="w", padx=6, pady=(8, 4))

        self._history_box = tk.Text(
            tab, bg=BG_CARD, fg=TEXT_MAIN,
            font=(FONT_SANS, 10),
            wrap="word", relief="flat",
            insertbackground=ACCENT,
            padx=10, pady=8,
            state="disabled",
        )
        self._history_box.pack(fill="both", expand=True, padx=0, pady=(0, 6))

        make_btn(tab, "🗑 Clear History", WARN,
                 self._clear_history).pack(pady=4)

    # ──────────────────────────────────────────────────────────────────
    #  Tab 3: Settings
    # ──────────────────────────────────────────────────────────────────
    def _build_settings_tab(self, nb):
        tab = tk.Frame(nb, bg=BG_DARK)
        nb.add(tab, text=" Settings ")

        def section(text):
            tk.Label(tab, text=text, bg=BG_DARK, fg=TEXT_DIM,
                     font=(FONT_SANS, 7, "bold")
                     ).pack(anchor="w", padx=6, pady=(12, 2))
            tk.Frame(tab, bg=BORDER, height=1).pack(fill="x", padx=6, pady=(0, 6))

        def slider_row(parent, label, from_, to, init, fmt, callback):
            row = tk.Frame(parent, bg=BG_DARK)
            row.pack(fill="x", padx=6, pady=3)
            tk.Label(row, text=label, width=18, anchor="w",
                     bg=BG_DARK, fg=TEXT_MAIN,
                     font=(FONT_SANS, 9)).pack(side="left")
            dv  = tk.DoubleVar(value=init)
            val_lbl = tk.Label(row, text=fmt(init), width=6,
                               bg=BG_DARK, fg=ACCENT,
                               font=(FONT_MONO, 9))
            val_lbl.pack(side="right")
            def _upd(*_):
                v = dv.get()
                val_lbl.configure(text=fmt(v))
                callback(v)
            dv.trace_add("write", _upd)
            ttk.Scale(row, from_=from_, to=to, variable=dv,
                      orient="horizontal", length=120).pack(side="left", padx=6)
            return dv

        # TTS section
        section("TEXT-TO-SPEECH")
        self._tts_rate_dv = slider_row(tab, "Speech Rate (WPM)",
                                       80, 260, 160,
                                       lambda v: f"{int(v)} wpm",
                                       lambda v: self.tts.set_rate(int(v)))
        self._tts_vol_dv  = slider_row(tab, "Volume",
                                       0.1, 1.0, 0.95,
                                       lambda v: f"{v:.0%}",
                                       lambda v: self.tts.set_volume(v))

        tts_row = tk.Frame(tab, bg=BG_DARK)
        tts_row.pack(fill="x", padx=6, pady=4)
        tk.Label(tts_row, text="Voice Quality", width=18, anchor="w",
                 bg=BG_DARK, fg=TEXT_MAIN,
                 font=(FONT_SANS, 9)).pack(side="left")
        self._online_var = tk.BooleanVar(value=True)
        tk.Checkbutton(tts_row, text="Use Google TTS (online, natural)",
                       variable=self._online_var,
                       bg=BG_DARK, fg=TEXT_DIM,
                       selectcolor=BG_CARD,
                       activebackground=BG_DARK,
                       font=(FONT_SANS, 9),
                       command=lambda:
                           setattr(self.tts, "_prefer_online",
                                   self._online_var.get())
                       ).pack(side="left")

        # Sign Recognition section
        section("SIGN RECOGNITION")
        self._stable_dv = slider_row(tab, "Stability Frames",
                                     5, 40, 15,
                                     lambda v: f"{int(v)} frm",
                                     lambda v: setattr(
                                         self.builder, "STABLE_FRAMES", int(v)))
        self._cool_dv   = slider_row(tab, "Sign Cooldown (s)",
                                     0.3, 3.0, 1.2,
                                     lambda v: f"{v:.1f}s",
                                     lambda v: setattr(
                                         self.builder, "COOLDOWN_SEC", v))

        # Test TTS button
        section("TEST")
        make_btn(tab, "🔊 Test TTS — speak 'Hello, I am the Sign Language Translator'",
                 ACCENT,
                 lambda: self.tts.speak(
                     "Hello, I am the Sign Language Translator")
                 ).pack(fill="x", padx=6, pady=4)

    # ──────────────────────────────────────────────────────────────────
    #  Tab 4: Help
    # ──────────────────────────────────────────────────────────────────
    def _build_help_tab(self, nb):
        tab = tk.Frame(nb, bg=BG_DARK)
        nb.add(tab, text=" Help ")

        help_text = (
            "HOW TO USE\n"
            "══════════\n\n"
            "SIGN → TEXT MODE\n"
            "─────────────────\n"
            "1. Train the model first:\n"
            "   • python collect_data.py   (collect samples)\n"
            "   • python train_model.py    (train classifier)\n\n"
            "2. Hold a sign in front of your webcam.\n"
            "3. Hold it STILL for 15 frames (watch the\n"
            "   stability bar fill up green).\n"
            "4. The sign is committed to the sentence.\n"
            "5. Press 🔊 Speak or hit Enter to hear it.\n\n"
            "SPECIAL SIGNS TO TRAIN\n"
            "──────────────────────\n"
            "  SPACE   → inserts a word space\n"
            "  DELETE  → removes last character\n"
            "  CLEAR   → wipes the sentence\n"
            "  PERIOD  → adds '. '\n"
            "  COMMA   → adds ', '\n\n"
            "VOICE → TEXT MODE\n"
            "─────────────────\n"
            "Switch to Voice mode, click 🎤 Start Listening,\n"
            "then speak normally.  Requires PyAudio.\n\n"
            "KEYBOARD SHORTCUTS\n"
            "──────────────────\n"
            "  Enter        Speak sentence\n"
            "  Ctrl+Z       Undo last sign\n"
            "  Ctrl+Space   Add space\n"
            "  Escape       Clear sentence\n"
            "  Ctrl+C       Copy to clipboard\n"
            "  F1           Toggle camera mirror\n\n"
            "TIPS\n"
            "────\n"
            "• Good lighting → better accuracy.\n"
            "• Keep your hand in the GREEN box.\n"
            "• Aim for 200+ samples per sign.\n"
            "• Retrain if accuracy drops.\n"
        )

        txt = scrolledtext.ScrolledText(
            tab, bg=BG_CARD, fg=TEXT_MAIN,
            font=(FONT_MONO, 8),
            wrap="word", relief="flat",
            padx=12, pady=10,
            state="normal",
        )
        txt.insert("1.0", help_text)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)

    # ──────────────────────────────────────────────────────────────────
    #  Status bar
    # ──────────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL, height=24)
        bar.pack(fill="x", side="bottom")

        self._status_var = tk.StringVar(value="Starting…")
        tk.Label(bar, textvariable=self._status_var,
                 bg=BG_PANEL, fg=TEXT_DIM,
                 font=(FONT_MONO, 8)).pack(side="left", padx=12)

        self._count_var = tk.StringVar(value="Signs committed: 0")
        tk.Label(bar, textvariable=self._count_var,
                 bg=BG_PANEL, fg=TEXT_DIM,
                 font=(FONT_MONO, 8)).pack(side="right", padx=12)

    # ──────────────────────────────────────────────────────────────────
    #  Master UI builder
    # ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_status_bar()  # pack to bottom BEFORE main
        self._build_main()

    # ──────────────────────────────────────────────────────────────────
    #  Keyboard shortcuts
    # ──────────────────────────────────────────────────────────────────
    def _bind_shortcuts(self):
        self.root.bind("<Return>",         lambda _: self._speak_sentence())
        self.root.bind("<Control-z>",      lambda _: self._undo())
        self.root.bind("<Control-Z>",      lambda _: self._undo())
        self.root.bind("<Control-space>",  lambda _: self._add_space())
        self.root.bind("<Escape>",         lambda _: self._clear())
        self.root.bind("<Control-c>",      lambda _: self._copy_sentence())
        self.root.bind("<F1>",             lambda _: self._toggle_mirror())

    def _toggle_mirror(self):
        self._flip_cam = not self._flip_cam
        self._flip_var.set(self._flip_cam)

    # ──────────────────────────────────────────────────────────────────
    #  Camera management
    # ──────────────────────────────────────────────────────────────────
    def _start_camera(self, index: int = 0):
        self._cam_running = False
        if self.cap:
            time.sleep(0.15)
            self.cap.release()

        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self._status_var.set(
                f"⚠  Camera {index} not found. Try a different index.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS,          self.FPS)
        self._cam_index   = index
        self._cam_running = True

        self.cam_thread = threading.Thread(
            target=self._camera_loop, daemon=True)
        self.cam_thread.start()

    def _change_camera(self):
        idx = self._cam_idx_var.get()
        self._start_camera(idx)

    # ──────────────────────────────────────────────────────────────────
    #  Camera thread (runs in background)
    # ──────────────────────────────────────────────────────────────────
    def _camera_loop(self):
        while self._cam_running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            if self._flip_cam:
                frame = cv2.flip(frame, 1)

            # ── Extract landmarks & predict ───────────────────────────
            features, raw_lm = self.recognizer.extract_landmarks(frame)
            label, confidence = None, 0.0

            if features is not None:
                label, confidence = self.recognizer.predict(features)
                if confidence < self._conf_threshold:
                    label = None

            # ── Update sentence builder (SIGN mode only) ──────────────
            if self._mode == "SIGN":
                # Capture the CURRENT candidate BEFORE update() may reset it
                candidate_before = self.builder.current_candidate

                committed = self.builder.update(label)
                if committed:
                    self._sign_count    += 1
                    # The just-committed sign is what was stable
                    self._last_committed = candidate_before or label or ""

            # ── Draw overlays on frame ────────────────────────────────
            frame = self.recognizer.draw_landmarks(frame, raw_lm)
            frame = self.recognizer.draw_prediction(frame, label, confidence)
            self._draw_stability_overlay(frame)

            # ── FPS tracking ──────────────────────────────────────────
            now = time.time()
            self._fps_counter.append(now)
            self._fps_counter = [t for t in self._fps_counter
                                  if now - t < 1.0]

            # ── Write shared data (thread-safe) ───────────────────────
            with self._frame_lock:
                self._latest_frame    = frame.copy()
                self._latest_label    = label    or ""
                self._latest_conf     = confidence
                self._latest_progress = self.builder.stable_progress
                self._latest_building = self.builder.current_candidate

    def _draw_stability_overlay(self, frame: np.ndarray):
        """Green bar at bottom + hand bounding guide."""
        h, w = frame.shape[:2]
        pct   = self.builder.stable_progress

        # Bottom progress bar
        bar_w = int(w * pct)
        cv2.rectangle(frame, (0, h - 7), (bar_w, h), (0, 229, 160), -1)

        # Faint guide box to position hand
        cx, cy = w // 2, h // 2
        box    = 180
        alpha_color = (40, 100, 70) if pct < 0.5 else (0, 229, 160)
        cv2.rectangle(frame, (cx - box, cy - box),
                      (cx + box, cy + box), alpha_color, 1)

    # ──────────────────────────────────────────────────────────────────
    #  UI refresh (Tkinter main thread, ~30 fps)
    # ──────────────────────────────────────────────────────────────────
    def _update_ui(self):
        try:
            # ── Grab latest shared state ──────────────────────────────
            with self._frame_lock:
                frame    = self._latest_frame
                label    = self._latest_label
                conf     = self._latest_conf
                progress = self._latest_progress
                building = self._latest_building

            # ── Camera image ──────────────────────────────────────────
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                lw  = max(self.cam_label.winfo_width(),  320)
                lh  = max(self.cam_label.winfo_height(), 240)
                img = img.resize((lw, lh), Image.BILINEAR)
                imgtk = ImageTk.PhotoImage(image=img)
                self.cam_label.imgtk = imgtk   # prevent GC
                self.cam_label.configure(image=imgtk, text="")

            # ── Current sign card ─────────────────────────────────────
            if label:
                self._sign_lbl.configure(text=label,  fg=ACCENT)
                self._conf_badge.configure(text=f"{conf:.0%}")
            else:
                self._sign_lbl.configure(text="–",    fg=TEXT_DIM)
                self._conf_badge.configure(text="")

            # ── Stability bar + building label ────────────────────────
            self._prog_var.set(progress)
            self._building_lbl.configure(
                text=f"building: {building}" if building else "")

            # ── Sentence box ──────────────────────────────────────────
            sentence = self.builder.get_sentence()
            self._sentence_box.configure(state="normal")
            self._sentence_box.delete("1.0", "end")
            self._sentence_box.insert("end", sentence)
            self._sentence_box.see("end")
            self._sentence_box.configure(state="disabled")

            # ── Last committed label ──────────────────────────────────
            if self._last_committed:
                self._last_sign_lbl.configure(
                    text=f"✓ Last added: {self._last_committed}",
                    fg=ACCENT2)

            # ── FPS ───────────────────────────────────────────────────
            fps = len(self._fps_counter)
            self._fps_lbl.configure(text=f"FPS: {fps:2d}")

            # ── Status bar ────────────────────────────────────────────
            model_ok  = self.recognizer.model is not None
            mode_str  = "Sign→Text" if self._mode == "SIGN" else "Voice→Text"
            signs_str = (f"Model: {len(self.recognizer.labels)} signs"
                         if model_ok else "⚠ No model – train first")
            cam_str   = f"Cam{self._cam_index}: active" if self._cam_running else "Camera: error"
            self._status_var.set(
                f"Mode: {mode_str}  │  {signs_str}  │  {cam_str}")
            self._count_var.set(f"Signs committed: {self._sign_count}")

            # Confidence label
            self._conf_lbl.configure(
                text=f"{int(self._conf_dv.get() * 100)}%")

            # ── Auto-speak on sentence-ending punctuation ─────────────
            if (self._auto_var.get()
                    and sentence.endswith((".", "?", "!"))
                    and sentence != self._prev_auto_sentence
                    and not self.tts.is_speaking()):
                self._prev_auto_sentence = sentence
                self.tts.speak(sentence)

        except Exception:
            pass  # never crash the UI loop

        finally:
            self.root.after(33, self._update_ui)   # ≈30 fps

    # ──────────────────────────────────────────────────────────────────
    #  Button / keyboard callbacks
    # ──────────────────────────────────────────────────────────────────
    def _speak_sentence(self):
        text = self.builder.get_sentence()
        if not text:
            return
        # Log to history
        self._sentence_history.insert(0, text)
        self._sentence_history = self._sentence_history[:20]
        self._refresh_history()
        self._status_var.set("🔊 Speaking…")
        self.tts.speak(text)

    def _undo(self):
        self.builder.undo()

    def _clear(self):
        self.builder.clear()
        self._last_committed = ""
        self._last_sign_lbl.configure(text="")

    def _add_space(self):
        self.builder.manual_add(" ")

    def _delete_char(self):
        self.builder.undo()

    def _copy_sentence(self):
        text = self.builder.get_sentence()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._status_var.set("✓ Copied to clipboard!")

    def _clear_history(self):
        self._sentence_history.clear()
        self._refresh_history()

    def _refresh_history(self):
        self._history_box.configure(state="normal")
        self._history_box.delete("1.0", "end")
        for i, s in enumerate(self._sentence_history, 1):
            self._history_box.insert("end", f"{i:2}. {s}\n")
        self._history_box.configure(state="disabled")

    # ── Settings callbacks ────────────────────────────────────────────
    def _on_conf_change(self, *_):
        v = self._conf_dv.get()
        self._conf_threshold = v
        self.recognizer.CONFIDENCE_THRESHOLD = v
        self._conf_lbl.configure(text=f"{int(v*100)}%")

    # ── Mode toggle ───────────────────────────────────────────────────
    def _on_mode_change(self):
        self._mode = self._mode_var.get()
        if self._mode == "VOICE":
            self._voice_frame.pack(fill="x", pady=(8, 0))
        else:
            self._voice_frame.pack_forget()
            if self.stt.is_listening:
                self.stt.stop_listening()
                self._mic_btn.configure(text="🎤 Start Listening", bg=ACCENT)
                self._mic_status.configure(text="Microphone: off", fg=TEXT_DIM)

    def _toggle_mic(self):
        if self.stt.is_listening:
            self.stt.stop_listening()
            self._mic_btn.configure(text="🎤 Start Listening", bg=ACCENT)
            self._mic_status.configure(text="Microphone: off", fg=TEXT_DIM)
        else:
            if not self.stt.is_available:
                messagebox.showwarning(
                    "Microphone Unavailable",
                    "SpeechRecognition or PyAudio is not installed.\n\n"
                    "Install steps:\n"
                    "  Windows : pip install pipwin && pipwin install pyaudio\n"
                    "  Linux   : sudo apt install python3-pyaudio portaudio19-dev\n"
                    "  macOS   : brew install portaudio && pip install pyaudio\n\n"
                    "Then:  pip install SpeechRecognition")
                return
            self.stt.start_listening()
            self._mic_btn.configure(text="⏹ Stop Listening", bg=WARN)
            self._mic_status.configure(text="🎤 Listening…", fg=ACCENT)

    # ── STT callback (background thread → main thread) ────────────────
    def _on_speech_recognised(self, text: str):
        print(f"[STT] {text}")
        self.root.after(0, lambda: self.builder.manual_add(text + " "))

    # ──────────────────────────────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._cam_running = False
        if self.stt.is_listening:
            self.stt.stop_listening()
        time.sleep(0.15)          # let camera thread exit
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.recognizer.release()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(MODEL_PATH):
        print("""
╔══════════════════════════════════════════════════════════╗
║  ⚠  No trained model found at models/gesture_model.pkl  ║
║                                                          ║
║  Sign recognition will be DISABLED until you train.     ║
║  Steps:                                                  ║
║    1. python collect_data.py                             ║
║    2. python train_model.py                              ║
║                                                          ║
║  Starting app in read-only / Voice-only mode…           ║
╚══════════════════════════════════════════════════════════╝
        """)

    root = tk.Tk()

    # Dark title bar on Windows 10/11
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(ctypes.c_int(2)), 4)
    except Exception:
        pass

    SignLanguageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
