# =============================================================
#  src/tts_engine.py
#  Text-to-Speech: tries gTTS (online) then falls back to pyttsx3
# =============================================================

import os
import threading


class TTSEngine:
    """
    Speaks text aloud.

    Priority
    --------
    1. gTTS  (Google Text-to-Speech – requires internet, natural voice)
    2. pyttsx3 (offline, built-in system voice)

    Both engines are driven from a background thread so the GUI never blocks.
    """

    def __init__(self, prefer_online: bool = True):
        self._prefer_online = prefer_online
        self._lock          = threading.Lock()
        self._speaking      = False

        # ── Pre-load offline engine (always available) ────────────────
        self._offline_engine = None
        try:
            import pyttsx3
            self._offline_engine = pyttsx3.init()
            self._offline_engine.setProperty("rate",   160)
            self._offline_engine.setProperty("volume", 0.95)
            # Try to pick a female voice if available
            voices = self._offline_engine.getProperty("voices")
            for v in voices:
                if "female" in v.name.lower() or "zira" in v.name.lower():
                    self._offline_engine.setProperty("voice", v.id)
                    break
        except Exception as e:
            print(f"[TTS] pyttsx3 unavailable: {e}")

        # ── Check gTTS availability ───────────────────────────────────
        self._gtts_available = False
        try:
            from gtts import gTTS
            import pygame
            pygame.mixer.init()
            self._gtts_available = True
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────
    def speak(self, text: str, blocking: bool = False):
        """
        Speak *text* asynchronously (or synchronously if blocking=True).
        """
        if not text.strip():
            return

        if blocking:
            self._do_speak(text)
        else:
            t = threading.Thread(target=self._do_speak, args=(text,), daemon=True)
            t.start()

    def is_speaking(self) -> bool:
        return self._speaking

    def set_rate(self, rate: int):
        """Words per minute (offline engine)."""
        if self._offline_engine:
            self._offline_engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        """0.0 – 1.0"""
        if self._offline_engine:
            self._offline_engine.setProperty("volume", volume)

    # ──────────────────────────────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────────────────────────────
    def _do_speak(self, text: str):
        with self._lock:
            self._speaking = True
            try:
                if self._prefer_online and self._gtts_available:
                    self._speak_gtts(text)
                else:
                    self._speak_offline(text)
            except Exception as e:
                print(f"[TTS] Error: {e} – retrying offline")
                self._speak_offline(text)
            finally:
                self._speaking = False

    def _speak_gtts(self, text: str):
        from gtts import gTTS
        import pygame
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name

        try:
            gTTS(text=text, lang="en", slow=False).save(tmp)
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                import time; time.sleep(0.05)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _speak_offline(self, text: str):
        if self._offline_engine:
            self._offline_engine.say(text)
            self._offline_engine.runAndWait()
        else:
            print(f"[TTS] (no engine) Would speak: {text}")
