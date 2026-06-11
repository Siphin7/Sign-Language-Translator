# =============================================================
#  src/stt_engine.py
#  Speech-to-Text: microphone → text using SpeechRecognition
# =============================================================

import threading
from typing import Callable


class STTEngine:
    """
    Listens to the microphone and calls *callback(text)* whenever
    speech is recognised.

    Uses Google Web Speech API by default (free, no key needed for
    limited use); falls back to Sphinx for fully offline operation.
    """

    def __init__(self, callback: Callable[[str], None] = None,
                 language: str = "en-US", use_sphinx: bool = False):
        self._callback   = callback
        self._language   = language
        self._use_sphinx = use_sphinx
        self._listening  = False
        self._thread     = None

        try:
            import speech_recognition as sr
            self._sr      = sr
            self._rec     = sr.Recognizer()
            self._rec.energy_threshold        = 300
            self._rec.dynamic_energy_threshold = True
            self._available = True
        except ImportError:
            self._available = False
            print("[STT] SpeechRecognition not installed – STT disabled")

    # ──────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────
    def start_listening(self):
        """Begin continuous background listening."""
        if not self._available or self._listening:
            return
        self._listening = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[STT] Listening started")

    def stop_listening(self):
        """Stop the background listener."""
        self._listening = False
        print("[STT] Listening stopped")

    def listen_once(self, timeout: float = 5.0) -> str:
        """
        Block until one utterance is captured (or timeout).
        Returns the recognised string or "" on failure.
        """
        if not self._available:
            return ""
        try:
            with self._sr.Microphone() as source:
                self._rec.adjust_for_ambient_noise(source, duration=0.5)
                audio = self._rec.listen(source, timeout=timeout)
            return self._recognise(audio)
        except Exception as e:
            print(f"[STT] listen_once error: {e}")
            return ""

    @property
    def is_listening(self) -> bool:
        return self._listening

    @property
    def is_available(self) -> bool:
        return self._available

    # ──────────────────────────────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────────────────────────────
    def _listen_loop(self):
        try:
            with self._sr.Microphone() as source:
                self._rec.adjust_for_ambient_noise(source, duration=1)
                print("[STT] Ambient noise calibrated")

                while self._listening:
                    try:
                        audio = self._rec.listen(source, timeout=3, phrase_time_limit=8)
                        text  = self._recognise(audio)
                        if text and self._callback:
                            self._callback(text)
                    except self._sr.WaitTimeoutError:
                        pass  # silence – keep waiting
                    except Exception as e:
                        if self._listening:
                            print(f"[STT] Loop error: {e}")
        except Exception as e:
            print(f"[STT] Microphone error: {e}")
            self._listening = False

    def _recognise(self, audio) -> str:
        try:
            if self._use_sphinx:
                return self._rec.recognize_sphinx(audio)
            return self._rec.recognize_google(audio, language=self._language)
        except self._sr.UnknownValueError:
            return ""
        except self._sr.RequestError as e:
            print(f"[STT] Recognition service error: {e}")
            return ""
