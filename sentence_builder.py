# =============================================================
#  src/sentence_builder.py
#  Converts a stream of detected signs into coherent sentences
# =============================================================

import time
from collections import deque


# Special sign tokens handled outside the classifier
SPECIAL_TOKENS = {
    "SPACE"     : " ",
    "DELETE"    : "\b",     # backspace one character
    "CLEAR"     : None,     # wipe the whole sentence
    "PERIOD"    : ". ",
    "COMMA"     : ", ",
    "QUESTION"  : "? ",
    "EXCLAIM"   : "! ",
}


class SentenceBuilder:
    """
    Accumulates individual sign predictions into words and sentences.

    Logic
    -----
    • A sign must be *stable* (same label for STABLE_FRAMES consecutive frames)
      before it is accepted.
    • Once accepted, the same sign won't be accepted again for COOLDOWN_SEC
      seconds (prevents a held gesture from spamming).
    • Recognised ASL letters are appended to the current word buffer.
    • Special tokens (SPACE, DELETE, CLEAR …) are handled immediately.
    • call get_sentence() to retrieve the current sentence string.
    """

    STABLE_FRAMES = 15        # how many consecutive identical predictions needed
    COOLDOWN_SEC  = 1.2       # seconds before same sign can fire again
    MAX_HISTORY   = 200       # max characters kept in the sentence buffer

    def __init__(self):
        self._buffer       : list[str] = []   # list of accepted chars / words
        self._last_label   : str       = ""
        self._stable_count : int       = 0
        self._last_accepted_time: float = 0.0
        self._last_accepted_label: str = ""
        self._history      : deque     = deque(maxlen=50)  # undo stack

    # ──────────────────────────────────────────────────────────────────
    #  Core method called every frame
    # ──────────────────────────────────────────────────────────────────
    def update(self, label: str | None) -> bool:
        """
        Feed the latest prediction.

        Parameters
        ----------
        label : predicted sign string or None (no hand / low confidence)

        Returns
        -------
        True if a new sign was *committed* to the buffer this frame.
        """
        if label is None:
            self._stable_count = 0
            self._last_label   = ""
            return False

        # Track stability
        if label == self._last_label:
            self._stable_count += 1
        else:
            self._stable_count = 1
            self._last_label   = label

        # Not stable yet
        if self._stable_count < self.STABLE_FRAMES:
            return False

        # Cooldown: same sign fired too recently
        now = time.time()
        if (label == self._last_accepted_label and
                now - self._last_accepted_time < self.COOLDOWN_SEC):
            return False

        # Commit the sign
        self._commit(label)
        self._last_accepted_time  = now
        self._last_accepted_label = label
        self._stable_count        = 0       # reset so it won't re-fire immediately
        return True

    # ──────────────────────────────────────────────────────────────────
    #  Internal commit
    # ──────────────────────────────────────────────────────────────────
    def _commit(self, label: str):
        upper = label.upper()

        if upper == "CLEAR":
            self._history.append(list(self._buffer))
            self._buffer.clear()
            return

        if upper == "DELETE":
            if self._buffer:
                self._history.append(list(self._buffer))
                self._buffer.pop()
            return

        if upper in SPECIAL_TOKENS:
            char = SPECIAL_TOKENS[upper]
            self._history.append(list(self._buffer))
            self._buffer.append(char)
            return

        # Plain letter or word token
        self._history.append(list(self._buffer))
        # If it's a multi-char word token, add surrounding spaces
        if len(label) > 1 and not label.startswith(" "):
            self._buffer.append(" " + label + " ")
        else:
            self._buffer.append(label.upper() if len(label) == 1 else label)

    # ──────────────────────────────────────────────────────────────────
    #  Public accessors
    # ──────────────────────────────────────────────────────────────────
    def get_sentence(self) -> str:
        """Return the current accumulated text."""
        return "".join(self._buffer).strip()

    def clear(self):
        """Wipe sentence buffer."""
        self._history.append(list(self._buffer))
        self._buffer.clear()

    def undo(self):
        """Revert the last committed sign."""
        if self._history:
            self._buffer = self._history.pop()

    def manual_add(self, text: str):
        """Directly append text (used by Voice→Text mode)."""
        self._history.append(list(self._buffer))
        for ch in text:
            self._buffer.append(ch)

    @property
    def char_count(self) -> int:
        return len(self._buffer)

    @property
    def stable_progress(self) -> float:
        """0..1 progress toward the stability threshold (for UI progress bar)."""
        return min(self._stable_count / self.STABLE_FRAMES, 1.0)

    @property
    def current_candidate(self) -> str:
        """The sign currently building toward stability."""
        return self._last_label or ""
