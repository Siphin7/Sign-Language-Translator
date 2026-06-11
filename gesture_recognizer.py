# =============================================================
#  src/gesture_recognizer.py
#  Core module: hand landmark extraction + sign classification
# =============================================================

import os
import cv2
import numpy as np
import pickle
import mediapipe as mp


class GestureRecognizer:
    """
    Wraps MediaPipe Hands for landmark detection and uses a trained
    scikit-learn classifier to predict ASL signs from those landmarks.
    """

    CONFIDENCE_THRESHOLD = 0.70   # minimum confidence to accept a prediction

    def __init__(self, model_path: str = None):
        # ── MediaPipe setup ──────────────────────────────────────────
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode        = False,   # optimised for video stream
            max_num_hands            = 1,
            min_detection_confidence = 0.70,
            min_tracking_confidence  = 0.50,
        )

        # ── ML model ─────────────────────────────────────────────────
        self.model  = None
        self.labels = []          # list of class names from training

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    # ──────────────────────────────────────────────────────────────────
    #  Model I/O
    # ──────────────────────────────────────────────────────────────────
    def load_model(self, path: str):
        """Load a previously saved model (.pkl) produced by train_model.py."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model  = data["model"]
            self.labels = data["labels"]
            print(f"[GestureRecognizer] Model loaded — {len(self.labels)} signs: {self.labels}")
        except Exception as e:
            print(f"[GestureRecognizer] Could not load model: {e}")

    # ──────────────────────────────────────────────────────────────────
    #  Landmark extraction
    # ──────────────────────────────────────────────────────────────────
    def extract_landmarks(self, frame: np.ndarray):
        """
        Run MediaPipe on one BGR frame.

        Returns
        -------
        features : np.ndarray shape (63,)  or  None  if no hand found
        hand_landmarks : mediapipe hand landmark object (for drawing)
        """
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None

        raw = results.multi_hand_landmarks[0]

        # ── Build raw 63-d vector (x, y, z for each of 21 landmarks) ──
        pts = np.array([[lm.x, lm.y, lm.z] for lm in raw.landmark])  # (21, 3)

        # ── Normalise: translate so wrist = origin ────────────────────
        pts -= pts[0]          # wrist is landmark 0

        # ── Scale: divide by max absolute value → range ≈ [−1, 1] ────
        scale = np.max(np.abs(pts)) + 1e-7
        pts  /= scale

        return pts.flatten(), raw          # (63,), raw landmarks

    # ──────────────────────────────────────────────────────────────────
    #  Prediction
    # ──────────────────────────────────────────────────────────────────
    def predict(self, features: np.ndarray):
        """
        Predict sign label + confidence from a (63,) feature vector.

        Returns
        -------
        label      : str   or  None
        confidence : float (0..1)
        """
        if self.model is None or features is None:
            return None, 0.0

        proba      = self.model.predict_proba(features.reshape(1, -1))[0]
        confidence = float(np.max(proba))
        label      = self.model.classes_[int(np.argmax(proba))]

        if confidence < self.CONFIDENCE_THRESHOLD:
            return None, confidence

        return label, confidence

    # ──────────────────────────────────────────────────────────────────
    #  Drawing helpers
    # ──────────────────────────────────────────────────────────────────
    def draw_landmarks(self, frame: np.ndarray, hand_landmarks) -> np.ndarray:
        """Overlay skeleton + dots on the frame (in-place copy)."""
        if hand_landmarks is None:
            return frame

        out = frame.copy()
        self.mp_drawing.draw_landmarks(
            out,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 220, 90),  thickness=2, circle_radius=5),
            self.mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2),
        )
        return out

    def draw_prediction(self, frame: np.ndarray, label: str, confidence: float) -> np.ndarray:
        """Draw sign label + confidence badge in top-left corner."""
        out = frame.copy()
        h, w = out.shape[:2]

        if label:
            # Dark translucent pill
            cv2.rectangle(out, (10, 10), (300, 65), (0, 0, 0), -1)
            cv2.rectangle(out, (10, 10), (300, 65), (0, 220, 90), 2)

            text  = f"{label}"
            conf  = f"Confidence: {confidence:.0%}"
            cv2.putText(out, text, (20, 45),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 120), 2)
            cv2.putText(out, conf, (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return out

    # ──────────────────────────────────────────────────────────────────
    #  Clean-up
    # ──────────────────────────────────────────────────────────────────
    def release(self):
        self.hands.close()
