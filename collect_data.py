#!/usr/bin/env python3
# =============================================================
#  collect_data.py  ──  STEP 1: Build your training dataset
# =============================================================
#
#  HOW TO USE
#  ----------
#  1. Run:  python collect_data.py
#  2. Pick a sign label from the on-screen menu (or type custom).
#  3. Hold the sign in front of your webcam.
#  4. Press  SPACE  to start a 3-second recording session.
#     The tool captures 60 frames of hand landmarks automatically.
#  5. Repeat for every sign you want to teach the model.
#  6. Saved to:  dataset/gesture_data.csv
#
#  RECOMMENDED SIGNS TO COLLECT
#  ─────────────────────────────
#  ASL Letters  : A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
#  Common Words : HELLO, THANKYOU, YES, NO, HELP, WATER, I, YOU,
#                 PLEASE, SORRY, GOOD, BAD, MORE, STOP, GO
#  Sentence ctrl: SPACE, DELETE, CLEAR
#
#  Aim for 200+ samples per sign for good accuracy.
# =============================================================

import os
import sys
import csv
import time
import cv2
import numpy as np
import mediapipe as mp

# ── Paths ────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join("dataset", "gesture_data.csv")
os.makedirs("dataset", exist_ok=True)

# ── Predefined sign list (extend as needed) ──────────────────────────
PRESET_SIGNS = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    ["HELLO", "THANKYOU", "YES", "NO", "HELP", "WATER",
     "I", "YOU", "PLEASE", "SORRY", "GOOD", "MORE", "STOP",
     "SPACE", "DELETE", "CLEAR"]
)

# ── MediaPipe setup ───────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.70, min_tracking_confidence=0.50
)


# ═════════════════════════════════════════════════════════════════════
def extract_landmarks(frame: np.ndarray):
    """Return normalised (63,) feature vector or None."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None, None

    raw = results.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in raw.landmark])  # (21,3)
    pts -= pts[0]                            # centre on wrist
    pts /= (np.max(np.abs(pts)) + 1e-7)     # scale
    return pts.flatten(), raw


def draw_hand(frame, raw_lm):
    if raw_lm:
        mp_drawing.draw_landmarks(
            frame, raw_lm, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 220, 90),  thickness=2, circle_radius=5),
            mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2),
        )


def count_existing_samples(label: str) -> int:
    if not os.path.exists(DATASET_PATH):
        return 0
    count = 0
    with open(DATASET_PATH, "r") as f:
        for row in csv.reader(f):
            if row and row[0] == label:
                count += 1
    return count


def save_sample(label: str, features: np.ndarray):
    """Append one sample row to the CSV."""
    with open(DATASET_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + features.tolist())


# ═════════════════════════════════════════════════════════════════════
def choose_sign() -> str:
    """Terminal menu for sign selection."""
    print("\n" + "═" * 60)
    print("  SIGN LANGUAGE TRANSLATOR — Data Collection")
    print("═" * 60)

    for i, s in enumerate(PRESET_SIGNS):
        existing = count_existing_samples(s)
        marker   = "✓" if existing >= 100 else " "
        end      = "\n" if (i + 1) % 6 == 0 else ""
        print(f"  [{marker}] {s:<12}", end=end)

    print("\n" + "═" * 60)
    print("  ✓ = 100+ samples collected")
    print("  Type a sign label (or press Enter to quit): ", end="")
    choice = input().strip().upper()
    return choice


def record_sign(label: str, cap: cv2.VideoCapture):
    """
    Interactive recording session for one sign label.
    Press SPACE to begin capturing, Q to quit/go back.
    """
    CAPTURE_FRAMES   = 60     # frames per session
    COUNTDOWN_SECS   = 3

    collected_this_session = 0
    state  = "WAITING"        # WAITING | COUNTDOWN | RECORDING | DONE
    t0     = 0.0

    existing = count_existing_samples(label)
    print(f"\n  → Recording '{label}' (existing: {existing} samples)")
    print("    Hold the sign and press  SPACE  to record, Q to go back.\n")

    frame_buf = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        features, raw_lm = extract_landmarks(frame)
        draw_hand(frame, raw_lm)

        # ── Overlay UI ───────────────────────────────────────────────
        # Background pill
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)

        # Sign label
        cv2.putText(frame, f"Sign: {label}", (15, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 220, 90), 2)

        # Sample counter
        total = count_existing_samples(label)
        cv2.putText(frame, f"Total samples: {total}", (15, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── State machine ────────────────────────────────────────────
        if state == "WAITING":
            cv2.putText(frame, "Hold sign + press SPACE to record | Q = back",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 50), 1)

        elif state == "COUNTDOWN":
            elapsed   = time.time() - t0
            remaining = COUNTDOWN_SECS - elapsed
            if remaining <= 0:
                state     = "RECORDING"
                t0        = time.time()
                frame_buf = []
            else:
                msg = f"Get ready! {int(remaining) + 1}"
                cv2.putText(frame, msg, (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 200, 255), 3)

        elif state == "RECORDING":
            elapsed = time.time() - t0
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 200), 4)   # red border

            # Progress bar
            pct    = len(frame_buf) / CAPTURE_FRAMES
            bar_w  = int(w * pct)
            cv2.rectangle(frame, (0, h - 12), (bar_w, h), (0, 220, 90), -1)

            cv2.putText(frame, f"RECORDING... {len(frame_buf)}/{CAPTURE_FRAMES}",
                        (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)

            if features is not None:
                save_sample(label, features)
                frame_buf.append(features)

            if len(frame_buf) >= CAPTURE_FRAMES:
                state = "DONE"
                collected_this_session += len(frame_buf)

        elif state == "DONE":
            cv2.putText(frame, f"✓ Saved {collected_this_session} samples!",
                        (15, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                        (0, 220, 90), 2)
            cv2.putText(frame, "SPACE = record more | Q = choose another sign",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 50), 1)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            if state in ("WAITING", "DONE"):
                state = "COUNTDOWN"
                t0    = time.time()

    print(f"  Collected {collected_this_session} samples for '{label}'.")


# ═════════════════════════════════════════════════════════════════════
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check your camera index.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("""
╔══════════════════════════════════════════════════════════╗
║   SIGN LANGUAGE TRANSLATOR — DATA COLLECTION TOOL       ║
╠══════════════════════════════════════════════════════════╣
║  This tool records your hand landmarks for each sign.    ║
║  Aim for 150-300 samples per sign for best accuracy.     ║
║  Dataset saved to:  dataset/gesture_data.csv             ║
╚══════════════════════════════════════════════════════════╝
    """)

    try:
        while True:
            label = choose_sign()
            if not label:
                print("Exiting data collection.")
                break
            record_sign(label, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\nData collection complete.")
        print(f"Dataset saved to: {os.path.abspath(DATASET_PATH)}")


if __name__ == "__main__":
    main()
