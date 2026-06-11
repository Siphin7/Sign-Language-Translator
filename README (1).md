# 🤟 AI Sign Language Translator

A real-time **American Sign Language (ASL) → Text → Speech** translator built with
Python, MediaPipe, and scikit-learn.

---

## 📁 Project Structure

```
sign_language_translator/
├── app.py                  ← STEP 3 — Main application (run this!)
├── collect_data.py         ← STEP 1 — Webcam data collection
├── train_model.py          ← STEP 2 — Model training
│
├── src/
│   ├── gesture_recognizer.py   Hand landmark detection + prediction
│   ├── sentence_builder.py     Signs → sentences (stability + cooldown)
│   ├── tts_engine.py           Text-to-Speech (Google TTS / pyttsx3)
│   └── stt_engine.py           Speech-to-Text (microphone)
│
├── dataset/
│   └── gesture_data.csv    ← Pre-built! 1,680 real ASL samples (A-Y)
│
├── models/
│   └── gesture_model.pkl   ← Generated after training
│
├── requirements.txt
├── setup.bat               ← Windows one-click setup
└── setup.sh                ← Linux/macOS one-click setup
```

---

## 🚀 Quick Start (3 Steps)

### Step 0 — Setup (once)

**Windows:**
```bat
setup.bat
```

**Linux / macOS:**
```bash
chmod +x setup.sh && ./setup.sh
```

**Manual:**
```bash
python -m venv venv
# Windows:   venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

---

### Step 1 — (Optional) Collect Your Own Data

The `dataset/gesture_data.csv` already contains **1,680 real ASL samples** (24 letters,
70 images each, extracted from real hand photos using MediaPipe).

To **add your own samples** or train on custom gestures:

```bash
python collect_data.py
```

- Pick a sign label (A-Z or custom words like HELLO, THANKYOU, WATER…)
- Hold the sign in front of your webcam
- Press **SPACE** to record 60 frames
- Repeat for every sign you want

---

### Step 2 — Train the Model

```bash
python train_model.py
```

This:
- Loads `dataset/gesture_data.csv`
- Trains 3 classifiers (RandomForest, MLP, GradientBoosting)
- Picks the best one automatically
- Saves `models/gesture_model.pkl`
- Shows a confusion matrix and accuracy report

Expected accuracy with the provided data: **92–97%**

---

### Step 3 — Run the App

```bash
python app.py
```

---

## 🎮 App Features

| Feature | How to Use |
|---------|-----------|
| Sign → Text | Hold sign still for 15 frames (watch green stability bar) |
| Sign → Voice | Click **🔊 Speak** or press **Enter** |
| Voice → Text | Switch to Voice mode, click **🎤 Start Listening** |
| Undo last sign | Click **↩ Undo** or press **Ctrl+Z** |
| Clear sentence | Click **✕ Clear** or press **Escape** |
| Copy to clipboard | Click **📋 Copy** or press **Ctrl+C** |
| Add space | Click **＋ Space** or press **Ctrl+Space** |
| Mirror camera | Press **F1** |
| History | Click the **History** tab |
| Settings | Click the **Settings** tab (TTS rate, volume, etc.) |

---

## 📦 Pre-loaded Dataset Details

| Property | Value |
|----------|-------|
| Source | Real ASL hand photos (7 users × 24 letters × 10 images) |
| Total samples | 1,680 |
| Letters covered | A B C D E F G H I K L M N O P Q R S T U V W X Y |
| Not included | J, Z (require hand motion, not static) |
| Feature format | 63 normalised MediaPipe landmarks (x,y,z × 21 points) |
| Samples per letter | 70 (perfectly balanced) |

---

## 🔧 Installation Details

### PyAudio (for Voice → Text feature)

**Windows:**
```bat
pip install pipwin
pipwin install pyaudio
```
or download the `.whl` from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-pyaudio portaudio19-dev
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

---

## 🧪 Adding Custom Signs (beyond A-Z)

You can train the model on any custom signs:

1. Run `python collect_data.py`
2. Type a custom label like: `HELLO`, `THANKYOU`, `WATER`, `SPACE`, `DELETE`
3. Collect 150–300 samples for each
4. Re-run `python train_model.py`

**Special labels** handled automatically by the sentence builder:
- `SPACE` → inserts a space character
- `DELETE` → removes last character
- `CLEAR` → wipes the whole sentence
- `PERIOD` → adds `. `
- `COMMA` → adds `, `

---

## 🎯 Tips for Best Accuracy

1. **Good lighting** — avoid backlit hands
2. **Plain background** — solid colour wall is ideal
3. **Centre your hand** in the green guide box on screen
4. **Hold still** — wait for the green stability bar to fill
5. **Collect more data** — 200+ samples per sign → better accuracy
6. **Retrain** when adding new signs

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'mediapipe'` | `pip install mediapipe==0.10.13` |
| `No module named 'cv2'` | `pip install opencv-python` |
| Camera not opening | Try `Cam: 1` or `Cam: 2` in the app toolbar |
| Low accuracy | Collect more samples; improve lighting |
| TTS not working | Check pyttsx3 or internet for gTTS |
| STT not working | Install PyAudio (see above) |
| `PIL` / `Pillow` error | `pip install Pillow` |

---

## 📜 Credits & Data Source

- **Hand landmark detection**: [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) by Google
- **ASL training images**: Mon95 / Sign-Language-and-Static-gesture-recognition-using-sklearn (public GitHub dataset)
- **ML classifiers**: scikit-learn
- **TTS**: pyttsx3 (offline) + gTTS (online)
- **STT**: SpeechRecognition + Google Web Speech API
