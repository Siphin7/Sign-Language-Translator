# 🤟 Sign Language Translator — Complete Beginner's Setup Guide

> **Who is this for?**  
> Complete beginners who have never used VS Code or Python before.  
> Every single step is written out. Nothing is skipped.

---

## 📋 What You Will End Up With

A real-time app that:
- Watches your webcam
- Recognises ASL hand signs (A to Y)
- Converts them to text on screen
- Reads them aloud through your speakers

---

## 🗂️ PART 1 — Create the Folder & File Structure

### Step 1 — Create the main project folder

Go to any place on your computer (Desktop is fine).  
Create a new folder and name it exactly:

```
sign_language_translator
```

### Step 2 — Create the sub-folders inside it

Open that folder. Inside it, create these **3 folders**:

```
sign_language_translator/
│
├── 📁 src/
├── 📁 dataset/
└── 📁 models/          ← This one can stay empty for now
```

> **How to create a folder on Windows:**  
> Right-click inside the folder → New → Folder → type the name → Enter

### Step 3 — Create all the files

You need to create these files exactly as shown below.  
Copy the content from each file provided to you and save them.

```
sign_language_translator/
│
├── 📄 app.py                 ← Main app (you will run this)
├── 📄 collect_data.py        ← For recording your own signs
├── 📄 train_model.py         ← For training the AI model
├── 📄 requirements.txt       ← List of libraries to install
├── 📄 setup.bat              ← Windows auto-installer (optional)
├── 📄 setup.sh               ← Mac/Linux auto-installer (optional)
│
├── 📁 src/
│   ├── 📄 __init__.py        ← (Can be empty OR copy provided content)
│   ├── 📄 gesture_recognizer.py
│   ├── 📄 sentence_builder.py
│   ├── 📄 tts_engine.py
│   └── 📄 stt_engine.py
│
├── 📁 dataset/
│   └── 📄 gesture_data.csv   ← Copy the provided CSV file here(DM ME FOR THE FILE)
│
└── 📁 models/                ← Leave this empty (auto-filled after training)
```

> ✅ **Tip:** The `gesture_data.csv` file is NOT provided to you, to get that file dm me.  
> Just copy it into the `dataset/` folder. No need to collect data yourself!

---

## 💻 PART 2 — Install VS Code

### Step 1 — Download VS Code

1. Open your browser
2. Go to: **https://code.visualstudio.com**
3. Click the big blue **Download** button
4. Run the installer and click **Next** on every screen
5. Make sure to check ✅ **"Add to PATH"** during install

### Step 2 — Open your project in VS Code

1. Open VS Code
2. Click **File** → **Open Folder**
3. Find and select your `sign_language_translator` folder
4. Click **Select Folder**

You should now see all your files listed on the left side panel.

---

## 🧩 PART 3 — Install VS Code Extensions

Extensions are like plug-ins that make VS Code smarter.  
You need these:

### Extension 1 — Python (REQUIRED)

1. Click the **Extensions icon** on the left sidebar  
   (It looks like 4 squares, one slightly detached)
2. In the search box type: `Python`
3. The first result should be **"Python"** by **Microsoft**
4. Click **Install**

### Extension 2 — Pylance (REQUIRED)

1. Same Extensions panel
2. Search: `Pylance`
3. Install **"Pylance"** by **Microsoft**
4. This gives you code suggestions and error highlights

### Extension 3 — Python Indent (HELPFUL)

1. Search: `Python Indent`
2. Install it
3. This auto-indents your code correctly

### Extension 4 — Better Comments (OPTIONAL but nice)

1. Search: `Better Comments`
2. Install it
3. Makes code comments colourful and readable

---

## 🐍 PART 4 — Install Python

### Step 1 — Download Python

1. Go to: **https://www.python.org/downloads/**
2. Click the big yellow **Download Python 3.x.x** button
3. Run the installer

### ⚠️ VERY IMPORTANT during Python install:

At the very first screen, check this box:  
✅ **"Add Python to PATH"**  
(It is at the bottom of the installer window)

Then click **Install Now**.

### Step 2 — Verify Python is installed

1. In VS Code, press `Ctrl + J` to open the **Terminal**
2. Type this and press Enter:

```
python --version
```

You should see something like:
```
Python 3.11.4
```

If you see that, Python is installed correctly. ✅

---

## 🔧 PART 5 — Set Up a Virtual Environment

A virtual environment is like a **separate room** just for this project's libraries.  
It keeps things clean and prevents conflicts.

### Step 1 — Open the Terminal in VS Code

Press `Ctrl + J`  
(Or click **View** → **Terminal**)

Make sure the terminal shows you are inside your project folder.  
It should look like:

```
C:\Users\YourName\Desktop\sign_language_translator>
```

If not, type:
```
cd Desktop\sign_language_translator
```
(adjust the path to where you saved the folder)

### Step 2 — Create the virtual environment

Type this and press Enter:

```
python -m venv venv
```

Wait a few seconds. A new folder called `venv` will appear in your project. ✅

### Step 3 — Activate the virtual environment

**On Windows:**
```
venv\Scripts\activate
```

**On Mac/Linux:**
```
source venv/bin/activate
```

After activating, you will see `(venv)` at the start of your terminal line:
```
(venv) C:\Users\YourName\Desktop\sign_language_translator>
```

That means the virtual environment is active. ✅

> ⚠️ **Every time you open VS Code** and want to run the project,  
> you must activate the virtual environment again with the command above.

### Step 4 — Tell VS Code to use this Python environment

1. Press `Ctrl + Shift + P` (opens the Command Palette)
2. Type: `Python: Select Interpreter`
3. Press Enter
4. Choose the option that says **`./venv/...`** or has `venv` in it
5. Click it

VS Code will now use the correct Python for this project. ✅

---

## 📦 PART 6 — Install All Required Libraries

Libraries are pre-built tools other people made that your code uses.

### Step 1 — Make sure your virtual environment is active

Your terminal should still show `(venv)` at the start.  
If not, run the activate command again (from Part 5 Step 3).

### Step 2 — Install everything with one command

In the terminal, type this and press Enter:

```
pip install opencv-python mediapipe==0.10.13 numpy scikit-learn pandas Pillow pyttsx3 gTTS pygame SpeechRecognition matplotlib seaborn tqdm
```

This will install **all the libraries at once**.  
It may take **3 to 5 minutes**. You will see text scrolling. That is normal. ✅

### What each library does (simple explanation):

| Library | What it does |
|---------|-------------|
| `opencv-python` | Opens your webcam and reads video frames |
| `mediapipe` | Detects your hand and finds 21 landmark points on it |
| `numpy` | Does fast maths on numbers |
| `scikit-learn` | The AI/ML engine that learns which sign is which |
| `pandas` | Reads and manages the CSV data file |
| `Pillow` | Displays the webcam image inside the app window |
| `pyttsx3` | Speaks text aloud (offline, no internet needed) |
| `gTTS` | Speaks text aloud using Google (better voice, needs internet) |
| `pygame` | Plays the audio from gTTS |
| `SpeechRecognition` | Listens to your microphone and converts speech to text |
| `matplotlib` | Draws graphs and charts after training |
| `seaborn` | Makes the accuracy chart look nice |
| `tqdm` | Shows a progress bar while processing images |

### Step 3 — Install PyAudio (for microphone feature)

PyAudio is separate because it needs special steps.

**Windows only:**
```
pip install pipwin
pipwin install pyaudio
```

**Ubuntu/Debian Linux:**
```
sudo apt-get install python3-pyaudio portaudio19-dev
pip install pyaudio
```

**macOS:**
```
brew install portaudio
pip install pyaudio
```

> ⚠️ If PyAudio fails to install, that is OK.  
> The app still works — you just cannot use the Voice → Text feature.

---

## 🚀 PART 7 — Run the Project (Step by Step)

There are **3 steps** to run this project. You only need to do Step 1 once.

---

### ▶️ STEP A — Train the AI Model (DO THIS FIRST)

The `gesture_data.csv` file in your `dataset/` folder already has **1,680 real ASL hand samples**.  
You do not need to collect your own data. Just train the model on this data.

In the terminal, type:

```
python train_model.py
```

What will happen:
- It reads the CSV file
- Trains 3 different AI models
- Picks the best one automatically
- Saves it to `models/gesture_model.pkl`
- Shows you an accuracy report and a chart

This takes **30 seconds to 2 minutes** depending on your computer.

You should see something like:
```
[RandomForest] Test accuracy: 0.9524 (95.2%)
[MLP]          Test accuracy: 0.9381 (93.8%)
★ Best model: RandomForest (95.2% accuracy)
✓ Model saved → models/gesture_model.pkl
```

That means the AI is trained. ✅

---

### ▶️ STEP B — Run the Main App

In the terminal, type:

```
python app.py
```

A window will open showing your webcam feed. ✅

---

### ▶️ STEP C — (Optional) Add Your Own Signs

If you want to add your name, or signs not in the dataset:

```
python collect_data.py
```

This opens an interactive tool where you can record your own signs.

---

## 🎮 PART 8 — How to Use the App

Once the app is open:

### Sign → Text mode (default)

1. Show your hand to the webcam
2. Hold the ASL sign **still**
3. Watch the **green bar at the bottom** fill up
4. When it fills completely, the letter is added to the sentence
5. Click **🔊 Speak** (or press **Enter**) to hear the sentence aloud

### Voice → Text mode

1. Click the **Voice → Text** radio button at the top right
2. Click **🎤 Start Listening**
3. Speak into your microphone
4. Your words appear in the sentence box

### Keyboard shortcuts

| Key | What it does |
|-----|-------------|
| **Enter** | Speak the sentence aloud |
| **Ctrl + Z** | Undo the last added letter |
| **Escape** | Clear the whole sentence |
| **Ctrl + Space** | Add a space between words |
| **Ctrl + C** | Copy the sentence to clipboard |
| **F1** | Flip the camera mirror |

---

## ❓ PART 9 — Common Problems and Fixes

| Problem | What to do |
|---------|-----------|
| `ModuleNotFoundError: No module named 'cv2'` | Run: `pip install opencv-python` |
| `ModuleNotFoundError: No module named 'mediapipe'` | Run: `pip install mediapipe==0.10.13` |
| `No module named 'PIL'` | Run: `pip install Pillow` |
| Camera not opening / black screen | In the app, try changing **Cam: 0** to **Cam: 1** or **Cam: 2** |
| `No trained model found` | You must run `python train_model.py` first |
| Signs not being detected | Ensure good lighting; keep hand in the green box on screen |
| App crashes on startup | Make sure your virtual environment is activated (`venv\Scripts\activate`) |
| PyAudio failed to install | Voice mode is disabled but everything else still works |
| `(venv)` disappeared from terminal | Run the activate command again |

---

## 🗂️ PART 10 — Final Checklist

Before running, make sure every item below is checked:

- [ ] `sign_language_translator/` folder exists
- [ ] `src/` folder exists inside it with all 5 `.py` files
- [ ] `dataset/` folder exists with `gesture_data.csv` inside
- [ ] `models/` folder exists (can be empty)
- [ ] All 3 root `.py` files exist (`app.py`, `collect_data.py`, `train_model.py`)
- [ ] `requirements.txt` exists
- [ ] VS Code is open at the project folder
- [ ] Python extension is installed in VS Code
- [ ] Virtual environment was created (`venv/` folder exists)
- [ ] Virtual environment is **activated** (`(venv)` shows in terminal)
- [ ] All libraries installed (ran the `pip install ...` command)
- [ ] Model trained (ran `python train_model.py` and saw `gesture_model.pkl` created)
- [ ] App opened (ran `python app.py`)

---

## 📌 PART 11 — The Complete File List (What to Create and Where)

Here is every single file, what folder it goes in, and where to get the content:

```
sign_language_translator/           ← ROOT FOLDER
│
├── app.py                          ← Copy from provided files
├── collect_data.py                 ← Copy from provided files
├── train_model.py                  ← Copy from provided files
├── requirements.txt                ← Copy from provided files
├── setup.bat                       ← Copy from provided files (Windows)
├── setup.sh                        ← Copy from provided files (Mac/Linux)
│
├── src/                            ← Create this folder
│   ├── __init__.py                 ← Copy from provided files
│   ├── gesture_recognizer.py       ← Copy from provided files
│   ├── sentence_builder.py         ← Copy from provided files
│   ├── tts_engine.py               ← Copy from provided files
│   └── stt_engine.py               ← Copy from provided files
│
├── dataset/                        ← Create this folder
│   └── gesture_data.csv            ← Copy the provided CSV file here
│
└── models/                         ← Create this folder, leave it EMPTY
                                      (gesture_model.pkl is auto-created
                                       when you run train_model.py)
```

---

## 🏁 Quick Reference — Commands Summary

Open VS Code terminal (`Ctrl + J`) and run these in order:

```bash
# 1. Create and activate virtual environment (do once)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 2. Install all libraries (do once)
pip install opencv-python mediapipe==0.10.13 numpy scikit-learn pandas Pillow pyttsx3 gTTS pygame SpeechRecognition matplotlib seaborn tqdm

# 3. Train the AI model (do once, or again if you add data)
python train_model.py

# 4. Run the app (do every time you want to use it)
python app.py
```

---

*Built with ❤️ using Python, MediaPipe, and scikit-learn.*  
*Dataset: 1,680 real ASL hand photos processed into 63-point hand landmarks.*
