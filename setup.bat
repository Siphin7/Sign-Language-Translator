@echo off
:: ============================================================
::  setup.bat  —  One-click setup for Windows
::  Usage: Double-click this file OR run in terminal
:: ============================================================

echo.
echo  ============================================
echo   Sign Language Translator — Windows Setup
echo  ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found!
    echo  Download from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause & exit /b 1
)
echo  [OK] Python found
python --version

:: Create virtual environment
if not exist "venv" (
    echo.
    echo  Creating virtual environment...
    python -m venv venv
)
echo  [OK] Virtual environment ready

:: Activate
call venv\Scripts\activate.bat

:: Upgrade pip
echo.
echo  Upgrading pip...
python -m pip install --upgrade pip -q

:: Install core requirements (excluding PyAudio)
echo.
echo  Installing packages (this may take 2-3 minutes)...
pip install opencv-python mediapipe==0.10.13 numpy scikit-learn pandas ^
            Pillow pyttsx3 gTTS pygame SpeechRecognition matplotlib ^
            seaborn tqdm -q

:: Attempt PyAudio via pipwin
echo.
echo  Installing PyAudio (needed for Voice->Text feature)...
pip install pipwin -q
pipwin install pyaudio
if errorlevel 1 (
    echo.
    echo  NOTE: PyAudio install failed. Voice->Text will be disabled.
    echo  To fix manually, download PyAudio wheel from:
    echo  https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
    echo  Then: pip install PyAudio‑0.x.x‑cpXX‑cpXX‑win_amd64.whl
)

:: Done
echo.
echo  ============================================
echo   Setup complete!
echo  ============================================
echo.
echo  NEXT STEPS:
echo    1. Activate environment:  venv\Scripts\activate
echo    2. Train the model:       python train_model.py
echo    3. Run the app:           python app.py
echo.
echo  The dataset is already in dataset\gesture_data.csv
echo  (1680 real ASL samples, letters A-Y)
echo.
pause
