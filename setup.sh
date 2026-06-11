#!/usr/bin/env bash
# ============================================================
#  setup.sh  —  One-click setup for Linux / macOS
#  Usage: chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "  ============================================"
echo "   Sign Language Translator — Linux/macOS Setup"
echo "  ============================================"
echo ""

# ── Detect OS ───────────────────────────────────────────────
OS="$(uname -s)"
echo "  Detected OS: $OS"

# ── Check Python ────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo -e "  ${RED}ERROR: python3 not found.${NC}"
    if [[ "$OS" == "Darwin" ]]; then
        echo "  Install with: brew install python3"
    else
        echo "  Install with: sudo apt-get install python3 python3-venv python3-pip"
    fi
    exit 1
fi
echo -e "  ${GREEN}[OK]${NC} Python: $(python3 --version)"

# ── System dependencies for PyAudio ─────────────────────────
echo ""
echo "  Installing system dependencies for audio..."

if [[ "$OS" == "Linux" ]]; then
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y python3-venv python3-pip \
             portaudio19-dev python3-pyaudio \
             libsm6 libxext6 libxrender-dev \
             espeak espeak-ng \
             2>/dev/null || true
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3-venv python3-pip \
             portaudio-devel python3-pyaudio espeak 2>/dev/null || true
    fi
elif [[ "$OS" == "Darwin" ]]; then
    if command -v brew &>/dev/null; then
        brew install portaudio 2>/dev/null || true
    else
        echo -e "  ${YELLOW}Homebrew not found — PyAudio may not install.${NC}"
        echo "  Install Homebrew from: https://brew.sh"
    fi
fi

# ── Virtual environment ──────────────────────────────────────
echo ""
if [[ ! -d "venv" ]]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
fi
echo -e "  ${GREEN}[OK]${NC} Virtual environment ready"

# Activate
source venv/bin/activate

# ── pip upgrade ──────────────────────────────────────────────
echo ""
echo "  Upgrading pip..."
pip install --upgrade pip -q

# ── Python packages ──────────────────────────────────────────
echo ""
echo "  Installing Python packages (this may take 2–3 minutes)..."
pip install \
    opencv-python \
    "mediapipe==0.10.13" \
    numpy \
    scikit-learn \
    pandas \
    Pillow \
    pyttsx3 \
    gTTS \
    pygame \
    SpeechRecognition \
    matplotlib \
    seaborn \
    tqdm \
    -q

# ── PyAudio ──────────────────────────────────────────────────
echo ""
echo "  Installing PyAudio..."
pip install pyaudio -q || {
    echo -e "  ${YELLOW}PyAudio install failed — Voice→Text will be disabled.${NC}"
    echo "  To fix on Ubuntu: sudo apt-get install python3-pyaudio portaudio19-dev"
    echo "  To fix on macOS:  brew install portaudio && pip install pyaudio"
}

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "  ============================================"
echo -e "  ${GREEN}Setup complete!${NC}"
echo "  ============================================"
echo ""
echo "  NEXT STEPS:"
echo "    1. Activate environment:  source venv/bin/activate"
echo "    2. Train the model:       python train_model.py"
echo "    3. Run the app:           python app.py"
echo ""
echo "  The dataset is already in dataset/gesture_data.csv"
echo "  (1680 real ASL samples, letters A-Y)"
echo ""
