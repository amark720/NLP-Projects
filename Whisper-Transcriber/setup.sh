#!/usr/bin/env bash
#
# setup.sh - One-time setup for Whisper Transcriber (Linux / macOS)
#
# What this script does:
#   1. Checks that Python 3 is installed.
#   2. Warns if ffmpeg is not on PATH (needed to read media files).
#   3. Creates a virtual environment in the ".venv" folder.
#   4. Installs everything from requirements.txt into it.
#
# How to run (from this project folder):
#     chmod +x setup.sh
#     ./setup.sh
#
# After it finishes, activate the venv in any new terminal with:
#     source .venv/bin/activate

set -e

echo "=== Whisper Transcriber setup ==="

# 1. Check Python 3
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 was not found on PATH."
    echo "Install Python 3.9+ and try again."
    exit 1
fi
echo "Found Python: $(python3 --version)"

# 2. Check ffmpeg (warn only, not fatal)
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg was not found on PATH."
    echo "It is needed to read media files and for video_to_mp3.py."
    echo "Install it (e.g. 'sudo apt install ffmpeg' or 'brew install ffmpeg')."
else
    echo "Found ffmpeg."
fi

# 3. Create virtual environment
if [ -d ".venv" ]; then
    echo "Virtual environment '.venv' already exists. Reusing it."
else
    echo "Creating virtual environment in '.venv'..."
    python3 -m venv .venv
fi

# 4. Install dependencies into the venv
echo "Installing dependencies from requirements.txt..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate the virtual environment with:"
echo "    source .venv/bin/activate"
echo "Then run, for example:"
echo "    python transcribe.py \"/path/to/meeting.mkv\""
