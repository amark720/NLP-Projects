<#
    setup.ps1 - One-time setup for Whisper Transcriber (Windows / PowerShell)

    What this script does:
      1. Checks that Python is installed.
      2. Warns if ffmpeg is not on PATH (needed to read media files).
      3. Creates a virtual environment in the ".venv" folder.
      4. Activates it and installs everything from requirements.txt.

    How to run (from this project folder):
        powershell -ExecutionPolicy Bypass -File .\setup.ps1

    After it finishes, activate the venv in any new terminal with:
        .\.venv\Scripts\Activate.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "=== Whisper Transcriber setup ===" -ForegroundColor Cyan

# 1. Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python was not found on PATH." -ForegroundColor Red
    Write-Host "Install Python 3.9+ from https://www.python.org/downloads/ and try again."
    exit 1
}
Write-Host "Found Python: $((python --version) 2>&1)" -ForegroundColor Green

# 2. Check ffmpeg (warn only, not fatal)
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host "WARNING: ffmpeg was not found on PATH." -ForegroundColor Yellow
    Write-Host "It is needed to read media files and for video_to_mp3.py."
    Write-Host "Get it from https://ffmpeg.org/download.html and add it to PATH."
} else {
    Write-Host "Found ffmpeg." -ForegroundColor Green
}

# 3. Create virtual environment
if (Test-Path ".venv") {
    Write-Host "Virtual environment '.venv' already exists. Reusing it." -ForegroundColor Yellow
} else {
    Write-Host "Creating virtual environment in '.venv'..."
    python -m venv .venv
}

# 4. Install dependencies into the venv
Write-Host "Installing dependencies from requirements.txt..."
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Activate the virtual environment with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "Then run, for example:"
Write-Host "    python transcribe.py `"C:\path\to\meeting.mkv`""
