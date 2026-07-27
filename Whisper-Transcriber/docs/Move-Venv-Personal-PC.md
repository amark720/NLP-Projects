# Move `.venv` into Whisper-Transcriber (Personal PC steps)

This guide is for the **personal laptop** (the one with the NVIDIA RTX 3060 GPU where
transcription actually runs). On that machine the `.venv` is at the **repo root**
(`NLP-Projects/.venv`) and it is a **working** environment (all packages + GPU set up).

We already changed the repo so that:
- The root `.gitignore` is removed.
- Only `Whisper-Transcriber/.gitignore` remains, and it ignores `.venv/`.

So on the personal PC we need to **move `.venv` into the `Whisper-Transcriber` folder**
and then **repair the venv paths** (a moved venv has old hardcoded paths in its
activate scripts, so it must be fixed). Follow the steps below exactly.

> IMPORTANT: Run every command from the **repo root** folder `NLP-Projects`
> (the folder that contains `Whisper-Transcriber`, `check_cublas11.py`, etc.).
> Use PowerShell.

---

## Step 0 - Get the latest code first

```powershell
git pull origin main
```

This brings the updated `.gitignore` setup. Your local root `.venv` will stay on disk
(git no longer tracks it), so nothing is lost.

---

## Step 1 - Make sure the venv is NOT active / not in use

1. If your terminal prompt shows `(.venv)` at the start, deactivate it:

   ```powershell
   deactivate
   ```

2. Close any other terminal that has the venv active.
3. In VS Code, do not run any Python file right now (so no process is holding the venv).

If the move fails later with an "in use" error, it means something is still using the
venv - close all terminals / VS Code Python processes and try again.

---

## Step 2 - Move `.venv` into the Whisper-Transcriber folder

```powershell
Move-Item -Path ".venv" -Destination "Whisper-Transcriber\.venv"
```

Check it moved:

```powershell
"root .venv exists: "  + (Test-Path ".venv")
"inside folder exists: " + (Test-Path "Whisper-Transcriber\.venv")
```

Expected output:

```
root .venv exists: False
inside folder exists: True
```

---

## Step 3 - Find the base Python path

The moved venv still points to the old location, so we must regenerate its scripts using
the **base Python** it was built from. Read the config to find it:

```powershell
Get-Content "Whisper-Transcriber\.venv\pyvenv.cfg"
```

Look at the `home =` line. On the personal PC it should be something like:

```
home = C:\Users\amark\AppData\Local\Programs\Python\Python311
```

Note that path - that folder must contain `python.exe`. Confirm it exists:

```powershell
Test-Path "C:\Users\amark\AppData\Local\Programs\Python\Python311\python.exe"
```

If this prints `True`, continue to Step 4. If it prints `False`, your Python is installed
somewhere else - find `python.exe` for Python 3.11 and use that path in Step 4.

---

## Step 4 - Repair (relocate) the venv

Re-run `venv` creation on the moved folder using the base Python. This regenerates the
activate scripts and `pyvenv.cfg` with the **new correct path**, while keeping all your
already-installed packages.

```powershell
& "C:\Users\amark\AppData\Local\Programs\Python\Python311\python.exe" -m venv "Whisper-Transcriber\.venv"
```

> Do NOT add `--clear` to this command. `--clear` would delete all installed packages.
> Plain `-m venv` on an existing folder only fixes the scripts and keeps packages.

---

## Step 5 - Activate and verify the packages are still there

```powershell
.\Whisper-Transcriber\.venv\Scripts\Activate.ps1
python --version
python -c "import faster_whisper, ctranslate2; print('faster-whisper OK')"
```

Expected: it prints the Python version and `faster-whisper OK`.

If you get `ModuleNotFoundError`, the packages did not survive - go to the
"Fallback" section at the bottom.

---

## Step 6 - Verify the GPU still works

Run a quick transcription on any short clip (replace the path with a real file), or the
whole folder, using the GPU:

```powershell
python "Whisper-Transcriber\transcribe.py" "C:\path\to\any-short-clip.mp4" --model medium --device cuda
```

At the start it should print:

```
Using CUDA libraries from NVIDIA pip wheels:
  - ...\Whisper-Transcriber\.venv\Lib\site-packages\nvidia\cublas\bin
  - ...\Whisper-Transcriber\.venv\Lib\site-packages\nvidia\cudnn\bin
```

In a second terminal, watch the GPU:

```powershell
nvidia-smi -l 2
```

You should see GPU-Util go up and memory usage rise - that confirms GPU is working from
the new venv location.

---

## Step 7 - Point VS Code to the new interpreter

1. Press `Ctrl+Shift+P`.
2. Type and select **Python: Select Interpreter**.
3. Choose the one at:
   `.\Whisper-Transcriber\.venv\Scripts\python.exe`
   (Use **Enter interpreter path** if it is not listed, and browse to that file.)

---

## Step 8 - Confirm git is clean (venv is ignored, not tracked)

```powershell
git status --short
git check-ignore -v "Whisper-Transcriber\.venv\pyvenv.cfg"
```

- `git status` should NOT list any `.venv/...` files.
- `git check-ignore` should print a line pointing to
  `Whisper-Transcriber/.gitignore:2:.venv/` - that means it is correctly ignored.

Nothing to commit here for the venv (it is intentionally not tracked). You are done.

---

## Fallback - if the repair fails or packages are missing

If Step 5 shows missing packages, just recreate a fresh venv inside the folder and
reinstall. Run from the repo root:

```powershell
# 1. Remove the broken venv folder
Remove-Item -Recurse -Force "Whisper-Transcriber\.venv"

# 2. Create a fresh venv inside the folder
& "C:\Users\amark\AppData\Local\Programs\Python\Python311\python.exe" -m venv "Whisper-Transcriber\.venv"

# 3. Activate it
.\Whisper-Transcriber\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install project requirements
python -m pip install --upgrade pip
pip install -r "Whisper-Transcriber\requirements.txt"

# 5. Install the GPU (CUDA 12) libraries so --device cuda works
pip install -U ctranslate2 faster-whisper
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Then repeat Step 6 (GPU verify). See also `docs/CUDA-GPU-Setup.md` in this same folder
for the full CUDA/GPU explanation.

---

## Quick summary (for a smart-enough agent to just do)

1. `git pull origin main`
2. `deactivate` (if active), close other terminals using the venv.
3. `Move-Item ".venv" "Whisper-Transcriber\.venv"`
4. Read `Whisper-Transcriber\.venv\pyvenv.cfg` -> get the `home =` base Python path.
5. `<basePython>\python.exe -m venv "Whisper-Transcriber\.venv"`  (no `--clear`).
6. Activate, verify `import faster_whisper` works, verify GPU with a test run + `nvidia-smi -l 2`.
7. VS Code: Python: Select Interpreter -> `Whisper-Transcriber\.venv\Scripts\python.exe`.
8. `git status` must show no `.venv` files (it is ignored).
