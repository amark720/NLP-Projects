# CUDA / GPU Setup and Fixes (Windows)

This note explains how we got `transcribe.py` running on the GPU (NVIDIA CUDA) on
Windows, the errors we hit, how we fixed them, and the exact commands that finally
worked. Keep this for future reference or when setting up on a new machine.

## Goal

Run Whisper transcription on an NVIDIA GPU (`--device cuda`) instead of the CPU, so
larger models like `medium` / `large-v3` run much faster. We wanted to verify the
GPU and CUDA actually work, so we did NOT fall back to CPU.

## Background

`transcribe.py` uses **faster-whisper**, which internally uses **CTranslate2** as the
engine. On the GPU, CTranslate2 needs two sets of NVIDIA runtime libraries at runtime:

- **cuBLAS** (and its dependency `cublasLt`, plus `cudart`)
- **cuDNN**

The CTranslate2 version decides which CUDA version it needs:

- CTranslate2 **3.x** -> needs CUDA **11** (looks for `cublas64_11.dll`)
- CTranslate2 **4.x** -> needs CUDA **12** (looks for `cublas64_12.dll`)

## The error we hit

```
RuntimeError: Library cublas64_11.dll is not found or cannot be loaded
```

### Why it happened

1. **"cannot be loaded" part** - `cublas64_11.dll` does not work alone. It depends on
   other DLLs like `cublasLt64_11.dll` and `cudart64_110.dll`. Copying only
   `cublas64_11.dll` by hand was not enough, so Windows found the file but could not
   load it because its dependencies were missing.
2. **Version mismatch** - the missing DLL was `cublas64_11.dll` (CUDA 11), but the
   machine had CUDA 12. Manually hunting DLLs is unreliable exactly because of these
   transitive dependencies and version mismatches.

## The fix (what actually worked)

Instead of copying DLLs by hand, we installed the **official NVIDIA pip wheels**, which
ship a complete and matching set of DLLs (cuBLAS + cuBLASLt + cuDNN together) under
`site-packages/nvidia/<lib>/bin`. We also moved to the modern CUDA 12 stack.

### Working commands (run inside the activated virtual environment)

```powershell
# 1. Upgrade to the modern CUDA 12 stack (CTranslate2 4.x + faster-whisper)
pip install -U ctranslate2 faster-whisper

# 2. Install the matching NVIDIA CUDA 12 runtime libraries (cuBLAS + cuDNN)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

> Note: pick ONE CUDA line. Since we upgraded CTranslate2 to 4.x (CUDA 12), we used the
> `cu12` wheels. If you were on CTranslate2 3.x (CUDA 11) instead, the matching wheels
> would be `nvidia-cublas-cu11 nvidia-cudnn-cu11`. Do not mix cu11 and cu12.

### Code change we made in `transcribe.py`

Windows does not automatically search the pip wheel folders for DLLs, so we added a
helper that registers them at runtime:

- `add_nvidia_pip_dll_directories()` - scans `site-packages/nvidia/*/bin` and calls
  `os.add_dll_directory(...)` on each, so the wheel-provided cuBLAS/cuDNN DLLs are found.
- It is called from `add_cuda_dll_directories()` before loading the model, and it prints
  which folders it registered, e.g.:

  ```
  Using CUDA libraries from NVIDIA pip wheels:
    - ...\.venv\Lib\site-packages\nvidia\cublas\bin
    - ...\.venv\Lib\site-packages\nvidia\cudnn\bin
  ```

## Running on the GPU

```powershell
python transcribe.py "C:\path\to\Recordings" --model medium --device cuda
```

## Verifying the GPU is actually being used

In a second terminal, watch the GPU live (refreshes every 2 seconds):

```powershell
nvidia-smi -l 2
```

While transcription runs you should see:

- **GPU-Util** going up (e.g. 60-70%) - GPU is doing work (idle would be ~0-2%).
- **Memory-Usage** rising (e.g. ~3.6 GB for the `medium` model in float16).
- A `python.exe` process listed as using the GPU.

Example (RTX 3060, 6 GB) while running the `medium` model:

```
| GPU-Util 66%   Memory-Usage 3652MiB / 6144MiB   CUDA Version: 12.7 |
```

## Notes / tips

- **No CPU fallback for verification**: we run without `--allow-cpu-fallback` so that if
  the GPU fails, the script errors out instead of silently switching to CPU. This makes
  the GPU test trustworthy.
- **Temperature**: on laptops the GPU can get hot (85C+) during long batch runs. Keep the
  vents clear, and take a short break between files if it stays very high.
- **Where models are cached**: first run downloads the model from Hugging Face to
  `C:\Users\<username>\.cache\huggingface\hub`; after that it works offline.
- **Machines without an NVIDIA GPU** (for example an Intel UHD integrated graphics PC)
  cannot use `--device cuda` at all - CUDA needs an NVIDIA card. On those use
  `--device cpu` with a smaller model like `small` for practical speed.

## Quick command reference

| Purpose | Command |
| --- | --- |
| Upgrade engine (CUDA 12 stack) | `pip install -U ctranslate2 faster-whisper` |
| Install CUDA 12 runtime libs | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` |
| Run on GPU | `python transcribe.py "<path>" --model medium --device cuda` |
| Watch GPU usage live | `nvidia-smi -l 2` |
