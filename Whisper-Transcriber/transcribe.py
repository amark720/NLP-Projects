"""
Transcribe a video/audio file (or a whole folder of them) to text transcripts
and SRT subtitle files using faster-whisper (local, offline speech recognition).

Usage:
    # Basic run - just pass the video/audio path (auto-detect language & default model):
    python transcribe.py "C:\\path\\to\\video.mkv"

    # Transcribe EVERY video/audio inside a folder, one by one:
    python transcribe.py "C:\\path\\to\\recordings_folder"
    python transcribe.py "C:\\path\\to\\recordings_folder" --recursive

    # FAST but less accurate (small/tiny model):
    python transcribe.py "C:\\path\\to\\video.mkv" --model tiny
    python transcribe.py "C:\\path\\to\\video.mkv" --model small

    # SLOWER but MORE accurate (medium / large-v3 model):
    python transcribe.py "C:\\path\\to\\video.mkv" --model medium
    python transcribe.py "C:\\path\\to\\video.mkv" --model large-v3

    # Force a specific language (e.g. English or Hindi):
    python transcribe.py "C:\\path\\to\\video.mkv" --language en
    python transcribe.py "C:\\path\\to\\video.mkv" --language hi

    # HINDI + ENGLISH MIXED audio -> get everything in English text:
    python transcribe.py "C:\\path\\to\\video.mkv" --task translate --model medium

Using your GPU (much faster):
    By default the script auto-detects an NVIDIA GPU and uses it if available,
    otherwise it falls back to the CPU. You can force the choice:
        python transcribe.py "video.mkv" --device cuda   # force GPU
        python transcribe.py "video.mkv" --device cpu    # force CPU
    On GPU it automatically uses float16 (fast); on CPU it uses int8.
    Note: GPU use needs an NVIDIA card plus the matching CUDA/cuDNN libraries.
    If the GPU cannot be initialised, the script prints a warning and uses the CPU.

Where are the models stored?
    Models are downloaded from Hugging Face the FIRST time you use them and cached
    locally (not inside this code folder). On Windows the default location is:
        C:\\Users\\<your-username>\\.cache\\huggingface\\hub
    Each model sits in its own folder, e.g. "models--Systran--faster-whisper-medium".
    After the first download it loads from this cache and works offline.

Model choice (speed vs accuracy):
    tiny  -> fastest,  least accurate
    base  -> fast
    small -> balanced (default)
    medium-> slower,   more accurate  (good for Hindi/mixed audio)
    large-v3 -> slowest, most accurate

Outputs (next to each input file):
    <name>.txt   -> plain transcript with timestamps
    <name>.srt   -> subtitle file usable in any video player
"""

import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel

# Media file types the folder mode will pick up.
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}


def format_timestamp(seconds: float, srt: bool = False) -> str:
    """Convert seconds to HH:MM:SS,mmm (SRT) or HH:MM:SS (txt)."""
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    if srt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def cuda_is_available() -> bool:
    """Return True if an NVIDIA GPU usable by CTranslate2 is present."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device_and_compute(requested_device: str, requested_compute: str | None):
    """Decide which device (cpu/cuda) and compute type to use.

    - device 'auto' picks the GPU when one is available, else the CPU.
    - compute type, when not given, defaults to float16 on GPU and int8 on CPU.
    """
    device = requested_device
    if device == "auto":
        device = "cuda" if cuda_is_available() else "cpu"

    if requested_compute:
        compute_type = requested_compute
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def collect_input_files(path: Path, recursive: bool) -> list[Path]:
    """Return the media files to transcribe.

    - A single file -> just that file.
    - A folder -> every supported media file inside it (optionally recursive),
      skipping files that are not recognised media types.
    """
    if path.is_file():
        return [path]

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p
        for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
    )
    return files


def progress_bar(pct: float, width: int = 20) -> str:
    """Render a simple text progress bar like |########------------|."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    return "|" + "#" * filled + "-" * (width - filled) + "|"


def transcribe_one(
    model: WhisperModel, input_path: Path, args, index: int, total: int
) -> None:
    """Transcribe a single file and write its .txt and .srt next to it.

    Prints a live per-file progress percentage (based on how far into the
    media's total duration we are) plus, in folder mode, an X-of-N counter.
    """
    txt_path = input_path.with_suffix(".txt")
    srt_path = input_path.with_suffix(".srt")

    counter = f"[file {index}/{total}] " if total > 1 else ""
    print(f"{counter}Transcribing: {input_path.name}")
    print("This runs locally and may take a while depending on length and model size.\n")

    segments, info = model.transcribe(
        str(input_path),
        task=args.task,
        language=args.language,
        vad_filter=True,  # skip long silences -> faster, cleaner output
        beam_size=5,
    )

    # Total audio length (seconds) so we can show a percentage as we go.
    duration = getattr(info, "duration", 0) or 0

    print(
        f"Detected language: {info.language} "
        f"(probability {info.language_probability:.2f})"
        + (f" | length {format_timestamp(duration)}" if duration else "")
        + "\n"
    )

    txt_lines = []
    srt_lines = []

    for i, segment in enumerate(segments, start=1):
        text = segment.text.strip()

        # How far into the file we are, as a percentage of total duration.
        pct = (segment.end / duration * 100.0) if duration else 0.0

        # Live progress line: percentage + bar + timestamp + text.
        start_disp = format_timestamp(segment.start)
        end_disp = format_timestamp(segment.end)
        if duration:
            print(f"{progress_bar(pct)} {pct:5.1f}% [{end_disp}] {text}")
        else:
            print(f"[{start_disp} -> {end_disp}] {text}")

        txt_lines.append(f"[{start_disp} - {end_disp}] {text}")

        srt_lines.append(str(i))
        srt_lines.append(
            f"{format_timestamp(segment.start, srt=True)} --> "
            f"{format_timestamp(segment.end, srt=True)}"
        )
        srt_lines.append(text)
        srt_lines.append("")

    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    srt_path.write_text("\n".join(srt_lines) + "\n", encoding="utf-8")

    if duration:
        print(f"{progress_bar(100)} 100.0% - finished {input_path.name}")
    print("Done.")
    print(f"Transcript: {txt_path}")
    print(f"Subtitles : {srt_path}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe a video/audio file.")
    parser.add_argument(
        "input",
        help="Path to a video/audio file, OR a folder containing several of them.",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size: tiny, base, small, medium, large-v3 "
        "(bigger = more accurate but slower). Default: small.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force a language code (e.g. 'en'). Default: auto-detect.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="'transcribe' keeps the original spoken language; 'translate' "
        "renders everything into English (useful for Hindi/mixed-language "
        "audio). Default: transcribe.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Where to run: 'auto' uses an NVIDIA GPU if present else the CPU; "
        "'cuda' forces GPU; 'cpu' forces CPU. Default: auto.",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Compute type: int8 (CPU friendly), int8_float16, float16, float32. "
        "Default: float16 on GPU, int8 on CPU.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When the input is a folder, also look inside sub-folders.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: path not found: {input_path}", file=sys.stderr)
        return 1

    files = collect_input_files(input_path, args.recursive)
    if not files:
        print(
            f"ERROR: no supported media files found in: {input_path}",
            file=sys.stderr,
        )
        return 1

    device, compute_type = resolve_device_and_compute(args.device, args.compute_type)

    print(f"Loading model '{args.model}' on {device.upper()} (compute_type={compute_type})...")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as exc:
        # GPU may be unusable (no CUDA/cuDNN, out of memory, etc.) -> fall back to CPU.
        if device == "cuda":
            print(
                f"WARNING: could not use the GPU ({exc}). Falling back to CPU.",
                file=sys.stderr,
            )
            device, compute_type = "cpu", args.compute_type or "int8"
            print(
                f"Loading model '{args.model}' on CPU (compute_type={compute_type})..."
            )
            model = WhisperModel(args.model, device=device, compute_type=compute_type)
        else:
            raise

    total = len(files)
    if total > 1:
        print(f"\nFound {total} media file(s) to transcribe.\n")

    for index, media_file in enumerate(files, start=1):
        if total > 1:
            print(f"===== [{index}/{total}] {media_file.name} =====")
        transcribe_one(model, media_file, args, index, total)

    if total > 1:
        print(f"All done. Transcribed {total} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copilot Change Log
#   - [2026-06-26] technical-writer: Added --task option (transcribe/translate) to support English output for Hindi/mixed-language audio
#   - [2026-07-27] technical-writer: Added optional GPU (--device auto/cpu/cuda) with automatic CPU fallback, and folder input to batch-transcribe all media files (with --recursive)
#   - [2026-07-27] technical-writer: Added user-friendly progress tracking - per-file X-of-N counter and a live percentage + progress bar based on media duration
