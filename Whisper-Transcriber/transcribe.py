"""
Transcribe a video/audio file to a text transcript and an SRT subtitle file
using faster-whisper (local, offline speech recognition).

Usage:
    # Basic run - just pass the video/audio path (auto-detect language & default model):
    python transcribe.py "C:\\path\\to\\video.mkv"

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

Outputs (next to the input file):
    <name>.txt   -> plain transcript with timestamps
    <name>.srt   -> subtitle file usable in any video player
"""

import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def format_timestamp(seconds: float, srt: bool = False) -> str:
    """Convert seconds to HH:MM:SS,mmm (SRT) or HH:MM:SS (txt)."""
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    if srt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe a video/audio file.")
    parser.add_argument("input", help="Path to the video or audio file.")
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
        "--compute-type",
        default="int8",
        help="Compute type: int8 (CPU friendly), int8_float16, float16, float32. "
        "Default: int8.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        return 1

    txt_path = input_path.with_suffix(".txt")
    srt_path = input_path.with_suffix(".srt")

    print(f"Loading model '{args.model}' (compute_type={args.compute_type}) on CPU...")
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)

    print(f"Transcribing: {input_path.name}")
    print("This runs locally and may take a while depending on length and model size.\n")

    segments, info = model.transcribe(
        str(input_path),
        task=args.task,
        language=args.language,
        vad_filter=True,  # skip long silences -> faster, cleaner output
        beam_size=5,
    )

    print(
        f"Detected language: {info.language} "
        f"(probability {info.language_probability:.2f})\n"
    )

    txt_lines = []
    srt_lines = []

    for i, segment in enumerate(segments, start=1):
        text = segment.text.strip()

        # Live progress so you can watch it work.
        start_disp = format_timestamp(segment.start)
        end_disp = format_timestamp(segment.end)
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

    print("\nDone.")
    print(f"Transcript: {txt_path}")
    print(f"Subtitles : {srt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copilot Change Log
#   - [2026-06-26] technical-writer: Added --task option (transcribe/translate) to support English output for Hindi/mixed-language audio
