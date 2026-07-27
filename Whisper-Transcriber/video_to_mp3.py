"""
Convert a video (or any media) file to an MP3 audio file using ffmpeg.

Usage:
    python video_to_mp3.py "C:\\path\\to\\video.mkv"
    python video_to_mp3.py "C:\\path\\to\\video.mkv" --output "C:\\path\\to\\audio.mp3"
    python video_to_mp3.py "C:\\path\\to\\video.mkv" --bitrate 320k

Output (next to the input file unless --output is given):
    <name>.mp3   -> extracted audio track
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a video/audio file to MP3.")
    parser.add_argument("input", help="Path to the video or audio file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP3 path. Default: same folder/name as input with .mp3.",
    )
    parser.add_argument(
        "--bitrate",
        default="192k",
        help="MP3 bitrate (e.g. 128k, 192k, 320k). Default: 192k.",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg not found on PATH. Install it or add it to PATH.",
            file=sys.stderr,
        )
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else input_path.with_suffix(".mp3")

    print(f"Converting: {input_path.name} -> {output_path.name}")
    print(f"Bitrate: {args.bitrate}\n")

    cmd = [
        "ffmpeg",
        "-y",                 # overwrite output if it exists
        "-i", str(input_path),
        "-vn",                # drop the video stream
        "-acodec", "libmp3lame",
        "-b:a", args.bitrate,
        str(output_path),
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nERROR: ffmpeg failed to convert the file.", file=sys.stderr)
        return result.returncode

    print("\nDone.")
    print(f"Audio: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copilot Change Log
#   - [2026-07-01] technical-writer: Created video-to-mp3 converter using ffmpeg
