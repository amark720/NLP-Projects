'''
Transcribe a video/audio file (or a whole folder of them) to text transcripts
and SRT subtitle files using faster-whisper (local, offline speech recognition).

Usage:
    # Basic run - just pass the video/audio path (auto-detect language & default model):
    python transcribe.py "C:\\path\\to\\video.mkv"

    # Transcribe EVERY video/audio inside a folder, one by one:
    python transcribe.py "C:\\path\\to\\recordings_folder"
    python transcribe.py "C:\\Users\\amark\\Videos\\Interview Recordings\\More Videos" --model large-v3 --device cuda --language en
    python transcribe.py "path of video" --model large-v3 --device cuda --language en --task translate
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
    small -> balanced
    medium-> slower,   more accurate
    large-v3-turbo -> DEFAULT. Near large-v3 accuracy at roughly medium speed.
                      Cannot translate, only transcribe.
    large-v3 -> slowest, best accuracy, and the one to use with --task translate.

Getting technical words right (GenAI / Azure / data-science / scrum jargon):
    Two files next to this script drive this, and both are used automatically:

        domain_vocab.txt  -> terms fed to the model as "hotwords" while it decodes,
                             so it expects to hear LangChain, NL2SQL, Azure AI Search...
                             Only the first ~800 characters fit, so order matters.
        corrections.txt   -> "wrong => right" rules applied to the finished text,
                             e.g. "rack approach => RAG approach". No size limit.

    Edit those files to match your own vocabulary. To turn either off:
        python transcribe.py "video.mkv" --vocab "" --corrections ""

    Segments that are only filler ("um um um") or an exact repeat of the previous
    line are dropped, because those are what Whisper invents over silence. Keep
    them with --keep-noise.

Outputs (next to each input file):
    <name>.txt   -> plain transcript with timestamps
    <name>.srt   -> subtitle file usable in any video player
'''

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from faster_whisper import WhisperModel

# Media file types the folder mode will pick up.
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VOCAB_FILE = PROJECT_DIR / "domain_vocab.txt"
DEFAULT_CORRECTIONS_FILE = PROJECT_DIR / "corrections.txt"

# Best accuracy for plain transcription; turbo models cannot translate, so
# Hindi/mixed audio that needs English output falls back to the full model.
DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_TRANSLATE_MODEL = "large-v3"

# Whisper truncates the hotword prompt at ~220 tokens, so only roughly this
# many characters of domain_vocab.txt ever reach the model.
HOTWORD_CHAR_BUDGET = 800

# Text Whisper invents when it is decoding silence, music or crosstalk.
JUNK_PHRASES = {
    "thanks for watching",
    "thank you for watching",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "subtitles by the amara.org community",
    "amara.org",
    "transcription by castingwords",
    "www.mooji.org",
}
FILLER_ONLY = re.compile(r"^(?:(?:um|uh|hmm+|mm+|mhm|ah|eh|hm)\b[\s,.!?\-]*)+$", re.IGNORECASE)


def format_timestamp(seconds: float, srt: bool = False) -> str:
    """Convert seconds to HH:MM:SS,mmm (SRT) or HH:MM:SS (txt)."""
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    if srt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_vocabulary(vocab_path: Path) -> str:
    """Build the comma-separated hotword string that biases Whisper's decoding.

    Terms are taken in file order and stopped once the model's prompt budget is
    reached, so the most important words must be listed first.
    """
    if not vocab_path.exists():
        return ""

    terms: list[str] = []
    used = 0
    dropped = 0
    for raw in vocab_path.read_text(encoding="utf-8").splitlines():
        term = raw.split("#", 1)[0].strip()
        if not term:
            continue
        if used + len(term) + 2 > HOTWORD_CHAR_BUDGET:
            dropped += 1
            continue
        terms.append(term)
        used += len(term) + 2

    if terms:
        print(f"Domain vocabulary: using {len(terms)} term(s) from {vocab_path.name}")
        if dropped:
            print(
                f"  ({dropped} term(s) skipped - Whisper's hotword limit was reached. "
                "Move important ones higher up, or put them in corrections.txt.)"
            )
    return ", ".join(terms)


def load_corrections(corrections_path: Path) -> list[tuple[re.Pattern, str]]:
    """Read the 'wrong => right' glossary used to clean up the decoded text."""
    if not corrections_path.exists():
        return []

    pairs: list[tuple[str, str]] = []
    for raw in corrections_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        wrong, right = line.split("=>", 1)
        wrong, right = wrong.strip(), right.strip()
        if wrong and right:
            pairs.append((wrong, right))

    # Longest first so "rack based" wins over a shorter overlapping entry.
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    rules = [
        (re.compile(rf"(?<!\w){re.escape(wrong)}(?!\w)", re.IGNORECASE), right)
        for wrong, right in pairs
    ]
    if rules:
        print(f"Glossary: loaded {len(rules)} correction(s) from {corrections_path.name}")
    return rules


def apply_corrections(text: str, rules: list[tuple[re.Pattern, str]]) -> str:
    """Replace misheard technical terms with their canonical spelling."""
    for pattern, replacement in rules:
        text = pattern.sub(replacement.replace("\\", r"\\"), text)
    return text


def is_noise_segment(text: str, no_speech_prob: float) -> bool:
    """Detect the filler and boilerplate Whisper produces on near-silent audio."""
    stripped = text.strip()
    if not stripped:
        return True
    if FILLER_ONLY.match(stripped):
        return True
    normalised = re.sub(r"[^a-z0-9.\s]", "", stripped.lower()).strip()
    return normalised in JUNK_PHRASES and no_speech_prob > 0.4


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


def _looks_like_cuda_runtime_error(exc: Exception) -> bool:
    """Return True when the error likely comes from missing CUDA runtime libraries."""
    message = str(exc).lower()
    return (
        "cublas" in message
        or "cudnn" in message
        or "cudart" in message
        or "cuda" in message and any(token in message for token in ("not found", "cannot be loaded", "failed to load", "failed to initialize"))
        or ("dll" in message and "not found" in message)
    )


def ensure_cuda_dll_available() -> None:
    """Make the expected CUDA runtime DLL discoverable by copying it locally if needed."""
    dll_name = "cublas64_11.dll"
    project_dir = Path(__file__).resolve().parent
    local_dll_path = project_dir / dll_name

    if local_dll_path.exists():
        return

    candidates: list[Path] = []
    for env_name in (
        "CUDA_PATH",
        "CUDA_PATH_V12_7",
        "CUDA_PATH_V12_6",
        "CUDA_PATH_V12_5",
        "CUDA_PATH_V12_4",
        "CUDA_PATH_V12_3",
        "CUDA_PATH_V12_2",
        "CUDA_PATH_V12_1",
        "CUDA_PATH_V12_0",
        "CUDA_PATH_V11_8",
        "CUDA_PATH_V11_7",
        "CUDA_PATH_V11_6",
        "CUDA_PATH_V11_5",
        "CUDA_PATH_V11_4",
        "CUDA_PATH_V11_3",
        "CUDA_PATH_V11_2",
        "CUDA_PATH_V11_1",
        "CUDA_PATH_V11_0",
    ):
        if env_name in os.environ:
            path = Path(os.environ[env_name])
            if path.exists():
                candidates.append(path / "bin")

    custom_locations = [
        Path(r"C:\CUDA\v12.7\bin"),
        Path(r"C:\CUDA\v12.6\bin"),
        Path(r"C:\CUDA\v12.5\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"),
    ]
    for path in custom_locations:
        if path.exists():
            candidates.append(path)

    for bin_dir in candidates:
        dll_path = bin_dir / dll_name
        if dll_path.exists():
            try:
                shutil.copy2(dll_path, local_dll_path)
                print(f"Copied {dll_name} to {local_dll_path}")
            except Exception:
                pass
            break

    if not local_dll_path.exists():
        print(
            f"WARNING: could not find {dll_name} in common CUDA locations. Continuing without a local copy.",
            file=sys.stderr,
        )


def add_nvidia_pip_dll_directories() -> list[Path]:
    """Expose CUDA DLLs shipped by the official NVIDIA pip wheels.

    faster-whisper/CTranslate2 need cuBLAS and cuDNN (plus their own dependencies like
    cublasLt and cudart) at runtime. The most reliable way to get a *complete and
    matching* set of these DLLs on Windows is via the NVIDIA pip wheels
    (e.g. nvidia-cublas-cuXX, nvidia-cudnn-cuXX), which drop everything together under
    site-packages/nvidia/<lib>/bin. Windows does not search those folders automatically,
    so we register each of them here. Returns the directories that were added.
    """
    added: list[Path] = []
    if not hasattr(os, "add_dll_directory"):
        return added

    search_roots: list[Path] = []
    try:
        import site

        for entry in site.getsitepackages():
            search_roots.append(Path(entry))
        user_site = site.getusersitepackages()
        if user_site:
            search_roots.append(Path(user_site))
    except Exception:
        pass

    # Fallback: derive site-packages from this interpreter's own location.
    search_roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    seen: set[Path] = set()
    for root in search_roots:
        nvidia_dir = root / "nvidia"
        if not nvidia_dir.exists():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            resolved = bin_dir.resolve()
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            try:
                os.add_dll_directory(str(resolved))
                added.append(resolved)
            except OSError:
                pass
    return added


def add_cuda_dll_directories() -> None:
    """Expose the project folder and common CUDA bin folders to Windows DLL resolution."""
    if not hasattr(os, "add_dll_directory"):
        return

    project_dir = Path(__file__).resolve().parent
    try:
        os.add_dll_directory(str(project_dir))
    except OSError:
        pass

    # DLLs installed via NVIDIA pip wheels (most reliable on Windows) take priority.
    nvidia_dirs = add_nvidia_pip_dll_directories()
    if nvidia_dirs:
        print("Using CUDA libraries from NVIDIA pip wheels:")
        for bin_dir in nvidia_dirs:
            print(f"  - {bin_dir}")

    candidates: list[Path] = []
    for env_name in (
        "CUDA_PATH",
        "CUDA_PATH_V12_7",
        "CUDA_PATH_V12_6",
        "CUDA_PATH_V12_5",
        "CUDA_PATH_V12_4",
        "CUDA_PATH_V12_3",
        "CUDA_PATH_V12_2",
        "CUDA_PATH_V12_1",
        "CUDA_PATH_V12_0",
        "CUDA_PATH_V11_8",
        "CUDA_PATH_V11_7",
        "CUDA_PATH_V11_6",
        "CUDA_PATH_V11_5",
        "CUDA_PATH_V11_4",
        "CUDA_PATH_V11_3",
        "CUDA_PATH_V11_2",
        "CUDA_PATH_V11_1",
        "CUDA_PATH_V11_0",
    ):
        if env_name in os.environ:
            path = Path(os.environ[env_name])
            if path.exists():
                candidates.append(path / "bin")

    custom_locations = [
        Path(r"C:\CUDA\v12.7\bin"),
        Path(r"C:\CUDA\v12.6\bin"),
        Path(r"C:\CUDA\v12.5\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"),
    ]
    for path in custom_locations:
        if path.exists():
            candidates.append(path)

    for bin_dir in candidates:
        if not bin_dir.exists():
            continue
        try:
            os.add_dll_directory(str(bin_dir))
        except OSError:
            pass


def load_whisper_model(model_name: str, requested_device: str, requested_compute: str | None):
    """Create a WhisperModel, falling back to CPU only when CUDA cannot be initialized at all."""
    ensure_cuda_dll_available()
    add_cuda_dll_directories()
    device, compute_type = resolve_device_and_compute(requested_device, requested_compute)

    if device == "cuda":
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as exc:
            if "out of memory" in str(exc).lower() and compute_type != "int8_float16":
                print(
                    f"WARNING: '{model_name}' did not fit in GPU memory as {compute_type}. "
                    "Retrying as int8_float16.",
                    file=sys.stderr,
                )
                compute_type = "int8_float16"
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
            elif _looks_like_cuda_runtime_error(exc):
                print(
                    "WARNING: CUDA runtime libraries are unavailable. Falling back to CPU.",
                    file=sys.stderr,
                )
                device, compute_type = "cpu", requested_compute if requested_compute is not None else "int8"
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
            else:
                raise
    else:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

    if hasattr(model, "__dict__"):
        setattr(model, "_runtime_device", device)
    return model, device, compute_type


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


def build_decode_options(args, hotwords: str) -> dict:
    """Decoding settings tuned for long technical calls.

    condition_on_previous_text is off because a single bad guess otherwise
    snowballs into the repeated-sentence loops Whisper is famous for, and the
    hotwords are re-applied to every window instead of only the first one.
    """
    options = {
        "task": args.task,
        "language": args.language,
        "beam_size": 5,
        "vad_filter": True,  # skip long silences -> faster, cleaner output
        "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        "condition_on_previous_text": False,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "word_timestamps": True,
        "hallucination_silence_threshold": 2.0,
    }
    if hotwords:
        options["hotwords"] = hotwords
    return options


def transcribe_one(
    model: WhisperModel,
    input_path: Path,
    args,
    index: int,
    total: int,
    hotwords: str = "",
    corrections: list[tuple[re.Pattern, str]] | None = None,
) -> None:
    """Transcribe a single file and write its .txt and .srt next to it.

    Prints a live per-file progress percentage (based on how far into the
    media's total duration we are) plus, in folder mode, an X-of-N counter.
    """
    corrections = corrections or []
    txt_path = input_path.with_suffix(".txt")
    srt_path = input_path.with_suffix(".srt")

    counter = f"[file {index}/{total}] " if total > 1 else ""
    print(f"{counter}Transcribing: {input_path.name}")
    print("This runs locally and may take a while depending on length and model size.\n")

    decode_options = build_decode_options(args, hotwords)

    try:
        segments, info = model.transcribe(str(input_path), **decode_options)
    except Exception as exc:
        if (
            getattr(model, "_runtime_device", None) == "cuda"
            and _looks_like_cuda_runtime_error(exc)
            and getattr(args, "allow_cpu_fallback", False)
        ):
            print(
                f"WARNING: transcription hit a CUDA runtime issue ({exc}). Retrying this file on CPU because --allow-cpu-fallback was requested.",
                file=sys.stderr,
            )
            cpu_model, _, _ = load_whisper_model(args.model, "cpu", "int8")
            segments, info = cpu_model.transcribe(str(input_path), **decode_options)
        else:
            raise

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
    kept = 0
    skipped = 0
    previous_text = None

    for segment in segments:
        text = apply_corrections(segment.text.strip(), corrections)

        if not args.keep_noise and (
            is_noise_segment(text, segment.no_speech_prob)
            or text.lower() == (previous_text or "").lower()
        ):
            skipped += 1
            continue
        previous_text = text
        kept += 1

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

        srt_lines.append(str(kept))
        srt_lines.append(
            f"{format_timestamp(segment.start, srt=True)} --> "
            f"{format_timestamp(segment.end, srt=True)}"
        )
        srt_lines.append(text)
        srt_lines.append("")

    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    srt_path.write_text("\n".join(srt_lines) + "\n", encoding="utf-8")

    if skipped:
        print(f"\nFiltered out {skipped} silent/repeated segment(s).")

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
        default=None,
        help="Whisper model size: tiny, base, small, medium, large-v3, large-v3-turbo "
        f"(bigger = more accurate but slower). Default: {DEFAULT_MODEL}, or "
        f"{DEFAULT_TRANSLATE_MODEL} with --task translate because turbo models cannot translate.",
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
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If CUDA hits a runtime issue during a file, retry that file on CPU. Default: off, to keep GPU sequential processing intact.",
    )
    parser.add_argument(
        "--vocab",
        default=str(DEFAULT_VOCAB_FILE),
        help="File of domain terms (one per line) used to bias the model towards your "
        f"jargon. Default: {DEFAULT_VOCAB_FILE.name} next to this script. Use '' to disable.",
    )
    parser.add_argument(
        "--corrections",
        default=str(DEFAULT_CORRECTIONS_FILE),
        help="File of 'wrong => right' rules applied to the finished text. "
        f"Default: {DEFAULT_CORRECTIONS_FILE.name} next to this script. Use '' to disable.",
    )
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="Keep filler-only ('um um um') and repeated segments that the model invents "
        "over silence. Default: off, they are dropped.",
    )
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_TRANSLATE_MODEL if args.task == "translate" else DEFAULT_MODEL
    elif args.task == "translate" and "turbo" in args.model:
        print(
            f"WARNING: '{args.model}' was not trained for translation. "
            f"Use --model {DEFAULT_TRANSLATE_MODEL} for Hindi/mixed audio.",
            file=sys.stderr,
        )

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

    hotwords = load_vocabulary(Path(args.vocab)) if args.vocab else ""
    corrections = load_corrections(Path(args.corrections)) if args.corrections else []

    device, compute_type = resolve_device_and_compute(args.device, args.compute_type)
    print(f"Loading model '{args.model}' on {device.upper()} (compute_type={compute_type})...")
    model, device, compute_type = load_whisper_model(args.model, args.device, args.compute_type)
    print(f"Using model '{args.model}' on {device.upper()} (compute_type={compute_type})")

    total = len(files)
    if total > 1:
        print(f"\nFound {total} media file(s) to transcribe.\n")
        print("Processing files sequentially, one by one. No parallel batch processing is used.\n")

    for index, media_file in enumerate(files, start=1):
        if total > 1:
            print(f"===== [{index}/{total}] {media_file.name} =====")
        transcribe_one(model, media_file, args, index, total, hotwords, corrections)

    if total > 1:
        print(f"All done. Transcribed {total} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copilot Change Log
#   - [2026-06-26] technical-writer: Added --task option (transcribe/translate) to support English output for Hindi/mixed-language audio
#   - [2026-07-27] technical-writer: Added optional GPU (--device auto/cpu/cuda) with automatic CPU fallback, and folder input to batch-transcribe all media files (with --recursive)
#   - [2026-07-27] technical-writer: Added user-friendly progress tracking - per-file X-of-N counter and a live percentage + progress bar based on media duration
#   - [2026-07-27] technical-writer: Register CUDA DLLs shipped by NVIDIA pip wheels (site-packages/nvidia/*/bin) so GPU runs find a complete, matching cuBLAS/cuDNN set on Windows
#   - [2026-08-22] technical-writer: Accuracy pass for technical vocabulary - domain hotwords (domain_vocab.txt), post-transcription glossary (corrections.txt), anti-hallucination decoding, silence/repeat filtering, and large-v3-turbo as the new default model
