'''
Transcribe a video/audio file (or a whole folder of them) to text transcripts
and SRT subtitle files using faster-whisper (local, offline speech recognition).

Usage:
    # Basic run - just pass the video/audio path (auto-detect language & default model):
    python transcribe.py "C:\\path\\to\\video.mkv"

    # Transcribe EVERY video/audio inside a folder, one by one:
    python transcribe.py "C:\\path\\to\\recordings_folder"
    python transcribe.py "C:\\Users\\amark\\Videos\\Interview Recordings\\More Videos" --model large-v3 --device cuda --language en
    python transcribe.py "C:\\path\\to\\recordings_folder" --recursive

    # READY-TO-USE - pick the vocabulary profile that matches the recording:
    # INTERVIEW / GenAI call (RAG, LangChain, NL2SQL, Azure AI Search...):
    python transcribe.py "path of video" --model large-v3 --device cuda --language en --task translate --vocab domain_vocab.txt
    # WORK / scrum call (Power BI, Azure DevOps, Kusto, IcM + colleague names):
    python transcribe.py "path of video" --model large-v3 --device cuda --language en --task translate --vocab vocab_work.txt

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
    python transcribe.py "C:\\path\\to\\video.mkv" --task translate

Hindi / mixed-language calls - what actually works:
    --language is the language SPOKEN in the audio, not the language you want out.
    --task translate is what turns non-English speech into English text.

    Use just  --task translate  and let the language auto-detect. When --language
    is not forced the script re-detects it on every window, so a call that opens
    in English and switches to Hindi still comes out as readable English.

    Do NOT combine "--language en --task translate". Whisper cannot translate
    English into English, so the translation is cancelled and the Hindi parts get
    forced into English sounds instead. Verified on a real recording:

        --language en --task translate -> "agar my good luck may be who told me
                                           that he's our application automate
                                           karna hake jaha pecha ke unko..."
        --task translate               -> "they are collecting and putting it into
                                           the excel, then they are customizing
                                           and doing some calculation on top..."

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
    large-v3-turbo -> roughly medium speed, but its decoder is distilled down to
                      4 layers, so rare technical words suffer. Cannot translate.
    large-v3 -> DEFAULT. Slowest, best accuracy, and the one that handles
                --task translate. Needs about 4.7 GB of GPU memory as float16.

Getting technical words right (GenAI / Azure / Power BI / scrum jargon):
    Two kinds of file drive this, and both are picked up automatically:

        <vocab>.txt       -> terms fed to the model as "hotwords" while it decodes,
                             so it expects to hear LangChain, NL2SQL, Power BI...
                             Only ~220 tokens fit, so put important terms FIRST.
        corrections.txt   -> "wrong => right" rules applied to the finished text,
                             e.g. "rack approach => RAG approach". No size limit.

    Two vocabulary profiles ship with the tool - pick the one that matches the
    recording, because they cannot both fit in the hotword prompt:

        domain_vocab.txt  -> GenAI / LLM / interview vocabulary (the default)
        vocab_work.txt    -> Power BI, Azure DevOps, Kusto, IcM, scrum vocabulary

        python transcribe.py "scrum call.mkv" --vocab vocab_work.txt --language en

    For any file X.txt a sibling X.local.txt is loaded FIRST when it exists.
    Those .local.txt files are git-ignored, so colleague names and internal
    project names go there and never end up in a shared repository.

    To turn either off:
        python transcribe.py "video.mkv" --vocab --corrections

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

# Most accurate option, and the only large model that can also translate.
DEFAULT_MODEL = "large-v3"

# Fallback only - the real budget is read from the loaded model's tokenizer.
HOTWORD_CHAR_BUDGET = 700

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


def resolve_source_files(paths: list[str]) -> list[Path]:
    """Expand each configured file into the list that is actually read.

    For "vocab_work.txt" a sibling "vocab_work.local.txt" is read first when it
    exists. Those .local files are git-ignored, which is where private things
    like colleague names belong so they never reach a shared repository.
    """
    resolved: list[Path] = []
    for raw in paths:
        if not raw:
            continue
        path = Path(raw)
        local = path.with_name(f"{path.stem}.local{path.suffix}")
        for candidate in (local, path):
            if candidate.exists() and candidate not in resolved:
                resolved.append(candidate)
    return resolved


def load_vocabulary(vocab_paths: list[Path]) -> list[str]:
    """Read the domain terms, in file order, dropping blanks and duplicates."""
    terms: list[str] = []
    seen: set[str] = set()
    for vocab_path in vocab_paths:
        for raw in vocab_path.read_text(encoding="utf-8").splitlines():
            term = raw.split("#", 1)[0].strip()
            if term and term.lower() not in seen:
                seen.add(term.lower())
                terms.append(term)
    return terms


def build_hotwords(terms: list[str], model, source_names: str) -> str:
    """Trim the term list to what actually fits in Whisper's hotword prompt.

    faster-whisper silently truncates anything past half the context window, and
    proper nouns tokenize badly, so the cut-off is measured with the model's own
    tokenizer rather than estimated. Terms are kept in order - most important first.
    """
    if not terms:
        return ""

    tokenizer = getattr(model, "hf_tokenizer", None)
    token_budget = getattr(model, "max_length", 448) // 2 - 1

    kept: list[str] = []
    for index, term in enumerate(terms):
        candidate = kept + [term]
        if tokenizer is not None:
            size = len(tokenizer.encode(" " + ", ".join(candidate), add_special_tokens=False).ids)
            over = size > token_budget
        else:
            over = len(", ".join(candidate)) > HOTWORD_CHAR_BUDGET
        if over:
            print(
                f"Domain vocabulary: using the first {len(kept)} of {len(terms)} term(s) "
                f"from {source_names}"
            )
            print(
                f"  ({len(terms) - index} term(s) did not fit Whisper's hotword limit. "
                "Move important ones higher up, or put them in the corrections file.)"
            )
            break
        kept = candidate
    else:
        print(f"Domain vocabulary: using all {len(kept)} term(s) from {source_names}")

    return ", ".join(kept)


def load_corrections(correction_paths: list[Path]) -> list[tuple[re.Pattern, str]]:
    """Read the 'wrong => right' glossary used to clean up the decoded text."""
    pairs: list[tuple[str, str]] = []
    for corrections_path in correction_paths:
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
        names = ", ".join(p.name for p in correction_paths)
        print(f"Glossary: loaded {len(rules)} correction(s) from {names}")
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
    if args.language is None:
        # Re-detect per window, otherwise a call that opens in English and later
        # switches to Hindi keeps the English token and comes out as gibberish.
        options["multilingual"] = True
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
        f"(bigger = more accurate but slower). Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force the language SPOKEN in the audio (e.g. 'hi'). Default: auto-detect, "
        "re-checked on every window so mixed Hindi/English calls stay readable. Do not "
        "combine 'en' with --task translate; that cancels the translation.",
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
        nargs="*",
        default=[str(DEFAULT_VOCAB_FILE)],
        help="One or more files of domain terms (one per line) used to bias the model "
        f"towards your jargon. Default: {DEFAULT_VOCAB_FILE.name} next to this script. "
        "A matching *.local.txt sibling is picked up automatically. Pass --vocab with "
        "no value to disable.",
    )
    parser.add_argument(
        "--corrections",
        nargs="*",
        default=[str(DEFAULT_CORRECTIONS_FILE)],
        help="One or more files of 'wrong => right' rules applied to the finished text. "
        f"Default: {DEFAULT_CORRECTIONS_FILE.name} next to this script. A matching "
        "*.local.txt sibling is picked up automatically. Pass --corrections with no "
        "value to disable.",
    )
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="Keep filler-only ('um um um') and repeated segments that the model invents "
        "over silence. Default: off, they are dropped.",
    )
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODEL
    elif args.task == "translate" and "turbo" in args.model:
        print(
            f"WARNING: '{args.model}' was not trained for translation. "
            f"Use --model {DEFAULT_MODEL} for Hindi/mixed audio.",
            file=sys.stderr,
        )

    if args.language == "en" and args.task == "translate":
        print(
            "WARNING: '--language en --task translate' cancels the translation - Whisper "
            "cannot translate English into English, so any Hindi in the call is forced "
            "into English sounds instead of being translated. Drop '--language en' and "
            "keep '--task translate'.",
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

    vocab_files = resolve_source_files(args.vocab)
    terms = load_vocabulary(vocab_files)
    corrections = load_corrections(resolve_source_files(args.corrections))

    device, compute_type = resolve_device_and_compute(args.device, args.compute_type)
    print(f"Loading model '{args.model}' on {device.upper()} (compute_type={compute_type})...")
    model, device, compute_type = load_whisper_model(args.model, args.device, args.compute_type)
    print(f"Using model '{args.model}' on {device.upper()} (compute_type={compute_type})")

    hotwords = build_hotwords(terms, model, ", ".join(p.name for p in vocab_files))

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
#   - [2026-08-22] technical-writer: Accuracy pass for technical vocabulary - domain hotwords (domain_vocab.txt), post-transcription glossary (corrections.txt), anti-hallucination decoding, silence/repeat filtering, and large-v3 as the new default model
#   - [2026-08-22] technical-writer: Added a work vocabulary profile (Power BI / Azure DevOps / Kusto / scrum), multi-file --vocab and --corrections, token-accurate hotword trimming, and git-ignored *.local.txt files for private names
#   - [2026-08-22] technical-writer: Per-window language re-detection for mixed Hindi/English calls, plus a warning that '--language en --task translate' cancels the translation
