# 🎙️ Whisper Transcriber

Turn your recorded meetings and call videos into clean, searchable **text transcripts** and **subtitles**, fully offline on your own machine.

No cloud upload. No paid API. Just run one command and get a transcript you can feed to any LLM (like Microsoft 365 Copilot or ChatGPT).

---

## 🤔 The problem this solves

We all record meetings and calls (OBS, Teams, Zoom, etc.) so we can go back to them later. But when you actually need something from that recording, the real pain starts:

- You have to **re-watch the whole video** just to find one discussion or one task someone assigned to you.
- Scrubbing back and forth to find "what was I supposed to do?" wastes a lot of time.
- Long recordings are impossible to search. There is no Ctrl+F for a video.
- You cannot easily give a video to an AI assistant and ask questions about it.

**This tool fixes that.** It converts your recording into a plain text transcript. Once it is text, everything becomes easy.

---

## 💡 How I actually use it (real use case)

1. I record my calls / important meetings using **Bandicam Screen Recorder**.
2. Later, if I need to go back and find out the discussion or the tasks that were given to me, instead of watching the full video again, I run this tool on the recording.
3. It generates a **transcript** of the whole meeting.
4. I paste that transcript as **context** into an LLM like **Microsoft 365 Copilot** or **ChatGPT**.
5. Now I can:
   - Ask it to **summarize** the meeting.
   - Ask it **"what tasks were assigned to me?"**
   - Ask **any question** related to that meeting and get an instant answer.

So instead of spending 40 minutes re-watching a call, I get my answers in a couple of minutes. 🚀

---

## ✨ What is inside

This project has two small, focused scripts:

| Script | What it does |
| --- | --- |
| [`transcribe.py`](transcribe.py) | Transcribes a video/audio file into a `.txt` transcript and a `.srt` subtitle file using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (runs locally). |
| [`video_to_mp3.py`](video_to_mp3.py) | Optional helper that extracts the audio from a video into an `.mp3` using `ffmpeg` (smaller file, faster to process). |

You usually just need `transcribe.py`. The MP3 converter is a handy extra.

Setup helpers are also included:

| File | What it does |
| --- | --- |
| [`requirements.txt`](requirements.txt) | List of Python packages needed by the project. |
| [`setup.ps1`](setup.ps1) | One-command setup for Windows (creates `.venv` and installs everything). |
| [`setup.sh`](setup.sh) | Same one-command setup for Linux / macOS. |

And the plain-text files that control **transcription accuracy** (see [Getting technical words right](#-getting-technical-words-right)):

| File | What it does |
| --- | --- |
| [`domain_vocab.txt`](domain_vocab.txt) | **Default profile.** GenAI / LLM jargon (LangChain, NL2SQL, Azure AI Search...) fed to the model **while** it listens. |
| [`vocab_work.txt`](vocab_work.txt) | **Work profile.** Power BI, Azure DevOps, Kusto, IcM and scrum vocabulary for scrum calls and KT sessions. |
| [`corrections.txt`](corrections.txt) | `wrong => right` rules applied **after** transcription, e.g. `rack approach => RAG approach`. |
| `*.local.txt` | Optional, **git-ignored**. Your private additions - colleague names, internal project names. Loaded automatically. |

---

## 🛠️ Setup

### 1. Requirements

- **Python 3.9+**
- **ffmpeg** on your PATH (needed for reading media files and for `video_to_mp3.py`)

Check ffmpeg is installed:

```powershell
ffmpeg -version
```

If it is not installed, get it from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH.

### 2. Automatic setup (recommended)

A setup script does everything for you: it creates a virtual environment in `.venv` and installs all Python dependencies from `requirements.txt`.

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

**Linux / macOS:**

```bash
chmod +x setup.sh
./setup.sh
```

After it finishes, activate the virtual environment in any new terminal:

```powershell
# Windows
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

### 3. Manual setup (if you prefer)

If you would rather do it by hand:

```powershell
# create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to use

### Transcribe a recording (most common)

Just point it at your video or audio file:

```powershell
python transcribe.py "C:\path\to\meeting.mkv"
```

You get two files next to your recording:

- `meeting.txt` -> plain transcript with timestamps (paste this into Copilot/ChatGPT)
- `meeting.srt` -> subtitle file you can load in any video player

### Transcribe a whole folder at once

Point it at a **folder** and it will transcribe every video/audio inside it, one by one. The model loads only once, so it is efficient.

```powershell
python transcribe.py "C:\path\to\recordings_folder"

# Also look inside sub-folders:
python transcribe.py "C:\path\to\recordings_folder" --recursive
```

Each file gets its own `.txt` and `.srt` next to it. Supported types include `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.mp3`, `.wav`, `.m4a` and more.

### Use your GPU for faster processing (optional)

By default the script **auto-detects** an NVIDIA GPU and uses it if available, otherwise it falls back to the CPU. You can also force it:

```powershell
python transcribe.py "C:\path\to\meeting.mkv" --device cuda   # force GPU
python transcribe.py "C:\path\to\meeting.mkv" --device cpu    # force CPU
python transcribe.py "C:\path\to\meeting.mkv" --device auto   # default (auto pick)
```

On GPU it automatically uses `float16` (fast); on CPU it uses `int8`. If the GPU cannot start (missing CUDA libraries, out of memory, etc.), the script prints a warning and quietly continues on the CPU.

> **One-time GPU setup:** an NVIDIA card alone is not enough - `faster-whisper` also needs the CUDA runtime libraries. If you have an NVIDIA GPU and want to use it, install them once inside your virtual environment:
>
> ```powershell
> pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
> ```
>
> A 6 GB card comfortably runs `tiny`, `base`, `small` and `medium`. If a big model runs out of memory, use a smaller `--model` or add `--compute-type int8_float16`.

### Faster vs more accurate

The model size decides the speed vs accuracy trade-off:

```powershell
# Faster, lighter (good for quick checks)
python transcribe.py "C:\path\to\meeting.mkv" --model tiny
python transcribe.py "C:\path\to\meeting.mkv" --model small

# Slower, more accurate (good for important meetings)
python transcribe.py "C:\path\to\meeting.mkv" --model medium
python transcribe.py "C:\path\to\meeting.mkv" --model large-v3
```

| Model | Speed | Accuracy | Notes |
| --- | --- | --- | --- |
| `tiny` | Fastest | Lowest | |
| `base` | Fast | Ok | |
| `small` | Balanced | Good | |
| `medium` | Slower | Better | |
| `large-v3-turbo` | About `medium` speed | Good, but weaker on rare words | Distilled 4-layer decoder. Cannot translate. |
| `large-v3` **(default)** | Slowest | **Best** | Use this. Also the only one that handles `--task translate`. |

> **Why `large-v3` and not `large-v3-turbo`?** Turbo is `large-v3` with the decoder distilled from 32 layers down to 4. That decoder is exactly what produces rare words - acronyms, product names, people's names. For technical calls the accuracy loss shows up precisely where you care most, so turbo is only worth it if `large-v3` is too slow on your machine.
>
> On an NVIDIA GPU with 6 GB, `large-v3` runs at a comfortable speed in `float16` (about 4.7 GB). If it ever runs out of memory the script automatically retries with `int8_float16`, which halves the memory need with almost no accuracy loss.

---

## 🎯 Getting technical words right

This is the part that matters if your calls are full of **GenAI, Azure, Power BI or scrum vocabulary**. Out of the box Whisper turns `RAG` into `rack`, `LangChain` into `slam chain`, `Power BI` into `Power Via`, `Kusto` into `custo` and `IcM` into `I see him`.

Three things fix that, and all three are already switched on:

### 1. Vocabulary files - tell the model what to expect

Every term in these files is fed to the model as a **hotword** on every 30-second window of audio, so it is actively looking out for your jargon while it decodes.

Two profiles ship with the tool. **Pick the one that matches the recording** - they cannot both fit in the hotword prompt:

| Profile | Use it for | Command |
| --- | --- | --- |
| [`domain_vocab.txt`](domain_vocab.txt) *(default)* | Interviews, GenAI / LLM discussions | `python transcribe.py "call.mkv"` |
| [`vocab_work.txt`](vocab_work.txt) | Scrum calls, KT sessions, Power BI / ADO / Kusto work | `python transcribe.py "call.mkv" --vocab vocab_work.txt` |

> ⚠️ Whisper only accepts **223 tokens** of hotwords, which works out to roughly **55-60 terms**. The script measures this with the model's own tokenizer and prints exactly how many fit, for example:
>
> ```
> Domain vocabulary: using the first 59 of 102 term(s) from vocab_work.local.txt, vocab_work.txt
>   (43 term(s) did not fit Whisper's hotword limit...)
> ```
>
> Terms are taken **in file order**, so put the ones you care about most at the **top**. Anything that does not fit still gets handled by the corrections file.

### 2. `corrections.txt` - fix whatever still slips through

This runs **after** transcription and has no size limit. One rule per line:

```text
rack approach => RAG approach
slam chain => LangChain
power via => Power BI
custo cluster => Kusto cluster
m2tr => MTTR
```

Matching is case-insensitive and respects word boundaries, so `rack based` will never match inside `racked`. Longer rules are applied first. Both the `.txt` and the `.srt` get the corrections.

> Only add a phrase here if the wrong form is unlikely to appear as normal English. Single common words (`rack`, `our`, `meter`) are too risky on their own - always write them as part of a longer phrase. The file has a commented-out `RISKY` section at the bottom for exactly these cases.

### 3. Silence and repetition are dropped automatically

When Whisper is fed silence or crosstalk it starts inventing text - long runs of `um um um`, the same sentence repeated three times, or random unrelated sentences. The script now:

- turns off `condition_on_previous_text`, which is what causes those repeat loops in the first place,
- uses a temperature fallback plus compression-ratio and log-probability checks to reject low-confidence guesses,
- drops filler-only segments and exact repeats of the previous line.

Pass `--keep-noise` if you want them back.

### 🔒 Private terms: colleague names and internal project names

Person names and internal project names should **never** be committed to a shared repository. So for any vocabulary or corrections file `X.txt`, the script also loads a sibling **`X.local.txt`** if it exists - and `*.local.txt` is git-ignored.

```
vocab_work.txt         <- shared: Power BI, Azure DevOps, Kusto, sprint...   (committed)
vocab_work.local.txt   <- private: colleague names, internal report names    (never committed)
corrections.txt        <- shared: power via => Power BI                      (committed)
corrections.local.txt  <- private: sangeera => Sangeeta                      (never committed)
```

The `.local.txt` file is loaded **first**, so your private terms get priority in the hotword budget. This is what makes scrum call transcripts spell teammate names correctly instead of inventing new ones.

### Using your own files

```powershell
# a different profile (its .local.txt sibling is picked up automatically)
python transcribe.py "meeting.mkv" --vocab vocab_work.txt

# combine several files - order decides hotword priority
python transcribe.py "meeting.mkv" --vocab vocab_work.txt domain_vocab.txt

# turn them off completely (pass the flags with no value)
python transcribe.py "meeting.mkv" --vocab --corrections
```

### Recommended workflow

1. Transcribe once with the right profile.
2. Skim the `.txt` and note every technical word or name that came out wrong.
3. Add the important ones to the **top** of the vocabulary file, and add the rest as `wrong => right` lines in the corrections file. Names and internal project names go in the `.local.txt` versions.
4. Next recording is already better - and the glossary keeps paying off on every future call.

### Hindi / mixed language meetings

If your meeting is in Hindi or a Hindi + English mix and you want the transcript in **English**:

```powershell
python transcribe.py "C:\path\to\meeting.mkv" --task translate --model medium
```

You can also force a language instead of auto-detecting:

```powershell
python transcribe.py "C:\path\to\meeting.mkv" --language en
python transcribe.py "C:\path\to\meeting.mkv" --language hi
```

### Convert video to MP3 first (optional)

If you want a smaller audio-only file (for example to archive or to speed things up):

```powershell
python video_to_mp3.py "C:\path\to\meeting.mkv"
python video_to_mp3.py "C:\path\to\meeting.mkv" --bitrate 320k
python video_to_mp3.py "C:\path\to\meeting.mkv" --output "C:\path\to\audio.mp3"
```

Then you can run `transcribe.py` on the `.mp3` file.

---

## 📦 Where do the models go?

The first time you use a model, it is downloaded from Hugging Face and cached locally. On Windows the default location is:

```
C:\Users\<your-username>\.cache\huggingface\hub
```

After the first download it loads from this cache and works **fully offline**.

---

## 🔒 Privacy

Everything runs **locally on your machine**. Your recordings and transcripts never leave your computer. This makes it safe for internal meetings and sensitive discussions.

---

## 💬 Typical workflow at a glance

```
Record meeting (OBS)  ->  transcribe.py  ->  meeting.txt  ->  paste into M365 Copilot / ChatGPT
                                                              ->  "Summarize this meeting"
                                                              ->  "What tasks were given to me?"
                                                              ->  ask any question about the call
```

---

## 📝 Tips

- The default `large-v3` is the most accurate option. Drop to `small` only when you just need a rough idea of what was said, or to `large-v3-turbo` if your machine is too slow for `large-v3`.
- The `.txt` file already has timestamps, so you can jump to that point in the video if needed.
- If a technical word or a name is misheard, do not just accept it - add it to the vocabulary or corrections file so it is fixed on every future recording too.
- `translate` mode is great when you only need the meaning in English and not the exact original words. It needs `large-v3`; turbo models cannot translate.
- Audio quality beats model size. A headset mic, and recording each participant's audio cleanly, helps more than any model change.

---

Happy transcribing. Spend less time re-watching, more time getting things done. ✅
