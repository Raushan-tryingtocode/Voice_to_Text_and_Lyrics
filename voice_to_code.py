"""
S.I.X  —  Enhanced Voice-to-Text
=================================
Improvements over the original:
  • VAD filter removes silence/noise for better accuracy
  • Word-level timestamps used to detect natural pauses
  • Punctuation post-processing: capitalisation, sentence endings, artefact cleanup
  • Song / lyrics mode: gaps between words split output into lines and verses
  • Optional large-v3 model for maximum accuracy (slower, auto-downloads)
  • Language can be forced to avoid mis-detection on short clips

Dependencies (same ecosystem as original):
  pip install faster-whisper sounddevice scipy numpy
"""

import re
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from faster_whisper import WhisperModel

# ── tunables ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
# fast + accurate; using "large-v3-turbo" for max quality and speed on CPU
DEFAULT_MODEL = "large-v3-turbo"
# Can also use "small", "medium", "large-v2" for faster performance with some accuracy tradeoff
# For the fastest performance (but lowest accuracy), use "tiny" or "tiny.en" (English-only)
# Faster models like "turbo" are very good for English but lacking when it comes to other languages, so "large-v3-turbo" is a good all-rounder for multilingual support without sacrificing too much speed.

# Song-mode gap thresholds (seconds)
LINE_GAP = 0.45             # pause this long  → new lyric line
VERSE_GAP = 1.80             # pause this long  → blank line between verses

# Transcription quality knobs
BEAM_SIZE = 5
PATIENCE = 1.0              # raise to ~1.5 for better accuracy at cost of speed
VAD_FILTER = True             # strip silence / background noise before decoding
# word-level timestamps (needed for song mode + pauses)
WORD_TS = True
# ──────────────────────────────────────────────────────────────────────────────


# ─── model ────────────────────────────────────────────────────────────────────

def load_model(model_size: str = DEFAULT_MODEL) -> WhisperModel:
    """Load (and cache) the Whisper model. Downloads on first run."""
    print(f"\n[*] Loading Whisper model — {model_size}")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("[+] Model ready.\n")
    return model


# ─── audio capture ────────────────────────────────────────────────────────────

def record_audio(duration: float | None = None) -> np.ndarray:
    """Record from microphone. Press Enter to stop when duration is None."""
    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata, _frames, _time, _status):
        if not stop_event.is_set():
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    )

    if duration:
        print(f"[*] Recording for {duration:.1f} s… speak now!")
        with stream:
            sd.sleep(int(duration * 1000))
    else:
        print("[*] Recording… press ENTER to stop.")
        with stream:
            input()
            stop_event.set()

    if not frames:
        return np.array([], dtype="float32")
    return np.concatenate(frames, axis=0).flatten()


def _array_to_wav(audio: np.ndarray) -> str:
    """Write float32 audio to a temp WAV and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    write_wav(path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return path


# ─── punctuation post-processing ──────────────────────────────────────────────

# Artefacts whisper sometimes produces at the start/end of a clip.
_JUNK_PATTERNS = re.compile(
    r"^\s*(you|thank you|thanks for watching|subscribe|\.+)\s*$",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """
    Light punctuation / capitalisation pass.
    1. Strip leading/trailing whitespace.
    2. Capitalise the first character of every sentence.
    3. Ensure the text ends with sentence-ending punctuation.
    4. Remove known whisper hallucination artefacts on very short clips.
    """
    text = text.strip()
    if not text:
        return text

    # Remove single-word hallucination artefacts
    if _JUNK_PATTERNS.match(text):
        return ""

    # Split on sentence-ending punctuation and capitalise each sentence start
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        # Capitalise first real character
        s = s[0].upper() + s[1:]
        cleaned.append(s)
    text = " ".join(cleaned)

    # Ensure terminal punctuation
    if text and text[-1] not in ".!?,;:":
        text += "."

    # Tidy up repeated punctuation produced by hallucinations
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


# ─── core transcription helpers ───────────────────────────────────────────────

def _run_transcription(
    model: WhisperModel,
    audio_path: str,
    language: str | None = None,
    song_mode: bool = False,
) -> tuple[list, object]:
    """
    Run faster-whisper and return (segments_list, info).
    Word timestamps are always requested so we can use pause data.
    """
    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=BEAM_SIZE,
        patience=PATIENCE,
        vad_filter=VAD_FILTER,
        word_timestamps=WORD_TS,
        language=language,       # None → auto-detect
    )
    # Materialise the generator (segments are lazy)
    segments = list(segments_iter)
    return segments, info


# ─── speech transcription ─────────────────────────────────────────────────────

def transcribe_audio(
    model: WhisperModel,
    audio: np.ndarray,
    language: str | None = None,
) -> str:
    """Transcribe a recorded NumPy array as normal speech."""
    path = _array_to_wav(audio)
    try:
        segments, info = _run_transcription(model, path, language=language)
    finally:
        Path(path).unlink(missing_ok=True)

    _print_lang_info(info)
    raw = " ".join(seg.text for seg in segments)
    return _clean_text(raw)


def transcribe_file(
    model: WhisperModel,
    file_path: str,
    language: str | None = None,
    song_mode: bool = False,
) -> str:
    """Transcribe an existing audio file (speech or song)."""
    path = Path(file_path)
    if not path.exists():
        print(f"[!] File not found: {file_path}")
        return ""

    print(f"[*] Transcribing: {path.name}")
    segments, info = _run_transcription(
        model, str(path), language=language, song_mode=song_mode
    )
    _print_lang_info(info)

    if song_mode:
        return _format_lyrics(segments)
    raw = " ".join(seg.text for seg in segments)
    return _clean_text(raw)


def _print_lang_info(info) -> None:
    print(
        f"[i] Language: {info.language}  "
        f"({info.language_probability:.0%} confidence)"
    )


# ─── song / lyrics formatter ──────────────────────────────────────────────────

def _format_lyrics(segments: list) -> str:
    """
    Convert whisper segments (with word timestamps) into neat lyric lines.

    Algorithm:
      • Iterate over every word across all segments.
      • Gap ≥ VERSE_GAP  → blank separator line (new verse)
      • Gap ≥ LINE_GAP   → start a new lyric line
      • Each lyric line is capitalised and ends with the correct punctuation.
    """
    # Flatten all words from all segments
    words: list[dict] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({
                    "word":  w.word.strip(),
                    "start": w.start,
                    "end":   w.end,
                })

    if not words:
        # Fallback: no word timestamps available — use segment-level gaps
        return _format_lyrics_by_segment(segments)

    lines: list[str] = []
    current_line_words: list[str] = []
    prev_end: float = words[0]["end"] if words else 0.0

    def _flush_line(words_in_line: list[str]) -> str:
        if not words_in_line:
            return ""
        line = " ".join(words_in_line).strip()
        # Clean stray punctuation at the start
        line = re.sub(r"^[,;:\s]+", "", line)
        if not line:
            return ""
        # Capitalise first word
        line = line[0].upper() + line[1:]
        # Ensure the line ends with punctuation
        if line[-1] not in ".!?,;:":
            line += "."
        # Clean up double spaces
        return re.sub(r"\s{2,}", " ", line)

    for w in words:
        word_text = w["word"]
        gap = w["start"] - prev_end

        if gap >= VERSE_GAP:
            # Flush current line, then add blank verse separator
            flushed = _flush_line(current_line_words)
            if flushed:
                lines.append(flushed)
            lines.append("")   # blank line = verse break
            current_line_words = [word_text]
        elif gap >= LINE_GAP:
            # Flush current line, start a new one
            flushed = _flush_line(current_line_words)
            if flushed:
                lines.append(flushed)
            current_line_words = [word_text]
        else:
            current_line_words.append(word_text)

        prev_end = w["end"]

    # Flush the last line
    flushed = _flush_line(current_line_words)
    if flushed:
        lines.append(flushed)

    # Remove leading/trailing blank lines, collapse multiple blanks to one
    output_lines: list[str] = []
    prev_blank = False
    for ln in lines:
        if ln == "":
            if not prev_blank:
                output_lines.append("")
            prev_blank = True
        else:
            output_lines.append(ln)
            prev_blank = False

    return "\n".join(output_lines).strip()


def _format_lyrics_by_segment(segments: list) -> str:
    """Fallback lyrics formatter that works at segment granularity."""
    lines: list[str] = []
    prev_end: float | None = None

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?,;:":
            text += "."

        if prev_end is not None:
            gap = seg.start - prev_end
            if gap >= VERSE_GAP:
                lines.append("")
        lines.append(text)
        prev_end = seg.end

    return "\n".join(lines).strip()


# ─── interactive menu ─────────────────────────────────────────────────────────

def interactive_mode(model: WhisperModel) -> None:
    print("=" * 58)
    print("   S.I.X  —  Enhanced Voice-to-Text")
    print("=" * 58)
    print()
    print("  [1]  Record speech (press Enter to stop)")
    print("  [2]  Record speech for a set duration")
    print("  [3]  Transcribe an audio file (speech)")
    print("  [4]  Transcribe an audio file as SONG / LYRICS")
    print("  [5]  Toggle language lock (currently: auto-detect)")
    print("  [q]  Quit")
    print()

    forced_lang: str | None = None

    while True:
        choice = input(">> ").strip().lower()

        # ── quit ──────────────────────────────────────────────────────────────
        if choice == "q":
            print("[*] Goodbye!")
            break

        # ── record (open-ended) ───────────────────────────────────────────────
        elif choice == "1":
            audio = record_audio()
            if audio.size == 0:
                print("[!] No audio captured.\n")
                continue
            text = transcribe_audio(model, audio, language=forced_lang)
            _print_result(text)

        # ── record (timed) ────────────────────────────────────────────────────
        elif choice == "2":
            try:
                dur = float(input("   Duration (seconds): "))
            except ValueError:
                print("[!] Invalid number.\n")
                continue
            audio = record_audio(duration=dur)
            if audio.size == 0:
                print("[!] No audio captured.\n")
                continue
            text = transcribe_audio(model, audio, language=forced_lang)
            _print_result(text)

        # ── file: speech ──────────────────────────────────────────────────────
        elif choice == "3":
            file_path = input("   File path: ").strip().strip('"')
            text = transcribe_file(model, file_path, language=forced_lang)
            if text:
                _print_result(text)

        # ── file: song / lyrics ───────────────────────────────────────────────
        elif choice == "4":
            file_path = input("   File path: ").strip().strip('"')
            text = transcribe_file(
                model, file_path, language=forced_lang, song_mode=True
            )
            if text:
                _print_lyrics(text)

        # ── language lock ─────────────────────────────────────────────────────
        elif choice == "5":
            if forced_lang:
                print(f"   Language lock removed (was: {forced_lang}).")
                forced_lang = None
            else:
                forced_lang = input(
                    "   Enter language code (e.g. en, es, fr) or leave blank to cancel: "
                ).strip().lower() or None
                if forced_lang:
                    print(f"   Language locked to: {forced_lang}")
                else:
                    print("   Language lock not set.")
            print()

        else:
            print("[!] Unknown option. Enter 1–5 or q.\n")


# ─── output helpers ───────────────────────────────────────────────────────────

def _print_result(text: str) -> None:
    if not text:
        print("[!] Nothing transcribed.\n")
        return
    bar = "─" * 52
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}\n")


def _print_lyrics(text: str) -> None:
    if not text:
        print("[!] Nothing transcribed.\n")
        return
    bar = "═" * 52
    print(f"\n{bar}")
    for line in text.splitlines():
        print(f"  {line}" if line else "")
    print(f"{bar}\n")


# ─── CLI entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI usage:
      python s_i_x_vtt.py                          → interactive menu
      python s_i_x_vtt.py <audio_file>             → transcribe file (speech)
      python s_i_x_vtt.py <audio_file> --song      → transcribe file (lyrics)
      python s_i_x_vtt.py <audio_file> large-v3    → use a specific model size
    """
    args = sys.argv[1:]

    if not args:
        model = load_model()
        interactive_mode(model)
        return

    file_path = args[0]
    song_mode = "--song" in args
    model_size = next(
        (a for a in args[1:] if not a.startswith("--")), DEFAULT_MODEL
    )

    model = load_model(model_size)
    text = transcribe_file(model, file_path, song_mode=song_mode)

    if text:
        if song_mode:
            _print_lyrics(text)
        else:
            print(f"\n{text}\n")


if __name__ == "__main__":
    main()
