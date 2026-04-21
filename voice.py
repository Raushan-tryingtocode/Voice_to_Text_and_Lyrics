import re
import sys
import tempfile
import threading
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from faster_whisper import WhisperModel

# --- Config & Tunables ---
SAMPLE_RATE = 16000
CHANNELS = 1
DEFAULT_MODEL = "large-v3-turbo"

# Lyrics mode timing (seconds)
LINE_GAP = 0.45   
VERSE_GAP = 1.80  

BEAM_SIZE = 5
PATIENCE = 1.0  
VAD_FILTER = True
WORD_TS = True

# --- Engine Logic ---

def load_model(model_size: str = DEFAULT_MODEL) -> WhisperModel:
    print(f"[*] Loading {model_size}...")
    # cpu/int8 is best for local dev machines without a beefy GPU
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model

def record_audio(duration: float | None = None) -> np.ndarray:
    frames = []
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
        print(f"[*] Recording for {duration}s...")
        with stream:
            sd.sleep(int(duration * 1000))
    else:
        print("[*] Recording... [Press Enter to Stop]")
        with stream:
            input()
            stop_event.set()

    return np.concatenate(frames, axis=0).flatten() if frames else np.array([], dtype="float32")

def _array_to_wav(audio: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    write_wav(path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return path

# --- Text Cleaning ---

# Filter out common Whisper "hallucinations" on silence/short clips
_JUNK_PATTERNS = re.compile(
    r"^\s*(you|thank you|thanks for watching|subscribe|\.+)\s*$",
    re.IGNORECASE,
)

def _clean_text(text: str) -> str:
    text = text.strip()
    if not text or _JUNK_PATTERNS.match(text):
        return ""

    # Fix casing for sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = [s[0].upper() + s[1:] for s in sentences if s.strip()]
    text = " ".join(cleaned)

    if text and text[-1] not in ".!?,;:":
        text += "."

    # Cleanup messy punctuation hallucinations
    text = re.sub(r"([.!?]){2,}", r"\1", text)
    return re.sub(r"\s{2,}", " ", text)

# --- Transcription Core ---

def _run_transcription(model, audio_path, language=None):
    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=BEAM_SIZE,
        patience=PATIENCE,
        vad_filter=VAD_FILTER,
        word_timestamps=WORD_TS,
        language=language
    )
    return list(segments_iter), info

def transcribe_audio(model, audio, language=None):
    path = _array_to_wav(audio)
    try:
        segments, info = _run_transcription(model, path, language=language)
    finally:
        Path(path).unlink(missing_ok=True)
    
    print(f"[i] Detected: {info.language} ({info.language_probability:.2f})")
    return _clean_text(" ".join(seg.text for seg in segments))

def transcribe_file(model, file_path, language=None, song_mode=False):
    if not Path(file_path).exists():
        return ""

    segments, info = _run_transcription(model, file_path, language=language)
    if song_mode:
        return _format_lyrics(segments)
    return _clean_text(" ".join(seg.text for seg in segments))

# --- Formatting ---

def _format_lyrics(segments):
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

    if not words: return ""

    lines, current_line = [], []
    prev_end = words[0]["end"]

    for w in words:
        gap = w["start"] - prev_end
        if gap >= VERSE_GAP or gap >= LINE_GAP:
            if current_line:
                line_text = " ".join(current_line).strip()
                line_text = line_text[0].upper() + line_text[1:] + "."
                lines.append(line_text)
                if gap >= VERSE_GAP: lines.append("")
            current_line = [w["word"]]
        else:
            current_line.append(w["word"])
        prev_end = w["end"]

    return "\n".join(lines).strip()

# --- Entry ---

def main():
    args = sys.argv[1:]
    model = load_model()

    if not args:
        print("-- Interactive Mode --\n1: Mic\n2: File\n3: Lyrics\nq: Quit")
        while True:
            cmd = input("> ").lower()
            if cmd == 'q': break
            if cmd == '1':
                print(transcribe_audio(model, record_audio()))
            elif cmd == '2':
                p = input("Path: ")
                print(transcribe_file(model, p))
            elif cmd == '3':
                p = input("Path: ")
                print(transcribe_file(model, p, song_mode=True))
    else:
        # Simple CLI: python s_i_x.py <file> [--song]
        song = "--song" in args
        print(transcribe_file(model, args[0], song_mode=song))

if __name__ == "__main__":
    main()
