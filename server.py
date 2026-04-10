"""
S.I.X  —  Flask API Server
===========================
Wraps the voice-to-text engine in a tiny HTTP API so the browser
frontend can POST audio blobs and receive transcription results.

Install extra dependency:
    pip install flask flask-cors

Run:
    python server.py
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# Import from the voice-to-text module (must be in the same folder)
from voice_to_code import load_model, transcribe_file

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)                          # allows the HTML page to call the API

# Load the model once at startup — no per-request loading delay
print("[*] Initialising model…")
MODEL = load_model()               # change to "large-v3" for higher accuracy
print("[+] Server ready.\n")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac", ".mp4"}


def _save_upload(file) -> str | None:
    """Save an uploaded file to a temp path and return its path."""
    suffix = Path(file.filename).suffix.lower() if file.filename else ".webm"
    if suffix not in ALLOWED_EXTENSIONS:
        return None
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    file.save(path)
    return path


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend HTML directly."""
    return app.send_static_file("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    POST /transcribe
    Body: multipart/form-data  →  field 'audio' containing the audio file.
    Returns: { "text": "…", "language": "en", "confidence": 0.99 }
    """
    if "audio" not in request.files:
        return jsonify(error="No audio field in request."), 400

    path = _save_upload(request.files["audio"])
    if path is None:
        return jsonify(error="Unsupported file type."), 415

    try:
        # Re-implement with info to expose language data
        from faster_whisper import WhisperModel
        segments, info = MODEL.transcribe(
            path,
            beam_size=5,
            patience=1.0,
            vad_filter=True,
            word_timestamps=True,
        )
        segments = list(segments)
        import re

        raw = " ".join(seg.text for seg in segments).strip()

        # Light punctuation pass
        sentences = re.split(r"(?<=[.!?])\s+", raw)
        cleaned = []
        for s in sentences:
            s = s.strip()
            if s:
                s = s[0].upper() + s[1:]
                cleaned.append(s)
        text = " ".join(cleaned)
        if text and text[-1] not in ".!?,;:":
            text += "."
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        return jsonify(
            text=text,
            language=info.language,
            confidence=round(info.language_probability, 3),
        )
    finally:
        Path(path).unlink(missing_ok=True)


@app.route("/transcribe-song", methods=["POST"])
def transcribe_song():
    """
    POST /transcribe-song
    Same as /transcribe but returns lyrics-formatted text.
    Returns: { "text": "verse\\nlines\\n\\nnext verse", "language": "en", "confidence": 0.99 }
    """
    if "audio" not in request.files:
        return jsonify(error="No audio field in request."), 400

    path = _save_upload(request.files["audio"])
    if path is None:
        return jsonify(error="Unsupported file type."), 415

    try:
        text = transcribe_file(MODEL, path, song_mode=True)
        # transcribe_file already prints lang info; get info separately for JSON
        segments_iter, info = MODEL.transcribe(path, beam_size=1)
        list(segments_iter)        # consume to get info
        return jsonify(
            text=text,
            language=info.language,
            confidence=round(info.language_probability, 3),
        )
    finally:
        Path(path).unlink(missing_ok=True)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
