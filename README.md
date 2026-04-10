# S.I.X — Voice to Text

> **Mini-Project** | B.Tech / BCA — Minor Project Submission

A local, offline voice-to-text transcription tool built with Python and a browser-based frontend. Powered by [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper), it supports **99 languages**, real-time waveform visualisation, and a dedicated **song/lyrics mode** that formats transcribed audio into neat verse-structured output. No API keys, no cloud services, no cost — everything runs on your own machine.

---

## License

Copyright (c) 2025 Your Name. Source available under the
[MIT + Commons Clause License](./LICENSE).  
Free to use and modify for personal and academic purposes.  
Commercial use requires explicit written permission from the author.

---

## Features

- 🎙️ **Live microphone recording** with real-time oscilloscope waveform
- 📁 **File upload & drag-and-drop** support (wav, mp3, ogg, m4a, flac, webm)
- 🌍 **99 languages** with automatic language detection
- 🎵 **Song / Lyrics mode** — formats output into lines and verses based on natural pauses
- ✏️ **Punctuation post-processing** — capitalisation, sentence endings, artefact cleanup
- 🔒 **Fully offline** — audio never leaves your machine
- ⚡ **VAD filter** — strips silence and background noise before decoding for better accuracy
- 🖥️ **Clean web UI** — Speech/Lyrics toggle, language badge, one-click copy

---

## Project Structure

```
s-i-x-voice-to-text/
│
├── s_i_x_vtt.py      # Core transcription engine (Faster-Whisper wrapper)
├── server.py         # Flask API server — bridges the frontend and engine
│
├── index.html        # Frontend — HTML structure
├── styles.css        # Frontend — all styling (CSS variables for easy theming)
├── app.js            # Frontend — all JavaScript (Web Audio API, fetch, drag-drop)
│
└── README.md
```

---

## Tech Stack

| Layer               | Technology                                                                   | Cost                |
| ------------------- | ---------------------------------------------------------------------------- | ------------------- |
| Transcription model | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI Whisper) | Free / Open-source  |
| Web framework       | [Flask](https://flask.palletsprojects.com/)                                  | Free / Open-source  |
| Audio capture       | Web Audio API + MediaRecorder API                                            | Built into browser  |
| Frontend            | Vanilla HTML, CSS, JavaScript                                                | No framework needed |
| Fonts               | Google Fonts (Cormorant Garamond, Space Mono)                                | Free                |

No paid APIs, no external services, no subscriptions.

---

## Requirements

- Python 3.9 or higher
- A modern browser (Chrome, Firefox, Edge)
- ~2 GB disk space for the model (downloaded automatically on first run)
- 4 GB+ RAM minimum — 8 GB+ recommended

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/s-i-x-voice-to-text.git
cd s-i-x-voice-to-text
```

**2. Install Python dependencies**

```bash
pip install faster-whisper sounddevice scipy numpy flask flask-cors
```

**3. Run the server**

```bash
python server.py
```

The first run will automatically download the Whisper model (~1.6 GB). Subsequent runs start instantly from cache.

**4. Open the app**

Navigate to `http://127.0.0.1:5000` in your browser.

---

## Usage

### Web Interface

| Action                | How                                                           |
| --------------------- | ------------------------------------------------------------- |
| Record from mic       | Click **Record**, speak, click **Stop**                       |
| Upload a file         | Click **↑ Upload file** or drag-and-drop onto the page        |
| Switch to lyrics mode | Toggle **Lyrics** in the mode pill before recording/uploading |
| Copy result           | Click **Copy** in the output card                             |

### Terminal (standalone, no server needed)

```bash
# Interactive menu
python s_i_x_vtt.py

# Transcribe a file directly
python s_i_x_vtt.py audio.mp3

# Transcribe a song/audio file in lyrics mode
python s_i_x_vtt.py audio.mp3 --song

# Use a specific model
python s_i_x_vtt.py audio.mp3 large-v3
```

---

## Model Options

Change the model in `server.py` by editing this line:

```python
MODEL = load_model("large-v3-turbo")   # recommended
```

| Model            | Speed (CPU) | Accuracy  | Size    | Best for                           |
| ---------------- | ----------- | --------- | ------- | ---------------------------------- |
| `turbo`          | Fastest     | Good      | ~1.5 GB | English, fast demos                |
| `large-v3-turbo` | Moderate    | Very good | ~1.6 GB | Multilingual — **recommended**     |
| `large-v3`       | Slow        | Best      | ~3 GB   | Maximum accuracy, no time pressure |
| `medium`         | Fast        | Decent    | ~1.5 GB | Fallback if others unavailable     |

---

## Supported Languages (select)

Whisper supports 99 languages. A few examples:

`en` English · `hi` Hindi · `te` Telugu · `ta` Tamil · `kn` Kannada · `ml` Malayalam · `mr` Marathi · `bn` Bengali · `ur` Urdu · `es` Spanish · `fr` French · `de` German · `ja` Japanese · `zh` Chinese · `ar` Arabic · `ru` Russian · `pt` Portuguese · `ko` Korean

Language is auto-detected by default. To force a specific language, use the **[5] Language lock** option in the terminal menu, or pass the `language` parameter directly in `server.py`.

---

## API Endpoints

| Method | Endpoint           | Description                     |
| ------ | ------------------ | ------------------------------- |
| `POST` | `/transcribe`      | Transcribe speech audio         |
| `POST` | `/transcribe-song` | Transcribe and format as lyrics |
| `GET`  | `/`                | Serves the frontend             |

Both POST endpoints accept `multipart/form-data` with a single field `audio` containing the audio file.

**Response format:**

```json
{
  "text": "Transcribed text here.",
  "language": "en",
  "confidence": 0.99
}
```

---

## Customisation

**Retheming the UI** — all colours and fonts live in the `:root` block at the top of `styles.css`:

```css
:root {
  --accent: #c8f542; /* change this to recolour all buttons and highlights */
  --bg: #0c0c0f; /* page background */
  --ff-display: "Cormorant Garamond", serif; /* heading font */
  --ff-mono: "Space Mono", monospace; /* UI and output font */
}
```

**Adjusting lyrics sensitivity** — edit the gap thresholds in `s_i_x_vtt.py`:

```python
LINE_GAP  = 0.45   # seconds of silence → new lyric line
VERSE_GAP = 1.80   # seconds of silence → new verse (blank line)
```

---

## Known Limitations

- Transcription speed depends on CPU performance — a slower machine will take longer on large files
- Punctuation post-processing is optimised for Latin scripts; other scripts are transcribed correctly but may not get added punctuation
- Browser microphone access requires either `localhost` or an HTTPS connection
- The server is not intended for production deployment — this is a local-only college project

---

## License

This project is submitted as a college mini-project for academic purposes.  
Faster-Whisper and the underlying Whisper model are released under the [MIT License](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE).

---

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — the underlying speech recognition model
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — optimised CTranslate2 inference wrapper
- [Flask](https://flask.palletsprojects.com/) — lightweight Python web framework
