/* ============================================================
   S.I.X — Voice to Text  |  app.js
   ============================================================
   Sections (Ctrl+F to jump):
     1.  Config
     2.  State
     3.  Mode Toggle
     4.  Recording — start / stop
     5.  File Upload & Drag-and-Drop
     6.  API — send audio to Flask backend
     7.  Output Display
     8.  Copy to Clipboard
     9.  Status Helper
    10.  Waveform Visualiser (Web Audio API)
   ============================================================ */


/* ── 1. Config ────────────────────────────────────────────────
   Change API_BASE if you run Flask on a different port/host.
   ─────────────────────────────────────────────────────────── */
const API_BASE = 'http://127.0.0.1:5000';


/* ── 2. State ─────────────────────────────────────────────────
   All mutable runtime state lives here — no hidden globals.
   ─────────────────────────────────────────────────────────── */
let mode      = 'speech';   // 'speech' | 'song'  (set by mode toggle)
let recording = false;      // true while MediaRecorder is active
let mediaRec  = null;       // MediaRecorder instance
let audioCtx  = null;       // Web Audio context (created on first record)
let analyser  = null;       // AnalyserNode feeding the waveform canvas
let animFrame = null;       // requestAnimationFrame handle for the waveform loop
let timerInt  = null;       // setInterval handle for the recording clock
let elapsed   = 0;          // seconds elapsed since recording started
let chunks    = [];         // raw Blob chunks from MediaRecorder


/* ── 3. Mode Toggle ───────────────────────────────────────────
   Called by the Speech / Lyrics pill buttons in the HTML.
   ─────────────────────────────────────────────────────────── */
function setMode(m) {
  mode = m;
  document.getElementById('btnSpeech').classList.toggle('active', m === 'speech');
  document.getElementById('btnSong').classList.toggle('active',   m === 'song');
}


/* ── 4. Recording ─────────────────────────────────────────────
   toggleRecord() is wired to the Record button.
   startRecord() asks for mic permission and begins capture.
   stopRecord()  finalises the chunks and sends them to the API.
   ─────────────────────────────────────────────────────────── */
function toggleRecord() {
  if (recording) stopRecord();
  else           startRecord();
}

async function startRecord() {
  try {
    // Request microphone access — browser will prompt the user
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Wire up the waveform visualiser to the same stream
    setupAnalyser(stream);

    // Initialise MediaRecorder with the best supported MIME type
    chunks  = [];
    mediaRec = new MediaRecorder(stream, { mimeType: getSupportedMime() });

    // Accumulate data chunks every 250 ms
    mediaRec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };

    // When stopped: release the mic track, then fire off the API call
    mediaRec.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      sendAudio();
    };

    mediaRec.start(250);  // emit a chunk every 250 ms

    // Update UI state
    recording = true;
    elapsed   = 0;
    document.getElementById('btnRecord').classList.add('recording');
    document.getElementById('recordLabel').textContent      = 'Stop';
    document.getElementById('idleLabel').style.opacity = '0';
    setStatus('recording', 'Recording…');

    // Start the MM:SS timer
    timerInt = setInterval(() => {
      elapsed++;
      const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const ss = String(elapsed % 60).padStart(2, '0');
      const el = document.getElementById('timer');
      el.textContent = `${mm}:${ss}`;
      el.classList.add('active');
    }, 1000);

  } catch (err) {
    // Most common cause: user denied mic permission
    setStatus('error', 'Microphone access denied');
  }
}

function stopRecord() {
  if (!mediaRec) return;

  mediaRec.stop();
  recording = false;

  // Clear timer
  clearInterval(timerInt);
  document.getElementById('timer').classList.remove('active');
  document.getElementById('timer').textContent = '';

  // Stop waveform animation
  cancelAnimationFrame(animFrame);
  clearCanvas();
  if (audioCtx) { audioCtx.close(); audioCtx = null; }

  // Reset button appearance
  document.getElementById('btnRecord').classList.remove('recording');
  document.getElementById('recordLabel').textContent      = 'Record';
  document.getElementById('idleLabel').style.opacity = '1';

  setStatus('processing', 'Transcribing…');
}

/**
 * Returns the best audio MIME type supported by this browser.
 * Opus inside WebM/Ogg gives the smallest files and best quality.
 */
function getSupportedMime() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  return candidates.find(t => MediaRecorder.isTypeSupported(t)) || '';
}


/* ── 5. File Upload & Drag-and-Drop ───────────────────────────
   handleFile() is the single entry point for any local file,
   whether chosen via the button or dragged onto the page.
   ─────────────────────────────────────────────────────────── */
function handleFile(file) {
  if (!file) return;
  setStatus('processing', `Transcribing "${file.name}"…`);
  hideOutput();
  postAudio(file);
}

// Drag-and-drop: show overlay when something is dragged onto the page
document.addEventListener('dragenter', e => {
  e.preventDefault();
  document.getElementById('dropOverlay').classList.add('active');
});

// Hide overlay when the drag leaves the window entirely
document.addEventListener('dragleave', e => {
  if (e.relatedTarget === null)
    document.getElementById('dropOverlay').classList.remove('active');
});

document.addEventListener('dragover', e => e.preventDefault());

document.addEventListener('drop', e => {
  e.preventDefault();
  document.getElementById('dropOverlay').classList.remove('active');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('audio/')) {
    handleFile(file);
  } else {
    setStatus('error', 'Please drop an audio file');
  }
});


/* ── 6. API — send audio to Flask backend ─────────────────────
   sendAudio()  packs recorded chunks into a File and calls postAudio().
   postAudio()  POSTs a FormData blob to /transcribe or /transcribe-song.
   ─────────────────────────────────────────────────────────── */
function sendAudio() {
  const mime = getSupportedMime() || 'audio/webm';
  const blob = new Blob(chunks, { type: mime });

  // Pick a file extension that matches the MIME type so Flask can read it
  const ext  = mime.includes('ogg') ? '.ogg' : mime.includes('mp4') ? '.mp4' : '.webm';
  const file = new File([blob], `recording${ext}`, { type: mime });

  postAudio(file);
}

async function postAudio(file) {
  // Choose endpoint based on current mode
  const endpoint = mode === 'song' ? '/transcribe-song' : '/transcribe';

  const fd = new FormData();
  fd.append('audio', file);   // 'audio' must match request.files['audio'] in server.py

  try {
    const res  = await fetch(API_BASE + endpoint, { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Server error');

    showOutput(data);
    setStatus('done', 'Done');

  } catch (err) {
    // Provide a friendlier message for the most common error (server not running)
    const msg = err.message.startsWith('Failed to fetch')
      ? 'Cannot reach server — is server.py running?'
      : err.message;
    setStatus('error', msg);
  }
}


/* ── 7. Output Display ────────────────────────────────────────
   showOutput() populates and reveals the output card.
   hideOutput() hides it (called before a new transcription starts).

   Expected data shape from the API:
     { text: string, language: string, confidence: number }
   ─────────────────────────────────────────────────────────── */
function showOutput(data) {
  const card   = document.getElementById('outputCard');
  const txtEl  = document.getElementById('outputText');
  const lyrEl  = document.getElementById('outputLyrics');
  const label  = document.getElementById('outputLabel');
  const badge  = document.getElementById('langBadge');

  // Language badge — e.g. "EN · 99%"
  badge.textContent = data.language
    ? `${data.language.toUpperCase()} · ${Math.round((data.confidence || 0) * 100)}%`
    : '';

  if (mode === 'song') {
    // Lyrics mode: monospace pre block with verse spans
    label.textContent   = 'Lyrics';
    txtEl.style.display = 'none';
    lyrEl.style.display = 'block';
    lyrEl.innerHTML     = formatLyrics(data.text || '');
  } else {
    // Speech mode: large serif paragraph
    label.textContent   = 'Transcription';
    lyrEl.style.display = 'none';
    txtEl.style.display = 'block';
    txtEl.textContent   = data.text || '';
  }

  card.classList.add('visible');
}

function hideOutput() {
  document.getElementById('outputCard').classList.remove('visible');
}

/**
 * Converts plain-text lyrics (verse blocks separated by blank lines)
 * into HTML with .verse spans for the left-border styling.
 *
 * @param  {string} text  Raw text from the /transcribe-song endpoint.
 * @return {string}       HTML string safe to set as innerHTML.
 */
function formatLyrics(text) {
  const verses = text.split(/\n\n+/);   // split on one or more blank lines
  return verses
    .map(verse => {
      const lines = verse.split('\n').map(escHtml).join('\n');
      return `<span class="verse">${lines}</span>`;
    })
    .join('');
}

/** Escapes < > & so user text is safe to inject as innerHTML. */
function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}


/* ── 8. Copy to Clipboard ─────────────────────────────────── */
function copyText() {
  // Grab plain text from whichever output element is currently visible
  const src = mode === 'song'
    ? document.getElementById('outputLyrics').textContent
    : document.getElementById('outputText').textContent;

  navigator.clipboard.writeText(src).then(() => {
    const btn = document.getElementById('btnCopy');
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.textContent = 'Copy';
      btn.classList.remove('copied');
    }, 1800);
  });
}


/* ── 9. Status Helper ─────────────────────────────────────────
   setStatus(state, message)
     state — one of: 'recording' | 'processing' | 'done' | 'error'
             (maps to CSS class names on .status)
   ─────────────────────────────────────────────────────────── */
function setStatus(state, msg) {
  const el = document.getElementById('status');
  el.className = `status ${state}`;
  document.getElementById('statusText').textContent = msg;
}


/* ── 10. Waveform Visualiser ──────────────────────────────────
   Uses the Web Audio API to draw a real-time oscilloscope line
   on a <canvas> element while the microphone is active.

   setupAnalyser(stream)  — called once when recording starts.
   drawWave()             — kicks off the rAF render loop.
   clearCanvas()          — wipes the canvas when recording stops.
   ─────────────────────────────────────────────────────────── */
function setupAnalyser(stream) {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;   // increase for more detail; must be power of 2

  // Connect: mic stream → analyser (no output, just analysis)
  audioCtx.createMediaStreamSource(stream).connect(analyser);

  drawWave();
}

function drawWave() {
  const canvas = document.getElementById('waveCanvas');
  const ctx    = canvas.getContext('2d');

  // Match canvas pixel dimensions to its CSS size × device pixel ratio
  // so the line stays crisp on high-DPI (Retina) screens
  canvas.width  = canvas.offsetWidth  * devicePixelRatio;
  canvas.height = canvas.offsetHeight * devicePixelRatio;

  // Buffer for raw time-domain waveform data (values 0–255)
  const buf = new Uint8Array(analyser.frequencyBinCount);

  function loop() {
    animFrame = requestAnimationFrame(loop);

    analyser.getByteTimeDomainData(buf);  // fill buf with current waveform

    // Clear previous frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Style the waveform line
    ctx.lineWidth   = 2 * devicePixelRatio;
    ctx.strokeStyle = '#c8f542';    // matches --accent; change here to recolour
    ctx.shadowColor = '#c8f542';
    ctx.shadowBlur  = 8;            // glow radius — set to 0 to remove glow

    // Draw polyline across the canvas width
    ctx.beginPath();
    const sliceW = canvas.width / buf.length;
    let x = 0;

    for (let i = 0; i < buf.length; i++) {
      const v = buf[i] / 128;              // normalise 0-255 → 0-2
      const y = (v * canvas.height) / 2;  // centre on half-height
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);  // close to the right edge
    ctx.stroke();
    ctx.shadowBlur = 0;   // reset so shadow doesn't bleed into next frame
  }

  loop();
}

function clearCanvas() {
  const canvas = document.getElementById('waveCanvas');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}
