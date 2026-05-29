import os
import uuid
import tempfile
import re
from pathlib import Path
from datetime import datetime

import ffmpeg
import google.generativeai as genai
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="WhisperSaaS API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase ──────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Gemini ────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── Whisper ───────────────────────────────────────────────────
MODEL_SIZE    = os.getenv("WHISPER_MODEL", "medium")   # medium is best for Hindi
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))

print(f"[startup] Loading faster-whisper model: {MODEL_SIZE}")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("[startup] Model ready.")

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".ogg"}
VIDEO_EXT   = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German",  "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic",  "pt": "Portuguese", "ru": "Russian", "it": "Italian",
    "bn": "Bengali", "ur": "Urdu", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "ml": "Malayalam",
}


# ── Helpers ───────────────────────────────────────────────────

def fmt_ts(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s  = int(seconds) % 60
    m  = (int(seconds) // 60) % 60
    h  = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{seg.text.strip()}\n")
    return "\n".join(lines)


def parse_srt(srt: str) -> list[dict]:
    """Parse SRT into list of {index, start, end, text}."""
    blocks = []
    for block in srt.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            blocks.append({
                "index": lines[0],
                "timecode": lines[1],
                "text": "\n".join(lines[2:]),
            })
    return blocks


def build_srt(blocks: list[dict]) -> str:
    parts = []
    for b in blocks:
        parts.append(f"{b['index']}\n{b['timecode']}\n{b['text']}\n")
    return "\n".join(parts)


def extract_audio(src: str, dst: str) -> None:
    (
        ffmpeg.input(src)
        .output(dst, ac=1, ar="16000", format="wav")
        .overwrite_output()
        .run(quiet=True)
    )


def persist(job_id, filename, transcript, srt, language, duration):
    try:
        supabase.table("transcriptions").upsert({
            "id": job_id,
            "filename": filename,
            "transcript": transcript,
            "srt_captions": srt,
            "language": language,
            "duration_seconds": round(duration, 2),
            "created_at": datetime.utcnow().isoformat(),
            "status": "completed",
        }).execute()
    except Exception as e:
        print(f"[warn] Supabase write failed: {e}")


def translate_with_gemini(texts: list[str], target_language: str) -> list[str]:
    """Translate a batch of subtitle texts using Gemini."""
    lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"Translate the following subtitle lines to {lang_name}.\n"
        f"Return ONLY the translated lines, numbered the same way.\n"
        f"Keep each translation on its own line. Do not add any explanation.\n\n"
        f"{numbered}"
    )
    gemini = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = gemini.generate_content(prompt)
    raw = response.text.strip()

    # Parse numbered lines back out
    translated = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading "1. " "2. " etc
        cleaned = re.sub(r"^\d+\.\s*", "", line)
        translated.append(cleaned)

    # Safety: if Gemini returned wrong count, fall back to originals
    if len(translated) != len(texts):
        print(f"[warn] Gemini returned {len(translated)} lines, expected {len(texts)}. Using originals.")
        return texts

    return translated


# ── Schemas ───────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str
    filename: str | None = None
    transcript: str | None = None
    srt_captions: str | None = None
    language: str | None = None
    duration_seconds: float | None = None
    created_at: str | None = None
    error: str | None = None


class TranslateRequest(BaseModel):
    srt: str
    transcript: str
    target_language: str   # e.g. "en", "fr", "hi"


class TranslateResponse(BaseModel):
    translated_srt: str
    translated_transcript: str
    target_language: str
    target_language_name: str


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "WhisperSaaS", "model": MODEL_SIZE, "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE}


@app.get("/languages")
def languages():
    return SUPPORTED_LANGUAGES


@app.post("/transcribe", response_model=JobStatus)
async def transcribe(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {MAX_UPLOAD_MB} MB")

    job_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, f"input{ext}")
        with open(input_path, "wb") as f:
            f.write(content)

        if ext in VIDEO_EXT:
            audio_path = os.path.join(tmp, "audio.wav")
            try:
                extract_audio(input_path, audio_path)
            except ffmpeg.Error as e:
                raise HTTPException(422, f"Audio extraction failed: {e.stderr.decode()}")
        else:
            audio_path = input_path

        # transcribe — force Hindi detection with language hint if needed
        segments_gen, info = model.transcribe(
            audio_path,
            beam_size=5,
            # Let Whisper auto-detect; medium model handles Hindi well
        )
        segments = list(segments_gen)

        language   = info.language
        duration   = info.duration
        transcript = " ".join(seg.text.strip() for seg in segments)
        srt        = to_srt(segments)

        persist(job_id, file.filename, transcript, srt, language, duration)

    return JobStatus(
        job_id=job_id,
        status="completed",
        filename=file.filename,
        transcript=transcript,
        srt_captions=srt,
        language=language,
        duration_seconds=round(duration, 2),
        created_at=datetime.utcnow().isoformat(),
    )


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY is not configured on the server")

    if req.target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Unsupported target language: {req.target_language}")

    # Parse SRT blocks
    blocks = parse_srt(req.srt)
    if not blocks:
        raise HTTPException(400, "Could not parse SRT content")

    # Translate all subtitle texts in one Gemini call
    texts = [b["text"] for b in blocks]
    translated_texts = translate_with_gemini(texts, req.target_language)

    # Rebuild SRT with translated text, original timecodes
    for b, t in zip(blocks, translated_texts):
        b["text"] = t
    translated_srt = build_srt(blocks)

    # Translate full transcript too
    translated_transcript_list = translate_with_gemini([req.transcript], req.target_language)
    translated_transcript = translated_transcript_list[0]

    return TranslateResponse(
        translated_srt=translated_srt,
        translated_transcript=translated_transcript,
        target_language=req.target_language,
        target_language_name=SUPPORTED_LANGUAGES[req.target_language],
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    res = supabase.table("transcriptions").select("*").eq("id", job_id).single().execute()
    if not res.data:
        raise HTTPException(404, "Job not found")
    r = res.data
    return JobStatus(job_id=r["id"], status=r["status"], filename=r["filename"],
                     transcript=r["transcript"], srt_captions=r["srt_captions"],
                     language=r["language"], duration_seconds=r["duration_seconds"],
                     created_at=r["created_at"])


@app.get("/jobs", response_model=list[JobStatus])
def list_jobs(limit: int = 20):
    res = (supabase.table("transcriptions")
           .select("*").order("created_at", desc=True).limit(min(limit, 100)).execute())
    return [
        JobStatus(job_id=r["id"], status=r["status"], filename=r["filename"],
                  transcript=r["transcript"], srt_captions=r["srt_captions"],
                  language=r["language"], duration_seconds=r["duration_seconds"],
                  created_at=r["created_at"])
        for r in (res.data or [])
    ]
