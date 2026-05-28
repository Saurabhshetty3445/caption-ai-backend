import os
import uuid
import tempfile
from pathlib import Path
from datetime import datetime

import whisper
import ffmpeg
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="WhisperSaaS API", version="1.0.0")

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

# ── Whisper model (loaded once at startup) ────────────────────
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))

print(f"[startup] Loading Whisper model: {MODEL_SIZE}")
model = whisper.load_model(MODEL_SIZE)
print("[startup] Whisper model ready.")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".ogg"}
VIDEO_EXTENSIONS   = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


# ── Helpers ───────────────────────────────────────────────────

def fmt_ts(seconds: float) -> str:
    """Seconds → SRT timestamp  HH:MM:SS,mmm"""
    ms = int((seconds % 1) * 1000)
    s  = int(seconds) % 60
    m  = (int(seconds) // 60) % 60
    h  = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(f"{i}\n{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}\n{seg['text'].strip()}\n")
    return "\n".join(lines)


def extract_audio(src: str, dst: str) -> None:
    (
        ffmpeg.input(src)
        .output(dst, ac=1, ar="16000", format="wav")
        .overwrite_output()
        .run(quiet=True)
    )


def persist(job_id: str, filename: str, transcript: str, srt: str, language: str, duration: float) -> None:
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


# ── Schema ────────────────────────────────────────────────────

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


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "WhisperSaaS", "model": MODEL_SIZE, "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE, "model_loaded": model is not None}


@app.post("/transcribe", response_model=JobStatus)
async def transcribe(request: Request, file: UploadFile = File(...)):
    # File type check
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Read file into memory with size guard
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_UPLOAD_MB} MB")

    job_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, f"input{ext}")
        with open(video_path, "wb") as f:
            f.write(content)

        # Extract audio from video formats
        if ext in VIDEO_EXTENSIONS:
            audio_path = os.path.join(tmp, "audio.wav")
            try:
                extract_audio(video_path, audio_path)
            except ffmpeg.Error as e:
                raise HTTPException(422, f"Could not extract audio: {e.stderr.decode()}")
        else:
            audio_path = video_path

        # Transcribe
        result   = model.transcribe(audio_path, verbose=False)
        segments = result["segments"]
        transcript = result["text"].strip()
        language   = result.get("language", "unknown")
        duration   = segments[-1]["end"] if segments else 0.0
        srt        = to_srt(segments)

        # Persist to Supabase (sync call — supabase-py is not async)
        try:
            persist(job_id, file.filename, transcript, srt, language, duration)
        except Exception as e:
            # Don't fail the request if DB write fails — still return result
            print(f"[warn] Supabase write failed: {e}")

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
