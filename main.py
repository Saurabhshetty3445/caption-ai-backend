import os
import uuid
import tempfile
from pathlib import Path
from datetime import datetime

import ffmpeg
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# ── faster-whisper model (loaded once at startup) ─────────────
MODEL_SIZE     = os.getenv("WHISPER_MODEL", "base")
MAX_UPLOAD_MB  = int(os.getenv("MAX_UPLOAD_MB", "500"))

print(f"[startup] Loading faster-whisper model: {MODEL_SIZE}")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("[startup] Model ready.")

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".ogg"}
VIDEO_EXT   = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


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
    return {"status": "ok", "model": MODEL_SIZE}


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

        # Extract audio from video
        if ext in VIDEO_EXT:
            audio_path = os.path.join(tmp, "audio.wav")
            try:
                extract_audio(input_path, audio_path)
            except ffmpeg.Error as e:
                raise HTTPException(422, f"Audio extraction failed: {e.stderr.decode()}")
        else:
            audio_path = input_path

        # Transcribe with faster-whisper
        # segments is a generator — consume it fully before tmp dir closes
        segments_gen, info = model.transcribe(audio_path, beam_size=5)
        segments = list(segments_gen)

        language = info.language
        duration = info.duration
        transcript = " ".join(seg.text.strip() for seg in segments)
        srt = to_srt(segments)

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
