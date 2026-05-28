# caption-ai-backend

FastAPI + OpenAI Whisper transcription API. Deployed on Railway.

## What it does

- Accepts video/audio uploads (mp4, mov, mkv, avi, webm, mp3, wav, m4a, ogg)
- Extracts audio with ffmpeg
- Transcribes with OpenAI Whisper
- Generates SRT caption file
- Stores result in Supabase
- Returns transcript + SRT to caller

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/transcribe` | Upload file → get transcript + SRT |
| GET | `/jobs` | List recent jobs |
| GET | `/jobs/{id}` | Get one job by ID |

## Local development

```bash
# 1. Create virtualenv
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install CPU-only PyTorch (smaller, faster install)
pip install torch==2.2.2+cpu torchaudio==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install app dependencies
pip install -r requirements.txt

# 4. Set env vars
cp .env.example .env
# Fill in SUPABASE_URL and SUPABASE_SERVICE_KEY

# 5. Run
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

## Deploy to Railway

See [DEPLOY.md](../DEPLOY.md) or the full steps below.

1. Push this repo to GitHub
2. Railway → New Project → Deploy from GitHub → select this repo
3. Railway auto-detects the Dockerfile (no config needed)
4. Add environment variables in Railway dashboard:

```
SUPABASE_URL         = https://your-project.supabase.co
SUPABASE_SERVICE_KEY = eyJhbG...   (service_role key, NOT anon)
WHISPER_MODEL        = base
MAX_UPLOAD_MB        = 500
ALLOWED_ORIGINS      = https://your-frontend.vercel.app
```

5. Set RAM to at least 2 GB (Railway dashboard → Service → Settings → Resources)

## Whisper model sizes

| Model | RAM | Speed | Accuracy |
|-------|-----|-------|----------|
| tiny | 1 GB | fastest | ok |
| base | 1 GB | fast | good ✓ |
| small | 2 GB | medium | better |
| medium | 5 GB | slow | great |
| large | 10 GB | slowest | best |

`base` is recommended for Railway Hobby. Set via `WHISPER_MODEL` env var.

## Supabase setup

Run `supabase_migration.sql` once in your Supabase SQL Editor before first deploy.
