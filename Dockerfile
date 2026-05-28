FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip + install setuptools/wheel into the GLOBAL env first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchaudio==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# PIP_NO_BUILD_ISOLATION=0 makes openai-whisper reuse the global env
# (where setuptools/pkg_resources is already installed) instead of
# spinning up a fresh isolated subprocess that lacks pkg_resources
RUN PIP_NO_BUILD_ISOLATION=0 pip install --no-cache-dir -r requirements.txt

COPY . .

ARG WHISPER_MODEL=base
RUN python -c "import whisper; whisper.load_model('${WHISPER_MODEL}')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
