FROM python:3.12-slim

# ffmpeg — конвертация голосовых; libgomp1 нужен ctranslate2 (Whisper) и
# onnxruntime (Piper); libasound2 — только Azure Speech SDK, если его включат.
RUN apt-get update && apt-get install --no-install-recommends -y \
        ffmpeg \
        libgomp1 \
        libasound2 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# HF_HOME и PIPER_MODEL_DIR указывают в том с базой: модели качаются один раз
# и переживают пересоздание контейнера.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_PATH=/app/data/bot.db \
    HF_HOME=/app/data/models \
    PIPER_MODEL_DIR=/app/data/voices

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot ./bot

RUN useradd --create-home --uid 1000 botuser \
    && mkdir -p /app/data \
    && chown -R botuser:botuser /app
USER botuser

VOLUME ["/app/data"]

CMD ["python", "-m", "bot"]
