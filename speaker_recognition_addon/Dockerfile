ARG BUILD_FROM
FROM ${BUILD_FROM}

ARG TEMPIO_VERSION BUILD_ARCH

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    libsndfile1 \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* && \
    curl -sSLf -o /usr/bin/tempio \
    "https://github.com/home-assistant/tempio/releases/download/${TEMPIO_VERSION}/tempio_${BUILD_ARCH}" && \
    chmod +x /usr/bin/tempio

COPY pyproject.toml ./
COPY speaker_recognition ./speaker_recognition

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache --break-system-packages -e ".[server]" && \
    python3 -c "import numpy, torch, resemblyzer; from speaker_recognition.api import app" && \
    python3 -m speaker_recognition --help >/dev/null

ENV HOST=0.0.0.0 \
    PORT=8099 \
    ACCESS_LOG=true \
    EMBEDDINGS_DIR=/share/speaker_recognition/embeddings

COPY rootfs /

RUN chmod a+x /etc/s6-overlay/s6-rc.d/speaker-recognition/run
