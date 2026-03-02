# ============================================================
# OratorAI Semantic Worker - Dockerfile
# Multi-stage build for optimized production image
# ============================================================

# ---------------------
# Stage 1: Builder
# ---------------------
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    pkg-config \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# ---------------------
# Stage 2: NLP Assets
# ---------------------
FROM python:3.11-slim AS nlp-downloader

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install only what's needed to download assets
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Pre-download sentence-transformers model to cache it
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV TRANSFORMERS_CACHE=/model-cache \
    SENTENCE_TRANSFORMERS_HOME=/model-cache

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
import os; \
model_name = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
print(f'Downloading model: {model_name}'); \
SentenceTransformer(model_name); \
print('Model downloaded successfully')"

# Download NLTK and spaCy assets
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')" && \
    python -m spacy download en_core_web_sm || true

# ---------------------
# Stage 3: Production
# ---------------------
FROM python:3.11-slim AS production

LABEL maintainer="OratorAI Team" \
      version="1.0.0" \
      description="OratorAI Semantic Analysis Worker"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    TRANSFORMERS_CACHE=/app/model-cache \
    SENTENCE_TRANSFORMERS_HOME=/app/model-cache \
    NLTK_DATA=/root/nltk_data \
    TEMP_DIR=/tmp/semantic-worker \
    LOG_LEVEL=INFO \
    LOG_FORMAT=colored

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libffi8 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r worker && useradd -r -g worker -d /app -s /sbin/nologin worker

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded ML models & NLP data
COPY --from=nlp-downloader /model-cache /app/model-cache
COPY --from=nlp-downloader /root/nltk_data /root/nltk_data

# Copy application source code
COPY --chown=worker:worker src/ ./src/
COPY --chown=worker:worker main.py .
COPY --chown=worker:worker health_check.py .

# Create required directories
RUN mkdir -p /tmp/semantic-worker && \
    chown -R worker:worker /app /tmp/semantic-worker

# Switch to non-root user
USER worker

# Health check using the health_check.py script
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python health_check.py || exit 1

# Expose no ports - this is a worker/consumer service
# (no HTTP server, communicates via SQS + Webhook)

# Run the semantic worker
CMD ["python", "main.py"]
