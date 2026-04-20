FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
# - libsndfile1, ffmpeg: required by librosa, soundfile, pydub for audio processing
# - build-essential, pkg-config: needed to compile some Python packages
# - libgomp1: OpenMP runtime required by sentence-transformers / sklearn
# - curl: for health check probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    build-essential \
    pkg-config \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spacy English model (used for NLP processing)
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Copy project files
COPY . /app/

# Create temp directory for audio files
RUN mkdir -p /tmp/semantic-worker

# Run the worker
CMD ["python", "main.py"]
