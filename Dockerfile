# =============================================================================
# Stage 1: Builder - Install dependencies and compile wheels
# =============================================================================
FROM python:3.12-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.12-slim-bookworm AS runtime

# Install runtime system dependencies
# - ffmpeg: video processing (ffprobe + ffmpeg commands)
# - libgl1: OpenCV dependency for image processing
# - libglib2.0-0: OpenCV dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser config.py .
COPY --chown=appuser:appuser interpreter.py .
COPY --chown=appuser:appuser reasoner.py .
COPY --chown=appuser:appuser batch_analyze.py .

# Copy system_test.py only if it exists (optional)
COPY --chown=appuser:appuser system_test.p[y] ./

# Create directories for volumes
RUN mkdir -p /app/videos /app/output /app/profiles && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables with container-appropriate defaults
# OLLAMA_HOST points to the ollama service name for container networking
ENV OLLAMA_HOST=http://ollama:11434 \
    INTERPRETER_MODEL=qwen3-vl:8b-instruct-q4_K_M \
    REASONER_MODEL=qwen3:14b-q4_K_M \
    QUALITY_MODE=high \
    VIDEO_DIR=/app/videos \
    OUTPUT_DIR=/app/output \
    PROFILE_PATH=/app/profiles/player_profile.json \
    BATCH_MODE=false \
    PYTHONUNBUFFERED=1

# Healthcheck to verify ollama is reachable (uses Python since it's already installed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('${OLLAMA_HOST}/api/version')" || exit 1

# Default command runs batch analysis (can be overridden)
CMD ["python", "batch_analyze.py"]