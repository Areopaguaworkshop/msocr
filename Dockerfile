# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies needed for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specs first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install native libraries needed by image processing and Kraken/Pillow runtimes
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

# Copy application source
COPY msocr/ ./msocr/

# Copy models directory. For production, mount a trained Sogdian Kraken model
# or bake models/kraken/sogdian_manuscript.mlmodel into the image.
COPY models/ ./models/

# Create non-root user for security
RUN groupadd --gid 1000 msocr && \
    useradd --uid 1000 --gid msocr --shell /bin/bash --create-home msocr && \
    chown -R msocr:msocr /app
USER msocr

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"] || exit 1

CMD ["python", "-m", "uvicorn", "msocr.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
