# ─── RL Trader Production Dockerfile ──────────────────────────────────────────
# Multi-stage build: slim runtime image (~800MB with torch)
#
# Build:  docker build -t rl_trader -f deployment/Dockerfile .
# Run:    docker run -p 8000:8000 -p 9090:9090 --env-file .env rl_trader

FROM python:3.11-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies stage ─────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
COPY deployment/requirements_prod.txt ./requirements_prod.txt

# Install CPU-only torch first (smaller image)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_prod.txt

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM deps AS runtime

# Copy project
COPY config/      config/
COPY data/        data/
COPY features/    features/
COPY environment/ environment/
COPY execution/   execution/
COPY monitoring/  monitoring/
COPY deployment/  deployment/
COPY train/       train/
COPY utils/       utils/
COPY scripts/     scripts/
COPY run_phase3.py .

# Create directories
RUN mkdir -p models/checkpoints logs/tensorboard data/cache data/processed state

# Non-root user for security
RUN useradd -m -u 1001 trader && chown -R trader:trader /app
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 9090

CMD ["python", "run_phase3.py", "--mode", "paper"]
