# ══════════════════════════════════════════════════════════
# IFRS 9 Risk Cockpit — Multi-Stage Dockerfile
# ══════════════════════════════════════════════════════════
#
# Stage 1 (train):  Image lourde avec CUDA/PyTorch pour entrainement GPU
# Stage 2 (serve):  Image slim pour inference CPU + dashboard
#
# Build :
#   docker build --target serve -t ifrs9-cockpit:serve .
#   docker build --target train -t ifrs9-cockpit:train .
#
# Run :
#   docker run -p 8501:8501 ifrs9-cockpit:serve           # Dashboard
#   docker run -p 8000:8000 ifrs9-cockpit:serve api       # API seule
#   docker run --gpus all ifrs9-cockpit:train              # Entrainement GPU
#
# ══════════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# STAGE 1 : TRAIN (GPU, image lourde ~8GB)
# ──────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS train

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Installer Python 3.13
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv python3.13-dev python3-pip && \
    ln -sf /usr/bin/python3.13 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dependances d'entrainement (GPU)
COPY ifrs9_cockpit/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m pip install fastapi uvicorn[standard] pandera structlog

# Copier le code source
COPY ifrs9_cockpit/ /app/ifrs9_cockpit/

# Point d'entree : entrainement offline
ENTRYPOINT ["python", "-m", "ifrs9_cockpit.training.train"]
CMD ["--n-clients", "1500000"]


# ──────────────────────────────────────────────
# STAGE 2 : SERVE (CPU, image legere ~1.5GB)
# ──────────────────────────────────────────────
FROM python:3.13-slim AS serve

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Dependances systeme minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Installer les dependances (CPU only — pas de torch GPU)
COPY ifrs9_cockpit/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install fastapi uvicorn[standard] pandera structlog

# Copier le code source
COPY ifrs9_cockpit/ /app/ifrs9_cockpit/

# Copier les modeles pre-entraines si disponibles
COPY ifrs9_cockpit/training/models/ /app/ifrs9_cockpit/training/models/ 2>/dev/null || true

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Exposer les ports
EXPOSE 8501 8000

# Entrypoint avec selection du mode
COPY ifrs9_cockpit/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["dashboard"]
