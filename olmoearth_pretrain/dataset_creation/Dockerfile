# Dockerfile for dataset creation (CPU-only — no GPU needed for data ingestion).

FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev ffmpeg libsm6 libxext6 git wget libgdal-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /opt/olmoearth_pretrain/
WORKDIR /opt/olmoearth_pretrain/

# Install CPU-only torch first (rslearn requires torch but we don't need CUDA),
# then the package + dataset-creation extras.
RUN pip install --no-cache-dir \
        torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[dataset-creation]"
