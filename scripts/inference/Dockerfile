# Dockerfile for running barebones server for generating embeddings
# To build: `docker build -t litserve -f scripts/litserve/Dockerfile .` from Helios directory
# To run:   `docker run -p 8000:8000 litserve`
# To test:  `python scripts/litserve/client.py`
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Only absolutely necessary packages
RUN pip install --no-cache-dir torch>=2.7 --index-url https://download.pytorch.org/whl/cpu 
RUN pip install \
    litserve==0.2.15 \
    ai2-olmo-core@git+https://github.com/allenai/OLMo-core.git@abc12e50ba756c21e575452cfc6f150dafa9509e \
    class-registry \
    cartopy \
    hdf5plugin \
    rasterio \
    universal_pathlib>=0.2.5

# additional packages
RUN pip install --no-cache-dir einops rasterio

WORKDIR /app
COPY olmoearth_pretrain /app/olmoearth_pretrain
COPY helios /app/helios

# Download model weights
# gcloud storage cp -r gs://helios-embeddings-bucket/latent_mim_tiny_shallow_decoder_lr2e-4_255000 .


# Model weights already here
COPY scripts/inference /app

EXPOSE 8080
CMD ["python", "/app/server2.py"]
