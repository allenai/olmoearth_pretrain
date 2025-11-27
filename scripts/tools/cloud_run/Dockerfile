# Dockerfile for running job.py
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    litserve==0.2.15 \
    ai2-olmo-core@git+https://github.com/allenai/OLMo-core.git@abc12e50ba756c21e575452cfc6f150dafa9509e \
    class-registry \
    cartopy \
    einops \
    hdf5plugin \
    rasterio \
    universal_pathlib>=0.2.5

WORKDIR /app
COPY olmoearth_pretrain /app/olmoearth_pretrain
COPY helios /app/helios
COPY scripts/tools/cloud_run /app

ARG GCLOUD_PROJECT
ARG IN_BUCKET
ARG OUT_BUCKET

ENV GCLOUD_PROJECT=${GCLOUD_PROJECT}
ENV IN_BUCKET=${IN_BUCKET}
ENV OUT_BUCKET=${OUT_BUCKET}

# Preloads the model
RUN python3 - <<EOF
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
load_model_from_id(ModelID.OLMOEARTH_V1_NANO)
EOF

ENTRYPOINT ["python", "job.py"]
