# Dockerfile for running job.py
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip==25.2 setuptools==69.0.3 wheel==0.42.0 \
    && pip install --no-cache-dir \
        ai2-olmo-core@git+https://github.com/allenai/OLMo-core.git@abc12e50ba756c21e575452cfc6f150dafa9509e \
        class-registry==2.1.2 \
        cartopy==0.25.0 \
        einops==0.8.1 \
        hdf5plugin==6.0.0 \
        rasterio==1.4.3 \
        universal_pathlib==0.3.4

WORKDIR /app
COPY olmoearth_pretrain /app/olmoearth_pretrain
COPY helios /app/helios
COPY scripts/tools/cloud_run /app

ARG GCLOUD_PROJECT
ARG IN_BUCKET
ARG OUT_BUCKET

ENV GCLOUD_PROJECT=${GCLOUD_PROJECT} \
    IN_BUCKET=${IN_BUCKET} \
    OUT_BUCKET=${OUT_BUCKET}

# Preloads the model
RUN python3 -c "from olmoearth_pretrain.model_loader import ModelID, load_model_from_id; load_model_from_id(ModelID.OLMOEARTH_V1_NANO)"


ENTRYPOINT ["python", "job.py"]
