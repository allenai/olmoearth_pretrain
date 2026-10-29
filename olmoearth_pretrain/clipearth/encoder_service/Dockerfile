# The ClipEarth text encoder (CPU), with the model baked in.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
# The revision transformers.js loads as "main" (unchanged since 2025-09-04).
ARG MODEL_REVISION=5090578d9565bb06545b4552f76e6bc2c93e4a66
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('onnx-community/embeddinggemma-300m-ONNX', revision='${MODEL_REVISION}', allow_patterns=['onnx/model_no_gather_q4.onnx*', 'tokenizer.json'], local_dir='/model')"
COPY *.py /app/
WORKDIR /app
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --no-access-log
