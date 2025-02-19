# The base image, which will be the starting point for the Docker image.
# We're using a PyTorch image built from https://github.com/allenai/docker-images
# because PyTorch is really big we want to install it first for caching.
FROM ghcr.io/allenai/pytorch:2.5.1-cuda12.1-python3.11

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR /stage/

# Copy the `requirements.txt` to `/stage/requirements.txt/` and then install them.
# We do this first because it's slow and each of these commands are cached in sequence.
RUN apt-get update && apt-get install --no-install-recommends -y git && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the folder `scripts` to `scripts/`
# You might need multiple of these statements to copy all the folders you need for your experiment.
COPY helios/ /stage/helios/
COPY scripts/latent_mim.py /stage/latent_mim.py
COPY data/norm_configs /stage/data/norm_configs
ENV PYTHONPATH="${PYTHONPATH}:/stage/"
