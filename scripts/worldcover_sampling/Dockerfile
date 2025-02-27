FROM python:3.10

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR /stage/
COPY requirements.txt /stage/
RUN pip install --no-cache-dir -r requirements.txt
COPY compute_worldcover_histograms.py /stage/
