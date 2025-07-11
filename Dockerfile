FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget

# Install rslearn and helios (need to be in local directory).
COPY ./rslearn /opt/rslearn
COPY ./helios /opt/helios
COPY requirements.txt /opt/rslearn_projects/requirements.txt
RUN pip install --no-cache-dir --upgrade /opt/rslearn[extra] /opt/helios -r /opt/rslearn_projects/requirements.txt

# Copy rslearn_projects and install it too.
COPY . /opt/rslearn_projects/
RUN pip install --no-cache-dir /opt/rslearn_projects

WORKDIR /opt/rslearn_projects
