#FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
# FROM nvcr.io/nvidia/pytorch:25.06-py3
FROM python:3.12


RUN apt update
RUN apt install -y libpq-dev ffmpeg libsm6 libxext6 git wget

# Install rslearn and helios (need to be in local directory).
COPY . /opt/helios
COPY ./rslearn /opt/rslearn
COPY ./rslearn_projects /opt/rslearn_projects/
COPY requirements.txt /opt/helios/requirements.txt

RUN pip install --no-cache-dir --upgrade /opt/rslearn[extra]
RUN pip install /opt/helios
RUN pip install --no-cache-dir /opt/rslearn_projects

WORKDIR /opt/rslearn_projects
