FROM python:3.12-alpine

COPY requirements.txt /opt/copy_files/
RUN pip install --no-cache-dir -r /opt/copy_files/requirements.txt

COPY copy_files.py move_weka_data_to_ssd.py /opt/copy_files/
WORKDIR /opt/copy_files/
