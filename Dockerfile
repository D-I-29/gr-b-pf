FROM python:3.10.7-slim-bullseye 
 
USER root

COPY ./requirements.txt /
RUN apt-get -y update && \
    apt-get -y install python3 && \
    apt-get -y install python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install -r /requirements.txt && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu