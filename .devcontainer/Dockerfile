FROM ubuntu:18.04

# Install dependencies

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    wget \
    unzip

# Install Python 3.8.0 and poetry

RUN apt-get update && apt-get install -y python3.8 python3.8-distutils

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py

RUN python3.8 -m pip install poetry
