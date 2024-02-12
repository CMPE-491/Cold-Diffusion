# Local Usage:
# ```
# docker build -t ghcr.io/cmpe-491/first-image:v6 .
# docker push ghcr.io/cmpe-491/first-image:v6
# ```""
# Get into server
# ssh ahmet.susuz@79.123.177.160
# Send file to remote server:
# scp training_script.sh ahmet.susuz@79.123.177.160:/users/ahmet.susuz
# Get file from remote server:
# scp ssh ahmet.susuz@79.123.177.160:/users/ahmet.susuz/slurm-1776.out .
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

LABEL maintainer="Abdullah Susuz"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        cmake \
        ninja-build \
        build-essential \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
        python3-opencv \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install --upgrade pip && \
    python3 -m venv /opt/python3/venv/base

COPY requirements.txt /opt/python3/venv/base/

RUN /opt/python3/venv/base/bin/python3 -m pip install --upgrade pip
RUN /opt/python3/venv/base/bin/python3 -m pip install wheel
RUN /opt/python3/venv/base/bin/python3 -m pip install -r /opt/python3/venv/base/requirements.txt

COPY snowification/ /opt/python3/venv/base/

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint to bash
ENTRYPOINT ["/entrypoint.sh"]