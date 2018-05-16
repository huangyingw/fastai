FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

COPY ./install_anaconda3.sh /root/setup/
COPY ./install_fastai.sh /root/setup/

WORKDIR /root/setup/
RUN apt-get update \
        && apt-get install -y \
        curl \
        git \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3-pip \
        realpath \
        && apt-get clean
RUN ./install_anaconda3.sh
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/root/anaconda3/bin:$PATH"
ENV PATH="/usr/local/bin:/opt/local/sbin:$PATH"
COPY ./environment.yml /root/setup/
RUN ./install_fastai.sh
RUN apt-get install -y
RUN rm -fr /root/setup/
