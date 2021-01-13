FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y gcc g++ ninja-build
RUN git clone -b non_local https://github.com/iShohei220/kaggle-pku-autonomous-driving.git
WORKDIR /workspace/kaggle-pku-autonomous-driving/lib/models
RUN git clone https://github.com/CharlesShang/DCNv2.git
WORKDIR /workspace/kaggle-pku-autonomous-driving/lib/models/DCNv2
RUN ./make.sh

WORKDIR /workspace
