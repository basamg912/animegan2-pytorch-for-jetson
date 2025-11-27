FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1

ENV DEBIAN_FRONTEND=nointeractive
ENV PYTHONUNBUFFERED=1

RUN apt update && \
    apt install -y --no-install-recommends \
        vim \
        python3-pip \
        libopenblas-base \
        libjpeg-dev \
        zlib1g-dev \
        libwebp-dev \
        git \
        cmake && \
    rm -rf /var/lib/apt/lists/* && \
    pip install onnx pycuda

WORKDIR /workspace
#VOLUME /workspace → 볼륨 인자로 지정된 데이터는 컨테이너가 삭제되어도 데이터가 남아있는다.
COPY . . 

#RUN pip install --no-cache-dir -r /workspace/animegan2-pytorch/requirements.txt // 아래에서 pytorch 및 torchvision 수동 설치

RUN wget https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl && \
    pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl && \
    rm torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

RUN git clone --branch release/0.16 https://github.com/pytorch/vision torchvision && \
    cd torchvision &&\
    pip install . -v --no-build-isolation &&\
    cd ../ &&\
    rm -rf torchvision

CMD ["/bin/bash"]