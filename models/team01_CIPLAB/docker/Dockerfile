FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/envs/sr_env/bin:/opt/conda/bin:$PATH
WORKDIR /workspace

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git curl wget ca-certificates tmux \
    bzip2 \
    build-essential ninja-build pkg-config \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
    && rm -f /tmp/miniconda.sh \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set channel_priority strict

RUN conda create -y -n sr_env python=3.10 pip

RUN conda run -n sr_env python -m pip install -U \
    "pip<25" \
    "setuptools<82" \
    wheel

RUN conda run -n sr_env pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

RUN conda run -n sr_env pip install \
    "numpy<2" \
    scipy \
    timm==0.9.16 \
    einops \
    pillow tqdm packaging pyyaml requests \
    "accelerate>=0.31.0" \
    "transformers>=4.41.2" \
    "peft>=0.11.1" \
    datasets \
    sentencepiece \
    ftfy \
    tensorboard \
    Jinja2 \
    bitsandbytes \
    prodigyopt \
    huggingface-hub \
    wandb \
    safetensors

RUN conda run -n sr_env pip install \
    opencv-python scikit-image \
    lmdb h5py addict future yapf \
    matplotlib

RUN conda run -n sr_env pip install --no-build-isolation \
    basicsr==1.3.4.9

RUN conda run -n sr_env pip install \
    "diffusers @ git+https://github.com/huggingface/diffusers.git"


FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/envs/sr_env/bin:/opt/conda/bin:$PATH
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git curl wget ca-certificates tmux \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda

RUN mkdir -p /opt/conda/envs/sr_env/etc/conda/activate.d \
    /opt/conda/envs/sr_env/etc/conda/deactivate.d \
    && printf '%s\n' \
    '#!/bin/sh' \
    'export _CONDA_OLD_TORCH_CUDNN_V8_API_DISABLED="${TORCH_CUDNN_V8_API_DISABLED-__UNSET__}"' \
    'export TORCH_CUDNN_V8_API_DISABLED=1' \
    > /opt/conda/envs/sr_env/etc/conda/activate.d/torch_cudnn_v8_api_disabled.sh \
    && printf '%s\n' \
    '#!/bin/sh' \
    'if [ "${_CONDA_OLD_TORCH_CUDNN_V8_API_DISABLED}" = "__UNSET__" ]; then' \
    '    unset TORCH_CUDNN_V8_API_DISABLED' \
    'else' \
    '    export TORCH_CUDNN_V8_API_DISABLED="${_CONDA_OLD_TORCH_CUDNN_V8_API_DISABLED}"' \
    'fi' \
    'unset _CONDA_OLD_TORCH_CUDNN_V8_API_DISABLED' \
    > /opt/conda/envs/sr_env/etc/conda/deactivate.d/torch_cudnn_v8_api_disabled.sh

RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate sr_env" >> /root/.bashrc

CMD ["bash"]
