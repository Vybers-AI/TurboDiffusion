FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel

# Optional: compile only for common RunPod GPUs (A100=8.0, L40S=8.9, H100=9.0)
ENV TORCH_CUDA_ARCH_LIST="12.0;12.0a+PTX"

# HF cache dirs (mount a volume to /models later if you want)
ENV HF_HOME=/models/hf
ENV TORCH_HOME=/models/torch

WORKDIR /workspace

# Clone TurboDiffusion with submodules
RUN git clone --recurse-submodules https://github.com/thu-ml/TurboDiffusion.git
WORKDIR /workspace/TurboDiffusion

# Patch setup.py to remove sm_120a which can break builds on builders without that nvcc support
# (Keep this unless you're specifically targeting Blackwell on the build environment)
# RUN sed -i '/compute_120a,code=sm_120a/d' setup.py

# Install TurboDiffusion (builds CUDA extension)
RUN pip install --no-cache-dir -e . --no-build-isolation

# SageSLA fast path dependency per README
RUN pip install --no-cache-dir git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation

# RunPod deps
COPY deploy/runpod/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Handler
COPY deploy/runpod/rp_handler.py /workspace/rp_handler.py

CMD ["python", "-u", "/workspace/rp_handler.py"]

# Fix HF dependency mismatch required by transformers
RUN pip install --no-cache-dir -U "huggingface-hub>=0.34.0,<1.0" "transformers>=4.45.0"
