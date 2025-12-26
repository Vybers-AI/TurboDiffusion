FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git git-lfs \
    ffmpeg \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

# Torch 2.8.0 is recommended by TurboDiffusion README (avoid >2.8.0 due to OOM risk).
# Install CUDA 12.1 wheels.
RUN pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Optional: set arch list for common RunPod GPUs (A100=8.0, L40S=8.9, H100=9.0)
# This reduces compile time & binary size.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"

# Hugging Face cache dirs (helps if you later mount a volume)
ENV HF_HOME=/models/hf
ENV TORCH_HOME=/models/torch

WORKDIR /workspace

# Clone TurboDiffusion source (with submodules)
RUN git clone --recurse-submodules https://github.com/thu-ml/TurboDiffusion.git
WORKDIR /workspace/TurboDiffusion

# Patch setup.py to remove compute_120a / sm_120a (Blackwell) which often breaks on non-Blackwell builds.
# If you later deploy on Blackwell GPUs, remove this line.
RUN sed -i '/compute_120a,code=sm_120a/d' setup.py

# Install TurboDiffusion from source (build CUDA extension)
RUN pip install --no-cache-dir -e . --no-build-isolation

# Install SpargeAttn to enable SageSLA fast path (per README)
RUN pip install --no-cache-dir git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation

# RunPod + helper deps
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy handler
COPY rp_handler.py /workspace/rp_handler.py

CMD ["python3", "-u", "/workspace/rp_handler.py"]
