FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System deps (+ ninja/cmake helps torch cpp_extension builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg ca-certificates build-essential ninja-build cmake \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel

# ---- Target GPU: RTX 5090 ----
# Compile only what you need first. (If this builds, you can add other arch later.)
ENV TURBODIFFUSION_CUDA_ARCHS="120"

# Put caches on mounted network volume (rp_handler also sets these)
ENV HF_HOME=/runpod-volume/hf
ENV HF_HUB_CACHE=/runpod-volume/hf/hub
ENV TORCH_HOME=/runpod-volume/torch
ENV TMPDIR=/runpod-volume/tmp
ENV CUDA_MODULE_LOADING=LAZY

WORKDIR /workspace

# Sanity prints (shows nvcc + torch CUDA runtime)
RUN nvcc --version && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Clone TurboDiffusion w/ submodules
RUN git clone https://github.com/thu-ml/TurboDiffusion.git /workspace/TurboDiffusion \
 && cd /workspace/TurboDiffusion \
 && git submodule update --init --recursive

WORKDIR /workspace/TurboDiffusion

# (Optional) If your setup.py STILL contains sm_120a and your nvcc chokes on it,
# you can delete it. This is safe for 5090 runtime.
RUN sed -i '/compute_120a,code=sm_120a/d' setup.py || true
RUN sed -i '/compute_120a,code=compute_120a/d' setup.py || true

# Build/install TurboDiffusion with verbose logs (so RunPod shows the REAL failure)
RUN rm -rf build \
 && pip install -v --no-cache-dir -e . --no-build-isolation

# SageSLA dependency
RUN pip install --no-cache-dir git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation

# RunPod deps
COPY deploy/runpod/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Ensure HF versions compatible (you already pinned these, but keep explicit)
RUN pip install --no-cache-dir -U "huggingface-hub>=0.34.0,<1.0" "transformers>=4.45.0,<5" "accelerate>=0.34.0,<1"

# Handler
COPY deploy/runpod/rp_handler.py /workspace/rp_handler.py

CMD ["python", "-u", "/workspace/rp_handler.py"]
