FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel

# -----------------------------------------------------------------------------
# IMPORTANT: Compile TurboDiffusion CUDA ops for RTX 5090 (sm_120)
# You can include additional fallbacks if your endpoint might land on other GPUs.
# -----------------------------------------------------------------------------
ENV TURBODIFFUSION_CUDA_ARCHS="120,120a,90,89,86,80"

# Put HF/Torch caches on mounted network volume by default on RunPod
# (rp_handler.py also forces these, but it's nice to keep consistent)
ENV HF_HOME=/runpod-volume/hf
ENV HF_HUB_CACHE=/runpod-volume/hf/hub
ENV TORCH_HOME=/runpod-volume/torch
ENV TMPDIR=/runpod-volume/tmp
ENV CUDA_MODULE_LOADING=LAZY

WORKDIR /workspace

# -----------------------------------------------------------------------------
# Clone TurboDiffusion with submodules
# -----------------------------------------------------------------------------
RUN git clone https://github.com/thu-ml/TurboDiffusion.git /workspace/TurboDiffusion \
 && cd /workspace/TurboDiffusion \
 && git submodule update --init --recursive

WORKDIR /workspace/TurboDiffusion

# -----------------------------------------------------------------------------
# Patch setup.py to a dynamic arch build (so it respects TURBODIFFUSION_CUDA_ARCHS)
# This avoids surprises if the upstream setup.py changes again.
# -----------------------------------------------------------------------------
RUN python - <<'PY'
from pathlib import Path
p = Path("setup.py")
s = p.read_text()

# If we've already patched (idempotent), do nothing.
if "TURBODIFFUSION_CUDA_ARCHS" in s:
    print("setup.py already patched for dynamic archs")
    raise SystemExit(0)

# Minimal patch: replace the static cc_flag definition block with a dynamic one.
# We look for "cc_flag = [" and replace until the closing "]" of that list.
import re

pattern = r"cc_flag\s*=\s*\[\s*.*?\]\s*"
m = re.search(pattern, s, flags=re.S)
if not m:
    raise RuntimeError("Could not find cc_flag list in setup.py to patch")

dynamic_block = r'''
import os

def _parse_arch_list() -> list[str]:
    raw = os.environ.get("TURBODIFFUSION_CUDA_ARCHS", "120,120a,90,89,86,80")
    return [a.strip() for a in raw.split(",") if a.strip()]

def _gencode_flags(archs: list[str]) -> list[str]:
    flags = []
    for a in archs:
        if a.endswith("a"):
            base = a[:-1]
            flags += ["-gencode", f"arch=compute_{base}a,code=sm_{base}a"]
            flags += ["-gencode", f"arch=compute_{base}a,code=compute_{base}a"]
        else:
            flags += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
            flags += ["-gencode", f"arch=compute_{a},code=compute_{a}"]
    return flags

cc_flag = _gencode_flags(_parse_arch_list())
'''

# Insert dynamic helper block right above cc_flag replacement location
start, end = m.span()
s = s[:start] + dynamic_block + s[end:]

p.write_text(s)
print("Patched setup.py to use TURBODIFFUSION_CUDA_ARCHS + PTX")
PY

# -----------------------------------------------------------------------------
# Install TurboDiffusion from source (builds CUDA extension)
# Force a clean rebuild by removing old build artifacts.
# -----------------------------------------------------------------------------
RUN rm -rf /workspace/TurboDiffusion/build \
 && pip install --no-cache-dir -e . --no-build-isolation

# SageSLA fast path dependency per README
RUN pip install --no-cache-dir git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation

# -----------------------------------------------------------------------------
# RunPod deps
# -----------------------------------------------------------------------------
COPY deploy/runpod/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Fix HF dependency mismatch required by transformers
RUN pip install --no-cache-dir -U "huggingface-hub>=0.34.0,<1.0" "transformers>=4.45.0"

# Handler
COPY deploy/runpod/rp_handler.py /workspace/rp_handler.py

# Entrypoint
CMD ["python", "-u", "/workspace/rp_handler.py"]
