"""
Copyright (c) 2025 by TurboDiffusion team.
Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025turbodiffusion,
  title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
  author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.16093},
  year={2025}
}
"""

import os
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ops_dir = Path(__file__).parent / "turbodiffusion" / "ops"
cutlass_dir = ops_dir / "cutlass"

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=--verbose,--warn-on-local-memory-usage",
    "-lineinfo",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-Xcompiler",
    "-fPIC",
]

def _parse_arch_list() -> list[str]:
    """
    Arch list can be overridden by env var TURBODIFFUSION_CUDA_ARCHS.
    Examples:
      TURBODIFFUSION_CUDA_ARCHS="120,120a,90,89,86,80"
      TURBODIFFUSION_CUDA_ARCHS="120"
    Default favors RTX 5090 serverless: 120 + 120a + a few common fallbacks.
    """
    raw = os.environ.get("TURBODIFFUSION_CUDA_ARCHS", "120,120a,90,89,86,80")
    archs = [a.strip() for a in raw.split(",") if a.strip()]
    return archs

def _gencode_flags(archs: list[str]) -> list[str]:
    flags = []
    for a in archs:
        # allow "120a" style and normal ints
        if a.endswith("a"):
            base = a[:-1]
            flags += ["-gencode", f"arch=compute_{base}a,code=sm_{base}a"]
            # also emit PTX for forward-compat
            flags += ["-gencode", f"arch=compute_{base}a,code=compute_{base}a"]
        else:
            flags += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
            # also emit PTX for forward-compat
            flags += ["-gencode", f"arch=compute_{a},code=compute_{a}"]
    return flags

cc_flag = _gencode_flags(_parse_arch_list())

ext_modules = [
    CUDAExtension(
        name="turbo_diffusion_ops",
        sources=[
            "turbodiffusion/ops/bindings.cpp",
            "turbodiffusion/ops/quant/quant.cu",
            "turbodiffusion/ops/norm/rmsnorm.cu",
            "turbodiffusion/ops/norm/layernorm.cu",
            "turbodiffusion/ops/gemm/gemm.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags + ["-DEXECMODE=0"] + cc_flag + ["--threads", "4"],
        },
        include_dirs=[
            cutlass_dir / "include",
            cutlass_dir / "tools" / "util" / "include",
            ops_dir,
        ],
        libraries=["cuda"],
    )
]

setup(
    packages=find_packages(exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks")),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
