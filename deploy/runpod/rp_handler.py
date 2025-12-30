import base64
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import runpod
from huggingface_hub import hf_hub_download


# -----------------------------------------------------------------------------
# Storage layout
# -----------------------------------------------------------------------------
# On RunPod Serverless, a mounted Network Volume appears at /runpod-volume.
# Persist checkpoints + outputs + HF cache + temp there.
PERSIST_ROOT = Path(os.environ.get("PERSIST_ROOT", "/runpod-volume"))

# TurboDiffusion repo lives in the container image
TD_ROOT = Path("/workspace/TurboDiffusion")

# Persisted directories (on network volume)
CHECKPOINT_DIR = PERSIST_ROOT / "checkpoints"
OUTPUT_DIR = PERSIST_ROOT / "outputs"
TMP_DIR = PERSIST_ROOT / "tmp"

# Hugging Face caches (also on network volume)
HF_HOME = Path(os.environ.get("HF_HOME", str(PERSIST_ROOT / "hf")))
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", str(HF_HOME / "hub")))
TORCH_HOME = Path(os.environ.get("TORCH_HOME", str(PERSIST_ROOT / "torch")))

for p in [CHECKPOINT_DIR, OUTPUT_DIR, TMP_DIR, HF_HOME, HF_HUB_CACHE, TORCH_HOME]:
    p.mkdir(parents=True, exist_ok=True)

# Critical: make sure temp downloads go to the mounted volume
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["TORCH_HOME"] = str(TORCH_HOME)

# Helpful for CUDA module loading on some environments
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

# ---- HF repos/files (from TurboDiffusion README) ----
WAN_BASE_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
WAN_VAE = "Wan2.1_VAE.pth"
UMT5 = "models_t5_umt5-xxl-enc-bf16.pth"

TURBOWAN_REPO = "TurboDiffusion/TurboWan2.1-T2V-14B-720P"
TURBOWAN_FP = "TurboWan2.1-T2V-14B-720P.pth"
TURBOWAN_QUANT = "TurboWan2.1-T2V-14B-720P-quant.pth"


def gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "no-cuda"
    return torch.cuda.get_device_name(0)


def gpu_cc() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def disk_free_gb(path: Path) -> float:
    try:
        total, used, free = shutil.disk_usage(str(path))
        return free / (1024 ** 3)
    except Exception:
        return -1.0


def download_required_checkpoints(use_quant: bool) -> Tuple[Path, Path, Path]:
    """
    Downloads (if missing) into CHECKPOINT_DIR on the network volume:
      - Wan2.1_VAE.pth
      - models_t5_umt5-xxl-enc-bf16.pth
      - TurboWan2.1-T2V-14B-720P(.pth or -quant.pth)
    Returns: (dit_path, vae_path, t5_path)
    """
    # Ensure tmpdir is on volume even if something resets env
    os.environ["TMPDIR"] = str(TMP_DIR)

    vae_path = CHECKPOINT_DIR / WAN_VAE
    if not vae_path.exists():
        vae_path = Path(
            hf_hub_download(
                repo_id=WAN_BASE_REPO,
                filename=WAN_VAE,
                cache_dir=str(HF_HOME),
                local_dir=str(CHECKPOINT_DIR),
                local_dir_use_symlinks=False,
            )
        )

    t5_path = CHECKPOINT_DIR / UMT5
    if not t5_path.exists():
        t5_path = Path(
            hf_hub_download(
                repo_id=WAN_BASE_REPO,
                filename=UMT5,
                cache_dir=str(HF_HOME),
                local_dir=str(CHECKPOINT_DIR),
                local_dir_use_symlinks=False,
            )
        )

    ckpt_name = TURBOWAN_QUANT if use_quant else TURBOWAN_FP
    dit_path = CHECKPOINT_DIR / ckpt_name
    if not dit_path.exists():
        dit_path = Path(
            hf_hub_download(
                repo_id=TURBOWAN_REPO,
                filename=ckpt_name,
                cache_dir=str(HF_HOME),
                local_dir=str(CHECKPOINT_DIR),
                local_dir_use_symlinks=False,
            )
        )

    return Path(dit_path), Path(vae_path), Path(t5_path)


def run_inference(args: Dict[str, Any]) -> Dict[str, Any]:
    prompt = args.get("prompt", "A cinematic corgi surfing at golden hour, film grain.")
    num_steps = int(args.get("num_steps", 4))          # 1–4
    num_frames = int(args.get("num_frames", 81))
    aspect_ratio = args.get("aspect_ratio", "16:9")
    seed = int(args.get("seed", 0))
    sla_topk = float(args.get("sla_topk", 0.15))       # README suggests ~0.15 for quality
    attention_type = args.get("attention_type", "sagesla")  # original|sla|sagesla
    resolution = args.get("resolution", "720p")

    # Quant decision: <40GB VRAM => quant + --quant_linear
    force_quant = args.get("force_quant", None)
    if force_quant is None:
        use_quant = gpu_mem_gb() < 40.0
    else:
        use_quant = bool(force_quant)

    disk_info = {
        "free_root_gb": round(disk_free_gb(Path("/")), 2),
        "free_tmp_gb": round(disk_free_gb(Path("/tmp")), 2),
        "free_persist_gb": round(disk_free_gb(PERSIST_ROOT), 2),
        "persist_root": str(PERSIST_ROOT),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "tmpdir": os.environ.get("TMPDIR", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE", ""),
    }

    # GPU debug
    gpu_info = {
        "gpu_name": gpu_name(),
        "gpu_mem_gb": round(gpu_mem_gb(), 2),
        "gpu_compute_capability": gpu_cc(),
        "use_quant": use_quant,
    }

    try:
        dit_path, vae_path, t5_path = download_required_checkpoints(use_quant=use_quant)
    except Exception as e:
        return {
            "ok": False,
            "stage": "download_checkpoints",
            "error_type": str(type(e)),
            "error_message": str(e),
            "disk": disk_info,
            **gpu_info,
        }

    out_name = args.get("save_name", f"generated_video_seed{seed}.mp4")
    save_path = OUTPUT_DIR / out_name
    if save_path.exists():
        save_path.unlink()

    env = os.environ.copy()
    # README uses: export PYTHONPATH=turbodiffusion (relative to TD_ROOT)
    env["PYTHONPATH"] = str(TD_ROOT / "turbodiffusion")

    cmd = [
        "python3",
        "turbodiffusion/inference/wan2.1_t2v_infer.py",
        "--model", "Wan2.1-14B",
        "--dit_path", str(dit_path),
        "--vae_path", str(vae_path),
        "--text_encoder_path", str(t5_path),
        "--resolution", resolution,
        "--prompt", prompt,
        "--num_samples", "1",
        "--num_steps", str(num_steps),
        "--num_frames", str(num_frames),
        "--aspect_ratio", aspect_ratio,
        "--seed", str(seed),
        "--save_path", str(save_path),
        "--attention_type", attention_type,
        "--sla_topk", str(sla_topk),
    ]
    if use_quant:
        cmd.append("--quant_linear")

    proc = subprocess.run(
        cmd,
        cwd=str(TD_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.returncode != 0 or not save_path.exists():
        return {
            "ok": False,
            "stage": "inference",
            "disk": disk_info,
            **gpu_info,
            "log_tail": proc.stdout[-12000:],
        }

    video_b64 = base64.b64encode(save_path.read_bytes()).decode("utf-8")
    return {
        "ok": True,
        "stage": "done",
        "disk": disk_info,
        **gpu_info,
        "video_mime": "video/mp4",
        "video_base64": video_b64,
        "log_tail": proc.stdout[-4000:],
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event.get("input", {}) or {}
    return run_inference(inp)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
