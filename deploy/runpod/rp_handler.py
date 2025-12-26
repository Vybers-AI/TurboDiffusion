import base64
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

import torch
import runpod
from huggingface_hub import hf_hub_download

# ---- Paths inside container ----
TD_ROOT = Path("/workspace/TurboDiffusion")
CHECKPOINT_DIR = TD_ROOT / "checkpoints"
OUTPUT_DIR = TD_ROOT / "output"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def download_required_checkpoints(use_quant: bool) -> Path:
    """
    Downloads:
      - Wan2.1_VAE.pth
      - models_t5_umt5-xxl-enc-bf16.pth
      - TurboWan2.1-T2V-14B-720P(.pth or -quant.pth)
    into ./checkpoints
    """
    hf_hub_download(
        repo_id=WAN_BASE_REPO,
        filename=WAN_VAE,
        local_dir=str(CHECKPOINT_DIR),
        local_dir_use_symlinks=False,
    )
    hf_hub_download(
        repo_id=WAN_BASE_REPO,
        filename=UMT5,
        local_dir=str(CHECKPOINT_DIR),
        local_dir_use_symlinks=False,
    )

    ckpt_name = TURBOWAN_QUANT if use_quant else TURBOWAN_FP
    ckpt_path = hf_hub_download(
        repo_id=TURBOWAN_REPO,
        filename=ckpt_name,
        local_dir=str(CHECKPOINT_DIR),
        local_dir_use_symlinks=False,
    )
    return Path(ckpt_path)


def run_inference(args: Dict[str, Any]) -> Dict[str, Any]:
    prompt = args.get("prompt", "A cinematic corgi surfing at golden hour, film grain.")
    num_steps = int(args.get("num_steps", 4))          # 1–4
    num_frames = int(args.get("num_frames", 81))
    aspect_ratio = args.get("aspect_ratio", "16:9")
    seed = int(args.get("seed", 0))
    sla_topk = float(args.get("sla_topk", 0.15))       # README recommends ~0.15 for quality
    attention_type = args.get("attention_type", "sagesla")  # original|sla|sagesla

    # Decide quant vs unquant
    # README rule: >40GB VRAM => unquant (no --quant_linear); else quant + --quant_linear
    force_quant = args.get("force_quant", None)
    if force_quant is None:
        use_quant = gpu_mem_gb() < 40.0
    else:
        use_quant = bool(force_quant)

    ckpt_path = download_required_checkpoints(use_quant=use_quant)

    out_name = args.get("save_name", "generated_video.mp4")
    save_path = OUTPUT_DIR / out_name
    if save_path.exists():
        save_path.unlink()

    # Ensure PYTHONPATH as README shows: export PYTHONPATH=turbodiffusion
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TD_ROOT / "turbodiffusion")

    cmd = [
        "python3",
        "turbodiffusion/inference/wan2.1_t2v_infer.py",
        "--model", "Wan2.1-14B",
        "--dit_path", str(ckpt_path),
        "--resolution", "720p",
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
            "use_quant": use_quant,
            "gpu_mem_gb": round(gpu_mem_gb(), 2),
            "log_tail": proc.stdout[-6000:],
        }

    video_b64 = base64.b64encode(save_path.read_bytes()).decode("utf-8")
    return {
        "ok": True,
        "use_quant": use_quant,
        "gpu_mem_gb": round(gpu_mem_gb(), 2),
        "video_mime": "video/mp4",
        "video_base64": video_b64,
        "log_tail": proc.stdout[-2000:],
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    # RunPod standard schema: {"input": {...}}
    inp = event.get("input", {}) or {}
    return run_inference(inp)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
