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
# We persist checkpoints + outputs + temp downloads there to avoid "No space left".
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

# Force temp files to the network volume (critical for hf_hub_download temp writes)
os.environ.setdefault("TMPDIR", str(TMP_DIR))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))

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

    vae_path = Path(CHECKPOINT_DIR / WAN_VAE)
    if not vae_path.exists():
        downloaded = hf_hub_download(
            repo_id=WAN_BASE_REPO,
            filename=WAN_VAE,
            cache_dir=str(HF_HOME),
            local_dir=str(CHECKPOINT_DIR),
            local_dir_use_symlinks=False,
        )
        vae_path = Path(downloaded)

    t5_path = Path(CHECKPOINT_DIR / UMT5)
    if not t5_path.exists():
        downloaded = hf_hub_download(
            repo_id=WAN_BASE_REPO,
            filename=UMT5,
            cache_dir=str(HF_HOME),
            local_dir=str(CHECKPOINT_DIR),
            local_dir_use_symlinks=False,
        )
        t5_path = Path(downloaded)

    ckpt_name = TURBOWAN_QUANT if use_quant else TURBOWAN_FP
    dit_path = Path(CHECKPOINT_DIR / ckpt_name)
    if not dit_path.exists():
        downloaded = hf_hub_download(
            repo_id=TURBOWAN_REPO,
            filename=ckpt_name,
            cache_dir=str(HF_HOME),
            local_dir=str(CHECKPOINT_DIR),
            local_dir_use_symlinks=False,
        )
        dit_path = Path(downloaded)

    return dit_path, vae_path, t5_path


def run_inference(args: Dict[str, Any]) -> Dict[str, Any]:
    prompt = args.get("prompt", "A cinematic corgi surfing at golden hour, film grain.")
    num_steps = int(args.get("num_steps", 4))          # 1–4
    num_frames = int(args.get("num_frames", 81))
    aspect_ratio = args.get("aspect_ratio", "16:9")
    seed = int(args.get("seed", 0))
    sla_topk = float(args.get("sla_topk", 0.15))       # README suggests ~0.15 for quality
    attention_type = args.get("attention_type", "sagesla")  # original|sla|sagesla

    # Decide quant vs unquant:
    # Rule-of-thumb: <40GB VRAM => quant (add --quant_linear), else unquant
    force_quant = args.get("force_quant", None)
    if force_quant is None:
        use_quant = gpu_mem_gb() < 40.0
    else:
        use_quant = bool(force_quant)

    # Helpful debug info for disk issues
    pre_disk = {
        "free_root_gb": round(disk_free_gb(Path("/")), 2),
        "free_tmp_gb": round(disk_free_gb(Path("/tmp")), 2),
        "free_persist_gb": round(disk_free_gb(PERSIST_ROOT), 2),
        "persist_root": str(PERSIST_ROOT),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "tmpdir": os.environ.get("TMPDIR", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE", ""),
    }

    try:
        dit_path, vae_path, t5_path = download_required_checkpoints(use_quant=use_quant)
    except Exception as e:
        return {
            "ok": False,
            "stage": "download_checkpoints",
            "error_type": str(type(e)),
            "error_message": str(e),
            "use_quant": use_quant,
            "gpu_mem_gb": round(gpu_mem_gb(), 2),
            "disk": pre_disk,
        }

    out_name = args.get("save_name", f"generated_video_seed{seed}.mp4")
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
        "--dit_path", str(dit_path),
        "--vae_path", str(vae_path),
        "--text_encoder_path", str(t5_path),
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
            "stage": "inference",
            "use_quant": use_quant,
            "gpu_mem_gb": round(gpu_mem_gb(), 2),
            "disk": pre_disk,
            "log_tail": proc.stdout[-8000:],
        }

    # Base64 return works for MVP but can get large; consider uploading and returning a URL later.
    video_b64 = base64.b64encode(save_path.read_bytes()).decode("utf-8")
    return {
        "ok": True,
        "use_quant": use_quant,
        "gpu_mem_gb": round(gpu_mem_gb(), 2),
        "video_mime": "video/mp4",
        "video_base64": video_b64,
        "disk": pre_disk,
        "log_tail": proc.stdout[-2000:],
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    # RunPod standard schema: {"input": {...}}
    inp = event.get("input", {}) or {}
    return run_inference(inp)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
