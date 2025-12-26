# TurboDiffusion RunPod Worker

RunPod settings:
- Dockerfile Path: deploy/runpod/Dockerfile
- Build Context: deploy/runpod

Example input:
{
  "input": {
    "prompt": "POV drone shot over a neon city, raining, reflections, cinematic.",
    "num_steps": 4,
    "num_frames": 81,
    "seed": 123,
    "sla_topk": 0.15
  }
}
