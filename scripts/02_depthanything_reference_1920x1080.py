#!/usr/bin/env python3
"""
Sprint 2 - Depth Anything 1080p Reference Stack Generation

Purpose:
Generate the frozen 1920x1080 transformer reference stack used by
Sprint-2 Plan-1. This script must follow the same current preprocessing
path as `03_resolution_sweep_metrics_timing.py` for the Depth Anything
model so that the 1080p transformer row trends to zero error against its
own reference baseline.

Run on the RTX machine:
  python scripts/02_depthanything_reference_1920x1080.py
"""

from __future__ import annotations

from pathlib import Path
import random
import hashlib

import cv2
import numpy as np
import torch
import yaml
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ------------------------------------------------------------
# Determinism / reproducibility lock
# ------------------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# Repo / config helpers
# ------------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(repo_root() / "configs" / "experiment.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------
# Device / preprocessing helpers
# ------------------------------------------------------------
def pick_device(cfg: dict) -> str:
    for d in cfg["model"]["device_priority"]:
        if d == "cuda" and torch.cuda.is_available():
            return "cuda"
        if d == "mps" and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def prepare_rgb_frame(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def prepare_depthanything_input(rgb: np.ndarray, processor, device: str):
    inputs = processor(
        images=rgb,
        return_tensors="pt",
        do_resize=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def to_uint8_preview(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (d * 255).astype(np.uint8)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
@torch.inference_mode()
def main() -> None:
    root = repo_root()
    cfg = load_config()

    device = pick_device(cfg)
    print("Seed:", SEED)
    print("Device:", device)

    if device != "cuda":
        raise RuntimeError(
            "This Sprint-2 transformer reference stack must be generated on the RTX CUDA machine."
        )

    ref_w, ref_h = cfg["resolutions"]["reference"]
    video_path = root / cfg["dataset"]["master_video_path"]

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf",
        revision="5426e4f0f36572d16453bbda7a8389317b1bef99",
        use_fast=False,
    )

    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf",
        revision="5426e4f0f36572d16453bbda7a8389317b1bef99",
    )
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    warmup = int(cfg["timing"]["warmup_frames"])
    measure = int(cfg["timing"]["measure_frames"])
    usable = frames[warmup:warmup + measure]

    if len(usable) != measure:
        raise RuntimeError(f"Expected {measure} measured frames, got {len(usable)}")

    print("Frames used:", len(usable))

    stack = []
    preview_depth = None

    for idx, frame in enumerate(usable, start=1):
        rgb = prepare_rgb_frame(frame, ref_w, ref_h)
        inputs = prepare_depthanything_input(rgb, processor, device)

        outputs = model(**inputs)
        depth = outputs.predicted_depth
        depth = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)

        depth_ref_size = cv2.resize(
            depth,
            (ref_w, ref_h),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)

        if preview_depth is None:
            preview_depth = depth_ref_size.copy()

        stack.append(depth_ref_size)

        if idx == 1 or idx == len(usable) or idx % 10 == 0:
            print(f"Processed frame {idx}/{len(usable)}")

    stack = np.stack(stack, axis=0).astype(np.float32)

    ref_dir = root / "reference_depth"
    ref_dir.mkdir(parents=True, exist_ok=True)

    out_npy = ref_dir / "reference_depth_depthanything_1080_stack.npy"
    out_png = ref_dir / "reference_depth_depthanything_1080_preview.png"
    out_txt = ref_dir / "reference_depth_depthanything_1080_info.txt"

    np.save(out_npy, stack)
    cv2.imwrite(str(out_png), to_uint8_preview(preview_depth))

    video_sha = sha256_file(video_path)
    ref_sha = sha256_file(out_npy)

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("model_label: depthanything\n")
        f.write("model_id: depth-anything/Depth-Anything-V2-Small-hf\n")
        f.write("revision: 5426e4f0f36572d16453bbda7a8389317b1bef99\n")
        f.write(f"device: {device}\n")
        f.write(f"video_path: {video_path}\n")
        f.write(f"master_video_sha256: {video_sha}\n")
        f.write(f"reference_depth_sha256: {ref_sha}\n")
        f.write(f"frames_n: {len(usable)}\n")
        f.write(f"stack_shape: {stack.shape}\n")
        f.write(f"stack_dtype: {stack.dtype}\n")
        f.write("preprocessing: current Sprint-2 Depth Anything path\n")
        f.write("do_resize: False\n")
        f.write("resize_interpolation: bicubic\n")

    print("Saved:")
    print(f"- {out_npy}")
    print(f"- {out_png}")
    print(f"- {out_txt}")
    print("Reference stack shape:", stack.shape, "dtype:", stack.dtype)
    print("Reference SHA256:", ref_sha)


if __name__ == "__main__":
    main()
