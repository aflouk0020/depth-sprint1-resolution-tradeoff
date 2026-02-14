#!/usr/bin/env python3
"""
Sprint 1 - Reference Depth Generation (LOCKED)
Reference: 1920x1080 landscape

- Loads model via HF pipeline with pinned revision from configs/experiment.yaml
- Reads one frame after skipping warmup frames
- Runs inference (inference-only timing)
- Saves reference depth as .npy float32 (NO normalisation)
- Saves a preview .png and timing .txt

Run:
  python scripts/02_depthanything_reference_1920x1080.py
"""

from __future__ import annotations

from pathlib import Path
import os
import time
import yaml
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict:
    root = repo_root()
    cfg_path = root / "configs" / "experiment.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(root: Path) -> None:
    (root / "outputs" / "baseline").mkdir(parents=True, exist_ok=True)


def to_uint8_preview(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (d * 255).astype(np.uint8)


def pick_device(cfg: dict) -> str:
    for d in cfg["model"]["device_priority"]:
        if d == "cuda" and torch.cuda.is_available():
            return "cuda"
        if d == "mps" and torch.backends.mps.is_available():
            return "mps"
        if d == "cpu":
            return "cpu"
    return "cpu"


def build_pipe(cfg: dict, device: str):
    model_id = cfg["model"]["model_id"]
    revision = cfg["model"]["revision"]

    # HF pipeline expects PIL / path / url (NOT raw numpy array)
    return pipeline(
        task="depth-estimation",
        model=model_id,
        revision=revision,
        device=0 if device == "cuda" else -1,
    )


def cuda_sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    root = repo_root()
    cfg = load_config()
    ensure_dirs(root)

    video_path = (root / cfg["dataset"]["master_video_path"]).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = pick_device(cfg)
    print(f"Device: {device}")

    depth_pipe = build_pipe(cfg, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    warmup = int(cfg["timing"]["warmup_frames"])
    for _ in range(warmup):
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Not enough frames to skip warmup.")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read a frame for reference.")

    # OpenCV -> RGB -> PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_pil = Image.fromarray(rgb)

    # inference-only timing
    cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    out = depth_pipe(rgb_pil)
    cuda_sync_if_needed(device)
    t1 = time.perf_counter()

    infer_s = (t1 - t0)
    approx_fps = (1.0 / infer_s) if infer_s > 0 else 0.0

    pred = out["predicted_depth"]  # torch tensor [1, H, W] or [H, W]
    depth = pred.squeeze().detach().cpu().numpy().astype(np.float32)

    # output paths (must match YAML)
    npy_path = (root / cfg["outputs"]["baseline"]["depth_npy"]).resolve()
    png_path = (root / cfg["outputs"]["baseline"]["preview_png"]).resolve()
    txt_path = (root / cfg["outputs"]["baseline"]["timing_txt"]).resolve()

    npy_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(npy_path, depth)  # float32, no normalisation
    cv2.imwrite(str(png_path), to_uint8_preview(depth))

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"model_id: {cfg['model']['model_id']}\n")
        f.write(f"revision: {cfg['model']['revision']}\n")
        f.write(f"device: {device}\n")
        f.write(f"inference_seconds: {infer_s:.6f}\n")
        f.write(f"approx_fps: {approx_fps:.2f}\n")

    print("Saved:")
    print(f"- {npy_path}")
    print(f"- {png_path}")
    print(f"- {txt_path}")
    print(f"Inference: {infer_s:.4f}s | Approx FPS: {approx_fps:.2f}")


if __name__ == "__main__":
    main()
