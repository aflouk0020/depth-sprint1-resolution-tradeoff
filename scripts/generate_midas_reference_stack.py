#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
import torch
import yaml


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_config():
    with open(repo_root() / "configs/experiment.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pad_to_multiple_of_32(rgb):
    h, w = rgb.shape[:2]
    new_h = int(np.ceil(h / 32) * 32)
    new_w = int(np.ceil(w / 32) * 32)

    padded = cv2.copyMakeBorder(
        rgb,
        0,
        new_h - h,
        0,
        new_w - w,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded, (h, w)


def crop_back(depth, shape):
    h, w = shape
    return depth[:h, :w]


def main():
    root = repo_root()
    cfg = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device != "cuda":
        raise RuntimeError("This script must be run on Lee's RTX CUDA machine.")

    video_path = root / cfg["dataset"]["master_video_path"]

    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()

    warmup = cfg["timing"]["warmup_frames"]
    measure = cfg["timing"]["measure_frames"]
    usable = frames[warmup:warmup + measure]

    if len(usable) != measure:
        raise RuntimeError(f"Expected {measure} measured frames, got {len(usable)}")

    print("Frames used:", len(usable))

    model = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        trust_repo=True,
    )
    model.to(device)
    model.eval()

    stack = []

    for idx, frame in enumerate(usable, start=1):
        resized = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        padded, shape = pad_to_multiple_of_32(rgb)

        tensor = (
            torch.from_numpy(padded)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        ).to(device)

        with torch.inference_mode():
            depth = model(tensor)

        depth = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)
        depth = crop_back(depth, shape)

        stack.append(depth)
        print(f"Processed frame {idx}/{len(usable)}")

    stack = np.stack(stack)

    out = root / "reference_depth/reference_depth_midasv2_1080_stack.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, stack)

    print("Saved reference stack:", out)
    print("Shape:", stack.shape)


if __name__ == "__main__":
    main()
