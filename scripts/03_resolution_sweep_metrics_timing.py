#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import math
import subprocess
import time

import cv2
import numpy as np
import torch
import yaml
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_config():
    with open(repo_root() / "configs/experiment.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_git_short_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                cwd=repo_root(),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def pick_device(cfg):
    for d in cfg["model"]["device_priority"]:
        if d == "cuda" and torch.cuda.is_available():
            return "cuda"
        if d == "mps" and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def get_device_name(device):
    if device == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda"
    if device == "mps":
        return "Apple Silicon MPS"
    return "cpu"


def pad_to_multiple_of_32(rgb):
    h, w = rgb.shape[:2]

    new_h = int(np.ceil(h / 32) * 32)
    new_w = int(np.ceil(w / 32) * 32)

    pad_bottom = new_h - h
    pad_right = new_w - w

    padded = cv2.copyMakeBorder(
        rgb,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    return padded, (h, w)


def crop_back(depth, shape):
    h, w = shape
    return depth[:h, :w]


def build_roi_mask(h, w):
    """
    Locked ROI from Sprint-1 / Sprint-2:
    bottom 40% x centre 60%
    """
    y_start = int(math.floor(0.60 * h))
    x_start = int(math.floor(0.20 * w))
    x_end = int(math.floor(0.80 * w))

    mask = np.zeros((h, w), dtype=bool)
    mask[y_start:h, x_start:x_end] = True
    return mask


def rmse(pred, ref, mask=None):
    diff = pred.astype(np.float32) - ref.astype(np.float32)
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def absrel(pred, ref, mask=None, eps=1e-6):
    p = pred.astype(np.float32)
    r = ref.astype(np.float32)

    if mask is not None:
        p = p[mask]
        r = r[mask]

    return float(np.mean(np.abs(p - r) / (np.abs(r) + eps)))


def load_model_and_processor(spec, device):
    """
    Keeps current working model-loading behaviour to avoid breaking your pipeline.

    Important:
    - depthanything uses Hugging Face with pinned revision
    - midasv2 currently uses torch.hub MiDaS_small as in your working script

    If later you want full model-lock alignment to qualcomm/Midas-V2 on HF,
    that is a separate loader change.
    """

    if spec["label"] == "midasv2":
        model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo=True,
        )
        model.to(device)
        model.eval()
        return model, None

    processor = AutoImageProcessor.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
        use_fast=False,
    )

    model = AutoModelForDepthEstimation.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
    )

    model.to(device)
    model.eval()

    return model, processor


@torch.inference_mode()
def run_model(model, inputs, label):
    if label == "midasv2":
        output = model(inputs)
        depth = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return depth

    outputs = model(**inputs)
    depth = outputs.predicted_depth
    depth = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return depth


def measure_inference(model, inp, label, device):
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        depth = run_model(model, inp, label)
        end.record()

        torch.cuda.synchronize()
        infer_ms = start.elapsed_time(end)
        return depth, infer_ms

    t0 = time.perf_counter()
    depth = run_model(model, inp, label)
    t1 = time.perf_counter()

    infer_ms = (t1 - t0) * 1000.0
    return depth, infer_ms


def main():
    root = repo_root()
    cfg = load_config()

    device = pick_device(cfg)
    backend = device
    device_name = get_device_name(device)
    git_hash = get_git_short_hash()

    print("Device:", device)

    ref_w, ref_h = cfg["resolutions"]["reference"]
    roi_mask = build_roi_mask(ref_h, ref_w)

    video_path = root / cfg["dataset"]["master_video_path"]
    master_video_sha256 = sha256_file(video_path)

    # Locked reference stack path
    ref_path = root / "reference_depth/reference_depth_1080_stack.npy"
    reference_depth_sha256 = sha256_file(ref_path)

    reference_stack = np.load(ref_path).astype(np.float32)

    if reference_stack.ndim != 3:
        raise ValueError(
            f"Expected reference stack shape (N,H,W), got {reference_stack.shape}"
        )

    if reference_stack.shape[1:] != (ref_h, ref_w):
        raise ValueError(
            f"Reference stack spatial shape mismatch. "
            f"Expected ({ref_h}, {ref_w}), got {reference_stack.shape[1:]}"
        )

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
        raise RuntimeError(
            f"Expected {measure} measured frames, got {len(usable)}"
        )

    if reference_stack.shape[0] != len(usable):
        raise RuntimeError(
            f"Reference stack frame count mismatch. "
            f"Expected {len(usable)}, got {reference_stack.shape[0]}"
        )

    out_dir = root / "outputs/csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in cfg["sprint2"]["models"]:
        label = spec["label"]

        print("\n======================")
        print("MODEL:", label)
        print("======================")

        model, processor = load_model_and_processor(spec, device)

        for w, h in cfg["resolutions"]["test_levels"]:
            # M1 safety skip
            if device == "mps" and label == "depthanything" and (w, h) == (1920, 1080):
                print("Skipping 1080p transformer on M1 due to memory limits")
                continue

            print("\nResolution:", w, "x", h)

            tensors = []
            shapes = []

            for frame in usable:
                resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                if label == "midasv2":
                    padded, shape = pad_to_multiple_of_32(rgb)

                    tensor = (
                        torch.from_numpy(padded)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        / 255.0
                    ).to(device)

                    tensors.append(tensor)
                    shapes.append(shape)

                else:
                    inputs = processor(
                        images=rgb,
                        return_tensors="pt",
                        do_resize=False,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    tensors.append(inputs)
                    shapes.append((h, w))

            # Mandatory processor bypass audit
            if label == "midasv2":
                print("inputs_shape =", tuple(tensors[0].shape))
            else:
                print("inputs_shape =", tuple(tensors[0]["pixel_values"].shape))

            times = []
            rmse_full_vals = []
            rmse_roi_vals = []
            absrel_full_vals = []
            absrel_roi_vals = []

            for i, inp in enumerate(tensors):
                depth, infer_ms = measure_inference(model, inp, label, device)
                times.append(infer_ms)

                if label == "midasv2":
                    depth = crop_back(depth, shapes[i])

                # Upscale prediction to locked reference size
                depth_ref_size = cv2.resize(
                    depth,
                    (ref_w, ref_h),
                    interpolation=cv2.INTER_CUBIC,
                ).astype(np.float32)

                ref_depth = reference_stack[i]

                rmse_full_vals.append(rmse(depth_ref_size, ref_depth))
                rmse_roi_vals.append(rmse(depth_ref_size, ref_depth, mask=roi_mask))
                absrel_full_vals.append(absrel(depth_ref_size, ref_depth))
                absrel_roi_vals.append(absrel(depth_ref_size, ref_depth, mask=roi_mask))

            # Keep same behaviour as your working script
            times_for_stats = times[1:] if len(times) > 1 else times

            mean_ms = float(np.mean(times_for_stats))
            std_ms = float(np.std(times_for_stats))
            fps = 1000.0 / mean_ms

            rmse_full_mean = float(np.mean(rmse_full_vals))
            rmse_roi_mean = float(np.mean(rmse_roi_vals))
            absrel_full_mean = float(np.mean(absrel_full_vals))
            absrel_roi_mean = float(np.mean(absrel_roi_vals))

            print("Mean Time (ms):", mean_ms)
            print("Std Time (ms):", std_ms)
            print("FPS:", fps)
            print("RMSE Full:", rmse_full_mean)
            print("RMSE ROI:", rmse_roi_mean)
            print("AbsRel Full:", absrel_full_mean)
            print("AbsRel ROI:", absrel_roi_mean)

            resolution_str = f"{w}x{h}"
            frames_n = len(usable)
            run_id = f"{label}_{backend}_{resolution_str}_{frames_n}_{git_hash}"
            out_csv = out_dir / f"s2_{run_id}.csv"

            # Use config metadata if present, otherwise safe fallback
            model_id = spec.get("model_id", label)
            revision = spec.get("revision", "not_recorded")

            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "resolution",
                        "mean_time_ms",
                        "std_time_ms",
                        "fps",
                        "rmse_full",
                        "rmse_roi",
                        "absrel_full",
                        "absrel_roi",
                        "device",
                        "backend",
                        "git_commit_hash",
                        "model_id",
                        "revision",
                        "frames_n",
                        "master_video_sha256",
                        "reference_depth_sha256",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "resolution": resolution_str,
                        "mean_time_ms": mean_ms,
                        "std_time_ms": std_ms,
                        "fps": fps,
                        "rmse_full": rmse_full_mean,
                        "rmse_roi": rmse_roi_mean,
                        "absrel_full": absrel_full_mean,
                        "absrel_roi": absrel_roi_mean,
                        "device": device_name,
                        "backend": backend,
                        "git_commit_hash": git_hash,
                        "model_id": model_id,
                        "revision": revision,
                        "frames_n": frames_n,
                        "master_video_sha256": master_video_sha256,
                        "reference_depth_sha256": reference_depth_sha256,
                    }
                )

            print("Results saved:", out_csv)


if __name__ == "__main__":
    main()
