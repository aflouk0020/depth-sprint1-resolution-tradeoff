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


# ------------------------------------------------------------
# Output policy
# ------------------------------------------------------------
WRITE_PER_RUN_CSVS = True
WRITE_MASTER_SUMMARY_CSV = True
WRITE_PAPER_TABLES = True


# ------------------------------------------------------------
# Paths / repo helpers
# ------------------------------------------------------------
def repo_root():
    return Path(__file__).resolve().parents[1]


def load_config():
    with open(repo_root() / "configs/experiment.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
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


# ------------------------------------------------------------
# Device helpers
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Image / metric helpers
# ------------------------------------------------------------
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
    Locked ROI:
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


# ------------------------------------------------------------
# Reference handling
# ------------------------------------------------------------
def get_reference_path(root: Path, label: str) -> Path:
    """
    Per-model frozen 1080p reference baseline.
    """
    mapping = {
        "depthanything": root / "reference_depth/reference_depth_depthanything_1080_stack.npy",
        "midasv2": root / "reference_depth/reference_depth_midasv2_1080_stack.npy",
    }

    if label not in mapping:
        raise KeyError(f"No reference mapping defined for model label: {label}")

    ref_path = mapping[label]
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Missing reference stack for {label}: {ref_path}\n"
            f"Expected per-model 1080p frozen reference."
        )

    return ref_path


def load_reference_stack(ref_path: Path, expected_h: int, expected_w: int, expected_n: int):
    reference_stack = np.load(ref_path).astype(np.float32)

    if reference_stack.ndim != 3:
        raise ValueError(
            f"Expected reference stack shape (N,H,W), got {reference_stack.shape}"
        )

    if reference_stack.shape[1:] != (expected_h, expected_w):
        raise ValueError(
            f"Reference stack spatial shape mismatch for {ref_path.name}. "
            f"Expected ({expected_h}, {expected_w}), got {reference_stack.shape[1:]}"
        )

    if reference_stack.shape[0] != expected_n:
        raise ValueError(
            f"Reference stack frame count mismatch for {ref_path.name}. "
            f"Expected {expected_n}, got {reference_stack.shape[0]}"
        )

    return reference_stack


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
def load_model_and_processor(spec, device):
    """
    Keeps your current working behaviour to avoid breaking the pipeline.

    depthanything:
      - Hugging Face
    midasv2:
      - current torch.hub MiDaS_small path from your working setup

    If you later switch midasv2 loader to a fully pinned HF implementation,
    that is a separate controlled change.
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


# ------------------------------------------------------------
# CSV writers
# ------------------------------------------------------------
CSV_FIELDS = [
    "model_label",
    "architecture",
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
    "reference_depth_file",
]


def write_csv_row(path: Path, row: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def write_master_summary_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_table_csv(path: Path, headers: list[str], rows: list[list]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
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

    out_csv_dir = root / "outputs/csv"
    out_tables_dir = root / "outputs/tables"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_tables_dir.mkdir(parents=True, exist_ok=True)

    master_rows = []

    for spec in cfg["sprint2"]["models"]:
        label = spec["label"]
        architecture = (
            "transformer" if label == "depthanything" else "cnn"
        )

        print("\n======================")
        print("MODEL:", label)
        print("======================")

        model, processor = load_model_and_processor(spec, device)

        # Per-model frozen reference
        ref_path = get_reference_path(root, label)
        reference_depth_sha256 = sha256_file(ref_path)
        reference_stack = load_reference_stack(
            ref_path,
            expected_h=ref_h,
            expected_w=ref_w,
            expected_n=len(usable),
        )

        for w, h in cfg["resolutions"]["test_levels"]:
            # M1 memory safeguard
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

            # Processor bypass audit
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

            # Same timing behaviour as before: discard first measured frame from stats
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

            row = {
                "model_label": label,
                "architecture": architecture,
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
                "model_id": spec.get("model_id", label),
                "revision": spec.get("revision", "not_recorded"),
                "frames_n": frames_n,
                "master_video_sha256": master_video_sha256,
                "reference_depth_sha256": reference_depth_sha256,
                "reference_depth_file": ref_path.name,
            }

            master_rows.append(row)

            if WRITE_PER_RUN_CSVS:
                out_csv = out_csv_dir / f"s2_{run_id}.csv"
                write_csv_row(out_csv, row)
                print("Per-run CSV saved:", out_csv)

    # --------------------------------------------------------
    # Consolidated outputs
    # --------------------------------------------------------
    if WRITE_MASTER_SUMMARY_CSV:
        master_csv_path = out_tables_dir / "sprint2_master_summary.csv"
        write_master_summary_csv(master_csv_path, master_rows)
        print("\nMaster summary CSV saved:", master_csv_path)

    if WRITE_PAPER_TABLES:
        # Table S2-1 style: performance
        perf_headers = [
            "Model",
            "Architecture",
            "Resolution",
            "Mean Time (ms)",
            "Std (ms)",
            "FPS",
            "Frames (N)",
            "Device",
            "Backend",
        ]
        perf_rows = [
            [
                r["model_label"],
                r["architecture"],
                r["resolution"],
                r["mean_time_ms"],
                r["std_time_ms"],
                r["fps"],
                r["frames_n"],
                r["device"],
                r["backend"],
            ]
            for r in master_rows
        ]
        perf_path = out_tables_dir / "table_s2_1_rtx_official_cross_model_performance.csv"
        write_table_csv(perf_path, perf_headers, perf_rows)
        print("Performance table saved:", perf_path)

        # Table S2-2 style: metrics
        metrics_headers = [
            "Model",
            "Resolution",
            "Device",
            "RMSE (Full)",
            "AbsRel (Full)",
            "RMSE (ROI)",
            "AbsRel (ROI)",
            "Reference File",
            "Reference SHA256",
        ]
        metrics_rows = [
            [
                r["model_label"],
                r["resolution"],
                r["device"],
                r["rmse_full"],
                r["absrel_full"],
                r["rmse_roi"],
                r["absrel_roi"],
                r["reference_depth_file"],
                r["reference_depth_sha256"],
            ]
            for r in master_rows
        ]
        metrics_path = out_tables_dir / "table_s2_2_reference_based_consistency.csv"
        write_table_csv(metrics_path, metrics_headers, metrics_rows)
        print("Metrics table saved:", metrics_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
