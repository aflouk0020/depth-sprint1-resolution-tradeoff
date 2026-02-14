#!/usr/bin/env python3
"""
Sprint 1 - Resolution Sweep + Metrics + Timing (LOCKED, Plan-Aligned)

Key fixes vs your previous version:
1) Uses per-frame 1080p reference depth (D_ref) for the SAME frame and SAME model.
2) Enforces draft scope = 3 resolutions only (1080, 720, 480) via config.
3) Uses processed H.264 master video + SHA256 verification (if provided).
4) Locks processor speed choice (use_fast) from YAML to avoid device mismatch.
5) Timing = inference-only, excludes resize + video I/O, includes CUDA sync on RTX.
6) 1080p row metrics are defined as 0.0000 (consistency vs itself).
7) Auto-captures git commit hash for audit traceability.
8) Prints ROI boundary values for 1080p (H=1080, W=1920) for audit clarity.
9) Timing loop reuses preloaded tensors (no per-iteration CPU→GPU transfer).

Run:
  python scripts/03_resolution_sweep_metrics_timing.py
"""

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import time
import yaml
import cv2
import numpy as np
import torch
import subprocess

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# -----------------------------
# Helpers
# -----------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = repo_root() / "configs" / "experiment.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device(cfg: dict) -> str:
    for d in cfg["model"]["device_priority"]:
        if d == "cuda" and torch.cuda.is_available():
            return "cuda"
        if d == "mps" and torch.backends.mps.is_available():
            return "mps"
        if d == "cpu":
            return "cpu"
    return "cpu"


def backend_label(device: str) -> str:
    return "CUDA" if device == "cuda" else ("MPS" if device == "mps" else "CPU")


def cuda_sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root()),
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "not_available"


def roi_mask(h: int, w: int, bottom_height_ratio: float, center_width_ratio: float) -> np.ndarray:
    # bottom 40% -> y_start = floor(0.60*H), y_end = H
    y_start = int(np.floor((1.0 - bottom_height_ratio) * h))
    y_end = h

    # central 60% width -> x_start=floor(0.20*W), x_end=floor(0.80*W)
    x_margin = (1.0 - center_width_ratio) / 2.0
    x_start = int(np.floor(x_margin * w))
    x_end = int(np.floor((1.0 - x_margin) * w))

    # ✅ Audit print (for 1080p reference this should print y[648:1080], x[384:1536])
    print(f"ROI boundaries for HxW={h}x{w}: y[{y_start}:{y_end}], x[{x_start}:{x_end}]")

    mask = np.zeros((h, w), dtype=bool)
    mask[y_start:y_end, x_start:x_end] = True
    return mask


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a - b).astype(np.float32)
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def absrel(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-6, mask: np.ndarray | None = None) -> float:
    num = np.abs(pred - ref).astype(np.float32)
    den = (np.abs(ref).astype(np.float32) + eps)
    val = num / den
    if mask is not None:
        val = val[mask]
    return float(np.mean(val))


# -----------------------------
# Model (locked)
# -----------------------------

@torch.inference_mode()
def infer_depth_from_inputs(model: AutoModelForDepthEstimation, inputs: dict, device: str) -> np.ndarray:
    """
    inputs: already on device, already preprocessed by processor (pixel_values)
    returns: HxW float32 depth prediction in model native scale
    """
    outputs = model(**inputs)
    depth = outputs.predicted_depth  # [1, H, W]
    depth = depth.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
    return depth


def load_model_and_processor(cfg: dict, device: str):
    model_id = cfg["model"]["model_id"]
    revision = cfg["model"]["revision"]
    use_fast = bool(cfg.get("model", {}).get("processor", {}).get("use_fast", False))

    processor = AutoImageProcessor.from_pretrained(
        model_id,
        revision=revision,
        use_fast=use_fast,
    )

    model = AutoModelForDepthEstimation.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float32,
    )

    model.eval()
    model.to(device)
    return model, processor


def build_device_inputs(
    processor: AutoImageProcessor,
    rgb_uint8_list: list[np.ndarray],
    device: str,
) -> list[dict]:
    """
    ✅ Preprocess once (CPU) and move once (CPU→GPU) for strict timing isolation.
    Timing loop will reuse these dicts with tensors already on device.
    """
    device_inputs: list[dict] = []
    for rgb in rgb_uint8_list:
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        device_inputs.append(inputs)
    return device_inputs


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    root = repo_root()
    cfg = load_config()

    git_commit = get_git_commit_hash()

    # Reference resolution
    ref_w, ref_h = map(int, cfg["resolutions"]["reference"])

    # Master video
    video_path = (root / cfg["dataset"]["master_video_path"]).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Master video not found: {video_path}")

    # Optional SHA256 verification
    expected_sha = cfg["dataset"].get("expected_sha256")
    actual_master_sha = sha256_file(video_path)
    if expected_sha:
        if actual_master_sha.lower() != str(expected_sha).lower():
            raise RuntimeError(
                "Master video SHA256 mismatch!\n"
                f"Expected: {expected_sha}\n"
                f"Actual:   {actual_master_sha}\n"
                "Fix master_video_path / expected_sha256 before any runs."
            )

    # ROI mask at reference size (H=1080, W=1920)
    roi_cfg = cfg["roi"]["definition"]
    roi = roi_mask(
        h=ref_h,
        w=ref_w,
        bottom_height_ratio=float(roi_cfg["bottom_height_ratio"]),
        center_width_ratio=float(roi_cfg["center_width_ratio"]),
    )

    # Save ROI preview
    roi_mask_path = (root / cfg["roi"]["output_mask_path"]).resolve()
    roi_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(roi_mask_path), (roi.astype(np.uint8) * 255))

    # Timing protocol
    warmup = int(cfg["timing"]["warmup_frames"])
    measure_frames = int(cfg["timing"]["measure_frames"])

    # Enforce draft scope
    test_levels = [tuple(map(int, r)) for r in cfg["resolutions"]["test_levels"]]
    allowed_draft = {(1920, 1080), (1280, 720), (854, 480)}
    if set(test_levels) != allowed_draft:
        raise RuntimeError(
            "Draft scope violation: test_levels must be exactly "
            "[1920x1080, 1280x720, 854x480] for the 22nd draft."
        )

    # Read frames once (I/O excluded from timing)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()

    if len(frames_bgr) < warmup + measure_frames:
        raise RuntimeError(
            f"Not enough frames for warmup+measure.\n"
            f"Frames total: {len(frames_bgr)}, warmup: {warmup}, measure: {measure_frames}"
        )

    usable_bgr = frames_bgr[warmup : warmup + measure_frames]

    # Device selection
    device = pick_device(cfg)
    print(f"Device: {device}")
    print("Timing protocol: ONLY model forward pass timed. No preprocessing, postprocessing, or I/O.")
    print("Timing protocol: Same preloaded tensor list reused across timing runs (no per-iteration CPU→GPU transfer).")
    if device == "cuda":
        print("Timing protocol: torch.cuda.synchronize() applied before and after timing block.")

    model, processor = load_model_and_processor(cfg, device)

    # Build reference RGB list at 1080p (no crop/trim)
    ref_rgb_list: list[np.ndarray] = []
    for frame_bgr in usable_bgr:
        # Ensure reference is exactly 1920x1080 (if decoder returns weird dims)
        if frame_bgr.shape[1] != ref_w or frame_bgr.shape[0] != ref_h:
            frame_bgr = cv2.resize(frame_bgr, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ref_rgb_list.append(rgb)

    # ✅ Precompute reference inputs ON DEVICE (not timed)
    ref_inputs_device = build_device_inputs(processor, ref_rgb_list, device)

    # Precompute per-frame 1080 reference depths (NOT timed; metrics reference)
    print("\nPrecomputing per-frame 1080 reference depths (for metrics)...")
    ref_depths: list[np.ndarray] = []
    for inp in ref_inputs_device:
        d_ref = infer_depth_from_inputs(model, inp, device)
        if d_ref.shape != (ref_h, ref_w):
            d_ref = cv2.resize(d_ref, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC)
        ref_depths.append(d_ref.astype(np.float32))

    # Output CSV
    results_csv = (root / cfg["outputs"]["tables"]["results_csv"]).resolve()
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = [
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
        "checkpoint_sha256",
        "model_id",
        "revision",
        "frames_n",
        "master_video_sha256",
    ]

    # If you later add a local checkpoint_path in YAML, you can hash it here.
    checkpoint_sha = cfg.get("model", {}).get("checkpoint_sha256", "")
    abs_eps = float(cfg["metrics"]["absrel_epsilon"])

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for (w, h) in test_levels:
            res_label = f"{w}x{h}"
            print(f"\n=== Running {res_label} ===")

            # ✅ Precompute resized RGB frames (not timed)
            rgb_list: list[np.ndarray] = []
            for frame_bgr in usable_bgr:
                resized = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
                rgb_list.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            # ✅ Precompute processor outputs ON DEVICE (not timed)
            inputs_device = build_device_inputs(processor, rgb_list, device)

            times_ms: list[float] = []
            metrics_accum: list[tuple[float, float, float, float]] = []

            for i, inp in enumerate(inputs_device):
                # ✅ Inference-only timing (strict)
                cuda_sync_if_needed(device)
                t0 = time.perf_counter()
                d_pred = infer_depth_from_inputs(model, inp, device)
                cuda_sync_if_needed(device)
                t1 = time.perf_counter()

                infer_ms = (t1 - t0) * 1000.0
                times_ms.append(infer_ms)

                # Upsample prediction to reference size (excluded from timing)
                if d_pred.shape != (ref_h, ref_w):
                    d_pred_up = cv2.resize(d_pred, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                else:
                    d_pred_up = d_pred.astype(np.float32)

                d_ref = ref_depths[i]

                # Reference row has zero degradation by definition
                if (w, h) == (ref_w, ref_h):
                    rmse_full_v = 0.0
                    rmse_roi_v = 0.0
                    abs_full_v = 0.0
                    abs_roi_v = 0.0
                else:
                    rmse_full_v = rmse(d_pred_up, d_ref, mask=None)
                    rmse_roi_v = rmse(d_pred_up, d_ref, mask=roi)
                    abs_full_v = absrel(d_pred_up, d_ref, eps=abs_eps, mask=None)
                    abs_roi_v = absrel(d_pred_up, d_ref, eps=abs_eps, mask=roi)

                metrics_accum.append((rmse_full_v, rmse_roi_v, abs_full_v, abs_roi_v))

            mean_ms = float(np.mean(times_ms))
            std_ms = float(np.std(times_ms))
            fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

            m = np.array(metrics_accum, dtype=np.float32)
            rmse_full_m = float(m[:, 0].mean())
            rmse_roi_m = float(m[:, 1].mean())
            abs_full_m = float(m[:, 2].mean())
            abs_roi_m = float(m[:, 3].mean())

            row = {
                "resolution": res_label,
                "mean_time_ms": f"{mean_ms:.4f}",
                "std_time_ms": f"{std_ms:.4f}",
                "fps": f"{fps:.4f}",
                "rmse_full": f"{rmse_full_m:.6f}",
                "rmse_roi": f"{rmse_roi_m:.6f}",
                "absrel_full": f"{abs_full_m:.6f}",
                "absrel_roi": f"{abs_roi_m:.6f}",
                "device": device,
                "backend": backend_label(device),
                "git_commit_hash": git_commit,
                "checkpoint_sha256": checkpoint_sha,
                "model_id": cfg["model"]["model_id"],
                "revision": cfg["model"]["revision"],
                "frames_n": str(len(usable_bgr)),
                "master_video_sha256": actual_master_sha,
            }
            writer.writerow(row)

            print(f"mean_time_ms={mean_ms:.2f} | fps={fps:.2f}")
            print(f"rmse_full={rmse_full_m:.6f} | rmse_roi={rmse_roi_m:.6f}")
            print(f"absrel_full={abs_full_m:.6f} | absrel_roi={abs_roi_m:.6f}")

    print("\n✅ Saved results CSV:", results_csv)
    print("✅ Saved ROI mask preview:", roi_mask_path)
    print("✅ Master video:", video_path)
    print("✅ Master SHA256:", actual_master_sha)
    print("✅ Git commit hash:", git_commit)
    print("NOTE: Official FPS must be produced on RTX (CUDA) per Sprint plan.")


if __name__ == "__main__":
    main()
