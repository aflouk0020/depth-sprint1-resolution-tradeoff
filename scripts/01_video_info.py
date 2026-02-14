#!/usr/bin/env python3
"""
Sprint 1 - Master Video Validation (LOCKED)

- Reads master video metadata (W, H, FPS, frames, duration)
- Optionally validates SHA256 if expected_sha256 is provided in configs/experiment.yaml
- Writes a validation report to outputs/tables/video_info.txt

Run:
  python scripts/01_video_info.py
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import cv2
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    root = repo_root()
    cfg_path = root / "configs" / "experiment.yaml"
    cfg = load_config(cfg_path)

    ds = cfg["dataset"]
    video_rel = ds["master_video_path"]
    video_path = (root / video_rel).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Master video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = (frames / fps) if fps > 0 else 0.0

    # Optional strict checks (recommended)
    mismatches: list[str] = []
    exp_w = ds.get("expected_width")
    exp_h = ds.get("expected_height")
    exp_fps = ds.get("expected_fps")
    exp_frames = ds.get("expected_frames")
    exp_dur = ds.get("expected_duration_seconds")

    if exp_w is not None and width != int(exp_w):
        mismatches.append(f"width expected {exp_w} got {width}")
    if exp_h is not None and height != int(exp_h):
        mismatches.append(f"height expected {exp_h} got {height}")
    if exp_fps is not None and abs(fps - float(exp_fps)) > 0.25:
        mismatches.append(f"fps expected {exp_fps} got {fps:.6f}")
    if exp_frames is not None and frames != int(exp_frames):
        mismatches.append(f"frames expected {exp_frames} got {frames}")
    if exp_dur is not None and abs(duration - float(exp_dur)) > 0.75:
        mismatches.append(f"duration expected {exp_dur} got {duration:.6f}")

    # Optional SHA256 check
    actual_hash = sha256_file(video_path)
    exp_hash = ds.get("expected_sha256")
    if exp_hash:
        if actual_hash.lower() != str(exp_hash).lower():
            mismatches.append(f"sha256 expected {exp_hash} got {actual_hash}")

    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "video_info.txt"

    with out_file.open("w", encoding="utf-8") as f:
        f.write("Video Information\n")
        f.write("------------------\n")
        f.write(f"Path: {video_path}\n")
        f.write(f"Width: {width}\n")
        f.write(f"Height: {height}\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"Frames: {frames}\n")
        f.write(f"Duration: {duration}\n")
        f.write(f"SHA256: {actual_hash}\n")

    print("Video Information")
    print("------------------")
    print("Path:", video_path)
    print("Width:", width)
    print("Height:", height)
    print("FPS:", fps)
    print("Frames:", frames)
    print("Duration:", duration)
    print("Saved:", out_file)

    if mismatches:
        print("\n❌ DATASET VALIDATION FAILED")
        for m in mismatches:
            print("-", m)
        raise SystemExit(1)

    print("\n✅ Dataset validated against experiment.yaml")


if __name__ == "__main__":
    main()
