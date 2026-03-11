#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import time
import yaml
import cv2
import numpy as np
import torch

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


def pick_device(cfg):
    for d in cfg["model"]["device_priority"]:
        if d == "cuda" and torch.cuda.is_available():
            return "cuda"
        if d == "mps" and torch.backends.mps.is_available():
            return "mps"
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


def load_model_and_processor(spec, device):

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

    else:

        t0 = time.perf_counter()

        depth = run_model(model, inp, label)

        t1 = time.perf_counter()

        infer_ms = (t1 - t0) * 1000

        return depth, infer_ms


def main():

    root = repo_root()

    cfg = load_config()

    device = pick_device(cfg)

    print("Device:", device)

    ref_w, ref_h = cfg["resolutions"]["reference"]

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

    usable = frames[warmup : warmup + measure]

    results = []

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

                resized = cv2.resize(frame, (w, h))

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

            times = []

            for i, inp in enumerate(tensors):

                depth, infer_ms = measure_inference(
                    model,
                    inp,
                    label,
                    device,
                )

                times.append(infer_ms)

                if label == "midasv2":
                    depth = crop_back(depth, shapes[i])

                depth = cv2.resize(depth, (ref_w, ref_h))

            # Remove first frame timing (stabilises FPS)
            times = times[1:]

            mean_ms = float(np.mean(times))

            fps = 1000.0 / mean_ms

            print("Mean Time (ms):", mean_ms)
            print("FPS:", fps)

            results.append([
                label,
                f"{w}x{h}",
                mean_ms,
                fps
            ])

    out_csv = root / cfg["outputs"]["tables"]["results_csv"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "model",
            "resolution",
            "mean_time_ms",
            "fps"
        ])

        writer.writerows(results)

    print("\nResults saved:", out_csv)


if __name__ == "__main__":
    main()
