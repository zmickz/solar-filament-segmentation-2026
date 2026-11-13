"""The inference pipeline end to end: weight files -> models, folder of frames -> submission rows."""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from .data import disk_mask, list_frames, read_gray
from .decision import Rule, instance_rles, instances
from .infer import predict_map
from .model import load_model


def load_models(weight_dir, device, encoder: str = "resnet34"):
    """Load every fold model (unet_<encoder>_fold*.pt) found in weight_dir."""
    files = sorted(Path(weight_dir).glob(f"unet_{encoder}_fold*.pt"))
    if not files:
        raise FileNotFoundError(f"no unet_{encoder}_fold*.pt in {weight_dir}")
    return [load_model(f, device, encoder) for f in files]


def segment_folder(image_dir, models, device, rule: Rule = Rule(), limit: int = 0, maps_dir=None, log=print):
    """Segment the frames of image_dir in file-name order.

    Returns the submission rows [(filament_id, RLE counts string)], filament_id = <frame stem>_<n>.
    If maps_dir is given, the probability map of each frame is saved there as an 8-bit PNG.
    """
    files = list_frames(image_dir)
    if limit:
        files = files[:limit]
    if maps_dir:
        Path(maps_dir).mkdir(parents=True, exist_ok=True)
    disk, rows, t0 = disk_mask(), [], time.time()
    for n, path in enumerate(files, 1):
        stem = Path(path).stem
        pm = predict_map(models, read_gray(path), device, disk)
        if maps_dir:
            cv2.imwrite(str(Path(maps_dir) / f"{stem}.png"), pm)
        lab, ids = instances(pm, rule)
        for j, rle in enumerate(instance_rles(lab, ids), 1):
            rows.append((f"{stem}_{j}", rle["counts"].decode("ascii")))
        if n % 10 == 0 or n == len(files):
            log(f"{n}/{len(files)} frames, {len(rows)} filaments, {time.time() - t0:.0f}s")
    return rows
