"""Reading GONG H-alpha frames and MAGFiLO annotations."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import cv2
import numpy as np

SIZE = 2048               # native frame size (pixels)
DISK_RADIUS = 950.0       # radius of the solar disk used for intensity normalisation


def disk_mask(size: int = SIZE) -> np.ndarray:
    """Boolean mask of the solar disk."""
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) / 2.0
    return (yy - c) ** 2 + (xx - c) ** 2 <= (DISK_RADIUS * size / SIZE) ** 2


def read_gray(path) -> np.ndarray:
    """Read a frame as an 8-bit grayscale array of SIZE x SIZE."""
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f"cannot decode {path}")
    if im.shape != (SIZE, SIZE):
        warnings.warn(f"{path}: frame is {im.shape}, resized to {SIZE} x {SIZE}; masks refer to the resized frame")
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return im


def frame_stats(frame: np.ndarray, disk: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of the on-disk intensities (in 0..1)."""
    v = frame[disk].astype(np.float32) / 255.0
    return float(v.mean()), float(v.std()) + 1e-6


def list_frames(folder) -> list[str]:
    """Paths of the .jpg / .jpeg frames of a folder, sorted by name."""
    files = sorted(str(p) for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if not files:
        raise FileNotFoundError(f"no .jpg/.jpeg frames in {folder}")
    return files


def load_annotations(json_path) -> dict[str, list]:
    """records[frame stem] = list of annotator records; a record is a list of filaments;
    a filament is a list of polygons ([x0, y0, x1, y1, ...])."""
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    by_image: dict[str, list] = {}
    for a in d["annotations"]:
        seg = a["segmentation"]
        if isinstance(seg, str):
            seg = json.loads(seg)
        polys = [[float(v) for v in p] for p in seg if len(p) >= 6]
        if polys:
            by_image.setdefault(str(a["image_id"]), []).append(polys)
    records: dict[str, list] = {}
    for im in d["images"]:
        records.setdefault(Path(im["file_name"]).stem, []).append(by_image.get(str(im["id"]), []))
    return records


def polygons_to_rle(polys, size: int = SIZE):
    """One annotated filament -> COCO RLE, rasterised with pycocotools as the host does."""
    from pycocotools import mask as mu
    r = mu.frPyObjects(polys, size, size)
    return mu.merge(r) if len(r) > 1 else r[0]
