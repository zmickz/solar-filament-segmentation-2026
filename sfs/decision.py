"""From a probability map to pixel-disjoint filament instances, and to the submission file.

An instance is an 8-connected region of p > t_lo. A region is kept when it is large enough and
confident enough. The four numbers of the rule were chosen on out-of-fold maps by maximising the
pooled Panoptic Quality of the competition (see scripts/evaluate_oof.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_CANDIDATE_AREA = 60     # regions below this never enter the search
MAX_CANDIDATES = 300        # safety cap per frame (a frame holds 5-40 filaments)


@dataclass(frozen=True)
class Rule:
    t_lo: float = 0.25      # a pixel belongs to a region when p > t_lo
    t_hi: float = 0.99      # the region must reach this probability somewhere
    min_area: float = 320   # pixels
    min_mean: float = 0.83  # mean probability inside the region


def candidates(pm: np.ndarray, t_lo: float):
    """-> (label image, region ids, features[n, 3] = area, mean p, peak p)."""
    n, lab = cv2.connectedComponents((pm > int(round(t_lo * 255))).astype(np.uint8), connectivity=8)
    if n <= 1:
        return lab, np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
    idx = np.flatnonzero(lab)
    fl, pv = lab.ravel()[idx], pm.ravel()[idx].astype(np.float64)
    area = np.bincount(fl, minlength=n)[1:].astype(np.float64)
    mean = np.bincount(fl, weights=pv, minlength=n)[1:] / np.maximum(area, 1) / 255.0
    order = np.argsort(fl, kind="stable")
    peak = np.maximum.reduceat(pv[order], np.searchsorted(fl[order], np.arange(1, n))) / 255.0
    feat = np.stack([area, mean, peak], 1).astype(np.float32)
    ids = np.nonzero(feat[:, 0] >= MIN_CANDIDATE_AREA)[0]
    if len(ids) > MAX_CANDIDATES:
        ids = np.sort(ids[np.argsort(-feat[ids, 0] * feat[ids, 1])[:MAX_CANDIDATES]])
    return lab, ids + 1, feat[ids]


def keep_mask(feat: np.ndarray, rule: Rule) -> np.ndarray:
    """Which candidate regions satisfy the rule."""
    if not len(feat):
        return np.zeros(0, bool)
    return (feat[:, 2] >= rule.t_hi) & (feat[:, 0] >= rule.min_area) & (feat[:, 1] >= rule.min_mean)


def instances(pm: np.ndarray, rule: Rule = Rule()):
    """-> (label image, list of kept region ids). Regions are disjoint by construction."""
    lab, ids, feat = candidates(pm, rule.t_lo)
    return lab, [int(k) for k in ids[keep_mask(feat, rule)]]


def instance_rles(lab: np.ndarray, ids) -> list[dict]:
    """COCO RLE objects of the kept regions."""
    from pycocotools import mask as mu
    return [mu.encode(np.asfortranarray((lab == k).astype(np.uint8))) for k in ids]


def write_submission(path, rows) -> None:
    """rows = [(filament_id, RLE counts string)]."""
    with open(path, "w", newline="") as f:
        f.write("filament_id,segmentation_rle\n")
        for fid, rle in rows:
            if "," in rle or "\n" in rle or chr(34) in rle:
                raise ValueError("RLE string would need quoting")
            f.write(f"{fid},{rle}\n")
