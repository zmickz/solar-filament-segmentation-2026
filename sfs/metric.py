"""The pooled Panoptic Quality used by the competition.

PQ = sum of IoU over matched pairs / (TP + FP/2 + FN/2); a pair matches when IoU > 0.5.
Every annotator record of a frame is scored as a separate ground truth, and the four totals are
pooled over all records before the ratio is taken.
"""
from __future__ import annotations

import numpy as np


def iou_matrix(pred_rles, gt_rles) -> np.ndarray:
    """IoU[pred, gt] between two lists of COCO RLE objects."""
    from pycocotools import mask as mu
    if not pred_rles or not gt_rles:
        return np.zeros((len(pred_rles), len(gt_rles)))
    return np.asarray(mu.iou(pred_rles, gt_rles, [0] * len(gt_rles))).reshape(len(pred_rles), len(gt_rles))


def match_parts(iou: np.ndarray) -> np.ndarray:
    """[sum IoU, TP, FP, FN] of one prediction set against one annotator record."""
    # Predictions are pixel-disjoint, so an annotated filament has at most one prediction with
    # IoU > 0.5; no assignment step is needed. Counted exactly as the competition scorer does.
    hit = iou > 0.5
    tp = float(hit.sum())
    return np.array([float(iou[hit].sum()), tp, iou.shape[0] - tp, iou.shape[1] - tp])


def pq(parts) -> float:
    """Panoptic Quality from pooled [sum IoU, TP, FP, FN]."""
    s, tp, fp, fn = parts
    return float(s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9))
