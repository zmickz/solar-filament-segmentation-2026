"""Score out-of-fold probability maps against the MAGFiLO training annotations.

    python scripts/evaluate_oof.py --annotations <MAGFiLO train json> --out artifacts/oof_evaluation.json

Every training frame was predicted by the fold model that never saw it (artifacts/oof_maps.zip),
so the numbers below are honest validation numbers. Reports the pooled Panoptic Quality with a
frame-level bootstrap interval, the per-fold values, the IoU / Dice distribution of the matched
filaments and two fragmentation counts.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfs.data import load_annotations, polygons_to_rle  # noqa: E402
from sfs.decision import Rule, instance_rles, instances  # noqa: E402
from sfs.metric import iou_matrix, match_parts, pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def covered_fraction(a_rles, b_rles) -> np.ndarray:
    """[a, b] = share of the area of a that lies inside b."""
    from pycocotools import mask as mu
    if not a_rles or not b_rles:
        return np.zeros((len(a_rles), len(b_rles)))
    return np.asarray(mu.iou(a_rles, b_rles, [1] * len(b_rles))).reshape(len(a_rles), len(b_rles))


def score_frame(args):
    """One frame -> pooled parts over its annotator records, matched IoUs and fragmentation counts."""
    stem, records, zip_path, rule = args
    with zipfile.ZipFile(zip_path) as z:
        pm = cv2.imdecode(np.frombuffer(z.read(stem + ".png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    lab, ids = instances(pm, rule)
    pred = instance_rles(lab, ids)
    parts, ious, split, merged, n_gt = np.zeros(4), [], 0, 0, 0
    for rec in records:
        if not rec:
            continue
        gt = [polygons_to_rle(p) for p in rec]
        iou = iou_matrix(pred, gt)
        parts += match_parts(iou)
        ious.extend(iou[iou > 0.5].tolist())
        n_gt += len(gt)
        split += int(((covered_fraction(pred, gt) > 0.5).sum(0) >= 2).sum())    # one filament, several masks
        merged += int(((covered_fraction(gt, pred) > 0.5).sum(0) >= 2).sum())   # one mask, several filaments
    return stem, parts, ious, split, merged, n_gt, len(pred), len([r for r in records if r])


def bootstrap_pq(parts: np.ndarray, n_boot: int, seed: int = 0):
    """95% interval of the pooled PQ when frames are resampled with replacement."""
    rng = np.random.default_rng(seed)
    vals = [pq(parts[rng.integers(0, len(parts), len(parts))].sum(0)) for _ in range(n_boot)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, help="MAGFiLO training annotation json")
    ap.add_argument("--maps", default=str(ROOT / "artifacts" / "oof_maps.zip"))
    ap.add_argument("--folds", default=str(ROOT / "artifacts" / "folds.json"))
    ap.add_argument("--out", default=None, help="optional json file for the full result")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    for name, value in Rule().__dict__.items():
        ap.add_argument(f"--{name}", type=float, default=value)
    args = ap.parse_args()
    rule = Rule(args.t_lo, args.t_hi, args.min_area, args.min_mean)

    records = load_annotations(args.annotations)
    with open(args.folds) as f:
        split = json.load(f)
    fold_of = dict(zip(split["stems"], split["folds"]))
    with zipfile.ZipFile(args.maps) as z:
        stems = sorted(Path(n).stem for n in z.namelist() if n.endswith(".png"))
    jobs = [(s, records[s], args.maps, rule) for s in stems]
    with Pool(args.workers) as pool:
        res = pool.map(score_frame, jobs, chunksize=8)

    parts = np.array([r[1] for r in res])
    total = parts.sum(0)
    folds = np.array([fold_of[r[0]] for r in res])
    n_rec = np.array([r[7] for r in res])
    ious = np.array([v for r in res for v in r[2]])
    n_gt = sum(r[5] for r in res)
    out = {
        "rule": rule.__dict__, "frames": len(res), "annotated_filaments": n_gt,
        "predicted_filaments": int(sum(r[6] for r in res)),
        "pq": pq(total), "pq_ci95": bootstrap_pq(parts, args.bootstrap),
        "sq": float(total[0] / max(total[1], 1)),
        "rq": float(total[1] / max(total[1] + 0.5 * total[2] + 0.5 * total[3], 1e-9)),
        "tp": float(total[1]), "fp": float(total[2]), "fn": float(total[3]),
        "pq_per_fold": [pq(parts[folds == f].sum(0)) for f in sorted(set(folds.tolist()))],
        "pq_by_annotators": {str(k): pq(parts[n_rec == k].sum(0)) for k in sorted(set(n_rec.tolist()))},
        "matched_iou_quantiles": dict(zip(("p10", "p25", "p50", "p75", "p90"),
                                          np.percentile(ious, (10, 25, 50, 75, 90)).round(4).tolist())),
        "matched_dice_median": float(np.median(2 * ious / (1 + ious))),
        "one_filament_several_masks": int(sum(r[3] for r in res)),
        "one_mask_several_filaments": int(sum(r[4] for r in res)),
        "matched_iou": ious.round(4).tolist(),
    }
    out["pq_fold_mean_std"] = [float(np.mean(out["pq_per_fold"])), float(np.std(out["pq_per_fold"], ddof=1))]
    for k, v in out.items():
        if k != "matched_iou":
            print(f"{k}: {v}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
