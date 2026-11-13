"""Write demo.ipynb from the cells below (keeps the notebook reviewable as plain Python).

    python scripts/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS = [
    ("md", """# Solar filament segmentation — end-to-end demo

This notebook loads the ten trained U-Net models stored in `weights/` (two encoders x five folds),
segments the test frames,
writes `submission.csv` in the competition format and compares it with the file we submitted to
Kaggle (`artifacts/submission_kaggle.csv`). No training happens here.

Run it from the repository root. On a GPU the 180 test frames take a few minutes; on CPU about
four minutes per frame, so set `SFS_LIMIT` to a small number for a quick look."""),
    ("code", """import glob, json, os, sys, time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Repository root: SFS_REPO if set, else the folder this notebook is run from.
REPO = Path(os.environ.get("SFS_REPO") or (Path.cwd() if (Path.cwd() / "sfs").is_dir() else Path.cwd().parent))
assert (REPO / "sfs").is_dir(), "run from the repository root or set SFS_REPO to it"
OUT = Path(os.environ.get("SFS_OUT") or Path.cwd())      # where submission.csv and the maps are written
sys.path.insert(0, str(REPO))

from sfs.data import list_frames, read_gray
from sfs.decision import Rule, write_submission
from sfs.pipeline import ENCODERS, load_models, segment_folder

# Folder with the test frames: set SFS_TEST_IMAGES, or attach the competition data on Kaggle.
hits = glob.glob("/kaggle/input/**/test/test_images", recursive=True)
IMAGES = os.environ.get("SFS_TEST_IMAGES") or (hits[0] if hits else None)
assert IMAGES and os.path.isdir(IMAGES), "set SFS_TEST_IMAGES to the folder with the test frames"
LIMIT = int(os.environ.get("SFS_LIMIT", 0))          # 0 = all frames
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("frames:", IMAGES, "| device:", device, "| limit:", LIMIT or "all")"""),
    ("md", """## 1. Load the weights

Two families of five models, one model per cross-validation fold (U-Net, 1 input channel):
ResNet-34 encoder (49 MB per 16-bit file) and ResNeXt-50 32x4d encoder (64 MB per file)."""),
    ("code", """families = load_models(REPO / "weights", device)
for enc, fam in zip(ENCODERS, families):
    n_par = sum(p.numel() for p in fam[0].parameters())
    print(f"{len(fam)} x U-Net {enc}: {n_par / 1e6:.1f} M parameters each")
rule = Rule()
print(rule)"""),
    ("md", """## 2. Segment the frames

For every frame: normalise by the mean and standard deviation of the solar disk, predict at the
native 2048 x 2048 resolution with 4 flips, average the five models of each family, average the two
families, then keep the connected
regions of `p > t_lo` that satisfy the rule. Regions are pixel-disjoint by construction."""),
    ("code", """t0 = time.time()
MAPS = OUT / "demo_maps"
rows = segment_folder(IMAGES, families, device, rule, limit=LIMIT, maps_dir=MAPS)
write_submission(OUT / "submission.csv", rows)
frames = sorted({r[0].rsplit("_", 1)[0] for r in rows})
n_done = len(list(MAPS.glob("*.png")))
print(f"{len(rows)} filaments on {len(frames)} frames; {(time.time() - t0) / max(n_done, 1):.1f} s per frame")"""),
    ("md", """## 3. Compare with the file submitted to Kaggle

The submitted file was produced on a Kaggle GPU with mixed precision. Other hardware can move a
probability by a grey level or two (1/255), so we report both exact row equality and mask agreement."""),
    ("code", """from pycocotools import mask as mu

def read_rows(path):
    out = {}
    for line in open(path).read().splitlines()[1:]:
        fid, rle = line.split(",", 1)
        out.setdefault(fid.rsplit("_", 1)[0], []).append({"size": [2048, 2048], "counts": rle.encode()})
    return out

mine = read_rows(OUT / "submission.csv")
ref = read_rows(REPO / "artifacts" / "submission_kaggle.csv")
done = sorted(p.stem for p in MAPS.glob("*.png"))
same = matched = total = 0
for stem in done:
    a, b = mine.get(stem, []), ref.get(stem, [])
    total += max(len(a), len(b))
    same += sum(x["counts"] == y["counts"] for x, y in zip(a, b)) if len(a) == len(b) else 0
    if a and b:
        matched += int((np.asarray(mu.iou(a, b, [0] * len(b))).reshape(len(a), len(b)) > 0.99).sum())
print(f"frames compared: {len(done)}")
print(f"masks identical byte for byte: {same}/{total}")
print(f"masks with IoU > 0.99 against the submitted ones: {matched}/{total}")"""),
    ("md", "## 4. Look at a few results"),
    ("code", """files = {Path(p).stem: p for p in list_frames(IMAGES)}
show = [s for s in done if s in mine][:3]
fig, axes = plt.subplots(1, len(show), figsize=(7 * len(show), 7), squeeze=False)
for ax, stem in zip(axes[0], show):
    img = cv2.cvtColor(read_gray(files[stem]), cv2.COLOR_GRAY2RGB)
    for k, r in enumerate(mine[stem]):
        cs, _ = cv2.findContours(mu.decode(r), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = tuple(int(c) for c in plt.cm.tab10(k % 10, bytes=True)[:3])
        cv2.drawContours(img, cs, -1, color, 3)
    ax.imshow(img); ax.set_title(f"{stem}: {len(mine[stem])} filaments"); ax.axis("off")
plt.tight_layout(); plt.show()"""),
    ("md", """## 5. Validation numbers

`artifacts/oof_maps.zip` holds, for every training frame, the ensemble map predicted by the fold models
that never saw it. `scripts/evaluate_oof.py` scores those maps with the competition metric; its saved
output is shown here (re-run the script with the MAGFiLO annotation file to reproduce it)."""),
    ("code", """ev = json.load(open(REPO / "artifacts" / "oof_evaluation.json"))
for k in ("frames", "annotated_filaments", "predicted_filaments", "pq", "pq_ci95", "sq", "rq", "pq_per_fold",
          "pq_by_annotators", "matched_iou_quantiles", "matched_dice_median",
          "one_filament_several_masks", "one_mask_several_filaments"):
    print(f"{k}: {ev[k]}")"""),
]


def main():
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = Path(__file__).resolve().parents[1] / "demo.ipynb"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
