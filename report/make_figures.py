"""Figures of the technical report.

    python report/make_figures.py --annotations <MAGFiLO train json> --images <train_images folder>

Writes report/figures/pipeline.pdf, examples.pdf and evaluation.pdf.
The example frames are training frames shown with their out-of-fold prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfs.data import load_annotations, read_gray  # noqa: E402
from sfs.decision import instances  # noqa: E402

FIG = ROOT / "report" / "figures"
ANNOTATOR_COLORS = ((255, 200, 0), (0, 220, 255), (255, 80, 200))
PRED_COLOR = (80, 255, 80)
CROP = 640


def pipeline_figure():
    steps = [("H-alpha frame\n2048 x 2048", "#dfe7f5"), ("disk mean/std\nnormalisation", "#dfe7f5"),
             ("2 x 5 U-Nets\nResNet-34 and\nResNeXt-50\n4 flips each", "#f5e6d3"), ("mean\nprobability\nmap", "#f5e6d3"),
             ("connected\nregions of\np > 0.25", "#e2f0d9"), ("keep rule\narea, mean p\n(tuned on PQ)", "#e2f0d9"),
             ("disjoint\ninstance masks\n(COCO RLE)", "#e2f0d9")]
    fig, ax = plt.subplots(figsize=(7.2, 1.25))
    w, gap = 0.92, 0.16
    for i, (text, color) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1, boxstyle="round,pad=0.02", fc=color, ec="0.3", lw=0.8))
        ax.text(x + w / 2, 0.5, text, ha="center", va="center", fontsize=6.5)
        if i:
            ax.annotate("", xy=(x - 0.02, 0.5), xytext=(x - gap + 0.02, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color="0.2"))
    ax.set_xlim(-0.05, len(steps) * (w + gap) - gap + 0.05)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")
    fig.savefig(FIG / "pipeline.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def draw_polygons(img, filaments, color):
    for polys in filaments:
        for p in polys:
            pts = np.asarray(p, np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)


def examples_figure(annotations, image_dir, stems):
    records = load_annotations(annotations)
    fig, axes = plt.subplots(len(stems), 3, figsize=(7.2, 2.4 * len(stems)), squeeze=False)
    with zipfile.ZipFile(ROOT / "artifacts" / "oof_maps.zip") as z:
        for row, stem in zip(axes, stems):
            gray = read_gray(next(Path(image_dir).glob(stem + ".jp*g")))
            pm = cv2.imdecode(np.frombuffer(z.read(stem + ".png"), np.uint8), cv2.IMREAD_GRAYSCALE)
            lab, ids = instances(pm)
            kept = np.isin(lab, ids)
            ys, xs = np.nonzero(kept)
            cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (1024, 1024)
            y0, x0 = min(max(cy - CROP // 2, 0), 2048 - CROP), min(max(cx - CROP // 2, 0), 2048 - CROP)
            window = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            ann = rgb.copy()
            recs = [r for r in records[stem] if r]
            for k, rec in enumerate(recs):
                draw_polygons(ann, rec, ANNOTATOR_COLORS[k % 3])
            pred = rgb.copy()
            cs, _ = cv2.findContours(kept.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(pred, cs, -1, PRED_COLOR, 2, cv2.LINE_AA)
            titles = (f"{stem} (crop)", f"annotations ({len(recs)} annotator{'s' if len(recs) > 1 else ''})",
                      f"our masks ({len(ids)} in frame)")
            for ax, im, title in zip(row, (rgb, ann, pred), titles):
                ax.imshow(im[window])
                ax.set_title(title, fontsize=7)
                ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "examples.pdf", dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def evaluation_figure():
    with open(ROOT / "artifacts" / "oof_evaluation.json") as f:
        ev = json.load(f)
    with open(ROOT / "artifacts" / "error_autopsy.json") as f:
        fn = json.load(f)["fn"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(width_ratios=(1, 0.8, 1.5)))

    ax = axes[0]
    ax.hist(ev["matched_iou"], bins=np.linspace(0.5, 1.0, 26), color="#4c72b0")
    ax.axvline(ev["matched_iou_quantiles"]["p50"], color="k", lw=0.8, ls="--")
    ax.set_xlabel("IoU of matched filaments", fontsize=7)
    ax.set_ylabel("count", fontsize=7)

    ax = axes[1]
    folds = ev["pq_per_fold"]
    ax.axhspan(*ev["pq_ci95"], color="#4c72b0", alpha=0.2, lw=0)
    ax.axhline(ev["pq"], color="#4c72b0", lw=1)
    ax.plot(range(len(folds)), folds, "o", color="k", ms=3.5)
    ax.set_xticks(range(len(folds)))
    ax.set_xlabel("fold", fontsize=7)
    ax.set_ylabel("PQ (out-of-fold)", fontsize=7)
    ax.set_ylim(0.39, 0.46)

    ax = axes[2]
    groups = {"visible, rule dropped it": ("1 ",), "mask much larger": ("5 ",), "not detected": ("2 ",),
              "near miss (IoU 0.3-0.5)": ("6 ",), "split in pieces": ("3 ",), "merged with neighbour": ("4 ",),
              "other": ("7 ",)}
    vals = [sum(v for k, v in fn.items() if k.startswith(pre)) for pre in groups.values()]
    total = sum(vals)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, [100 * v / total for v in vals], color="#c44e52")
    ax.set_yticks(y)
    ax.set_yticklabels(list(groups), fontsize=6.5)
    ax.set_xlabel("% of missed filaments", fontsize=7)

    for ax in axes:
        ax.tick_params(labelsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "evaluation.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--images", required=True, help="folder with training frames")
    ap.add_argument("--stems", nargs="+", default=["20110123174914Mh", "20110203082634Lh"])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    pipeline_figure()
    evaluation_figure()
    examples_figure(args.annotations, args.images, args.stems)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
