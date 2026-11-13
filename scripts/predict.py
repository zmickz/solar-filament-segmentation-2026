"""Segment every frame of a folder and write the competition submission file.

    python scripts/predict.py --images <folder with .jpg frames> --out submission.csv

Uses the five fold models in weights/ (mean of their probability maps), then the instance rule.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfs.decision import write_submission  # noqa: E402
from sfs.pipeline import load_models, segment_folder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with the H-alpha frames (.jpg / .jpeg)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--maps", default=None, help="optional folder to save the probability maps (PNG)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("no GPU found: running on CPU (about one minute per frame)")
    models = load_models(args.weights, device)
    rows = segment_folder(args.images, models, device, limit=args.limit, maps_dir=args.maps)
    write_submission(args.out, rows)
    print(f"wrote {args.out}: {len(rows)} filaments")


if __name__ == "__main__":
    main()
