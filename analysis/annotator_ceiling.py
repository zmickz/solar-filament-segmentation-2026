"""How high can the competition metric go? Score the human annotators against each other.

    python analysis/annotator_ceiling.py <MAGFiLO train json>

Two numbers:
  * human vs human: one annotator record used as the prediction for another record of the same frame;
  * perfect clone: the first record of a frame used as the prediction, scored against every record of
    that frame (itself included), which is how a submission is scored on multi-annotator frames.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfs.data import load_annotations, polygons_to_rle  # noqa: E402
from sfs.metric import iou_matrix, match_parts, pq  # noqa: E402


def main():
    records = load_annotations(sys.argv[1])
    human = np.zeros(4)
    clone = collections.defaultdict(lambda: np.zeros(4))
    for recs in records.values():
        rles = [[polygons_to_rle(p) for p in rec] for rec in recs if rec]
        n = len(rles)
        for i in range(n):
            for j in range(n):
                if i != j:
                    human += match_parts(iou_matrix(rles[i], rles[j]))
        for j in range(n):
            clone[n] += match_parts(iou_matrix(rles[0], rles[j]))
    print(f"human vs another human: PQ {pq(human):.4f}")
    for n in sorted(clone):
        print(f"perfect clone of one annotator, frames with {n} record(s): PQ {pq(clone[n]):.4f}")
    print(f"perfect clone, all training frames: PQ {pq(sum(clone.values())):.4f}")


if __name__ == "__main__":
    main()
