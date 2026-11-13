"""Grid search of the fixed keep rule (t_hi = 0.99) on a candidate file built by selector_candidates.py
(levels 0..2 = p>0.80/0.50/0.25). Used to pick the rule of the final two-family ensemble;
its output is artifacts/rule_search.txt.
Usage: python analysis/rule_grid.py <nodes.npz>
"""
import sys

import numpy as np

d = np.load(sys.argv[1])
X, S, TP, level = d["X"], d["S"].astype(float), d["TP"].astype(float), d["level"].astype(int)
off, nrec, n_gt, fold = d["offsets"], d["nrec"], d["n_gt"], d["fold"]
frame = np.repeat(np.arange(len(nrec)), np.diff(off))
nr = nrec[frame].astype(float)
nf = fold[frame]
tot = float(n_gt.sum())


def pq(ch, total=tot):
    s, tp = S[ch].sum(), TP[ch].sum()
    fp = nr[ch].sum() - tp
    return s / max(tp + 0.5 * fp + 0.5 * (total - tp), 1e-9), tp, fp, total - tp


res = []
for L, t in ((2, 0.25), (1, 0.50)):
    for a in (200, 260, 320, 400, 500, 650):
        for mm in (0.70, 0.75, 0.80, 0.83, 0.85, 0.87, 0.89, 0.91):
            ch = (level == L) & (X[:, 1] >= np.log(a)) & (X[:, 2] >= mm)
            res.append((pq(ch), t, a, mm, ch))
res.sort(key=lambda r: -r[0][0])
for (q, tp, fp, fn), t, a, mm, ch in res[:8]:
    pf = [round(float(pq(ch & (nf == f), float(n_gt[fold == f].sum()))[0]), 4) for f in sorted(set(fold.tolist()))]
    print("PQ %.4f  t_lo %.2f area>=%d mean>=%.2f  TP %.0f FP %.0f FN %.0f  kept %d  folds %s" % (q, t, a, mm, tp, fp, fn, ch.sum(), pf))
ch = (level == 2) & (X[:, 1] >= np.log(320)) & (X[:, 2] >= 0.85)
print("rule of the ResNet-34-only model (area>=320, mean>=0.85) on these maps: PQ %.4f" % pq(ch)[0])
