"""Selector, step 2: learn the expected PQ gain of every candidate region and pick the PQ-optimal disjoint set.

PQ = sum IoU(TP) / (TP + FP/2 + FN/2). Keeping one more region always adds 1/2 to the denominator (it either
turns a miss into a hit or becomes a false positive), and adds its IoU to the numerator only if it hits.
So a region is worth keeping iff  E[IoU * hit] > tau,  tau = PQ / 2.
Regions are nested in a tree, so the best disjoint set is an exact dynamic programme:
  best(node) = max( gain(node) - tau , sum best(children) , 0 ).

The gain model is fitted with the same date-grouped folds that produced the out-of-fold maps, so every number
printed here is out-of-fold end to end.

Usage: python analysis/selector_expected_pq.py <nodes.npz> <out.json>
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

TAUS = (0.12, 0.15, 0.18, 0.20, 0.22, 0.24, 0.27, 0.30)


def select(gain, tau, level, parent, nlev):
    """Exact tree DP. Returns boolean mask of the selected nodes."""
    own = gain - tau
    child = np.zeros(len(gain))
    for L in range(nlev):
        m = (level == L) & (parent >= 0)
        np.add.at(child, parent[m], np.maximum(np.maximum(own[m], child[m]), 0.0))
    take = (own >= child) & (own > 0)
    active = parent < 0
    chosen = np.zeros(len(gain), bool)
    for L in range(nlev - 1, -1, -1):
        m = level == L
        chosen |= m & active & take
        down = m & active & ~take
        nxt = (level == L - 1) & (parent >= 0)
        active[nxt] = down[parent[nxt]]
    return chosen


def pq(chosen, S, TP, nrec_node, total_gt):
    s, tp = float(S[chosen].sum()), float(TP[chosen].sum())
    fp = float(nrec_node[chosen].sum()) - tp
    fn = total_gt - tp
    return s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9), s, tp, fp, fn


def main():
    d = np.load(sys.argv[1])
    X, S, TP, level, parent = d["X"], d["S"].astype(np.float64), d["TP"].astype(np.float64), d["level"].astype(int), d["parent"]
    off, nrec, n_gt, fold = d["offsets"], d["nrec"], d["n_gt"], d["fold"]
    nlev = int(level.max()) + 1
    frame = np.repeat(np.arange(len(nrec)), np.diff(off))
    nrec_node = nrec[frame].astype(np.float64)
    node_fold = fold[frame]
    total_gt = float(n_gt.sum())
    y = S / np.maximum(nrec_node, 1)
    rep = {"nodes": int(len(y)), "frames": int(len(nrec))}

    base = (level == 2) & (X[:, 1] >= np.log(320)) & (X[:, 2] >= 0.85)
    rep["baseline_rule"] = pq(base, S, TP, nrec_node, total_gt)
    print("baseline rule (p>0.25, area>=320, mean>=0.85): PQ %.4f  TP %.0f FP %.0f FN %.0f" % tuple(np.array(rep["baseline_rule"])[[0, 2, 3, 4]]))

    tau = 0.25
    for _ in range(8):                       # oracle: true gains, tau at its fixed point
        r = pq(select(y, tau, level, parent, nlev), S, TP, nrec_node, total_gt)
        tau = r[0] / 2
    rep["oracle_tree"] = r
    print("oracle on this tree (perfect gain model):   PQ %.4f  TP %.0f FP %.0f FN %.0f" % (r[0], r[2], r[3], r[4]))
    for L in range(nlev):
        only = level == L
        t = 0.25
        for _ in range(6):
            r = pq(only & (y > t), S, TP, nrec_node, total_gt)
            t = r[0] / 2
        print("  oracle using level %d only: PQ %.4f" % (L, r[0]))

    pred = np.zeros(len(y))
    folds = sorted(set(fold.tolist()))
    for f in folds:
        tr, va = node_fold != f, node_fold == f
        m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=2026)
        m.fit(X[tr], y[tr], sample_weight=nrec_node[tr])
        pred[va] = m.predict(X[va])
        print("fold", f, "train nodes", int(tr.sum()), "val nodes", int(va.sum()), flush=True)
    rep["learned"] = {}
    best = None
    for tau in TAUS:
        ch = select(pred, tau, level, parent, nlev)
        r = pq(ch, S, TP, nrec_node, total_gt)
        pf = [round(pq(ch & (node_fold == f), S, TP, nrec_node, float(n_gt[fold == f].sum()))[0], 4) for f in folds]
        rep["learned"][str(tau)] = {"pq": r, "per_fold": pf, "levels": np.bincount(level[ch], minlength=nlev).tolist()}
        print("learned tau %.2f: PQ %.4f  TP %.0f FP %.0f FN %.0f  folds %s  picked per level %s"
              % (tau, r[0], r[2], r[3], r[4], pf, rep["learned"][str(tau)]["levels"]))
        if best is None or r[0] > best[1]:
            best = (tau, r[0])
    for tau in (0.18, 0.22):                 # ablation: learned keep on the plain p>0.25 regions, no tree
        ch = (level == 2) & (pred > tau)
        print("learned keep, no tree (p>0.25 regions only), tau %.2f: PQ %.4f" % (tau, pq(ch, S, TP, nrec_node, total_gt)[0]))
    rep["best"] = {"tau": best[0], "pq": best[1]}
    print("BEST out-of-fold PQ %.4f at tau %.2f (fixed rule: %.4f)" % (best[1], best[0], rep["baseline_rule"][0]))
    json.dump(rep, open(sys.argv[2], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
