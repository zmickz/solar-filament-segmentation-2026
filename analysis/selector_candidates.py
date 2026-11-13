"""Selector, step 1: build a tree of nested candidate regions for every frame, with features and PQ labels.

Levels, fine -> coarse (each region of a finer level lies inside exactly one region of the next):
  p>0.80, p>0.50, p>0.25, closing(p>0.25, 9px), closing(.., 19px), closing(.., 33px)
The finer levels let the selector shrink or split a region; the closings let it join pieces across a gap,
which is how the annotators draw fragmented filaments.

Usage: python analysis/selector_candidates.py <annotations.json | -> <maps.zip> <artifacts/folds.json> <out.npz>
With "-" as annotations the labels are left at zero (test maps).
"""
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

H = W = 2048
THR = (0.80, 0.50, 0.25)
CLOSE = (9, 19, 33)
NLEV = len(THR) + len(CLOSE)
MIN_AREA = 60
STATIONS = "BCLMTUZ"
FEATS = ["level", "log_area", "mean_p", "frac50", "frac80", "bw", "bh", "extent", "thin", "r_disk",
         "n_child", "max_child", "sum_child", "parent_ratio", "frame_n25", "frame_area25", "year", "station"]


def load_records(path):
    d = json.load(open(path, encoding="utf-8"))
    by_rec = {}
    for a in d["annotations"]:
        seg = a["segmentation"]
        if isinstance(seg, str):
            seg = json.loads(seg)
        polys = [[float(v) for v in p] for p in seg if len(p) >= 6]
        if polys:
            by_rec.setdefault(str(a["image_id"]), []).append(polys)
    records = {}
    for im in d["images"]:
        records.setdefault(Path(im["file_name"]).stem, []).append(by_rec.get(str(im["id"]), []))
    return records


def gt_crops(rec):
    out = []
    for polys in rec:
        r = mu.frPyObjects(polys, H, W)
        r = mu.merge(r) if len(r) > 1 else r[0]
        y, x, w, h = [int(v) for v in mu.toBbox(r)[[1, 0, 2, 3]]]
        m = mu.decode(r)[y:y + h + 1, x:x + w + 1].astype(bool)
        if m.any():
            out.append((y, x, m, float(m.sum())))
    return out


def level_masks(pm):
    b = [(pm > int(round(t * 255))).astype(np.uint8) for t in THR]
    for k in CLOSE:
        b.append(cv2.morphologyEx(b[len(THR) - 1], cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)))
    for i in range(len(THR), NLEV):          # guarantee nesting numerically
        b[i] |= b[i - 1]
    return b


def build_tree(pm):
    """-> labs (list of label images), per-node arrays: level, label, area, parent(local index), feature block."""
    masks = level_masks(pm)
    idx = np.flatnonzero(masks[-1])
    pv = pm.ravel()[idx].astype(np.float64)
    labs, per = [], []
    for L, m in enumerate(masks):
        n, lab, st, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
        labs.append(lab)
        fl = lab.ravel()[idx]
        area = st[:, cv2.CC_STAT_AREA].astype(np.float64)
        sp = np.bincount(fl, weights=pv, minlength=n)
        s50 = np.bincount(fl, weights=(pv > 127), minlength=n)
        s80 = np.bincount(fl, weights=(pv > 204), minlength=n)
        first = np.full(n, -1, np.int64)
        u, fi = np.unique(fl, return_index=True)
        first[u] = idx[fi]
        per.append((n, area, sp, s50, s80, st, first))
    level, label, parent_lab = [], [], []
    for L, (n, area, *_r, first) in enumerate(per):
        ids = np.nonzero(area[1:] >= MIN_AREA)[0] + 1
        level.append(np.full(len(ids), L)); label.append(ids)
        parent_lab.append(labs[L + 1].ravel()[first[ids]] if L + 1 < NLEV else np.zeros(len(ids), np.int64))
    off = np.cumsum([0] + [len(x) for x in label])
    lut = []
    for L in range(NLEV):
        t = np.full(per[L][0], -1, np.int64)
        t[label[L]] = off[L] + np.arange(len(label[L]))
        lut.append(t)
    parent = np.concatenate([lut[L + 1][parent_lab[L]] if L + 1 < NLEV else np.full(len(label[L]), -1, np.int64)
                             for L in range(NLEV)])
    level, label = np.concatenate(level), np.concatenate(label)
    N = len(level)
    X = np.zeros((N, len(FEATS)), np.float32)
    area_all = np.zeros(N)
    for L in range(NLEV):
        n, area, sp, s50, s80, st, _f = per[L]
        sl = slice(off[L], off[L + 1]); ids = label[sl]
        a = area[ids]; area_all[sl] = a
        bw, bh = st[ids, cv2.CC_STAT_WIDTH].astype(np.float64), st[ids, cv2.CC_STAT_HEIGHT].astype(np.float64)
        cx, cy = st[ids, cv2.CC_STAT_LEFT] + bw / 2, st[ids, cv2.CC_STAT_TOP] + bh / 2
        X[sl, 0] = L
        X[sl, 1] = np.log(a)
        X[sl, 2] = sp[ids] / a / 255.0
        X[sl, 3] = s50[ids] / a
        X[sl, 4] = s80[ids] / a
        X[sl, 5], X[sl, 6] = bw, bh
        X[sl, 7] = a / (bw * bh)
        X[sl, 8] = np.hypot(bw, bh) / np.sqrt(a)
        X[sl, 9] = np.hypot(cx - W / 2, cy - H / 2) / 950.0
    has = parent >= 0
    X[:, 10] = np.bincount(parent[has], minlength=N)
    mx = np.zeros(N); np.maximum.at(mx, parent[has], area_all[has])
    X[:, 11] = mx / area_all
    X[:, 12] = np.bincount(parent[has], weights=area_all[has], minlength=N) / area_all
    X[:, 13] = np.where(has, area_all / area_all[np.maximum(parent, 0)], 1.0)
    l25 = level == len(THR) - 1
    X[:, 14] = l25.sum()
    X[:, 15] = area_all[l25].sum() / 1e4
    return labs, level, label, area_all, parent, X


def one(args):
    stem, recs, zpath = args
    with zipfile.ZipFile(zpath) as z:
        pm = cv2.imdecode(np.frombuffer(z.read(stem + ".png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    labs, level, label, area, parent, X = build_tree(pm)
    X[:, 16] = int(stem[:4]) if stem[:4].isdigit() else 0
    X[:, 17] = STATIONS.find(stem[-2]) if len(stem) >= 2 else -1
    N = len(level)
    S, TP = np.zeros(N), np.zeros(N)
    recs = [gt_crops(r) for r in recs if r]
    n_gt = sum(len(r) for r in recs)
    sizes = [int(l.max()) + 1 for l in labs]
    for crops in recs:
        for (y, x, m, ga) in crops:
            for L in range(NLEV):
                sel = level == L
                cnt = np.bincount(labs[L][y:y + m.shape[0], x:x + m.shape[1]][m], minlength=sizes[L])[label[sel]]
                iou = cnt / (area[sel] + ga - cnt)
                h = iou > 0.5
                if h.any():
                    w = np.nonzero(sel)[0][h]
                    S[w] += iou[h]; TP[w] += 1
    return stem, level.astype(np.int8), parent.astype(np.int32), X, S.astype(np.float32), TP.astype(np.float32), len(recs), n_gt


def main():
    ann, zpath, report, out = sys.argv[1:5]
    with zipfile.ZipFile(zpath) as z:
        stems = sorted(Path(n).stem for n in z.namelist() if n.endswith(".png"))
    records = load_records(ann) if ann != "-" else {}
    meta = json.load(open(report))
    meta = meta.get("meta", meta)
    fold_of = dict(zip(meta["stems"], meta["folds"]))
    with Pool(4) as p:
        rs = p.map(one, [(s, records.get(s, []), zpath) for s in stems], chunksize=8)
    off = np.cumsum([0] + [len(r[1]) for r in rs])
    parent = np.concatenate([np.where(r[2] >= 0, r[2] + off[i], -1) for i, r in enumerate(rs)])
    np.savez_compressed(
        out, stems=np.array([r[0] for r in rs]), feats=np.array(FEATS), offsets=off,
        level=np.concatenate([r[1] for r in rs]), parent=parent, X=np.concatenate([r[3] for r in rs]),
        S=np.concatenate([r[4] for r in rs]), TP=np.concatenate([r[5] for r in rs]),
        nrec=np.array([r[6] for r in rs]), n_gt=np.array([r[7] for r in rs]),
        fold=np.array([fold_of.get(r[0], -1) for r in rs]))
    print("frames", len(rs), "nodes", off[-1], "per level", np.bincount(np.concatenate([r[1] for r in rs]), minlength=NLEV))


if __name__ == "__main__":
    main()
