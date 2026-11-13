"""Error autopsy of the out-of-fold maps: where is PQ lost, and what would each possible fix be worth.

Light CPU only (reads 707 PNG maps from oof_maps.zip + the training annotations). No GPU, no network.
Usage: python analysis/error_autopsy.py <annotations.json> <oof_maps.zip> <out.json>
"""
import json
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

H = W = 2048
MIN_AREA = 320
MEANS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93)
BASE_T, BASE_MEAN = 0.25, 0.85
K3 = np.ones((3, 3), np.uint8)


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
    """One annotator record -> list of (y0, y1, x0, x1, bool crop, area, n_polygons, n_components)."""
    out = []
    for polys in rec:
        r = mu.frPyObjects(polys, H, W)
        r = mu.merge(r) if len(r) > 1 else r[0]
        m = mu.decode(r)
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        c = np.ascontiguousarray(m[y0:y1, x0:x1])
        ncomp = cv2.connectedComponents(c, connectivity=8)[0] - 1
        out.append((y0, y1, x0, x1, c.astype(bool), int(c.sum()), len(polys), ncomp))
    return out


def inter_matrix(lab, n, crops):
    I = np.zeros((n, len(crops)), np.float64)
    for g, (y0, y1, x0, x1, c, *_r) in enumerate(crops):
        I[:, g] = np.bincount(lab[y0:y1, x0:x1][c], minlength=n)[:n]
    return I[1:]


def region_feats(lab, n, pm):
    fl = lab.ravel()
    area = np.bincount(fl, minlength=n)[1:].astype(np.float64)
    mean = np.bincount(fl, weights=pm.ravel().astype(np.float64), minlength=n)[1:] / np.maximum(area, 1) / 255.0
    return area, mean


def labels_cc(pm, t):
    return cv2.connectedComponents((pm > int(round(t * 255))).astype(np.uint8), connectivity=8)


def grow(lab, allowed, iters):
    lab = lab.astype(np.float32)
    for _ in range(iters):
        d = cv2.dilate(lab, K3)
        lab = np.where((lab == 0) & allowed, d, lab)
    return lab.astype(np.int32)


def variants(pm):
    out = {}
    for t in (0.10, 0.15, 0.20, 0.25, 0.35):
        out[f"cc>{t:.2f}"] = labels_cc(pm, t)
    n, lab = out["cc>0.25"]
    every = np.ones_like(pm, bool)
    out["cc>0.25 +dil2"] = (n, grow(lab, every, 2))
    out["cc>0.25 +dil4"] = (n, grow(lab, every, 4))
    out["cc>0.25 +dil7"] = (n, grow(lab, every, 7))
    core = cv2.erode((lab > 0).astype(np.uint8), K3, iterations=2)
    out["cc>0.25 -ero2"] = (n, lab * core)
    for tc in (0.5, 0.7):
        nc, labc = labels_cc(pm, tc)
        out[f"core>{tc} grow to >0.15"] = (nc, grow(labc, pm > int(0.15 * 255), 12))
    return out


def pq_terms(I, area, ga, keep):
    """sum IoU, TP, FP, FN for one record."""
    if I.shape[0] == 0 or I.shape[1] == 0:
        return np.array([0.0, 0.0, float(keep.sum()), float(I.shape[1])])
    iou = I / (area[:, None] + ga[None, :] - I)
    hit = (iou > 0.5) & keep[:, None]
    tp = float(hit.sum())
    return np.array([float(iou[hit].sum()), tp, float(keep.sum()) - tp, float(I.shape[1]) - tp])


def one(args):
    stem, recs, zpath = args
    with zipfile.ZipFile(zpath) as z:
        pm = cv2.imdecode(np.frombuffer(z.read(stem + ".png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    recs = [gt_crops(r) for r in recs if r]
    nrec = len(recs)
    res = {"nrec": nrec, "var": {}, "fn": Counter(), "fp": Counter(), "gt": Counter(), "ratio": []}
    # ---- variants x mean thresholds
    for name, (n, lab) in variants(pm).items():
        area, mean = region_feats(lab, n, pm)
        acc = np.zeros((len(MEANS), 4))
        for crops in recs:
            ga = np.array([c[5] for c in crops], np.float64)
            I = inter_matrix(lab, n, crops)
            for j, m in enumerate(MEANS):
                acc[j] += pq_terms(I, area, ga, (area >= MIN_AREA) & (mean >= m))
        res["var"][name] = acc
    # ---- autopsy of the baseline rule
    n, lab = labels_cc(pm, BASE_T)
    area, mean = region_feats(lab, n, pm)
    keep = (area >= MIN_AREA) & (mean >= BASE_MEAN)
    Is = [inter_matrix(lab, n, crops) for crops in recs]
    gas = [np.array([c[5] for c in crops], np.float64) for crops in recs]
    votes = np.zeros(len(area))
    orc = {k: np.zeros(4) for k in ("base", "oracle_keep_record", "oracle_keep_image", "oracle_group", "oracle_group_keep")}
    for I, ga in zip(Is, gas):
        if I.size:
            votes += ((I / (area[:, None] + ga[None, :] - I)) > 0.5).any(1)
    img_keep = (votes * 2 >= nrec) & (votes > 0)
    for r, (I, ga, crops) in enumerate(zip(Is, gas, recs)):
        ng = len(ga)
        iou = I / (area[:, None] + ga[None, :] - I) if I.size else np.zeros((len(area), ng))
        hit_any = iou > 0.5
        hit = hit_any & keep[:, None]
        orc["base"] += pq_terms(I, area, ga, keep)
        s_all = float(iou[hit_any].sum())
        tp_all = float(hit_any.sum())
        orc["oracle_keep_record"] += np.array([s_all, tp_all, 0.0, ng - tp_all])
        orc["oracle_keep_image"] += pq_terms(I, area, ga, img_keep)
        # oracle grouping: union of every region lying mostly inside the annotated filament
        inside = I > 0.5 * area[:, None]
        gi = (I * inside).sum(0)
        gu = ga + (area[:, None] * inside).sum(0) - gi
        giou = np.where(gu > 0, gi / np.maximum(gu, 1), 0)
        single = np.where(hit_any, iou, 0).max(0) if len(area) else np.zeros(ng)
        best = np.maximum(giou, single)
        ok = best > 0.5
        used = (inside & ok[None, :]).any(1) | hit_any.any(1)
        orc["oracle_group"] += np.array([best[ok].sum(), ok.sum(), float((keep & ~used).sum()), ng - ok.sum()])
        orc["oracle_group_keep"] += np.array([best[ok].sum(), ok.sum(), 0.0, ng - ok.sum()])
        other = np.zeros(len(area))
        for r2, I2 in enumerate(Is):
            if r2 != r and I2.size:
                other = np.maximum(other, I2.sum(1))
        cov_c = I.sum(1) / np.maximum(area, 1) if ng else np.zeros(len(area))
        for g in range(ng):
            a_g, npoly, ncomp = crops[g][5], crops[g][6], crops[g][7]
            res["gt"]["n"] += 1
            res["gt"]["multi_polygon"] += int(npoly > 1)
            res["gt"]["multi_component"] += int(ncomp > 1)
            res["gt"]["area<320"] += int(a_g < MIN_AREA)
            res["gt"]["area<1000"] += int(a_g < 1000)
            if hit[:, g].any():
                c = int(np.argmax(hit[:, g] * iou[:, g]))
                res["ratio"].append((area[c] / a_g, I[c, g] / area[c], I[c, g] / a_g, float(iou[c, g])))
                continue
            col = I[:, g]
            if hit_any[:, g].any():
                res["fn"]["1 hit existed but the keep rule dropped it"] += 1
            elif col.sum() < 0.2 * a_g:
                res["fn"]["2 model saw nothing there" + (" (tiny, <320px)" if a_g < MIN_AREA else "")] += 1
            elif giou[g] > 0.5 and inside[:, g].sum() >= 2:
                res["fn"]["3 model broke it in pieces (grouping would fix)" + (" [annotation itself multi-part]" if ncomp > 1 else "")] += 1
            else:
                big = np.nonzero(col > 0.5 * a_g)[0]
                if len(big):
                    c = big[0]
                    if int((I[c] > 0.5 * ga).sum()) >= 2:
                        res["fn"]["4 model glued it to a neighbour (splitting would fix)"] += 1
                    else:
                        res["fn"]["5 model mask much bigger than annotation"] += 1
                elif iou[:, g].max() > 0.3:
                    res["fn"]["6 near miss, IoU 0.3-0.5 (boundary)"] += 1
                else:
                    res["fn"]["7 partial / other"] += 1
        for c in np.nonzero(keep & ~hit.any(1))[0]:
            if ng == 0 or cov_c[c] < 0.1:
                if other[c] > 0.3 * area[c]:
                    res["fp"]["a nothing in this record, but ANOTHER annotator of the same frame marked it"] += 1
                else:
                    res["fp"]["b no annotator marked it" + ("" if nrec > 1 else " (single-annotator frame)")] += 1
            elif (I[c] > 0.5 * ga).sum() >= 2:
                res["fp"]["c glues 2+ annotated filaments"] += 1
            elif (I[c] > 0.5 * area[c]).any():
                res["fp"]["d fragment of a bigger annotated filament"] += 1
            elif iou[c].max() > 0.3:
                res["fp"]["e near miss, IoU 0.3-0.5"] += 1
            else:
                res["fp"]["f partial / other"] += 1
    res["orc"] = orc
    return res


def pq(v):
    s, tp, fp, fn = v
    return s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)


def main():
    ann, zpath, out = sys.argv[1:4]
    records = load_records(ann)
    with zipfile.ZipFile(zpath) as z:
        have = {Path(n).stem for n in z.namelist()}
    jobs = [(s, records[s], zpath) for s in sorted(records) if s in have]
    with Pool(6) as p:
        rs = p.map(one, jobs, chunksize=8)
    rep = {"frames": len(rs)}
    print("frames", len(rs))
    print("\n== segmentation variants (area>=320), PQ by mean-probability threshold", MEANS)
    rep["variants"] = {}
    for name in rs[0]["var"]:
        acc = sum(r["var"][name] for r in rs)
        row = [round(pq(a), 4) for a in acc]
        rep["variants"][name] = row
        print(f"{name:28s}", row, " best", max(row))
    print("\n== oracles (what a perfect version of each step would give)")
    rep["oracles"] = {}
    for k in rs[0]["orc"]:
        v = sum(r["orc"][k] for r in rs)
        rep["oracles"][k] = [round(pq(v), 4)] + [float(x) for x in v]
        print(f"{k:22s} PQ {pq(v):.4f}  sumIoU {v[0]:.0f} TP {v[1]:.0f} FP {v[2]:.0f} FN {v[3]:.0f}")
    print("\n== baseline PQ by number of annotators on the frame")
    for k in (1, 2, 3):
        v = sum(r["orc"]["base"] for r in rs if r["nrec"] == k)
        vk = sum(r["orc"]["oracle_keep_image"] for r in rs if r["nrec"] == k)
        print(f"{k} annotator(s): frames {sum(r['nrec'] == k for r in rs)}  PQ {pq(v):.4f}  TP {v[1]:.0f} FP {v[2]:.0f} FN {v[3]:.0f}   image-oracle-keep {pq(vk):.4f}")
    for key, title in (("gt", "annotated filaments"), ("fn", "missed annotated filaments (FN)"), ("fp", "unmatched predictions (FP)")):
        c = sum((r[key] for r in rs), Counter())
        tot = c["n"] if key == "gt" else sum(c.values())
        print(f"\n== {title}: {tot}")
        rep[key] = dict(c)
        for k, v in sorted(c.items()):
            print(f"  {v:6d} {100 * v / max(tot, 1):5.1f}%  {k}")
    ra = np.array([x for r in rs for x in r["ratio"]])
    print("\n== matched pairs: median pred/annot area %.3f, precision %.3f, recall %.3f, IoU %.3f" % tuple(np.median(ra, 0)))
    rep["matched_medians"] = [float(x) for x in np.median(ra, 0)]
    json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
