"""k2 — Solar Filament Segmentation 2026: 5-fold native-resolution U-Net (ResNet-34).

One script does everything, on a Kaggle T4 x2 session:
  1. cache    : decode the 707 training / 180 test frames, rasterise the annotator
                polygons into a soft target (share of annotators that marked a pixel)
  2. smoke    : a 2-minute end-to-end rehearsal of every later stage on a few images
  3. train    : 5 folds (grouped by observation date), two folds at a time (one per GPU)
  4. predict  : out-of-fold maps for training frames, 5-fold mean maps for test frames
  5. decide   : candidates = connected regions of p > t_lo; keep rule tuned on the
                out-of-fold maps with the host's pooled Panoptic Quality
  6. write    : submission.csv (pixel-disjoint COCO RLE), weights, maps, report

Environment knobs (used by the local CPU rehearsal): SFS_H, SFS_CROP, SFS_BUDGET_S,
SFS_MAX_IT, SFS_FOLDS, SFS_LIMIT, SFS_OUT, SFS_CACHE, SFS_ROOT, SFS_PRETRAINED.
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

H = W = int(os.environ.get("SFS_H", 2048))          # native frame size
CROP = int(os.environ.get("SFS_CROP", 640))         # training crop (multiple of 32)
BUDGET_S = float(os.environ.get("SFS_BUDGET_S", 40 * 60))  # training seconds per fold
MAX_IT = int(os.environ.get("SFS_MAX_IT", 12000))
MIN_IT = int(os.environ.get("SFS_MIN_IT", 2500))
NFOLD = int(os.environ.get("SFS_FOLDS", 5))
LIMIT = int(os.environ.get("SFS_LIMIT", 0))         # 0 = all images
BATCH = int(os.environ.get("SFS_BATCH", 8))
PRETRAINED = os.environ.get("SFS_PRETRAINED", "1") == "1"
ENCODER = os.environ.get("SFS_ENCODER", "resnet34")
OUT = Path(os.environ.get("SFS_OUT", "/kaggle/working"))
def _pick_cache() -> Path:
    import shutil
    if os.environ.get("SFS_CACHE"):
        return Path(os.environ["SFS_CACHE"])
    for base in ("/kaggle/temp", "/tmp", "/dev/shm"):
        try:
            if os.path.isdir(base) and shutil.disk_usage(base).free > 14e9:
                return Path(base) / "sfs"
        except OSError:
            pass
    raise RuntimeError("no scratch directory with 14 GB free")


CACHE = _pick_cache()
os.environ["SFS_CACHE"] = str(CACHE)                       # worker processes use the same directory
DISK_R = 950.0 * H / 2048.0                          # normalisation disk radius
SEED = 2026
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# --------------------------------------------------------------------------- data
def find_root() -> Path:
    if os.environ.get("SFS_ROOT"):
        return Path(os.environ["SFS_ROOT"])
    hits = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026", recursive=True)
    if not hits:
        raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")
    return Path(sorted(hits)[0])


def load_annotations(root: Path):
    """Return (stems, records): records[stem] = list of annotator records, each a list of polygons."""
    f = glob.glob(str(root / "train" / "*.json"))
    if len(f) != 1:
        raise FileNotFoundError(f"expected one annotation json, found {f}")
    d = json.load(open(f[0], encoding="utf-8"))
    by_rec: dict[str, list] = {}
    for a in d["annotations"]:
        seg = a["segmentation"]
        if isinstance(seg, str):
            seg = json.loads(seg)
        polys = [[float(v) for v in p] for p in seg if len(p) >= 6]
        if polys:
            by_rec.setdefault(str(a["image_id"]), []).append(polys)
    records: dict[str, list] = {}
    for im in d["images"]:
        stem = Path(im["file_name"]).stem
        records.setdefault(stem, []).append(by_rec.get(str(im["id"]), []))
    stems = sorted(records)
    return stems, records


def poly_rle(polys, scale=1.0):
    """One annotated filament -> COCO RLE (pycocotools rasterisation, as the host uses)."""
    from pycocotools import mask as mu
    if scale != 1.0:
        polys = [[v * scale for v in p] for p in polys]
    r = mu.frPyObjects(polys, H, W)
    return mu.merge(r) if len(r) > 1 else r[0]


def disk_mask():
    yy, xx = np.ogrid[:H, :W]
    return ((yy - (H - 1) / 2.0) ** 2 + (xx - (W - 1) / 2.0) ** 2) <= DISK_R ** 2


def read_gray(path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f"cannot decode {path}")
    if im.shape != (H, W):
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return im


def _cache_one(args):
    kind, i, path, recs, n, n_train = args
    from pycocotools import mask as mu
    global DISK
    if DISK is None:
        DISK = disk_mask()
    img = np.memmap(CACHE / f"{kind}_img.u8", np.uint8, "r+", shape=(n, H, W))
    g = read_gray(path)
    img[i] = g
    v = g[DISK].astype(np.float32) / 255.0
    stat = (float(v.mean()), float(v.std()) + 1e-6)
    pos = np.zeros((0, 2), np.int16)
    if kind == "train":
        tgt = np.memmap(CACHE / "train_tgt.u8", np.uint8, "r+", shape=(n_train, H, W))
        acc = np.zeros((H, W), np.float32)
        for rec in recs:                       # one annotator record
            if rec:
                rl = [poly_rle(p, H / 2048.0) for p in rec]
                acc += mu.decode(mu.merge(rl) if len(rl) > 1 else rl[0])
        t = np.round(acc / max(len(recs), 1) * 255).astype(np.uint8)
        tgt[i] = t
        ys, xs = np.nonzero(t)
        if len(ys):
            k = np.random.default_rng(i).choice(len(ys), min(len(ys), 4000), replace=False)
            pos = np.stack([ys[k], xs[k]], 1).astype(np.int16)
        tgt.flush()
    img.flush()
    return i, stat, pos


def build_cache(root: Path):
    global N_TRAIN, N_TEST
    from multiprocessing import Pool
    CACHE.mkdir(parents=True, exist_ok=True)
    stems, records = load_annotations(root)
    test_paths = sorted(glob.glob(str(root / "test" / "test_images" / "*.jp*g")))
    if LIMIT:
        stems, test_paths = stems[:LIMIT], test_paths[: max(2, LIMIT // 4)]
    N_TRAIN, N_TEST = len(stems), len(test_paths)
    if N_TEST != 180 and not LIMIT:
        raise RuntimeError(f"expected 180 test frames, found {N_TEST}")
    for kind, n in (("train", N_TRAIN), ("test", N_TEST)):
        np.memmap(CACHE / f"{kind}_img.u8", np.uint8, "w+", shape=(n, H, W)).flush()
    np.memmap(CACHE / "train_tgt.u8", np.uint8, "w+", shape=(N_TRAIN, H, W)).flush()
    train_dir = root / "train" / "train_images"
    jobs = [("train", i, _find_image(train_dir, s), records[s], N_TRAIN, N_TRAIN) for i, s in enumerate(stems)]
    jobs += [("test", i, p, None, N_TEST, N_TRAIN) for i, p in enumerate(test_paths)]
    stats = {"train": np.zeros((N_TRAIN, 2), np.float32), "test": np.zeros((N_TEST, 2), np.float32)}
    pos = [None] * N_TRAIN
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        for k, (i, st, ps) in enumerate(pool.imap(_cache_one, jobs, chunksize=4)):
            kind = jobs[k][0]
            stats[kind][i] = st
            if kind == "train":
                pos[i] = ps
            if k % 200 == 0:
                log(f"cache {k}/{len(jobs)}")
    # folds: all frames of one calendar date stay together
    dates = sorted({s[:8] for s in stems})
    perm = np.random.default_rng(SEED).permutation(len(dates))
    fold_of_date = {dates[j]: int(r % NFOLD) for r, j in enumerate(perm)}
    folds = [fold_of_date[s[:8]] for s in stems]
    np.save(CACHE / "train_stats.npy", stats["train"])
    np.save(CACHE / "test_stats.npy", stats["test"])
    np.save(CACHE / "pos.npy", np.array(pos, dtype=object), allow_pickle=True)
    meta = dict(stems=stems, folds=folds, test_stems=[Path(p).stem for p in test_paths],
                n_train=N_TRAIN, n_test=N_TEST, root=str(root))
    json.dump(meta, open(CACHE / "meta.json", "w"))
    log(f"cache done: {N_TRAIN} train frames, {N_TEST} test frames, folds {np.bincount(folds).tolist()}")
    return meta


def _find_image(d: Path, stem: str) -> str:
    for ext in (".jpeg", ".jpg"):
        p = d / f"{stem}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"{stem} not in {d}")


N_TRAIN = N_TEST = 0
DISK = None


# --------------------------------------------------------------------------- model
def build_model(pretrained: bool):
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Sequential):
        def __init__(self, cin, cout):
            super().__init__(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        """U-Net: ImageNet encoder (1 input channel), 5 skip levels, output at input resolution."""

        def __init__(self):
            super().__init__()
            self.enc = timm.create_model(ENCODER, features_only=True, pretrained=pretrained, in_chans=1)
            ch = self.enc.feature_info.channels()          # strides 2, 4, 8, 16, 32
            dec = [256, 128, 64, 32]
            self.blocks = nn.ModuleList()
            cin = ch[-1]
            for skip, cout in zip(ch[-2::-1], dec):
                self.blocks.append(Block(cin + skip, cout))
                cin = cout
            self.last = Block(cin, 16)
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            f = self.enc(x)
            y = f[-1]
            for blk, skip in zip(self.blocks, f[-2::-1]):
                y = F.interpolate(y, size=skip.shape[-2:], mode="nearest")
                y = blk(torch.cat([y, skip], 1))
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            return self.head(self.last(y))

    return UNet()


# --------------------------------------------------------------------------- worker
def worker(fold: int, smoke: bool):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    meta = json.load(open(CACHE / "meta.json"))
    n_tr, n_te = meta["n_train"], meta["n_test"]
    folds = np.array(meta["folds"])
    tr_idx = np.nonzero(folds != fold)[0]
    va_idx = np.nonzero(folds == fold)[0]
    if smoke:
        tr_idx, va_idx = tr_idx[:8], va_idx[:2]
    stats = np.load(CACHE / "train_stats.npy")
    tstats = np.load(CACHE / "test_stats.npy")
    pos = np.load(CACHE / "pos.npy", allow_pickle=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"[fold {fold}{' smoke' if smoke else ''}]"

    class Crops(Dataset):
        def __init__(self, n):
            self.n = n
            self.img = self.tgt = None

        def __len__(self):
            return self.n

        def __getitem__(self, k):
            if self.img is None:
                self.img = np.memmap(CACHE / "train_img.u8", np.uint8, "r", shape=(n_tr, H, W))
                self.tgt = np.memmap(CACHE / "train_tgt.u8", np.uint8, "r", shape=(n_tr, H, W))
            rng = np.random.default_rng([SEED, fold, k, int(smoke)])
            i = int(rng.choice(tr_idx))
            s = int(round(CROP * rng.uniform(0.8, 1.25)))
            s = min(s, H)
            if len(pos[i]) and rng.random() < 0.5:        # centre on a filament pixel
                cy, cx = pos[i][rng.integers(len(pos[i]))]
                y0 = int(cy) - s // 2 + int(rng.integers(-s // 4, s // 4 + 1))
                x0 = int(cx) - s // 2 + int(rng.integers(-s // 4, s // 4 + 1))
            else:
                y0, x0 = int(rng.integers(0, H - s + 1)), int(rng.integers(0, W - s + 1))
            y0, x0 = min(max(y0, 0), H - s), min(max(x0, 0), W - s)
            a = np.asarray(self.img[i, y0:y0 + s, x0:x0 + s])
            t = np.asarray(self.tgt[i, y0:y0 + s, x0:x0 + s])
            if s != CROP:
                a = cv2.resize(a, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
                t = cv2.resize(t, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
            kk = int(rng.integers(4))
            a, t = np.rot90(a, kk), np.rot90(t, kk)
            if rng.random() < 0.5:
                a, t = a[:, ::-1], t[:, ::-1]
            x = (a.astype(np.float32) / 255.0 - stats[i, 0]) / stats[i, 1]
            x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.15, 0.15)
            return (torch.from_numpy(np.ascontiguousarray(x))[None],
                    torch.from_numpy(np.ascontiguousarray(t.astype(np.float32) / 255.0))[None])

    max_it = 12 if smoke else MAX_IT
    nw = 0 if (smoke or os.name == "nt") else 2
    loader = DataLoader(Crops(max_it * BATCH), batch_size=BATCH, shuffle=False, num_workers=nw,
                        pin_memory=dev.type == "cuda", drop_last=True, persistent_workers=nw > 0)
    model = build_model(PRETRAINED).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    warm, total, t_start, t_probe = (2 if smoke else 300), max_it, time.time(), None
    budget = 20 if smoke else BUDGET_S
    model.train()
    run_loss, it = 0.0, 0
    for x, t in loader:
        if it == 10:
            t_probe = time.time()
        if it == 60 and not smoke:                         # fit the schedule to the time budget
            rate = 50 / (time.time() - t_probe)
            total = int(min(MAX_IT, max(MIN_IT, rate * budget)))
            print(tag, f"{rate:.2f} it/s -> {total} iterations planned", flush=True)
        if it >= total:
            break
        lr = 5e-4 * (min(1.0, (it + 1) / warm) if it < warm
                     else 0.5 * (1 + math.cos(math.pi * min(1.0, (it - warm) / max(1, total - warm)))) * 0.98 + 0.02)
        for g in opt.param_groups:
            g["lr"] = lr
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        t = t.to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            z = model(x)
        z = z.float()
        p = torch.sigmoid(z)
        dice = 1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
        loss = F.binary_cross_entropy_with_logits(z, t) + dice.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{tag} non-finite loss at iteration {it}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        run_loss += float(loss.detach())
        it += 1
        if it % 250 == 0 or (smoke and it % 4 == 0):
            print(tag, f"it {it}/{total} loss {run_loss / (250 if not smoke else 4):.4f} lr {lr:.2e} "
                       f"{time.time() - t_start:.0f}s", flush=True)
            run_loss = 0.0
    del loader
    name = f"smoke_fold{fold}.pt" if smoke else f"unet_{ENCODER}_fold{fold}.pt"
    wdir = CACHE if smoke else OUT
    half = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save(half, wdir / name)                           # 16-bit file (49 MB); inference below uses these exact values
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in half.items()})
    print(tag, f"trained {it} iterations in {time.time() - t_start:.0f}s, saved {name}", flush=True)

    # ---- full-frame inference with 4 flips
    model.eval()

    @torch.inference_mode()
    def predict(frame: np.ndarray, st) -> np.ndarray:
        x = torch.from_numpy((frame.astype(np.float32) / 255.0 - st[0]) / st[1])[None, None].to(dev)
        acc = torch.zeros((H, W), device=dev)
        for fy, fx in ((0, 0), (0, 1), (1, 0), (1, 1)):
            dims = [d for d, f in ((2, fy), (3, fx)) if f]
            xx = torch.flip(x, dims) if dims else x
            with torch.autocast("cuda", enabled=amp):
                zz = model(xx.contiguous(memory_format=torch.channels_last))
            pp = torch.sigmoid(zz.float())
            acc += (torch.flip(pp, dims) if dims else pp)[0, 0]
        return (acc / 4 * 255).round().clamp(0, 255).byte().cpu().numpy()

    sub = "smoke" if smoke else "run"
    oof_dir, test_dir = CACHE / sub / "oof", CACHE / sub / f"test_f{fold}"
    oof_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    img = np.memmap(CACHE / "train_img.u8", np.uint8, "r", shape=(n_tr, H, W))
    timg = np.memmap(CACHE / "test_img.u8", np.uint8, "r", shape=(n_te, H, W))
    inter = union = 0.0
    tgt = np.memmap(CACHE / "train_tgt.u8", np.uint8, "r", shape=(n_tr, H, W))
    for i in va_idx:
        pm = predict(np.asarray(img[i]), stats[i])
        if not cv2.imwrite(str(oof_dir / f"{meta['stems'][i]}.png"), pm):
            raise IOError("cannot write out-of-fold map")
        a, b = pm > 127, np.asarray(tgt[i]) > 127
        inter += float((a & b).sum())
        union += float(a.sum() + b.sum())
    print(tag, f"out-of-fold maps: {len(va_idx)} frames, pixel Dice@0.5 {2 * inter / max(union, 1):.4f}", flush=True)
    te = range(min(2, n_te)) if smoke else range(n_te)
    for j in te:
        pm = predict(np.asarray(timg[j]), tstats[j])
        cv2.imwrite(str(test_dir / f"{meta['test_stems'][j]}.png"), pm)
    print(tag, f"test maps: {len(te)} frames, total {time.time() - t_start:.0f}s", flush=True)
    (CACHE / sub / f"done_{fold}").write_text("ok")


# --------------------------------------------------------------------------- decision layer
T_LO = (0.25, 0.35, 0.45, 0.55)
GRID_HI = (0.5, 0.7, 0.85, 0.95, 0.99)
GRID_AREA = (60, 120, 200, 320, 500)
GRID_MEAN = (0.0, 0.55, 0.65, 0.75, 0.85)


MAX_CAND = 300          # a frame holds 5-40 filaments; the cap only matters for a barely trained (rehearsal) model


def candidates(pm: np.ndarray, t_lo: float):
    """8-connected regions of p > t_lo -> (label image, region ids, [area@2048, mean p, peak p] per region)."""
    n, lab = cv2.connectedComponents((pm > int(round(t_lo * 255))).astype(np.uint8), connectivity=8)
    if n <= 1:
        return lab, np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
    idx = np.flatnonzero(lab)
    fl, pv = lab.ravel()[idx], pm.ravel()[idx].astype(np.float64)
    area = np.bincount(fl, minlength=n)[1:].astype(np.float64)
    mean = np.bincount(fl, weights=pv, minlength=n)[1:] / np.maximum(area, 1) / 255.0
    order = np.argsort(fl, kind="stable")
    peak = np.maximum.reduceat(pv[order], np.searchsorted(fl[order], np.arange(1, n))) / 255.0
    feat = np.stack([area * (2048.0 / H) ** 2, mean, peak], 1).astype(np.float32)
    ids = np.nonzero(feat[:, 0] >= GRID_AREA[0])[0]
    if len(ids) > MAX_CAND:
        ids = ids[np.argsort(-feat[ids, 0] * feat[ids, 1])[:MAX_CAND]]
        ids.sort()
    return lab, ids + 1, feat[ids]


def region_rles(lab: np.ndarray, ids):
    from pycocotools import mask as mu
    return [mu.encode(np.asfortranarray((lab == k).astype(np.uint8))) for k in ids]


def _score_image(args):
    """For one training frame: candidates at every t_lo and their IoU hits against each annotator record."""
    from pycocotools import mask as mu
    stem, recs, png = args
    pm = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    gts = [[poly_rle(p, H / 2048.0) for p in rec] for rec in recs if rec]
    res = {}
    for t_lo in T_LO:
        lab, ids, feat = candidates(pm, t_lo)
        rl = region_rles(lab, ids)
        hits = []
        for g in gts:
            if rl:
                iou = np.asarray(mu.iou(rl, g, [0] * len(g))).reshape(len(rl), len(g))   # [pred, gt]
                pi, gi = np.nonzero(iou > 0.5)
                hits.append((len(g), pi.astype(np.int32), gi.astype(np.int32), iou[pi, gi].astype(np.float32)))
            else:
                hits.append((len(g), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)))
        res[t_lo] = (feat, hits)
    return stem, res


def pq_parts(feat, hits, t_hi, min_area, min_mean):
    """(sum IoU, TP, FP, FN) of one frame under a keep rule, pooled over its annotator records."""
    k = (feat[:, 2] >= t_hi) & (feat[:, 0] >= min_area) & (feat[:, 1] >= min_mean) if len(feat) else np.zeros(0, bool)
    s = tp = fp = fn = 0.0
    for n_gt, pi, gi, iou in hits:
        m = k[pi] if len(pi) else np.zeros(0, bool)
        s += float(iou[m].sum())
        tp += float(m.sum())
        fp += float(k.sum() - len(np.unique(pi[m])))
        fn += float(n_gt - len(np.unique(gi[m])))
    return s, tp, fp, fn


def tune_and_write(meta, sub: str):
    from multiprocessing import Pool
    from pycocotools import mask as mu
    _, records = load_annotations(Path(meta["root"]))
    oof_dir = CACHE / sub / "oof"
    jobs = [(s, records[s], str(oof_dir / f"{s}.png")) for s in meta["stems"] if (oof_dir / f"{s}.png").exists()]
    log(f"decision layer: scoring candidates on {len(jobs)} out-of-fold frames")
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        scored = dict(pool.imap_unordered(_score_image, jobs, chunksize=4))
    fold_of = dict(zip(meta["stems"], meta["folds"]))
    table = []
    for t_lo in T_LO:
        for t_hi in GRID_HI:
            if t_hi < t_lo:
                continue
            for a in GRID_AREA:
                for mn in GRID_MEAN:
                    tot = np.zeros((NFOLD, 4))
                    for stem, res in scored.items():
                        tot[fold_of[stem]] += pq_parts(*res[t_lo], t_hi, a, mn)
                    T = tot.sum(0)
                    pq = T[0] / max(T[1] + 0.5 * T[2] + 0.5 * T[3], 1e-9)
                    per = (tot[:, 0] / np.maximum(tot[:, 1] + 0.5 * tot[:, 2] + 0.5 * tot[:, 3], 1e-9)).round(4).tolist()
                    table.append(dict(t_lo=t_lo, t_hi=t_hi, min_area=a, min_mean=mn, pq=round(float(pq), 5),
                                      tp=T[1], fp=T[2], fn=T[3], sq=round(float(T[0] / max(T[1], 1)), 4), per_fold=per))
    table.sort(key=lambda r: -r["pq"])
    for r in table[:12]:
        log("rule", {k: r[k] for k in ("t_lo", "t_hi", "min_area", "min_mean", "pq", "sq", "tp", "fp", "fn")}, r["per_fold"])
    best = table[0]
    json.dump(dict(best=best, top=table[:200], n_frames=len(jobs)), open(OUT / "decision_report.json", "w"), indent=1)

    # ---- test: mean of the fold maps, then the chosen rule
    rows, n_img = [], 0
    tdir = OUT / "test_maps"
    tdir.mkdir(exist_ok=True)
    for stem in meta["test_stems"]:
        files = [CACHE / sub / f"test_f{f}" / f"{stem}.png" for f in range(NFOLD)]
        maps = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in files if p.exists()]
        if not maps:
            continue
        if sub == "run" and len(maps) != NFOLD:
            raise RuntimeError(f"{stem}: {len(maps)} fold maps, expected {NFOLD}")
        pm = np.round(np.mean(np.stack(maps).astype(np.float32), 0)).astype(np.uint8)
        cv2.imwrite(str(tdir / f"{stem}.png"), pm)
        lab, cid, feat = candidates(pm, best["t_lo"])
        ids = [int(cid[k]) for k in range(len(feat)) if feat[k, 2] >= best["t_hi"] and feat[k, 0] >= best["min_area"]
               and feat[k, 1] >= best["min_mean"]]
        if H != 2048:                                     # rehearsal only: submission masks are always 2048 x 2048
            lab = cv2.resize(lab.astype(np.int32), (2048, 2048), interpolation=cv2.INTER_NEAREST)
        n_img += bool(ids)
        for j, k in enumerate(ids, 1):
            m = np.asfortranarray((lab == k).astype(np.uint8))
            if not m.any():
                continue
            rle = mu.encode(m)["counts"].decode("ascii")
            back = mu.decode({"size": [2048, 2048], "counts": rle.encode("ascii")})
            if back.shape != (2048, 2048) or int(back.sum()) != int(m.sum()):
                raise RuntimeError("RLE round trip failed")
            rows.append((f"{stem}_{j}", rle))
    with open(OUT / "submission.csv", "w", newline="") as f:
        f.write("filament_id,segmentation_rle\n")
        for fid, rle in rows:
            if "," in rle or '"' in rle or "\n" in rle:
                raise RuntimeError("RLE needs quoting; refusing to write")
            f.write(f"{fid},{rle}\n")
    log(f"submission.csv: {len(rows)} filaments on {n_img}/{len(meta['test_stems'])} frames; "
        f"out-of-fold PQ {best['pq']:.4f} with rule {best['t_lo']}/{best['t_hi']}/{best['min_area']}/{best['min_mean']}")
    return best, len(rows)


# --------------------------------------------------------------------------- orchestration
def run_workers(folds, smoke: bool, n_gpu: int):
    """Run one worker process per fold, at most one per GPU at a time; abort everything if one dies."""
    queue, live = list(folds), {}
    while queue or live:
        for gpu in range(max(n_gpu, 1)):
            if gpu not in live and queue:
                f = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)) if n_gpu else dict(os.environ)
                cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(f)] + (["--smoke"] if smoke else [])
                live[gpu] = (f, subprocess.Popen(cmd, env=env))
                log(f"started fold {f} on gpu {gpu}{' (smoke)' if smoke else ''}")
        time.sleep(2 if smoke else 15)
        for gpu, (f, p) in list(live.items()):
            rc = p.poll()
            if rc is None:
                continue
            del live[gpu]
            if rc != 0:
                for _, q in live.values():
                    q.kill()
                raise RuntimeError(f"fold {f} worker exited with code {rc}; aborting the run")
            log(f"fold {f} finished")


def zip_dir(src: Path, dst: Path):
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(src.glob("*.png")):
            z.write(p, p.name)


def main():
    import torch
    n_gpu = torch.cuda.device_count()
    log(f"torch {torch.__version__}, GPUs {n_gpu}, H={H}, crop={CROP}, budget/fold {BUDGET_S:.0f}s, folds {NFOLD}")
    if "/kaggle" in str(OUT) and n_gpu < 1:
        raise RuntimeError("no GPU attached; refusing to run")
    OUT.mkdir(parents=True, exist_ok=True)
    root = find_root()
    build_model(PRETRAINED)                                # fails here, in seconds, if the pretrained weights are unreachable
    meta = build_cache(root)

    # ---- rehearsal: every later stage on a handful of frames
    log("smoke rehearsal")
    run_workers(range(NFOLD), True, n_gpu)
    tune_and_write(meta, "smoke")
    for f in ("submission.csv", "decision_report.json"):
        (OUT / f).unlink(missing_ok=True)
    log("smoke rehearsal passed")

    # ---- the real run
    run_workers(range(NFOLD), False, n_gpu)
    zip_dir(CACHE / "run" / "oof", OUT / "oof_maps.zip")
    best, n_rows = tune_and_write(meta, "run")
    zip_dir(OUT / "test_maps", OUT / "test_maps.zip")
    for p in (OUT / "test_maps").glob("*.png"):
        p.unlink()
    (OUT / "test_maps").rmdir()
    json.dump(dict(meta=dict(stems=meta["stems"], folds=meta["folds"], test_stems=meta["test_stems"]),
                   best_rule=best, rows=n_rows, seconds=time.time() - T0, encoder=ENCODER, crop=CROP),
              open(OUT / "run_report.json", "w"))
    log("all done")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(int(sys.argv[sys.argv.index("--worker") + 1]), "--smoke" in sys.argv)
    else:
        main()
