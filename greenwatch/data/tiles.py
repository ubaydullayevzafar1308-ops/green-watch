"""Build leakage-free train / val / test data with a buffered spatial block split.

Why blocks: neighbouring pixels (and tiles) are strongly correlated, so a random
per-tile split leaks information from train into test and inflates metrics.
The scene is cut into `block_size` blocks, each block goes entirely to one split.

Why a buffer: even block-wise, pixels right at a train/test border are near
duplicates of each other. Pixels closer than `buffer_px / 2` to a block of a
DIFFERENT split are removed from all splits, i.e. there is a `buffer_px` wide
no-man's-land between splits. Buffer pixels never carry a label (255).

Why folds: with ~25 blocks one random split gives a handful of test blocks, and
the score depends a lot on which blocks they are. We repeat the split with
several seeds (`dataset.fold_seeds`) and report mean ± std over folds.

Shared (data/tiles/):
  {train_year}_X.npy                  full-scene features (float16, memmap-able), same for all folds
Per fold (data/tiles/fold{k}/):
  {train_year}_train_y.npy            full-scene labels, 255 everywhere outside the train core
  {train_year}_train_origins.npy      top-left corners of every `tile_size` crop that lies
                                      ENTIRELY inside the train core -> random-crop sampler
  {train_year}_val.npz, _test.npz     fixed non-overlapping tiles (same as before)
  {temporal_test_year}_test.npz       other year, test blocks only (generalisation in time)
  summary.json                        block map, tile counts, green fraction
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage

from greenwatch.config import load_config, p, save_json
from greenwatch.features import compute_features
from greenwatch.raster import read_tif

SPLITS = ("train", "val", "test")
MAX_NODATA_FRAC = 0.5  # skip tiles / crops whose labels are mostly ignored


def labels_from_worldcover(wc: np.ndarray, green_classes: list[int]) -> np.ndarray:
    y = np.isin(wc, green_classes).astype(np.uint8)
    y[wc == 0] = 255  # nodata -> ignored by loss and metrics
    return y


def assign_blocks(shape, block, split, seed):
    """Randomly assign every block to train / val / test. Returns (nby, nbx) array of split names."""
    nby, nbx = int(np.ceil(shape[0] / block)), int(np.ceil(shape[1] / block))
    n = nby * nbx
    order = np.random.default_rng(seed).permutation(n)
    n_tr = max(1, int(round(split["train"] * n)))
    n_va = max(1, int(round(split["val"] * n))) if n >= 3 else 0
    names = np.empty(n, dtype=object)
    names[order[:n_tr]] = "train"
    names[order[n_tr:n_tr + n_va]] = "val"
    names[order[n_tr + n_va:]] = "test"
    return names.reshape(nby, nbx)


def split_masks(shape, blocks, block, buffer_px):
    """Per-pixel boolean mask for each split, with the inter-split buffer removed.

    A pixel is dropped when another split lies within buffer_px // 2 pixels (Chebyshev
    distance), so two different splits are always >= buffer_px pixels apart.
    """
    H, W = shape
    code = {s: i for i, s in enumerate(SPLITS)}
    block_codes = np.vectorize(code.get)(blocks).astype(np.int8)
    pix = np.kron(block_codes, np.ones((block, block), np.int8))[:H, :W]
    half = buffer_px // 2
    masks = {}
    for s, i in code.items():
        m = pix == i
        if half > 0:
            near_other = ndimage.maximum_filter((pix != i).astype(np.uint8), size=2 * half + 1, mode="nearest") > 0
            m &= ~near_other
        masks[s] = m
    return masks


def crop_origins(core: np.ndarray, label: np.ndarray, t: int, stride: int = 8) -> np.ndarray:
    """Top-left (r, c) of every t×t window fully inside `core` with mostly valid labels."""
    H, W = core.shape
    if H < t or W < t:
        return np.zeros((0, 2), np.int32)

    def window_sum(a):
        ii = np.pad(a.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return ii[t:, t:] - ii[:-t, t:] - ii[t:, :-t] + ii[:-t, :-t]  # (H-t+1, W-t+1)

    inside = window_sum(core) == t * t
    ok_lab = window_sum((label != 255) & core) >= (1 - MAX_NODATA_FRAC) * t * t
    good = inside & ok_lab
    sub = np.zeros_like(good)
    sub[::stride, ::stride] = True  # 8-px grid of origins: plenty of variety, small index
    rr, cc = np.nonzero(good & sub)
    return np.stack([rr, cc], 1).astype(np.int32)


def _load_year(cfg, year):
    raw = p(cfg, "raw")
    s2, _, _, _ = read_tif(raw / f"s2_{year}.tif")
    wc, _, _, _ = read_tif(raw / f"wc_{year}.tif")
    x, valid = compute_features(s2, cfg["imagery"]["bands"], cfg["model"]["extra_indices"])
    y = labels_from_worldcover(wc[0], cfg["labels"]["green_classes"])
    y[~valid] = 255
    return x, y


def fixed_tiles(x, y, mask, t):
    """Non-overlapping t×t tiles whose top-left pixel block belongs to `mask`; labels outside mask -> 255."""
    H, W = y.shape
    Xs, ys, origins = [], [], []
    for r in range(0, H - t + 1, t):
        for c in range(0, W - t + 1, t):
            m = mask[r:r + t, c:c + t]
            if not m.any():
                continue
            yt = np.where(m, y[r:r + t, c:c + t], 255).astype(np.uint8)
            if (yt == 255).mean() > MAX_NODATA_FRAC:
                continue
            Xs.append(x[:, r:r + t, c:c + t].astype(np.float16))
            ys.append(yt)
            origins.append((r, c))
    return Xs, ys, origins


def _green_frac(lab):
    return float((lab == 1).sum() / max((lab != 255).sum(), 1))


def build_fold(cfg, seed, out_dir, data=None):
    """Write one fold's train scene + crop origins, val/test tiles and summary. Returns the summary."""
    ds = cfg["dataset"]
    t, block, buf = ds["tile_size"], ds["block_size"], ds.get("buffer_px", 0)
    ty, tty = ds["train_year"], ds["temporal_test_year"]
    data = data or {}
    x, y = data.get(ty) or _load_year(cfg, ty)
    blocks = assign_blocks(y.shape, block, ds["split"], seed)
    masks = split_masks(y.shape, blocks, block, buf)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train: whole scene + valid crop origins (labels only inside the train core)
    y_tr = np.where(masks["train"], y, 255).astype(np.uint8)
    origins = crop_origins(masks["train"], y_tr, t)
    x_path = train_features_path(cfg)
    if not x_path.exists():
        np.save(x_path, x.astype(np.float16))
    np.save(out_dir / f"{ty}_train_y.npy", y_tr)
    np.save(out_dir / f"{ty}_train_origins.npy", origins)
    counts = {ty: {"train": {"crop_origins": int(len(origins)), "pixels": int((y_tr != 255).sum()),
                             "green_frac": _green_frac(y_tr)}}}

    def save_tiles(year, xx, yy, split):
        Xs, ys, org = fixed_tiles(xx, yy, masks[split], t)
        if not Xs:
            return None
        lab = np.stack(ys)
        np.savez_compressed(out_dir / f"{year}_{split}.npz", X=np.stack(Xs), y=lab, origin=np.array(org))
        return {"tiles": len(Xs), "green_frac": _green_frac(lab)}

    for split in ("val", "test"):
        counts[ty][split] = save_tiles(ty, x, y, split)
    xt, yt = data.get(tty) or _load_year(cfg, tty)
    counts[tty] = {"test": save_tiles(tty, xt, yt, "test")}

    summary = {"seed": int(seed), "block_size": block, "buffer_px": buf, "blocks": blocks.tolist(),
               "counts": {str(k): v for k, v in counts.items()}}
    save_json(summary, out_dir / "summary.json")
    return summary


def train_features_path(cfg):
    return p(cfg, "tiles") / f"{cfg['dataset']['train_year']}_X.npy"


def fold_dir(cfg, k):
    return p(cfg, "tiles") / f"fold{k}"


def main(argv=None, cfg=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = cfg or load_config(a.config)
    ds = cfg["dataset"]
    data = {yr: _load_year(cfg, yr) for yr in (ds["train_year"], ds["temporal_test_year"])}
    train_features_path(cfg).unlink(missing_ok=True)  # rebuilt from the current raw composite
    summaries = []
    for k, seed in enumerate(ds["fold_seeds"]):
        s = build_fold(cfg, seed, fold_dir(cfg, k), data)
        summaries.append(s)
        print(f"[tiles fold{k} seed={seed}] {s['counts']}")
    return summaries


if __name__ == "__main__":
    main()
