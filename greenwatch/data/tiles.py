"""Cut aligned imagery + labels into tiles with a leakage-free spatial split.

The scene is divided into large spatial blocks; every block goes entirely to
train, val or test. Neighbouring tiles are strongly correlated, so a random
per-tile split would inflate metrics — block split keeps the evaluation honest.

Two years are tiled:
  train_year          -> train / val / test   (same-year generalisation in space)
  temporal_test_year  -> test only            (generalisation in time: other year's imagery + labels)
"""
from __future__ import annotations

import argparse

import numpy as np

from greenwatch.config import load_config, p, save_json
from greenwatch.features import compute_features
from greenwatch.raster import read_tif


def labels_from_worldcover(wc: np.ndarray, green_classes: list[int]) -> np.ndarray:
    y = np.isin(wc, green_classes).astype(np.uint8)
    y[wc == 0] = 255  # nodata -> ignored by loss and metrics
    return y


def assign_blocks(shape, block, split, seed):
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


def tile_year(cfg, year, blocks, splits_wanted):
    ds = cfg["dataset"]
    t, block = ds["tile_size"], ds["block_size"]
    raw = p(cfg, "raw")
    s2, _, _, _ = read_tif(raw / f"s2_{year}.tif")
    wc, _, _, _ = read_tif(raw / f"wc_{year}.tif")
    x, valid = compute_features(s2, cfg["imagery"]["bands"], cfg["model"]["extra_indices"])
    y = labels_from_worldcover(wc[0], cfg["labels"]["green_classes"])
    y[~valid] = 255

    out = {s: {"X": [], "y": [], "origin": []} for s in splits_wanted}
    H, W = y.shape
    for r in range(0, H - t + 1, t):
        for c in range(0, W - t + 1, t):
            split = blocks[r // block, c // block]
            if split not in out:
                continue
            yt = y[r:r + t, c:c + t]
            if (yt == 255).mean() > 0.5:
                continue
            out[split]["X"].append(x[:, r:r + t, c:c + t].astype(np.float16))
            out[split]["y"].append(yt)
            out[split]["origin"].append((r, c))
    tiles = p(cfg, "tiles")
    counts = {}
    for s, d in out.items():
        if not d["X"]:
            continue
        np.savez_compressed(tiles / f"{year}_{s}.npz", X=np.stack(d["X"]), y=np.stack(d["y"]),
                            origin=np.array(d["origin"]))
        lab = np.stack(d["y"])
        counts[s] = {"tiles": len(d["X"]), "green_frac": float((lab == 1).sum() / max((lab != 255).sum(), 1))}
    return counts


def main(argv=None, cfg=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = cfg or load_config(a.config)
    ds = cfg["dataset"]
    s2, _, _, _ = read_tif(p(cfg, "raw") / f"s2_{ds['train_year']}.tif")
    blocks = assign_blocks(s2.shape[1:], ds["block_size"], ds["split"], ds["seed"])
    summary = {
        "blocks": blocks.tolist(),
        str(ds["train_year"]): tile_year(cfg, ds["train_year"], blocks, ["train", "val", "test"]),
        str(ds["temporal_test_year"]): tile_year(cfg, ds["temporal_test_year"], blocks, ["test"]),
    }
    save_json(summary, p(cfg, "tiles") / "summary.json")
    for k, v in summary.items():
        if k != "blocks":
            print(f"[tiles {k}] {v}")
    return summary


if __name__ == "__main__":
    main()
