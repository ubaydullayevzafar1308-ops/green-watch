"""Run the trained model on every year's composite.

Outputs:
  outputs/pred/prob_{year}.tif     uint8, P(green)*250, 255 = nodata
  outputs/overlays/green_{year}.png, rgb_{year}.png   web overlays (EPSG:3857)
  outputs/overlays/index.json      years, lat/lon bounds, threshold
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from greenwatch.config import get_device, load_config, p, save_json
from greenwatch.features import compute_features
from greenwatch.model import load_checkpoint, predict_scene
from greenwatch.raster import read_tif, to_web_overlay, write_tif

NODATA = 255


def available_years(cfg):
    raw = p(cfg, "raw")
    return [y for y in cfg["imagery"]["years"] if (raw / f"s2_{y}.tif").exists()]


def prob_to_uint8(prob, valid):
    out = np.round(np.clip(prob, 0, 1) * 250).astype(np.uint8)
    out[~valid] = NODATA
    return out


def read_prob(cfg, year):
    arr, tr, crs, _ = read_tif(p(cfg, "pred") / f"prob_{year}.tif")
    a = arr[0]
    valid = a != NODATA
    return np.where(valid, a / 250.0, np.nan).astype(np.float32), valid, tr, crs


def rgb_overlay(s2, bands, transform, crs):
    idx = [bands.index(b) for b in ("B04", "B03", "B02")]
    rgb = s2[idx].astype(np.float32) / 10000.0
    rgb = np.where(s2[idx] > 0, rgb, np.nan)
    web, bounds = to_web_overlay(rgb, transform, crs)
    img = np.clip((np.nan_to_num(web) * 3.5) ** (1 / 1.4), 0, 1)
    alpha = np.isfinite(web).all(0)
    rgba = np.dstack([(img.transpose(1, 2, 0) * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA"), bounds


def green_overlay(prob, transform, crs, threshold):
    web, bounds = to_web_overlay(prob, transform, crs)
    pr = np.nan_to_num(web[0])
    rgba = np.zeros(pr.shape + (4,), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 22, 163, 74  # green
    rgba[..., 3] = np.where(pr >= threshold, 170, (pr * 90).astype(np.uint8))
    return Image.fromarray(rgba, "RGBA"), bounds


def main(argv=None, cfg=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = cfg or load_config(a.config)
    device = get_device()
    model, meta = load_checkpoint(p(cfg, "models") / "best.pt", device)
    tile = cfg["dataset"]["tile_size"]
    index = {"years": [], "threshold": meta["threshold"], "bounds": None}
    for y in available_years(cfg):
        s2, tr, crs, _ = read_tif(p(cfg, "raw") / f"s2_{y}.tif")
        feats, valid = compute_features(s2, meta["bands"], meta["indices"])
        prob = predict_scene(model, feats, meta, device, tile=tile)
        write_tif(p(cfg, "pred") / f"prob_{y}.tif", prob_to_uint8(prob, valid), tr, crs, nodata=NODATA)
        prob_nan = np.where(valid, prob, np.nan)
        g_img, bounds = green_overlay(prob_nan, tr, crs, meta["threshold"])
        r_img, _ = rgb_overlay(s2, meta["bands"], tr, crs)
        g_img.save(p(cfg, "overlays") / f"green_{y}.png", optimize=True)
        r_img.save(p(cfg, "overlays") / f"rgb_{y}.png", optimize=True)
        frac = float((prob[valid] >= meta["threshold"]).mean()) if valid.any() else float("nan")
        print(f"[infer {y}] green cover {frac:.1%}")
        index["years"].append(y)
        index["bounds"] = bounds
    save_json(index, p(cfg, "overlays") / "index.json")


if __name__ == "__main__":
    main()
