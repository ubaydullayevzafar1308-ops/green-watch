"""Synthetic "Tashkent-like" scenes for offline demo and tests.

Generates multi-year Sentinel-2-like composites and WorldCover-like labels on the
real Tashkent UTM grid, with known ground truth: green patches are removed
("construction") in later years and a few new parks appear. This lets the whole
pipeline (train -> infer -> change -> aggregate -> API) run with no internet and
lets tests check that known losses are actually detected.
"""
from __future__ import annotations

import argparse

import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import transform

from greenwatch.config import load_config, p, save_json
from greenwatch.raster import write_tif

# mean reflectance per class for bands B02 B03 B04 B08 B11 B12
SPECTRA = {
    "tree": [0.030, 0.060, 0.040, 0.360, 0.180, 0.090],
    "grass": [0.045, 0.085, 0.070, 0.280, 0.240, 0.140],
    "built": [0.100, 0.115, 0.135, 0.210, 0.270, 0.240],
    "bare": [0.120, 0.150, 0.190, 0.250, 0.320, 0.280],
    "water": [0.050, 0.060, 0.040, 0.020, 0.010, 0.008],
}
WC_CODE = {"tree": 10, "grass": 30, "built": 50, "bare": 60, "water": 80}


def _smooth_noise(rng, shape, scale):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(rng.standard_normal(shape), scale)


def make_base(rng, size):
    """Return class map (H,W) of str keys encoded as ints and helper masks."""
    from scipy.ndimage import binary_dilation

    h = w = size
    cls = np.full((h, w), 2, dtype=np.uint8)  # 0 tree, 1 grass, 2 built, 3 bare, 4 water
    n = _smooth_noise(rng, (h, w), size / 40)
    parks = n > np.quantile(n, 0.82)  # ~18% of the city in parks / tree massifs
    cls[parks] = 0
    # grass patches (lawns, fields) — green but NOT trees: the model must separate them
    g = _smooth_noise(rng, (h, w), size / 60)
    grass = (g > np.quantile(g, 0.85)) & ~parks
    cls[grass & (rng.random((h, w)) > 0.3)] = 1
    # street tree lines along a road grid
    roads = np.zeros((h, w), bool)
    for c in range(30, w, 70):
        roads[:, c:c + 2] = True
    for r in range(30, h, 70):
        roads[r:r + 2, :] = True
    trees_line = binary_dilation(roads, iterations=2) & ~roads
    cls[trees_line & (rng.random((h, w)) > 0.35)] = 0
    cls[roads] = 2
    # bare patches at the periphery
    b = _smooth_noise(rng, (h, w), size / 30)
    cls[(b < np.quantile(b, 0.08)) & (cls == 2)] = 3
    # canal
    yy = (np.arange(w) * 0.35 + h * 0.3).astype(int)
    for x, y in enumerate(yy):
        if 0 <= y < h - 4:
            cls[y:y + 4, x] = 4
    return cls


def evolve(rng, base, years):
    """Yearly class maps with progressive construction on green and some new parks."""
    maps, events = {}, []
    cur = base.copy()
    h, w = base.shape
    for i, y in enumerate(years):
        if i > 0:
            for _ in range(rng.integers(3, 7)):  # construction sites on green
                green = np.argwhere(cur == 0)
                if len(green) == 0:
                    break
                r, c = green[rng.integers(len(green))]
                hh, ww = rng.integers(12, 40, size=2)
                r0, c0, r1, c1 = max(r - hh // 2, 0), max(c - ww // 2, 0), min(r + hh // 2, h), min(c + ww // 2, w)
                lost = (cur[r0:r1, c0:c1] == 0).sum()
                cur[r0:r1, c0:c1] = np.where(cur[r0:r1, c0:c1] == 4, 4, 2)
                events.append({"year": int(y), "type": "loss", "rows": [int(r0), int(r1)], "cols": [int(c0), int(c1)],
                               "green_px": int(lost)})
            if y >= 2023 and rng.random() < 0.7:  # a new park
                r, c = rng.integers(40, h - 40), rng.integers(40, w - 40)
                cur[r - 12:r + 12, c - 12:c + 12] = np.where(cur[r - 12:r + 12, c - 12:c + 12] == 4, 4, 0)
                events.append({"year": int(y), "type": "gain", "rows": [int(r - 12), int(r + 12)],
                               "cols": [int(c - 12), int(c + 12)]})
        maps[y] = cur.copy()
    return maps, events


def render(rng, cls_map):
    from scipy.ndimage import gaussian_filter

    names = ["tree", "grass", "built", "bare", "water"]
    spec = np.array([SPECTRA[n] for n in names], dtype=np.float32)  # (5, 6)
    img = spec[cls_map].transpose(2, 0, 1)  # (6,H,W)
    img = np.stack([gaussian_filter(b, 0.7) for b in img])  # mixed pixels at edges (10 m blur)
    gain = 1 + rng.normal(0, 0.04)  # year-to-year illumination / phenology jitter
    img = img * gain * (1 + rng.normal(0, 0.06, img.shape)) + rng.normal(0, 0.006, img.shape)
    return np.clip(img, 0.001, 1).astype(np.float32)


def worldcover_like(rng, cls_map):
    """WorldCover-style codes with realistic label noise (missing small trees, edge errors)."""
    from scipy.ndimage import binary_erosion

    codes = np.array([WC_CODE[n] for n in ["tree", "grass", "built", "bare", "water"]], dtype=np.uint8)
    wc = codes[cls_map]
    trees = cls_map == 0
    edge = trees & ~binary_erosion(trees)
    flip = edge & (rng.random(cls_map.shape) < 0.35)
    wc[flip] = 50
    wc[(cls_map == 2) & (rng.random(cls_map.shape) < 0.01)] = 10
    return wc


def generate(cfg, size=768, seed=0):
    rng = np.random.default_rng(seed)
    years = cfg["imagery"]["years"]
    aoi = cfg["aoi"]
    lon_c = (aoi["bbox"][0] + aoi["bbox"][2]) / 2
    lat_c = (aoi["bbox"][1] + aoi["bbox"][3]) / 2
    xs, ys = transform("EPSG:4326", aoi["crs"], [lon_c], [lat_c])
    res = aoi["resolution"]
    tr = from_origin(xs[0] - size * res / 2, ys[0] + size * res / 2, res, res)

    base = make_base(rng, size)
    maps, events = evolve(rng, base, years)
    raw = p(cfg, "raw")
    for y in years:
        img = render(rng, maps[y])
        write_tif(raw / f"s2_{y}.tif", (img * 10000).astype(np.uint16), tr, aoi["crs"], nodata=0,
                  band_names=cfg["imagery"]["bands"])
        write_tif(raw / f"truth_{y}.tif", (maps[y] == 0).astype(np.uint8), tr, aoi["crs"])
    for y in cfg["labels"]["versions"]:
        write_tif(raw / f"wc_{y}.tif", worldcover_like(rng, maps[int(y)]), tr, aoi["crs"], nodata=0)
    save_json({"size": size, "seed": seed, "events": events}, raw / "synthetic_truth.json")
    print(f"[synthetic] {len(years)} years, {size}x{size} px, {len(events)} change events -> {raw}")


def score_change(cfg, min_green_px=40):
    """Compare detected yearly losses against the known synthetic construction events."""
    import rasterio

    from greenwatch.config import load_json

    truth = load_json(p(cfg, "raw") / "synthetic_truth.json")["events"]
    years = cfg["imagery"]["years"]
    hit = total = fp = det = 0
    for y0, y1 in zip(years, years[1:]):
        with rasterio.open(p(cfg, "change") / f"loss_{y0}_{y1}.tif") as src:
            m = src.read(1).astype(bool)
        box = np.zeros_like(m)
        for e in truth:
            if e["year"] != y1 or e["type"] != "loss":
                continue
            (r0, r1), (c0, c1) = e["rows"], e["cols"]
            box[r0:r1, c0:c1] = True
            if e["green_px"] >= min_green_px:  # tiny events are below the 10 m detection limit
                total += 1
                hit += bool(m[r0:r1, c0:c1].any())
        fp += int((m & ~box).sum())
        det += int(m.sum())
    res = {"event_recall": round(hit / max(total, 1), 3), "events_detected": hit, "events_total": total,
           "pixel_precision": round(1 - fp / max(det, 1), 4)}
    save_json(res, p(cfg, "stats") / "change_eval.json")
    print(f"[synthetic eval] loss events found {hit}/{total}, pixel precision {res['pixel_precision']:.2%}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    generate(load_config(a.config), a.size, a.seed)


if __name__ == "__main__":
    main()
