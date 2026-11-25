"""Manual validation of detected changes (the only check against reality, not WorldCover).

`build` draws a reproducible random sample for a pair of years (default first -> last):
  * n_loss  loss patches (only confirmed == true ones if there are enough of them)
  * n_stable random pixels with data in both years and no detected loss/gain nearby
and renders RGB chips before / after for each of them. Samples are shuffled and the
page does NOT show which group a sample came from: the annotator labels blind, so
the answer cannot be nudged by knowing what the model said.

`score` turns the annotations into:
  * change precision = real / (real + false) over loss patches (+ Wilson 95% CI)
  * miss rate        = real loss among the "no change" points (a rough recall check)
"unclear" answers are reported but excluded from both ratios.

Files (outputs/validation/):
  change_sample.geojson     points + hidden metadata (kind, area, P before/after)
  chips/{sid}_{year}.png    RGB chips, 64 px = 640 m, upscaled x4, same marker for all
  labels.json               {sid: {"label": "real"|"false"|"unclear", "ts": ...}} (written by the page)
  report.json               output of `score`
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from PIL import Image, ImageDraw
from rasterio.warp import transform as warp_transform
from scipy import ndimage

from greenwatch.config import load_config, load_json, p, save_json
from greenwatch.infer import available_years, read_prob
from greenwatch.raster import apply, read_tif

LABELS = ("real", "false", "unclear")
CHIP_PX = 64        # 640 m window
CHIP_SCALE = 4      # -> 256 px image
MARKER_R_PX = 3     # 30 m radius circle around the sampled point
STABLE_MARGIN_PX = 5  # stable points are at least 50 m from any detected change


def validation_dir(cfg):
    d = p(cfg, "change").parent / "validation"
    (d / "chips").mkdir(parents=True, exist_ok=True)
    return d


def rgb_chip(s2, bands, r, c, half=CHIP_PX // 2, scale=CHIP_SCALE):
    """RGB chip centred on pixel (r, c), same stretch as the web overlay; out-of-scene = black."""
    idx = [bands.index(b) for b in ("B04", "B03", "B02")]
    H, W = s2.shape[1:]
    chip = np.zeros((3, 2 * half, 2 * half), np.float32)
    r0, c0 = r - half, c - half
    rr0, cc0 = max(r0, 0), max(c0, 0)
    rr1, cc1 = min(r0 + 2 * half, H), min(c0 + 2 * half, W)
    if rr1 > rr0 and cc1 > cc0:
        chip[:, rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = s2[idx, rr0:rr1, cc0:cc1] / 10000.0
    img = np.clip((chip * 3.5) ** (1 / 1.4), 0, 1)
    im = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB")
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    cx = cy = half * scale + scale // 2
    rad = MARKER_R_PX * scale
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(255, 221, 0), width=2)
    return im


def _pixel_to_lnglat(tr, crs, r, c):
    x, y = apply(tr, c + 0.5, r + 0.5)
    lon, lat = warp_transform(crs, "EPSG:4326", [x], [y])
    return float(lon[0]), float(lat[0])


def pick_loss_patches(features, n, rng):
    """Random patches; restricted to confirmed == true when at least n of them exist."""
    confirmed = [f for f in features if f["properties"].get("confirmed") is True]
    pool, used_confirmed = (confirmed, True) if len(confirmed) >= n else (features, False)
    if not pool:
        return [], used_confirmed, len(confirmed)
    sel = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(sel)], used_confirmed, len(confirmed)


def stable_candidates(valid0, valid1, change_mask, margin=STABLE_MARGIN_PX):
    """Pixels with data in both years and no detected change within `margin` px."""
    near = ndimage.binary_dilation(change_mask, iterations=margin) if margin else change_mask
    return valid0 & valid1 & ~near


def build(cfg, y0=None, y1=None, n_loss=50, n_stable=50, seed=0):
    import geopandas as gpd

    years = available_years(cfg)
    y0, y1 = y0 or years[0], y1 or years[-1]
    rng = np.random.default_rng(seed)
    out = validation_dir(cfg)
    chg = p(cfg, "change")
    bands = cfg["imagery"]["bands"]

    s2_0, tr, crs, _ = read_tif(p(cfg, "raw") / f"s2_{y0}.tif")
    s2_1, _, _, _ = read_tif(p(cfg, "raw") / f"s2_{y1}.tif")
    p0, v0, _, _ = read_prob(cfg, y0)
    p1, v1, _, _ = read_prob(cfg, y1)
    loss_mask = read_tif(chg / f"loss_{y0}_{y1}.tif")[0][0].astype(bool)
    gain_mask = read_tif(chg / f"gain_{y0}_{y1}.tif")[0][0].astype(bool)

    gdf = gpd.read_file(chg / f"loss_{y0}_{y1}.geojson")
    feats = [{"geometry": g, "properties": r.drop(labels="geometry").to_dict()} for g, (_, r) in
             zip(gdf.geometry, gdf.iterrows())]
    for f in feats:  # GeoPandas reads JSON null as NaN / None
        c = f["properties"].get("confirmed")
        f["properties"]["confirmed"] = c if isinstance(c, bool) else None
    chosen, used_confirmed, n_confirmed = pick_loss_patches(feats, n_loss, rng)

    inv = ~tr
    samples = []
    for f in chosen:
        pt = f["geometry"].representative_point()  # inside the polygon, unlike the centroid
        x, y = warp_transform("EPSG:4326", crs, [pt.x], [pt.y])
        col, row = inv * (x[0], y[0])
        samples.append({"kind": "loss", "r": int(row), "c": int(col), "lng": pt.x, "lat": pt.y,
                        "area_m2": float(f["properties"].get("area_m2", math.nan)),
                        "patch_id": int(f["properties"].get("id", -1)),
                        "confirmed": f["properties"]["confirmed"]})

    cand = stable_candidates(v0, v1, loss_mask | gain_mask)
    flat = np.flatnonzero(cand)
    for i in rng.choice(flat, size=min(n_stable, len(flat)), replace=False):
        r, c = divmod(int(i), cand.shape[1])
        lng, lat = _pixel_to_lnglat(tr, crs, r, c)
        samples.append({"kind": "stable", "r": r, "c": c, "lng": lng, "lat": lat, "area_m2": None,
                        "patch_id": None, "confirmed": None})

    order = rng.permutation(len(samples))  # blind: loss and stable samples interleaved
    features = []
    for k, j in enumerate(order, start=1):
        s = samples[j]
        sid = f"s{k:03d}"
        r, c = s["r"], s["c"]
        rgb_chip(s2_0, bands, r, c).save(out / "chips" / f"{sid}_{y0}.png", optimize=True)
        rgb_chip(s2_1, bands, r, c).save(out / "chips" / f"{sid}_{y1}.png", optimize=True)
        props = {"sid": sid, "kind": s["kind"], "y0": y0, "y1": y1, "lat": round(s["lat"], 6),
                 "lng": round(s["lng"], 6), "area_m2": s["area_m2"], "patch_id": s["patch_id"],
                 "confirmed": s["confirmed"],
                 "p_before": None if np.isnan(p0[r, c]) else round(float(p0[r, c]), 3),
                 "p_after": None if np.isnan(p1[r, c]) else round(float(p1[r, c]), 3),
                 "chip_before": f"/validation/chips/{sid}_{y0}.png",
                 "chip_after": f"/validation/chips/{sid}_{y1}.png"}
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [props["lng"], props["lat"]]},
                         "properties": props})
    meta = {"pair": [y0, y1], "seed": seed, "n_loss": sum(s["kind"] == "loss" for s in samples),
            "n_stable": sum(s["kind"] == "stable" for s in samples), "loss_patches_total": len(feats),
            "loss_patches_confirmed": n_confirmed, "loss_sample_confirmed_only": used_confirmed,
            "chip_size_m": CHIP_PX * cfg["aoi"]["resolution"], "marker_radius_m": MARKER_R_PX * cfg["aoi"]["resolution"]}
    save_json({"type": "FeatureCollection", "meta": meta, "features": features}, out / "change_sample.geojson")
    print(f"[validation] {meta['n_loss']} loss patches (of {len(feats)}; confirmed: {n_confirmed}, "
          f"confirmed-only sample: {used_confirmed}) + {meta['n_stable']} stable points, {y0}->{y1} "
          f"-> {out / 'change_sample.geojson'}")
    return meta


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k / n."""
    if n == 0:
        return None
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 3), round(centre + half, 3)]


def score_labels(sample: dict, labels: dict) -> dict:
    by_kind = {}
    for f in sample["features"]:
        pr = f["properties"]
        lab = (labels.get(pr["sid"]) or {}).get("label")
        d = by_kind.setdefault(pr["kind"], {"n": 0, "labelled": 0, **{k: 0 for k in LABELS}})
        d["n"] += 1
        if lab in LABELS:
            d["labelled"] += 1
            d[lab] += 1
    rep = {"pair": sample["meta"]["pair"], "groups": by_kind}
    loss = by_kind.get("loss")
    if loss:
        n = loss["real"] + loss["false"]
        rep["change_precision"] = round(loss["real"] / n, 3) if n else None
        rep["change_precision_ci95"] = wilson(loss["real"], n)
    st = by_kind.get("stable")
    if st:
        n = st["real"] + st["false"]
        rep["miss_rate"] = round(st["real"] / n, 3) if n else None
        rep["miss_rate_ci95"] = wilson(st["real"], n)
    return rep


def score(cfg):
    out = validation_dir(cfg)
    sample = load_json(out / "change_sample.geojson")
    lp = out / "labels.json"
    labels = load_json(lp) if lp.exists() else {}
    rep = score_labels(sample, labels)
    save_json(rep, out / "report.json")
    print(f"[validation] {rep}")
    return rep


def main(argv=None, cfg=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--score", action="store_true", help="score labels.json instead of building the sample")
    ap.add_argument("--y0", type=int)
    ap.add_argument("--y1", type=int)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = cfg or load_config(a.config)
    if a.score:
        return score(cfg)
    return build(cfg, a.y0, a.y1, a.n, a.n, a.seed)


if __name__ == "__main__":
    main()
