"""Green-cover change detection between years.

Loss  = P(green) >= high in year A  AND  P(green) <= low in year B
Gain  = the reverse.
Hysteresis (two thresholds) suppresses flicker of pixels near the decision
boundary; morphological opening + minimum area drop speckle noise. A loss is
marked `confirmed` when the following year (if any) is still not green —
single-year artefacts (a cloudy composite, a dry summer) rarely persist.

Outputs: outputs/change/{loss,gain}_{A}_{B}.geojson (+ .tif masks)
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage

from greenwatch.config import load_config, p, save_json
from greenwatch.infer import available_years, read_prob
from greenwatch.raster import write_tif


def clean_mask(mask, min_px, opening_iterations=1):
    if opening_iterations:
        mask = ndimage.binary_opening(mask, iterations=opening_iterations)
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask, lab, 0
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    mask = keep[lab]
    lab, n = ndimage.label(mask)
    return mask, lab, n


def detect(p0, p1, valid, cfg_change, res_m, kind="loss"):
    hi, lo = cfg_change["high"], cfg_change["low"]
    if kind == "loss":
        raw = (p0 >= hi) & (p1 <= lo)
    else:
        raw = (p0 <= lo) & (p1 >= hi)
    raw &= valid
    min_px = max(int(np.ceil(cfg_change["min_area_m2"] / res_m ** 2)), 1)
    return clean_mask(raw, min_px, cfg_change.get("opening_iterations", 1))


def vectorize(lab, n, transform, crs, p0, p1, p_next=None, kind="loss", thr=0.5):
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape

    if n == 0:
        return gpd.GeoDataFrame({"area_m2": []}, geometry=[], crs=crs).to_crs(4326)
    idx = np.arange(1, n + 1)
    m0 = ndimage.mean(np.nan_to_num(p0), lab, idx)
    m1 = ndimage.mean(np.nan_to_num(p1), lab, idx)
    mn = ndimage.mean(np.nan_to_num(p_next), lab, idx) if p_next is not None else None
    rows = []
    for geom, val in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=transform):
        i = int(val) - 1
        g = shape(geom)
        rec = {"id": int(val), "area_m2": round(g.area, 1), "p_before": round(float(m0[i]), 3),
               "p_after": round(float(m1[i]), 3), "geometry": g}
        if mn is not None:
            still = mn[i] < thr if kind == "loss" else mn[i] >= thr
            rec["confirmed"] = bool(still)
        rows.append(rec)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    gdf = gdf.dissolve(by="id", aggfunc="first").reset_index()  # one feature per blob
    gdf["area_m2"] = gdf.geometry.area.round(1)
    return gdf.sort_values("area_m2", ascending=False).to_crs(4326)


def year_pairs(years):
    pairs = [(a, b) for a, b in zip(years, years[1:])]
    if len(years) > 2:
        pairs.append((years[0], years[-1]))
    return pairs


def main(argv=None, cfg=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = cfg or load_config(a.config)
    years = available_years(cfg)
    res = cfg["aoi"]["resolution"]
    from greenwatch.config import load_json

    thr = load_json(p(cfg, "overlays") / "index.json")["threshold"]
    probs = {y: read_prob(cfg, y) for y in years}
    out_dir = p(cfg, "change")
    summary = []
    for y0, y1 in year_pairs(years):
        p0, v0, tr, crs = probs[y0]
        p1, v1, _, _ = probs[y1]
        nxt = years[years.index(y1) + 1] if years.index(y1) + 1 < len(years) and y1 - y0 == 1 else None
        p_next = probs[nxt][0] if nxt else None
        row = {"from": y0, "to": y1}
        for kind in ("loss", "gain"):
            mask, lab, n = detect(p0, p1, v0 & v1, cfg["change"], res, kind)
            write_tif(out_dir / f"{kind}_{y0}_{y1}.tif", mask.astype(np.uint8), tr, crs)
            gdf = vectorize(lab, n, tr, crs, p0, p1, p_next, kind, thr)
            gdf.to_file(out_dir / f"{kind}_{y0}_{y1}.geojson", driver="GeoJSON")
            row[f"{kind}_ha"] = round(float(mask.sum()) * res ** 2 / 1e4, 2)
            row[f"{kind}_patches"] = int(len(gdf))
        summary.append(row)
        print(f"[change {y0}->{y1}] loss {row['loss_ha']} ha ({row['loss_patches']} patches), "
              f"gain {row['gain_ha']} ha ({row['gain_patches']} patches)")
    save_json(summary, out_dir / "summary.json")
    return summary


if __name__ == "__main__":
    main()
