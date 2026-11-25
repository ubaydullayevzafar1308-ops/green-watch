"""Aggregate green cover and change to H3 hexagons, districts and a city summary.

Outputs (outputs/stats/):
  h3.geojson        per hexagon: green share per year, change in p.p., loss/gain area
  districts.json    same per district (only if a districts GeoJSON is provided)
  summary.json      city-level time series, totals, top hotspots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h3
import numpy as np
from rasterio.warp import transform as warp_transform

from greenwatch.change import year_pairs
from greenwatch.config import load_config, load_json, p, save_json
from greenwatch.infer import available_years, read_prob
from greenwatch.raster import apply, read_tif


def pixel_cells(shape, transform, crs, res, stride=5):
    """H3 cell index for every pixel (computed on a coarse stride, then upsampled)."""
    H, W = shape
    rr = np.minimum(np.arange(0, H, stride) + stride // 2, H - 1)
    cc = np.minimum(np.arange(0, W, stride) + stride // 2, W - 1)
    cgrid, rgrid = np.meshgrid(cc + 0.5, rr + 0.5)
    xs, ys = apply(transform, cgrid.ravel(), rgrid.ravel())
    lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
    cells = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(lats, lons)]
    uniq, inv = np.unique(np.array(cells), return_inverse=True)
    coarse = inv.reshape(len(rr), len(cc))
    full = np.repeat(np.repeat(coarse, stride, 0), stride, 1)[:H, :W]
    return uniq, full


def zonal(zone_idx, n_zones, values, weights=None):
    w = np.ones_like(values, dtype=np.float64) if weights is None else weights
    s = np.bincount(zone_idx.ravel(), weights=(values * w).ravel(), minlength=n_zones)
    c = np.bincount(zone_idx.ravel(), weights=w.ravel(), minlength=n_zones)
    return s, c


def compute_zone_stats(zone_idx, n_zones, years, probs, thr, changes, px_area):
    stats = {}
    for y in years:
        pr, valid, _, _ = probs[y]
        green = (np.nan_to_num(pr) >= thr) & valid
        s, c = zonal(zone_idx, n_zones, green.astype(np.float64), valid.astype(np.float64))
        stats[f"g_{y}"] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        stats.setdefault("valid_px", c)
    for (y0, y1), (loss, gain) in changes.items():
        stats[f"loss_m2_{y0}_{y1}"] = np.bincount(zone_idx.ravel(), weights=loss.ravel(), minlength=n_zones) * px_area
        stats[f"gain_m2_{y0}_{y1}"] = np.bincount(zone_idx.ravel(), weights=gain.ravel(), minlength=n_zones) * px_area
    return stats


def main(argv=None, cfg=None):
    import geopandas as gpd
    from shapely.geometry import Polygon

    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = cfg or load_config(a.config)
    ag, res = cfg["aggregate"], cfg["aoi"]["resolution"]
    years = available_years(cfg)
    thr = load_json(p(cfg, "overlays") / "index.json")["threshold"]
    probs = {y: read_prob(cfg, y) for y in years}
    _, _, tr, crs = probs[years[0]]
    shape = probs[years[0]][0].shape
    px_area = res ** 2
    changes = {}
    for y0, y1 in year_pairs(years):
        loss = read_tif(p(cfg, "change") / f"loss_{y0}_{y1}.tif")[0][0].astype(np.float64)
        gain = read_tif(p(cfg, "change") / f"gain_{y0}_{y1}.tif")[0][0].astype(np.float64)
        changes[(y0, y1)] = (loss, gain)
    first, last = years[0], years[-1]
    out = p(cfg, "stats")

    # --- H3 hexagons
    cells, zone = pixel_cells(shape, tr, crs, ag["h3_resolution"])
    st = compute_zone_stats(zone, len(cells), years, probs, thr, changes, px_area)
    feats = []
    for i, cell in enumerate(cells):
        if st["valid_px"][i] < 50:
            continue
        props = {"cell": cell}
        for k, v in st.items():
            if k == "valid_px":
                continue
            props[k] = None if np.isnan(v[i]) else round(float(v[i]), 4 if k.startswith("g_") else 1)
        if props.get(f"g_{first}") is not None and props.get(f"g_{last}") is not None:
            props["delta_pp"] = round((props[f"g_{last}"] - props[f"g_{first}"]) * 100, 2)
        boundary = [(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]
        feats.append({**props, "geometry": Polygon(boundary)})
    hex_gdf = gpd.GeoDataFrame(feats, geometry="geometry", crs=4326)
    hex_gdf.to_file(out / "h3.geojson", driver="GeoJSON")

    # --- districts (optional)
    districts = None
    dpath = Path(cfg["_root"]) / ag["districts_file"]
    if dpath.exists():
        from rasterio import features

        dg = gpd.read_file(dpath).to_crs(crs)
        dzone = features.rasterize(((g, i + 1) for i, g in enumerate(dg.geometry)), out_shape=shape,
                                   transform=tr, fill=0, dtype="int32")
        dst = compute_zone_stats(dzone, len(dg) + 1, years, probs, thr, changes, px_area)
        districts = []
        for i, name in enumerate(dg[ag["district_name_field"]], start=1):
            if dst["valid_px"][i] == 0:
                continue
            row = {"name": name, "area_km2": round(dst["valid_px"][i] * px_area / 1e6, 2)}
            for k, v in dst.items():
                if k != "valid_px":
                    row[k] = round(float(v[i]), 4)
            row["delta_pp"] = round((row[f"g_{last}"] - row[f"g_{first}"]) * 100, 2)
            districts.append(row)
        districts.sort(key=lambda r: r["delta_pp"])
        save_json(districts, out / "districts.json")

    # --- city summary
    series = []
    for y in years:
        pr, valid, _, _ = probs[y]
        green = (np.nan_to_num(pr) >= thr) & valid
        series.append({"year": y, "green_share": round(float(green.sum() / max(valid.sum(), 1)), 4),
                       "green_km2": round(float(green.sum()) * px_area / 1e6, 3)})
    key = f"loss_m2_{first}_{last}"
    top = hex_gdf.sort_values(key, ascending=False).head(10) if key in hex_gdf else hex_gdf.head(0)
    summary = {
        "aoi": cfg["aoi"]["name"], "years": years, "threshold": thr, "series": series,
        "changes": load_json(p(cfg, "change") / "summary.json"),
        "hotspots": [{"cell": r["cell"], "loss_m2": r[key], "delta_pp": r.get("delta_pp"),
                      "lat": h3.cell_to_latlng(r["cell"])[0], "lng": h3.cell_to_latlng(r["cell"])[1]}
                     for _, r in top.iterrows() if r[key] > 0],
        "has_districts": districts is not None,
    }
    mpath = p(cfg, "models") / "metrics.json"
    if mpath.exists():
        m = load_json(mpath)
        summary["model_metrics"] = m["results"]
    save_json(summary, out / "summary.json")
    print(f"[aggregate] {len(hex_gdf)} hexagons; green share {series[0]['green_share']:.1%} ({first}) -> "
          f"{series[-1]['green_share']:.1%} ({last})")
    return summary


if __name__ == "__main__":
    main()
