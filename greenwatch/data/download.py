"""Download Sentinel-2 summer composites and ESA WorldCover labels for the AOI.

Source: Microsoft Planetary Computer STAC (free, no API key needed).
All rasters are written on ONE common grid (same CRS, transform and shape),
so years and labels line up pixel-to-pixel.

Outputs (data/raw/):
  s2_{year}.tif   uint16, reflectance * 10000, bands = config imagery.bands, nodata 0
  wc_{year}.tif   uint8, raw WorldCover class codes (10 tree, 20 shrub, 50 built-up ...), nodata 0
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
from rasterio.warp import transform_bounds

from greenwatch.config import load_config, p
from greenwatch.raster import write_tif

# Sentinel-2 Scene Classification (SCL) values kept as "clear":
# 2 dark area, 4 vegetation, 5 bare soil, 6 water, 7 unclassified
CLEAR_SCL = [2, 4, 5, 6, 7]
# Processing baseline 04.00 (from 2022-01-25) adds +1000 offset to L2A digital numbers
BASELINE_CHANGE = datetime(2022, 1, 25)


def make_geobox(cfg):
    from odc.geo.geobox import GeoBox

    aoi = cfg["aoi"]
    bounds = transform_bounds("EPSG:4326", aoi["crs"], *aoi["bbox"])
    return GeoBox.from_bbox(bounds, crs=aoi["crs"], resolution=aoi["resolution"])


def open_catalog(cfg):
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(cfg["imagery"]["stac_url"], modifier=planetary_computer.sign_inplace)


def _needs_offset(item) -> bool:
    pb = item.properties.get("s2:processing_baseline")
    if pb:
        try:
            return float(pb) >= 4.0
        except ValueError:
            pass
    return item.datetime.replace(tzinfo=None) >= BASELINE_CHANGE


def composite_from_items(items, bands, geobox, chunks=None):
    """Cloud-masked, harmonised median composite -> (bands, H, W) float32 reflectance (NaN = no clear obs)."""
    import odc.stac
    import xarray as xr

    ds = odc.stac.load(items, bands=list(bands) + ["SCL"], geobox=geobox, chunks=chunks or {"x": 1024, "y": 1024},
                       groupby="time", resampling={"SCL": "nearest", "*": "bilinear"})

    # per-time-slice harmonisation offset (match slices back to items by timestamp)
    item_times = [(np.datetime64(it.datetime.replace(tzinfo=None), "ns"), _needs_offset(it)) for it in items]
    offsets = []
    for t in ds.time.values:
        near = [off for it_t, off in item_times if abs(it_t - t) <= np.timedelta64(timedelta(minutes=5))]
        offsets.append(1000.0 if (near and any(near)) else 0.0)
    offset = xr.DataArray(np.array(offsets, dtype=np.float32), dims=["time"], coords={"time": ds.time})

    clear = ds["SCL"].isin(CLEAR_SCL)
    stack = []
    for b in bands:
        refl = (ds[b].astype("float32") - offset) / 10000.0
        refl = refl.where(clear & (ds[b] > 0)).clip(0, 1)
        stack.append(refl.chunk({"time": -1}).median("time", skipna=True))
    return xr.concat(stack, dim="band").compute().values.astype(np.float32)


def download_s2_year(cfg, year: int, catalog=None, geobox=None) -> str:
    im = cfg["imagery"]
    catalog = catalog or open_catalog(cfg)
    geobox = geobox or make_geobox(cfg)
    start, end = (f"{year}-{d}" for d in im["season"])
    items = catalog.search(
        collections=[im["collection"]], bbox=cfg["aoi"]["bbox"], datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": im["max_cloud"]}},
    ).item_collection()
    items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100))[: im["max_scenes"]]
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes for {year}; relax max_cloud or season")
    print(f"[s2 {year}] {len(items)} scenes")

    comp = composite_from_items(items, im["bands"], geobox)

    valid_frac = float(np.isfinite(comp).all(0).mean())
    print(f"[s2 {year}] clear-pixel coverage: {valid_frac:.1%}")
    out = np.where(np.isfinite(comp), np.clip(comp * 10000, 1, 65535), 0).astype(np.uint16)
    path = p(cfg, "raw") / f"s2_{year}.tif"
    write_tif(path, out, geobox.affine, str(geobox.crs), nodata=0, band_names=im["bands"])
    return str(path)


def download_worldcover(cfg, year: int, catalog=None, geobox=None) -> str:
    import odc.stac

    lab = cfg["labels"]
    catalog = catalog or open_catalog(cfg)
    geobox = geobox or make_geobox(cfg)
    version = lab["versions"][year]
    items = [
        it for it in catalog.search(collections=[lab["collection"]], bbox=cfg["aoi"]["bbox"]).item_collection()
        if it.properties.get("esa_worldcover:product_version") == version
    ]
    if not items:  # fallback: match by item year
        items = [it for it in catalog.search(collections=[lab["collection"]], bbox=cfg["aoi"]["bbox"]).item_collection()
                 if it.datetime and it.datetime.year == year]
    if not items:
        raise RuntimeError(f"No WorldCover {year} (v{version}) items found")
    ds = odc.stac.load(items, bands=["map"], geobox=geobox, resampling="nearest", groupby="solar_day")
    wc = ds["map"].max("time").values.astype(np.uint8)
    path = p(cfg, "raw") / f"wc_{year}.tif"
    write_tif(path, wc, geobox.affine, str(geobox.crs), nodata=0, band_names=["worldcover"])
    print(f"[worldcover {year}] saved {path}")
    return str(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--years", type=int, nargs="*", help="override imagery years")
    ap.add_argument("--skip-labels", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    catalog, geobox = open_catalog(cfg), make_geobox(cfg)
    print(f"grid: {geobox.shape} px @ {cfg['aoi']['resolution']} m, {geobox.crs}")
    for y in args.years or cfg["imagery"]["years"]:
        download_s2_year(cfg, y, catalog, geobox)
    if not args.skip_labels:
        for y in cfg["labels"]["versions"]:
            download_worldcover(cfg, int(y), catalog, geobox)


if __name__ == "__main__":
    main()
