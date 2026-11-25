"""Raster IO helpers (GeoTIFF read/write, reprojection to web overlays)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds


def write_tif(path, array: np.ndarray, transform: Affine, crs, nodata=None, band_names=None):
    """array: (bands, H, W) or (H, W)."""
    arr = array[None] if array.ndim == 2 else array
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
        dtype=arr.dtype, crs=crs, transform=transform, compress="deflate", tiled=True,
        blockxsize=256, blockysize=256,
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        if band_names:
            for i, name in enumerate(band_names, 1):
                dst.set_band_description(i, name)


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(), src.transform, src.crs, list(src.descriptions)


def apply(tr: Affine, col, row):
    """Pixel (col,row) -> map (x,y); works for scalars and arrays."""
    return tr.c + tr.a * col + tr.b * row, tr.f + tr.d * col + tr.e * row


def to_web_overlay(array: np.ndarray, transform: Affine, crs, resampling=Resampling.bilinear):
    """Reproject (bands,H,W) to EPSG:3857 for a Leaflet imageOverlay.

    Returns (reprojected array, [[south, west], [north, east]] in lat/lon).
    """
    arr = array[None] if array.ndim == 2 else array
    h, w = arr.shape[1:]
    left, top = apply(transform, 0, 0)
    right, bottom = apply(transform, w, h)
    dst_crs = "EPSG:3857"
    dst_t, dw, dh = calculate_default_transform(crs, dst_crs, w, h, left, bottom, right, top)
    out = np.zeros((arr.shape[0], dh, dw), dtype=np.float32)
    for b in range(arr.shape[0]):
        reproject(
            arr[b].astype(np.float32), out[b], src_transform=transform, src_crs=crs,
            dst_transform=dst_t, dst_crs=dst_crs, resampling=resampling,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
    l3, t3 = apply(dst_t, 0, 0)
    r3, b3 = apply(dst_t, dw, dh)
    west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", l3, b3, r3, t3)
    return out, [[south, west], [north, east]]
