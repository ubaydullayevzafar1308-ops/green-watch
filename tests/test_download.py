"""Offline test of the Sentinel-2 compositing logic using local fake STAC items.

Checks: cloud masking via SCL, the +1000 processing-baseline offset, and the median composite.
"""
from datetime import datetime, timezone

import numpy as np
import pystac
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from greenwatch.data.download import composite_from_items
from greenwatch.raster import write_tif

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
TR = from_origin(500000, 4580000, 10, 10)
CRS = "EPSG:32642"


def make_item(tmp, idx, date, baseline, value, cloudy_rows):
    assets = {}
    dn = value + (1000 if baseline >= 4 else 0)
    for b in BANDS:
        arr = np.full((32, 32), dn, np.uint16)
        path = tmp / f"{idx}_{b}.tif"
        write_tif(path, arr, TR, CRS, nodata=0)
        assets[b] = pystac.Asset(str(path), media_type=pystac.MediaType.GEOTIFF, roles=["data"])
    scl = np.full((32, 32), 4, np.uint8)
    scl[cloudy_rows] = 9  # high-probability cloud
    write_tif(tmp / f"{idx}_SCL.tif", scl, TR, CRS)
    assets["SCL"] = pystac.Asset(str(tmp / f"{idx}_SCL.tif"), media_type=pystac.MediaType.GEOTIFF, roles=["data"])
    w, s_, e, n = transform_bounds(CRS, "EPSG:4326", 500000, 4580000 - 320, 500320, 4580000)
    ring = [[w, s_], [e, s_], [e, n], [w, n], [w, s_]]
    item = pystac.Item(f"it{idx}", geometry={"type": "Polygon", "coordinates": [ring]},
                       bbox=[w, s_, e, n], datetime=date,
                       properties={"s2:processing_baseline": f"{baseline:05.2f}", "eo:cloud_cover": 5})
    for k, a in assets.items():
        item.add_asset(k, a)
    return item


def test_composite_masks_clouds_and_harmonises(tmp_path):
    from odc.geo.geobox import GeoBox

    gb = GeoBox((32, 32), TR, CRS)
    items = [
        make_item(tmp_path, 0, datetime(2021, 7, 1, 6, tzinfo=timezone.utc), 3.0, 500, slice(0, 16)),
        make_item(tmp_path, 1, datetime(2022, 7, 6, 6, tzinfo=timezone.utc), 4.0, 500, slice(16, 32)),  # +1000 in DN
        make_item(tmp_path, 2, datetime(2022, 7, 11, 6, tzinfo=timezone.utc), 4.0, 9000, slice(0, 32)),  # fully cloudy, bright
    ]
    comp = composite_from_items(items, BANDS, gb, chunks={"x": 32, "y": 32})
    assert comp.shape == (6, 32, 32)
    # cloudy bright scene ignored, offset removed -> exactly 0.05 everywhere
    assert np.allclose(comp, 0.05, atol=1e-4), (np.nanmin(comp), np.nanmax(comp))
