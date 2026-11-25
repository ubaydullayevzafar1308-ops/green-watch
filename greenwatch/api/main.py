"""FastAPI service: serves model outputs and the map UI.

Run:  uvicorn greenwatch.api.main:app --reload
Env:  GW_OUTPUTS=outputs (or outputs/demo for the synthetic demo)
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
OUT = (ROOT / os.environ.get("GW_OUTPUTS", "outputs")).resolve()
WEB = ROOT / "web"

app = FastAPI(title="Tashkent Green Watch API", version="0.1.0",
              description="Green-cover segmentation & change detection for Tashkent (Sentinel-2 + U-Net)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def _json(path: Path):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found — run the pipeline first")
    with open(path) as f:
        return json.load(f)


@app.get("/api/health")
def health():
    return {"status": "ok", "outputs": str(OUT), "ready": (OUT / "stats" / "summary.json").exists()}


@app.get("/api/meta")
def meta():
    summary = _json(OUT / "stats" / "summary.json")
    index = _json(OUT / "overlays" / "index.json")
    ev = OUT / "stats" / "change_eval.json"
    return {**summary, "bounds": index["bounds"], "overlay_years": index["years"],
            "change_eval": _json(ev) if ev.exists() else None,
            "pairs": [[c["from"], c["to"]] for c in summary.get("changes", [])]}


@app.get("/api/hexagons")
def hexagons():
    return JSONResponse(_json(OUT / "stats" / "h3.geojson"))


@app.get("/api/districts")
def districts():
    return _json(OUT / "stats" / "districts.json")


@app.get("/api/changes/{kind}/{y0}/{y1}")
def changes(kind: str, y0: int, y1: int, min_area_m2: float = 0, confirmed_only: bool = False):
    if kind not in ("loss", "gain"):
        raise HTTPException(400, "kind must be loss or gain")
    gj = _json(OUT / "change" / f"{kind}_{y0}_{y1}.geojson")
    feats = [f for f in gj["features"] if f["properties"].get("area_m2", 0) >= min_area_m2
             and (not confirmed_only or f["properties"].get("confirmed") is True)]  # null = not checked
    return JSONResponse({"type": "FeatureCollection", "features": feats})


@lru_cache(maxsize=16)
def _prob_dataset(year: int):
    import rasterio

    path = OUT / "pred" / f"prob_{year}.tif"
    return rasterio.open(path) if path.exists() else None


@app.get("/api/point")
def point(lat: float = Query(..., ge=-90, le=90), lng: float = Query(..., ge=-180, le=180)):
    """P(green) time series at a location."""
    from rasterio.warp import transform

    index = _json(OUT / "overlays" / "index.json")
    series = []
    for y in index["years"]:
        ds = _prob_dataset(y)
        if ds is None:
            continue
        xs, ys = transform("EPSG:4326", ds.crs, [lng], [lat])
        r, c = ds.index(xs[0], ys[0])
        if not (0 <= r < ds.height and 0 <= c < ds.width):
            raise HTTPException(404, "point outside the area of interest")
        v = int(ds.read(1, window=((r, r + 1), (c, c + 1)))[0, 0])
        series.append({"year": y, "p_green": None if v == 255 else round(v / 250, 3)})
    return {"lat": lat, "lng": lng, "threshold": index["threshold"], "series": series}


if (OUT / "overlays").exists():
    app.mount("/overlays", StaticFiles(directory=OUT / "overlays"), name="overlays")


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


app.mount("/web", StaticFiles(directory=WEB), name="web")
