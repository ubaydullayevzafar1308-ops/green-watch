import importlib
import json
import os

import numpy as np
import pytest

from greenwatch.change import clean_mask, detect
from greenwatch.config import load_config
from greenwatch.features import compute_features
from greenwatch.metrics import best_threshold, evaluate


def test_metrics_ignore_label():
    pred = np.array([[1, 1], [0, 0]])
    lab = np.array([[1, 255], [0, 1]], dtype=np.uint8)
    m = evaluate(pred, lab)
    assert m["precision"] == 1.0 and m["recall"] == 0.5 and m["iou"] == 0.5


def test_best_threshold_finds_separation():
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    lab = np.array([0, 0, 1, 1], dtype=np.uint8)
    thr, f1 = best_threshold(prob, lab)
    assert 0.2 < thr <= 0.8 and f1 == 1.0


def test_features_ndvi_and_nodata():
    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
    s2 = np.full((6, 4, 4), 1000, np.uint16)
    s2[3] = 4000  # strong NIR -> high NDVI
    s2[:, 0, 0] = 0  # nodata pixel
    x, valid = compute_features(s2, bands, ["NDVI", "NDBI"])
    assert x.shape == (8, 4, 4)
    assert not valid[0, 0] and valid[1, 1]
    assert np.isclose(x[6, 1, 1], (0.4 - 0.1) / 0.5, atol=1e-4)


def test_clean_mask_removes_small_blobs():
    m = np.zeros((50, 50), bool)
    m[5:20, 5:20] = True   # 225 px, kept
    m[40, 40] = True       # 1 px, removed
    out, _, n = clean_mask(m, min_px=10, opening_iterations=0)
    assert n == 1 and out[10, 10] and not out[40, 40]


def test_hysteresis_detects_loss_only():
    p0 = np.full((30, 30), 0.9, np.float32)
    p1 = p0.copy()
    p1[5:15, 5:15] = 0.1   # real loss
    p1[20:28, 20:28] = 0.5  # ambiguous -> ignored
    cfg = {"high_offset": 0.15, "high_min": 0.5, "low_offset": 0.1, "low_max": 0.2,
           "min_area_m2": 500, "opening_iterations": 0}
    # model threshold 0.45 -> high max(0.60, 0.5) = 0.60 / low min(0.35, 0.2) = 0.20
    mask, _, n = detect(p0, p1, np.ones_like(p0, bool), cfg, res_m=10, kind="loss", thr=0.45)
    assert n == 1 and mask[10, 10] and not mask[24, 24]


@pytest.mark.slow
def test_end_to_end_synthetic(tmp_path):
    from greenwatch.__main__ import DEMO_OVERRIDES, run_pipeline
    from greenwatch.config import deep_merge
    from greenwatch.data import synthetic

    # small scene -> smaller tiles/blocks so that every split exists and train crops fit inside the core
    over = deep_merge(DEMO_OVERRIDES, {"train": {"epochs": 2, "crops_per_epoch": 32},
                                       "dataset": {"tile_size": 64, "block_size": 128, "buffer_px": 16},
                                       "imagery": {"years": [2020, 2021, 2022]}})
    over["paths"] = {k: str(tmp_path / k) for k in DEMO_OVERRIDES["paths"]}
    cfg = load_config(overrides=over)
    synthetic.generate(cfg, size=384)
    run_pipeline(cfg)
    synthetic.score_change(cfg)
    from greenwatch import validation

    vmeta = validation.build(cfg, n_loss=5, n_stable=5)
    assert vmeta["pair"] == [2020, 2022] and vmeta["n_stable"] == 5
    vsample = json.loads((tmp_path / "validation" / "change_sample.geojson").read_text())
    assert all((tmp_path / f["properties"]["chip_before"].lstrip("/")).exists() for f in vsample["features"])
    for f in ["models/best.pt", "models/metrics.json", "stats/h3.geojson", "stats/summary.json",
              "change/loss_2020_2022.geojson", "overlays/index.json", "pred/prob_2022.tif"]:
        sub, name = f.split("/")
        assert (tmp_path / {"models": "models", "stats": "stats", "change": "change", "overlays": "overlays",
                            "pred": "pred"}[sub] / name).exists(), f

    # API on top of these outputs
    os.makedirs(tmp_path / "out", exist_ok=True)
    for k in ["stats", "overlays", "change", "pred", "models", "validation"]:
        os.symlink(tmp_path / k, tmp_path / "out" / k)
    os.environ["GW_OUTPUTS"] = str(tmp_path / "out")
    from fastapi.testclient import TestClient

    import greenwatch.api.main as api

    importlib.reload(api)
    c = TestClient(api.app)
    assert c.get("/api/health").json()["ready"]
    meta = c.get("/api/meta").json()
    assert meta["years"] == [2020, 2021, 2022] and len(meta["bounds"]) == 2
    assert c.get("/api/hexagons").status_code == 200
    assert c.get("/api/changes/loss/2020/2021").status_code == 200
    lat = (meta["bounds"][0][0] + meta["bounds"][1][0]) / 2
    lng = (meta["bounds"][0][1] + meta["bounds"][1][1]) / 2
    pt = c.get(f"/api/point?lat={lat}&lng={lng}").json()
    assert len(pt["series"]) == 3
    assert len(c.get("/api/validation/sample").json()["features"]) == vmeta["n_loss"] + vmeta["n_stable"]
