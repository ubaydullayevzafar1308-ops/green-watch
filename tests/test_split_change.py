"""Second-iteration behaviour: buffered block split, random train crops, fold summary,
hysteresis tied to the model threshold, honest `confirmed`, per-year valid masks, API filter."""
import importlib
import json

import numpy as np
import pytest
from scipy import ndimage

from greenwatch.aggregate import compute_zone_stats
from greenwatch.change import confirm_patches, hysteresis_thresholds
from greenwatch.data.tiles import crop_origins, split_masks
from greenwatch.train import RandomCropDataset, summarize_folds


# ---------- split ----------
def test_split_masks_buffer_separates_splits():
    blocks = np.array([["train", "test"], ["val", "train"]], dtype=object)
    masks = split_masks((16, 16), blocks, block=8, buffer_px=4)
    tr, te, va = masks["train"], masks["test"], masks["val"]
    assert not (tr & te).any() and not (tr & va).any() and not (te & va).any()
    # every train pixel is >= 4 px (Chebyshev) away from every test / val pixel
    far = ndimage.maximum_filter((te | va).astype(np.uint8), size=2 * 4 - 1) == 0  # within 3 px of other
    assert (tr <= far).all()
    # buffer pixels exist and belong to no split
    assert (~(tr | te | va)).sum() > 0
    assert tr[0, 0] and not tr[0, 7] and te[0, 15] and not te[0, 8]


def test_split_masks_no_buffer_covers_everything():
    blocks = np.array([["train", "test"]], dtype=object)
    masks = split_masks((8, 16), blocks, block=8, buffer_px=0)
    assert (masks["train"] | masks["test"]).all()


def test_crop_origins_lie_fully_inside_core():
    rng = np.random.default_rng(0)
    core = np.zeros((64, 64), bool)
    core[4:40, 10:60] = True
    core[20, 30] = False  # a hole: windows covering it are not allowed
    lab = rng.integers(0, 2, core.shape).astype(np.uint8)
    lab[~core] = 255
    org = crop_origins(core, lab, t=16, stride=2)
    assert len(org) > 0
    for r, c in org:
        assert core[r:r + 16, c:c + 16].all()


def test_random_crop_dataset_draws_n_crops_from_core():
    X = np.random.default_rng(1).normal(size=(3, 40, 40)).astype(np.float16)
    y = np.full((40, 40), 255, np.uint8)
    y[0:20, 0:20] = 1
    org = crop_origins(y != 255, y, t=8, stride=1)
    ds = RandomCropDataset(X, y, org, [0, 0, 0], [1, 1, 1], n=37, t=8, augment=True, seed=0)
    assert len(ds) == 37
    for i in range(10):
        xb, yb = ds[i]
        assert xb.shape == (3, 8, 8) and yb.shape == (8, 8)
        assert (yb == 1).all()  # only train-core labels


def test_summarize_folds_mean_std():
    m = lambda v: {"iou": v, "f1": v, "precision": v, "recall": v, "accuracy": v}  # noqa: E731
    folds = [{"results": {"test/unet": m(0.2)}}, {"results": {"test/unet": m(0.4)}}]
    mean, std = summarize_folds(folds)
    assert mean["test/unet"]["iou"] == 0.3 and std["test/unet"]["iou"] == 0.1


# ---------- change ----------
def test_hysteresis_thresholds_follow_model_threshold_and_clip():
    cfg = {"high_offset": 0.15, "low_offset": 0.15}
    assert hysteresis_thresholds(0.65, cfg) == (0.8, 0.5)
    assert hysteresis_thresholds(0.9, cfg) == (0.95, 0.75)
    assert hysteresis_thresholds(0.1, cfg) == (0.25, 0.05)


def test_confirmed_ignores_nan_and_needs_half_the_patch():
    lab = np.zeros((10, 10), np.int32)
    lab[0:5, 0:5] = 1   # patch 1
    lab[5:10, 5:10] = 2  # patch 2
    p_next = np.full((10, 10), np.nan, np.float32)
    # patch 1: 60% no data -> unknown
    p_next[0:2, 0:5] = 0.1
    # patch 2: 40% no data, the rest is green (0.7) -> the loss is NOT confirmed.
    # (treating NaN as 0 would give mean 0.42 < 0.5 and wrongly confirm it)
    p_next[5:8, 5:10] = 0.7
    assert confirm_patches(lab, 2, p_next, "loss", thr=0.5) == [None, False]
    p_next[5:8, 5:10] = 0.1
    assert confirm_patches(lab, 2, p_next, "loss", thr=0.5) == [None, True]
    assert confirm_patches(lab, 2, None, "loss", thr=0.5) == [None, None]


# ---------- aggregate ----------
def test_zone_stats_use_each_years_own_valid_mask():
    zone = np.zeros((10, 10), np.int64)
    zone[:, 5:] = 1
    pa = np.full((10, 10), 0.9, np.float32)
    pb = pa.copy()
    pb[:, :3] = np.nan            # zone 0: 2020 has only 20 valid px...
    pb[:, 3:5] = 0.9              # ...all green
    pb[:, 5:] = np.nan            # zone 1: no data at all in 2021
    probs = {2020: (pa, np.isfinite(pa), None, None), 2021: (pb, np.isfinite(pb), None, None)}
    st = compute_zone_stats(zone, 2, [2020, 2021], probs, 0.5, {}, 100.0, min_valid_px=10)
    assert st["g_2020"][0] == 1.0 and st["g_2021"][0] == 1.0  # not 0.4: no-data pixels are not "not green"
    assert np.isnan(st["g_2021"][1])                          # no data -> no share, not 0 %
    assert st["valid_px_2021"][0] == 20 and st["valid_px_2020"][0] == 50


# ---------- API ----------
def test_api_confirmed_only_keeps_only_true(tmp_path, monkeypatch):
    (tmp_path / "change").mkdir()
    feats = [{"type": "Feature", "geometry": None, "properties": {"area_m2": 1000, "confirmed": c}}
             for c in (True, False, None)]
    (tmp_path / "change" / "loss_2020_2021.geojson").write_text(json.dumps({"type": "FeatureCollection",
                                                                          "features": feats}))
    monkeypatch.setenv("GW_OUTPUTS", str(tmp_path))
    from fastapi.testclient import TestClient

    import greenwatch.api.main as api

    importlib.reload(api)
    c = TestClient(api.app)
    assert len(c.get("/api/changes/loss/2020/2021").json()["features"]) == 3
    got = c.get("/api/changes/loss/2020/2021?confirmed_only=true").json()["features"]
    assert [f["properties"]["confirmed"] for f in got] == [True]
    assert c.get("/api/changes/other/2020/2021").status_code == 400
    monkeypatch.delenv("GW_OUTPUTS")
    importlib.reload(api)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
