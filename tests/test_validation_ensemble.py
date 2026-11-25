"""Third iteration: conservative change thresholds, fold ensemble, manual-validation sample + API."""
import importlib
import json

import numpy as np
import pytest
import torch

from greenwatch.change import hysteresis_thresholds
from greenwatch.model import deployed_threshold, ensemble_paths, predict_scene_ensemble, save_checkpoint
from greenwatch.validation import pick_loss_patches, rgb_chip, score_labels, stable_candidates, wilson

CONSERVATIVE = {"high_offset": 0.15, "high_min": 0.5, "low_offset": 0.1, "low_max": 0.2}


# ---------- thresholds ----------
def test_conservative_thresholds():
    assert hysteresis_thresholds(0.275, CONSERVATIVE) == (0.5, 0.175)   # high floored at 0.5
    assert hysteresis_thresholds(0.6, CONSERVATIVE) == (0.75, 0.2)      # low capped at 0.2
    assert hysteresis_thresholds(0.1, CONSERVATIVE) == (0.5, 0.05)      # low clipped to 0.05
    with pytest.raises(ValueError):
        hysteresis_thresholds(0.5, {"high_offset": -0.3, "low_offset": 0.0})


# ---------- ensemble ----------
class Const(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.logit)


def test_ensemble_averages_member_probabilities():
    feats = np.zeros((2, 64, 64), np.float32)
    meta = {"mean": [0, 0], "std": [1, 1]}
    members = [(Const(2.0), meta), (Const(-2.0), meta), (Const(0.0), meta)]
    prob = predict_scene_ensemble(members, feats, "cpu", tile=64)
    expected = (torch.sigmoid(torch.tensor(2.0)) + torch.sigmoid(torch.tensor(-2.0)) + 0.5).item() / 3
    assert np.allclose(prob, expected, atol=1e-5)


def test_ensemble_paths_and_threshold(tmp_path):
    lin = torch.nn.Linear(1, 1)
    save_checkpoint(tmp_path / "best.pt", lin, {"threshold": 0.9})
    assert ensemble_paths(tmp_path) == [tmp_path / "best.pt"]      # no folds -> single model
    for k, thr in enumerate([0.2, 0.3, 0.4]):
        save_checkpoint(tmp_path / f"fold{k}" / "best.pt", lin, {"threshold": thr})
    assert [p.parent.name for p in ensemble_paths(tmp_path)] == ["fold0", "fold1", "fold2"]
    assert deployed_threshold(tmp_path) == pytest.approx(0.3)      # mean of members, best.pt ignored


# ---------- validation sample ----------
def _feat(i, confirmed):
    return {"geometry": None, "properties": {"id": i, "confirmed": confirmed}}


def test_pick_loss_patches_prefers_confirmed_only_when_enough():
    rng = np.random.default_rng(0)
    feats = [_feat(i, i < 3) for i in range(20)]  # 3 confirmed
    chosen, only_conf, n_conf = pick_loss_patches(feats, 2, rng)
    assert only_conf and n_conf == 3 and all(f["properties"]["confirmed"] for f in chosen)
    chosen, only_conf, _ = pick_loss_patches(feats, 10, rng)
    assert not only_conf and len(chosen) == 10 and len({f["properties"]["id"] for f in chosen}) == 10
    assert pick_loss_patches([], 5, rng)[0] == []


def test_stable_candidates_keep_away_from_change():
    valid = np.ones((20, 20), bool)
    change = np.zeros((20, 20), bool)
    change[10, 10] = True
    v1 = valid.copy()
    v1[0, 0] = False
    cand = stable_candidates(valid, v1, change, margin=3)
    assert not cand[10, 13] and cand[10, 14]  # 3 px dilation (cross) around the change
    assert not cand[0, 0]                     # no data in one year -> not a candidate


def test_rgb_chip_size_and_edges():
    s2 = np.full((6, 10, 10), 800, np.uint16)
    im = rgb_chip(s2, ["B02", "B03", "B04", "B08", "B11", "B12"], 0, 0, half=8, scale=2)
    assert im.size == (32, 32)
    a = np.asarray(im)
    assert a[0, 0].sum() == 0 and a[-1, -1].sum() > 0  # outside the scene is black


def test_score_labels_precision_and_miss_rate():
    feats = [{"properties": {"sid": f"s{i}", "kind": "loss" if i < 4 else "stable"}} for i in range(8)]
    sample = {"meta": {"pair": [2019, 2025]}, "features": feats}
    labels = {"s0": {"label": "real"}, "s1": {"label": "real"}, "s2": {"label": "false"},
              "s3": {"label": "unclear"}, "s4": {"label": "false"}, "s5": {"label": "real"}}
    rep = score_labels(sample, labels)
    assert rep["groups"]["loss"] == {"n": 4, "labelled": 4, "real": 2, "false": 1, "unclear": 1}
    assert rep["change_precision"] == pytest.approx(0.667, abs=1e-3)   # unclear excluded
    assert rep["miss_rate"] == 0.5 and rep["groups"]["stable"]["labelled"] == 2
    lo, hi = wilson(2, 3)
    assert 0 < lo < 2 / 3 < hi < 1


# ---------- API ----------
def test_api_validation_roundtrip(tmp_path, monkeypatch):
    vdir = tmp_path / "validation"
    vdir.mkdir()
    sample = {"type": "FeatureCollection", "meta": {"pair": [2019, 2025]},
              "features": [{"type": "Feature", "geometry": None, "properties": {"sid": "s001", "kind": "loss"}}]}
    (vdir / "change_sample.geojson").write_text(json.dumps(sample))
    monkeypatch.setenv("GW_OUTPUTS", str(tmp_path))
    from fastapi.testclient import TestClient

    import greenwatch.api.main as api

    importlib.reload(api)
    c = TestClient(api.app)
    assert c.get("/api/validation/sample").json()["features"][0]["properties"]["sid"] == "s001"
    assert c.get("/api/validation/labels").json() == {}
    assert c.post("/api/validation/labels", json={"s001": {"label": "real", "ts": "t"}}).json() == {"saved": 1}
    assert json.loads((vdir / "labels.json").read_text())["s001"]["label"] == "real"
    assert c.post("/api/validation/labels", json={"s999": {"label": "real"}}).status_code == 422
    assert c.post("/api/validation/labels", json={"s001": {"label": "maybe"}}).status_code == 422
    assert c.get("/validate").status_code == 200
    monkeypatch.delenv("GW_OUTPUTS")
    importlib.reload(api)
