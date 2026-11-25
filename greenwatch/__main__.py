"""CLI: python -m greenwatch <command>

Commands
  download    Sentinel-2 composites + WorldCover labels (needs internet)
  tiles       cut training tiles with spatial block split
  train       train U-Net, evaluate vs NDVI baseline
  infer       predict every year, build web overlays
  change      detect green loss / gain between years
  aggregate   H3 / district / city statistics
  pipeline    tiles -> train -> infer -> change -> aggregate
  demo        full pipeline on synthetic data (offline, ~minutes on CPU)
  serve       start API + map (uvicorn)
"""
from __future__ import annotations

import argparse
import os
import sys

from greenwatch.config import load_config

DEMO_OVERRIDES = {
    "dataset": {"tile_size": 128, "block_size": 256, "fold_seeds": [42]},
    "model": {"encoder": "resnet18", "encoder_weights": None},
    "train": {"epochs": 12, "batch_size": 8, "patience": 5, "lr": 0.001, "crops_per_epoch": 160},
    "change": {"min_area_m2": 800},
    "paths": {
        "raw": "data/demo/raw", "tiles": "data/demo/tiles", "models": "outputs/demo/models",
        "pred": "outputs/demo/pred", "change": "outputs/demo/change", "stats": "outputs/demo/stats",
        "overlays": "outputs/demo/overlays",
    },
}


def run_pipeline(cfg):
    from greenwatch import aggregate, change, infer, train
    from greenwatch.data import tiles

    tiles.main([], cfg=cfg)
    train.train(cfg)
    infer.main([], cfg=cfg)
    change.main([], cfg=cfg)
    aggregate.main([], cfg=cfg)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="greenwatch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command")
    ap.add_argument("--config")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--size", type=int, default=768, help="demo scene size (px)")
    ap.add_argument("--port", type=int, default=8000)
    a, rest = ap.parse_known_args(argv)

    if a.command == "download":
        from greenwatch.data import download

        return download.main((["--config", a.config] if a.config else []) + rest)
    if a.command == "serve":
        import uvicorn

        uvicorn.run("greenwatch.api.main:app", host="0.0.0.0", port=a.port)
        return

    cfg = load_config(a.config, DEMO_OVERRIDES if a.command == "demo" else None)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    if a.command == "demo":
        from greenwatch.data import synthetic

        synthetic.generate(cfg, size=a.size)
        run_pipeline(cfg)
        synthetic.score_change(cfg)
        print("\nDemo ready. Start the map with:\n  GW_OUTPUTS=outputs/demo python -m greenwatch serve")
        return
    if a.command == "pipeline":
        return run_pipeline(cfg)

    from importlib import import_module

    modules = {"tiles": "greenwatch.data.tiles", "train": "greenwatch.train", "infer": "greenwatch.infer",
               "change": "greenwatch.change", "aggregate": "greenwatch.aggregate"}
    if a.command not in modules:
        ap.error(f"unknown command {a.command}")
    mod = import_module(modules[a.command])
    if a.command == "train":
        return mod.train(cfg)
    return mod.main([], cfg=cfg)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
