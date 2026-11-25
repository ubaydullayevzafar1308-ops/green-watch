"""Train the green-cover segmentation model and evaluate it against an NDVI baseline.

Evaluation protocol (written to outputs/models/metrics.json):
  * val        — threshold tuning + early stopping (train-year, held-out blocks)
  * test       — same year, unseen spatial blocks
  * test_time  — other year (imagery + labels), unseen blocks: does the model
                 transfer across years? This is what change detection relies on.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from greenwatch.config import get_device, load_config, p, save_json
from greenwatch.features import normalize
from greenwatch.metrics import IGNORE, best_threshold, evaluate
from greenwatch.model import build_model, save_checkpoint


class TileDataset(Dataset):
    def __init__(self, X, y, mean, std, augment=False):
        self.X, self.y, self.mean, self.std, self.augment = X, y, mean, std, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = normalize(self.X[i].astype(np.float32), self.mean, self.std)
        y = self.y[i].astype(np.int64)
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            # radiometric jitter: years differ in illumination / phenology
            x = x * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.05)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(np.ascontiguousarray(y))


def masked_loss(logits, target):
    valid = (target != IGNORE).float()
    t = (target == 1).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, t, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    prob = torch.sigmoid(logits) * valid
    inter = (prob * t).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + (t * valid).sum() + 1)
    return bce + dice


def load_split(cfg, year, split):
    path = p(cfg, "tiles") / f"{year}_{split}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return d["X"], d["y"]


def channel_stats(X, y):
    m = (y != IGNORE)
    xs = X.astype(np.float32).transpose(1, 0, 2, 3)[:, m]
    return xs.mean(1).tolist(), xs.std(1).tolist()


@torch.no_grad()
def predict_tiles(model, X, mean, std, device, bs=16):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(normalize_batch(X[i:i + bs], mean, std)).to(device)
        out.append(torch.sigmoid(model(xb))[:, 0].cpu().numpy())
    return np.concatenate(out)


def normalize_batch(X, mean, std):
    mean = np.asarray(mean, np.float32)[None, :, None, None]
    std = np.asarray(std, np.float32)[None, :, None, None]
    return ((X.astype(np.float32) - mean) / (std + 1e-6)).astype(np.float32)


def ndvi_channel(cfg):
    return len(cfg["imagery"]["bands"]) + cfg["model"]["extra_indices"].index("NDVI")


def _mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def train(cfg):
    ds, tr, mc = cfg["dataset"], cfg["train"], cfg["model"]
    year = ds["train_year"]
    torch.manual_seed(ds["seed"])
    np.random.seed(ds["seed"])
    Xtr, ytr = load_split(cfg, year, "train")
    va = load_split(cfg, year, "val") or load_split(cfg, year, "test")
    Xva, yva = va
    mean, std = channel_stats(Xtr, ytr)
    device = get_device()
    print(f"[train] device={device} train={len(Xtr)} val={len(Xva)} channels={Xtr.shape[1]}")

    model = build_model(mc["arch"], mc["encoder"], Xtr.shape[1], mc["encoder_weights"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["epochs"])
    dl = DataLoader(TileDataset(Xtr, ytr, mean, std, augment=True), batch_size=tr["batch_size"],
                    shuffle=True, num_workers=tr["num_workers"], drop_last=len(Xtr) > tr["batch_size"])

    meta = {"arch": mc["arch"], "encoder": mc["encoder"], "in_channels": int(Xtr.shape[1]), "mean": mean,
            "std": std, "bands": cfg["imagery"]["bands"], "indices": mc["extra_indices"], "threshold": 0.5,
            "train_year": year, "green_classes": cfg["labels"]["green_classes"]}
    ckpt = p(cfg, "models") / "best.pt"
    mlf = _mlflow()
    if mlf:
        mlf.set_experiment(cfg["project"])
        mlf.start_run()
        mlf.log_params({"arch": mc["arch"], "encoder": mc["encoder"], "lr": tr["lr"], "epochs": tr["epochs"],
                        "batch_size": tr["batch_size"], "tile": ds["tile_size"], "channels": Xtr.shape[1]})

    best_iou, bad, history = -1.0, 0, []
    for ep in range(tr["epochs"]):
        model.train()
        t0, losses = time.time(), []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = masked_loss(model(xb)[:, 0], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        prob = predict_tiles(model, Xva, mean, std, device)
        val = evaluate(prob >= 0.5, yva)
        history.append({"epoch": ep + 1, "loss": float(np.mean(losses)), **val})
        print(f"[train] ep {ep + 1:3d} loss {np.mean(losses):.4f} val IoU {val['iou']:.4f} F1 {val['f1']:.4f}"
              f" ({time.time() - t0:.1f}s)")
        if mlf:
            mlf.log_metrics({"loss": float(np.mean(losses)), "val_iou": val["iou"], "val_f1": val["f1"]}, step=ep)
        if val["iou"] > best_iou:
            best_iou, bad = val["iou"], 0
            save_checkpoint(ckpt, model, meta)
        else:
            bad += 1
            if bad >= tr["patience"]:
                print("[train] early stopping")
                break

    from greenwatch.model import load_checkpoint

    model, meta = load_checkpoint(ckpt, device)
    thr, _ = best_threshold(predict_tiles(model, Xva, mean, std, device), yva)
    meta["threshold"] = thr
    save_checkpoint(ckpt, model, meta)
    metrics = evaluate_all(cfg, model, meta, device)
    metrics["history"] = history
    save_json(metrics, p(cfg, "models") / "metrics.json")
    if mlf:
        for split, m in metrics["results"].items():
            for name, v in m.items():
                mlf.log_metric(f"{split}_{name}", v)
        mlf.log_artifact(str(ckpt))
        mlf.end_run()
    print_table(metrics)
    return metrics


def evaluate_all(cfg, model, meta, device):
    """Compare U-Net vs NDVI-threshold baseline on every available split."""
    ds = cfg["dataset"]
    ch = ndvi_channel(cfg)
    Xtr, ytr = load_split(cfg, ds["train_year"], "train")
    ndvi_thr, _ = best_threshold(Xtr[:, ch].astype(np.float32), ytr, grid=np.round(np.arange(0.05, 0.85, 0.01), 3))
    splits = {"val": (ds["train_year"], "val"), "test": (ds["train_year"], "test"),
              "test_time": (ds["temporal_test_year"], "test")}
    results = {}
    for name, (yr, sp) in splits.items():
        d = load_split(cfg, yr, sp)
        if d is None:
            continue
        X, y = d
        prob = predict_tiles(model, X, meta["mean"], meta["std"], device)
        results[f"{name}/unet"] = evaluate(prob >= meta["threshold"], y)
        results[f"{name}/ndvi"] = evaluate(X[:, ch].astype(np.float32) >= ndvi_thr, y)
    return {"threshold": meta["threshold"], "ndvi_threshold": ndvi_thr, "results": results}


def print_table(metrics):
    print(f"\n{'split/model':22s} {'IoU':>7s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s}")
    for k, m in metrics["results"].items():
        print(f"{k:22s} {m['iou']:7.4f} {m['f1']:7.4f} {m['precision']:7.4f} {m['recall']:7.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    train(cfg)


if __name__ == "__main__":
    main()
