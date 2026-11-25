"""Train the green-cover segmentation model and evaluate it against an NDVI baseline.

Training data: random `tile_size` crops drawn every epoch from anywhere inside the
train core (train blocks minus the inter-split buffer), `train.crops_per_epoch`
per epoch. Fixed non-overlapping tiles gave only ~60 training samples; random
crops give far more distinct views of the same pixels without touching val/test.

Evaluation protocol (per fold, then mean ± std over folds):
  * val        — threshold tuning + early stopping (train year, held-out blocks)
  * test       — same year, unseen spatial blocks
  * test_time  — other year (imagery + labels), the same unseen blocks: does the
                 model transfer across years? Change detection relies on this.
  * NDVI baseline: one threshold tuned for best F1 on train-core pixels.
Val is used for model selection, so val numbers are optimistic by design;
test and test_time are the honest ones.

Outputs (outputs/models/):
  fold{k}/best.pt, fold{k}/metrics.json   per fold
  best.pt                                 fold 0 model (configured seed), used by `infer`
  metrics.json                            mean / std / per-fold table
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from greenwatch.config import get_device, load_config, p, save_json
from greenwatch.features import normalize
from greenwatch.metrics import IGNORE, best_threshold, evaluate
from greenwatch.model import build_model, load_checkpoint, save_checkpoint

METRIC_KEYS = ("iou", "f1", "precision", "recall", "accuracy")


def augment_pair(x, y, rng):
    """Rotations, flips and radiometric jitter (years differ in illumination / phenology)."""
    k = rng.integers(4)
    x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    x = x * rng.uniform(0.9, 1.1) + rng.normal(0, 0.05)
    return x, y


class RandomCropDataset(Dataset):
    """`n` random t×t crops per epoch; every crop lies fully inside the train core."""

    def __init__(self, X, y, origins, mean, std, n, t, augment=True, seed=0):
        if len(origins) == 0:
            raise ValueError("no valid train crops — tile_size too large for block_size/buffer_px?")
        self.X, self.y, self.origins = X, y, origins
        self.mean, self.std, self.n, self.t, self.augment = mean, std, n, t, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        r, c = self.origins[self.rng.integers(len(self.origins))]
        t = self.t
        x = normalize(np.asarray(self.X[:, r:r + t, c:c + t], np.float32), self.mean, self.std)
        y = np.asarray(self.y[r:r + t, c:c + t]).astype(np.int64)
        if self.augment:
            x, y = augment_pair(x, y, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(np.ascontiguousarray(y))


def masked_loss(logits, target):
    """BCE + soft Dice, both computed only on pixels whose label != 255."""
    valid = (target != IGNORE).float()
    t = (target == 1).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, t, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    prob = torch.sigmoid(logits) * valid
    inter = (prob * t).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + (t * valid).sum() + 1)
    return bce + dice


def load_split(fdir, year, split):
    path = fdir / f"{year}_{split}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return d["X"], d["y"]


def load_train(cfg, fdir):
    from greenwatch.data.tiles import train_features_path

    ty = cfg["dataset"]["train_year"]
    X = np.load(train_features_path(cfg), mmap_mode="r")
    y = np.load(fdir / f"{ty}_train_y.npy")
    origins = np.load(fdir / f"{ty}_train_origins.npy")
    return X, y, origins


def sample_train_pixels(X, y, n=400_000, seed=0):
    """Random labelled train-core pixels -> (C, n) features, (n,) labels."""
    rr, cc = np.nonzero(y != IGNORE)
    if len(rr) > n:
        sel = np.random.default_rng(seed).choice(len(rr), n, replace=False)
        sel.sort()
        rr, cc = rr[sel], cc[sel]
    return np.asarray(X[:, rr, cc], np.float32), y[rr, cc]


def channel_stats(xs):
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


def train_fold(cfg, k, seed, fdir, mdir, mlf=None):
    """Train + evaluate one fold. Returns its metrics dict."""
    ds, tr, mc = cfg["dataset"], cfg["train"], cfg["model"]
    year, t = ds["train_year"], ds["tile_size"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y, origins = load_train(cfg, fdir)
    va = load_split(fdir, year, "val")
    if va is None:
        raise RuntimeError(f"fold{k}: no validation tiles — check dataset.split / block_size")
    Xva, yva = va
    px_x, px_y = sample_train_pixels(X, y, seed=seed)
    mean, std = channel_stats(px_x)
    device = get_device()
    print(f"[train fold{k}] device={device} crop_origins={len(origins)} crops/epoch={tr['crops_per_epoch']} "
          f"val_tiles={len(Xva)} channels={X.shape[0]}")

    model = build_model(mc["arch"], mc["encoder"], X.shape[0], mc["encoder_weights"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["epochs"])
    dset = RandomCropDataset(X, y, origins, mean, std, tr["crops_per_epoch"], t, augment=True, seed=seed)
    dl = DataLoader(dset, batch_size=tr["batch_size"], shuffle=False, num_workers=tr["num_workers"], drop_last=True)

    meta = {"arch": mc["arch"], "encoder": mc["encoder"], "in_channels": int(X.shape[0]), "mean": mean,
            "std": std, "bands": cfg["imagery"]["bands"], "indices": mc["extra_indices"], "threshold": 0.5,
            "train_year": year, "green_classes": cfg["labels"]["green_classes"], "fold": k, "split_seed": seed}
    ckpt = mdir / "best.pt"
    if mlf:
        mlf.start_run(run_name=f"fold{k}")
        mlf.log_params({"arch": mc["arch"], "encoder": mc["encoder"], "lr": tr["lr"], "epochs": tr["epochs"],
                        "batch_size": tr["batch_size"], "tile": t, "channels": X.shape[0], "fold": k,
                        "split_seed": seed, "block_size": ds["block_size"], "buffer_px": ds.get("buffer_px", 0),
                        "crops_per_epoch": tr["crops_per_epoch"]})

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
        print(f"[train fold{k}] ep {ep + 1:3d} loss {np.mean(losses):.4f} val IoU {val['iou']:.4f} "
              f"F1 {val['f1']:.4f} ({time.time() - t0:.1f}s)")
        if mlf:
            mlf.log_metrics({"loss": float(np.mean(losses)), "val_iou": val["iou"], "val_f1": val["f1"]}, step=ep)
        if val["iou"] > best_iou:
            best_iou, bad = val["iou"], 0
            save_checkpoint(ckpt, model, meta)
        else:
            bad += 1
            if bad >= tr["patience"]:
                print(f"[train fold{k}] early stopping")
                break

    model, meta = load_checkpoint(ckpt, device)
    thr, _ = best_threshold(predict_tiles(model, Xva, mean, std, device), yva)
    meta["threshold"] = thr
    save_checkpoint(ckpt, model, meta)
    metrics = evaluate_fold(cfg, fdir, model, meta, device, px_x, px_y)
    metrics.update({"fold": k, "split_seed": int(seed), "epochs_run": len(history), "history": history})
    save_json(metrics, mdir / "metrics.json")
    if mlf:
        for split, m in metrics["results"].items():
            for name, v in m.items():
                mlf.log_metric(f"{split.replace('/', '_')}_{name}", v)
        mlf.end_run()
    print_table(metrics["results"], title=f"fold{k} (seed {seed})")
    return metrics


def evaluate_fold(cfg, fdir, model, meta, device, px_x, px_y):
    """U-Net vs NDVI-threshold baseline on val / test / test_time of one fold."""
    ds = cfg["dataset"]
    ch = ndvi_channel(cfg)
    ndvi_thr, _ = best_threshold(px_x[ch], px_y, grid=np.round(np.arange(0.05, 0.85, 0.01), 3))
    splits = {"val": (ds["train_year"], "val"), "test": (ds["train_year"], "test"),
              "test_time": (ds["temporal_test_year"], "test")}
    results = {}
    for name, (yr, sp) in splits.items():
        d = load_split(fdir, yr, sp)
        if d is None:
            continue
        X, y = d
        prob = predict_tiles(model, X, meta["mean"], meta["std"], device)
        results[f"{name}/unet"] = evaluate(prob >= meta["threshold"], y)
        results[f"{name}/ndvi"] = evaluate(X[:, ch].astype(np.float32) >= ndvi_thr, y)
    return {"threshold": meta["threshold"], "ndvi_threshold": ndvi_thr, "results": results}


def summarize_folds(fold_metrics):
    """Mean and (population) std of every split/model metric over folds."""
    keys = list(dict.fromkeys(k for m in fold_metrics for k in m["results"]))  # val, test, test_time order
    mean, std = {}, {}
    for key in keys:
        vals = [m["results"][key] for m in fold_metrics if key in m["results"]]
        mean[key] = {mk: round(float(np.mean([v[mk] for v in vals])), 4) for mk in METRIC_KEYS}
        std[key] = {mk: round(float(np.std([v[mk] for v in vals])), 4) for mk in METRIC_KEYS}
    return mean, std


def train(cfg):
    ds = cfg["dataset"]
    from greenwatch.data.tiles import fold_dir

    mlf = _mlflow()
    if mlf:
        mlf.set_experiment(cfg["project"])
    folds = []
    for k, seed in enumerate(ds["fold_seeds"]):
        mdir = p(cfg, "models") / f"fold{k}"
        mdir.mkdir(parents=True, exist_ok=True)
        folds.append(train_fold(cfg, k, seed, fold_dir(cfg, k), mdir, mlf))
    mean, std = summarize_folds(folds)
    shutil.copy(p(cfg, "models") / "fold0" / "best.pt", p(cfg, "models") / "best.pt")
    metrics = {
        "protocol": {"folds": len(folds), "fold_seeds": ds["fold_seeds"], "block_size": ds["block_size"],
                     "buffer_px": ds.get("buffer_px", 0), "train_year": ds["train_year"],
                     "temporal_test_year": ds["temporal_test_year"],
                     "deployed_model": "fold0 (configured seed, not selected by score)"},
        "threshold": folds[0]["threshold"], "ndvi_threshold": folds[0]["ndvi_threshold"],
        "results": mean, "results_std": std,
        "folds": [{k: v for k, v in f.items() if k != "history"} for f in folds],
    }
    save_json(metrics, p(cfg, "models") / "metrics.json")
    print_table(mean, std, title=f"mean ± std over {len(folds)} folds")
    return metrics


def print_table(results, std=None, title=""):
    print(f"\n[{title}]\n{'split/model':18s} {'IoU':>15s} {'F1':>15s} {'Prec':>15s} {'Recall':>15s}")
    for k, m in results.items():
        cells = [f"{m[c]:.3f}" + (f" ± {std[k][c]:.3f}" if std else "") for c in ("iou", "f1", "precision", "recall")]
        print(f"{k:18s} " + " ".join(f"{c:>15s}" for c in cells))


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
