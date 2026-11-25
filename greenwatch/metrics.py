"""Segmentation metrics with an ignore label."""
from __future__ import annotations

import numpy as np

IGNORE = 255


def confusion(pred: np.ndarray, label: np.ndarray) -> dict[str, int]:
    m = label != IGNORE
    p, t = pred[m].astype(bool), label[m].astype(bool)
    return {"tp": int((p & t).sum()), "fp": int((p & ~t).sum()), "fn": int((~p & t).sum()), "tn": int((~p & ~t).sum())}


def scores(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "precision": prec,
        "recall": rec,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def evaluate(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    return {k: round(v, 4) for k, v in scores(confusion(pred, label)).items()}


def best_threshold(prob: np.ndarray, label: np.ndarray, grid=None) -> tuple[float, float]:
    """Threshold maximising F1 (ignoring IGNORE pixels)."""
    grid = np.round(np.arange(0.05, 0.96, 0.025), 3) if grid is None else grid
    m = label != IGNORE
    pr, lb = prob[m], label[m].astype(bool)
    best = (0.5, -1.0)
    for t in grid:
        f1 = scores(confusion(pr >= t, lb.astype(np.uint8)))["f1"]
        if f1 > best[1]:
            best = (float(t), f1)
    return best
