"""Model definition, checkpoint IO and sliding-window inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from greenwatch.features import normalize


def build_model(arch: str, encoder: str, in_channels: int, encoder_weights=None) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    cls = getattr(smp, arch)
    return cls(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=1)


def save_checkpoint(path, model, meta: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck["meta"]
    model = build_model(meta["arch"], meta["encoder"], meta["in_channels"], encoder_weights=None)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), meta


def _window(t: int) -> np.ndarray:
    w = np.hanning(t + 2)[1:-1]
    return np.clip(np.outer(w, w), 1e-3, None).astype(np.float32)


@torch.no_grad()
def predict_scene(model, feats: np.ndarray, meta: dict, device, tile=256, overlap=0.5, batch=8) -> np.ndarray:
    """Sliding-window probability map for a full scene. feats: (C,H,W) unnormalised."""
    x = normalize(feats, meta["mean"], meta["std"])
    C, H, W = x.shape
    tile = min(tile, H, W)
    tile -= tile % 32  # encoder needs multiples of 32
    step = max(int(tile * (1 - overlap)), 1)
    rows = list(range(0, max(H - tile, 0) + 1, step))
    cols = list(range(0, max(W - tile, 0) + 1, step))
    if rows[-1] != H - tile:
        rows.append(H - tile)
    if cols[-1] != W - tile:
        cols.append(W - tile)
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    win = _window(tile)
    coords = [(r, c) for r in rows for c in cols]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([x[:, r:r + tile, c:c + tile] for r, c in chunk])).float().to(device)
        prob = torch.sigmoid(model(xb))[:, 0].cpu().numpy()
        for (r, c), pr in zip(chunk, prob):
            acc[r:r + tile, c:c + tile] += pr * win
            wsum[r:r + tile, c:c + tile] += win
    return acc / np.maximum(wsum, 1e-6)
