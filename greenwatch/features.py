"""Model input features: Sentinel-2 bands + spectral indices."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def band_index(bands: list[str], name: str) -> int:
    return bands.index(name)


def compute_features(s2_uint16: np.ndarray, bands: list[str], indices: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(B,H,W) uint16 reflectance*1e4 -> (C,H,W) float32 features and (H,W) valid mask."""
    refl = s2_uint16.astype(np.float32) / 10000.0
    valid = (s2_uint16 > 0).all(0)
    feats = [refl]
    b = {n: refl[band_index(bands, n)] for n in bands}
    for name in indices:
        if name == "NDVI":
            idx = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + EPS)
        elif name == "NDBI":
            idx = (b["B11"] - b["B08"]) / (b["B11"] + b["B08"] + EPS)
        elif name == "NDWI":
            idx = (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + EPS)
        else:
            raise ValueError(f"unknown index {name}")
        feats.append(idx[None])
    x = np.concatenate(feats, 0)
    x[:, ~valid] = 0
    return x.astype(np.float32), valid


def ndvi(s2_uint16: np.ndarray, bands: list[str]) -> np.ndarray:
    refl = s2_uint16.astype(np.float32) / 10000.0
    nir, red = refl[band_index(bands, "B08")], refl[band_index(bands, "B04")]
    return (nir - red) / (nir + red + EPS)


def normalize(x: np.ndarray, mean, std) -> np.ndarray:
    mean = np.asarray(mean, np.float32)[:, None, None]
    std = np.asarray(std, np.float32)[:, None, None]
    return (x - mean) / (std + EPS)
