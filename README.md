# Tashkent Green Watch 🌳🛰️

**Green-cover segmentation and change detection for Tashkent from Sentinel-2 imagery.**
A U-Net model maps tree and shrub cover every summer (2019–2025), finds where green areas
were lost or gained, and aggregates the results to H3 hexagons and districts. A FastAPI
service and a map interface (UZ / RU / EN) present everything.

> 🇷🇺 Русская версия: [README.ru.md](README.ru.md)

---

## Why

Tashkent is building a tree-by-tree digital registry ([green.tashkent.uz](https://green.tashkent.uz)),
and from 2026 satellite monitoring is used nationally to detect illegal tree cutting. Field
inventory gives precise data for each tree, but it is slow and covers the city once.
**Satellite monitoring covers the whole city every season.** This project is a prototype of that
second layer:

- **where** green cover is shrinking (hotspot map, area in hectares),
- **how fast** (yearly time series per hexagon / district),
- **which patches to check first**: a list of loss polygons with area and confidence. A field team
  or the registry can then verify them.

## How it works

```mermaid
flowchart LR
  A[Sentinel-2 L2A<br/>summer scenes] -->|cloud mask SCL,<br/>median composite| B[Yearly composites<br/>10 m, 6 bands]
  W[ESA WorldCover<br/>2020 / 2021] -->|tree + shrub = green| L[Labels]
  B --> T[Tiles + spatial<br/>block split]
  L --> T
  T --> M[U-Net<br/>bands + NDVI + NDBI]
  M --> P["P(green) map<br/>per year"]
  P --> C[Change detection<br/>hysteresis + morphology]
  C --> G[Loss / gain polygons]
  P --> H[H3 / district stats]
  G --> API[FastAPI + map]
  H --> API
```

| Stage | What happens | Code |
|---|---|---|
| Data | Sentinel-2 L2A from Microsoft Planetary Computer (free). June–August scenes, clouds masked per pixel with SCL, processing-baseline 04.00 offset harmonised, median composite. ESA WorldCover on the same grid. | `greenwatch/data/download.py` |
| Dataset | Split by **spatial blocks** (512 px), not random tiles, with a **64 px buffer** between blocks of different splits (buffer pixels belong to no split), so neighbouring pixels cannot leak between train and test. Train = **random 256 px crops** fully inside the train core (500 per epoch); val / test = fixed 256 px tiles. Repeated for **3 spatial folds** (3 independent block assignments). | `greenwatch/data/tiles.py` |
| Model | U-Net (ResNet-34 encoder, ImageNet init, 8 input channels: B02 B03 B04 B08 B11 B12 + NDVI + NDBI). Masked BCE + Dice loss, flips/rotations, radiometric jitter to simulate year-to-year differences. Early stopping on val IoU; decision threshold tuned on val. Runs on Apple MPS. | `greenwatch/train.py`, `model.py` |
| Baseline | NDVI threshold, tuned for best F1 on train-core pixels. Every model number is reported next to it. | `train.py::evaluate_fold` |
| Inference | Sliding window with 50 % overlap and Hann blending (no tile seams). | `model.py::predict_scene` |
| Change | Hysteresis around the model's own threshold `thr` (from the checkpoint): loss = P ≥ thr + 0.15 before **and** P ≤ thr − 0.15 after, then morphological opening and minimum area 500 m². `confirmed` = the next year is still not green, judged only on next-year pixels with data; `null` if < 50 % of the patch has data or there is no next year. | `greenwatch/change.py` |
| Aggregation | Green share per year (each year on its own valid-pixel mask), loss/gain m² per H3 cell (res 8 ≈ 0.74 km²) and per district (optional GeoJSON). | `greenwatch/aggregate.py` |
| Serving | FastAPI (`/api/meta`, `/api/hexagons`, `/api/changes/...`, `/api/point`) and a Leaflet map with no CDN dependency. | `greenwatch/api`, `web/` |

### Evaluation protocol

| Split | Imagery & labels | Question it answers |
|---|---|---|
| `val` | 2021, held-out blocks | early stopping, threshold tuning |
| `test` | 2021, unseen blocks | does it generalise **in space**? |
| `test_time` | **2020** imagery + 2020 labels, unseen blocks | does it generalise **in time**? Change detection depends on this. |

Every number is the **mean ± std over 3 spatial folds** (block assignments with seeds 42 / 43 / 44).
`val` is used for early stopping and threshold tuning, so it is optimistic by construction; read
`test` and `test_time`. The deployed model (`outputs/models/best.pt`) is fold 0 (the configured seed),
not the best-scoring fold.

Change detection is also scored against **known synthetic construction events**: event recall and
pixel precision (`outputs/demo/stats/change_eval.json`).

## Results

Real data: Tashkent, Sentinel-2 summer composites (8 least-cloudy scenes per year), WorldCover labels
(green = classes 10 + 20). Trained on 2021, `test_time` = 2020. **Mean ± std over 3 spatial folds**
(512 px blocks, 64 px buffer, seeds 42 / 43 / 44; 10–12 test tiles of 256 px per fold). Metrics are
measured **against WorldCover**, which is itself only ~75–77 % accurate (see Limitations).

| Split | U-Net IoU | U-Net F1 | NDVI IoU | NDVI F1 |
|---|---|---|---|---|
| val (optimistic: used for early stopping + threshold) | 0.439 ± 0.051 | 0.608 ± 0.049 | 0.255 ± 0.042 | 0.405 ± 0.054 |
| **test** (2021, unseen blocks) | **0.463 ± 0.091** | **0.627 ± 0.089** | 0.305 ± 0.128 | 0.452 ± 0.163 |
| **test_time** (2020, unseen blocks) | **0.404 ± 0.104** | **0.567 ± 0.112** | 0.297 ± 0.145 | 0.437 ± 0.189 |

U-Net beats the NDVI baseline on **every** fold: test IoU +0.127 / +0.210 / +0.136, test_time IoU
+0.097 / +0.164 / +0.060. The spread between folds (± 0.09–0.10 IoU) is large: which city blocks land in
test matters, so a single split would be misleading. The drop from `test` to `test_time` (−0.06 IoU)
is the cost of moving to another year (and to WorldCover v1.0.0 labels instead of v2.0.0).

<details><summary>First iteration (for comparison, not directly comparable)</summary>

1024 px blocks, no buffer, one split, 60 fixed training tiles, 15 epochs: test was **5 tiles**.
U-Net test IoU 0.177 vs NDVI 0.216 (U-Net lost), test_time 0.172 vs 0.167. The test sets differ between
iterations, so compare the U-Net − NDVI gap rather than absolute numbers: −0.039 → **+0.158** on test,
+0.006 → **+0.107** on test_time. The gain comes from training (≈ 4 → 31 optimiser steps per epoch with
random crops, early stopping after 27–32 epochs) and from a test set large enough to be meaningful.
</details>

**Change 2020 → 2021 (deployed fold-0 model, threshold 0.275 → hysteresis 0.425 / 0.125):** 904 ha of
candidate loss (2 963 patches) and 151 ha of gain; city green share 12.0 % → 8.9 %. Treat this with
caution: part of it is year-to-year domain shift of the model (`test_time` is clearly worse than `test`)
and a different WorldCover version, not real change. With only two years there is no following year, so
every patch has `confirmed = null` (not checked). Download 2019 and 2022–2025 to get confirmations.

Synthetic offline demo (pipeline sanity check only): 21 / 27 construction events found, 99.7 % pixel
precision. (Events smaller than ~40 px of green fall below what 10 m imagery can resolve.)

## Quick start

```bash
conda create -n greenwatch python=3.11 -y && conda activate greenwatch
pip install -r requirements.txt

# 1) Offline demo on synthetic data (about 1 min)
make demo
make serve-demo          # → http://localhost:8000

# 2) Real data for Tashkent
make download            # Sentinel-2 2019–2025 + WorldCover (a few GB, 20–60 min)
make pipeline            # tiles → train (MPS) → infer → change → aggregate
make serve               # → http://localhost:8000
make mlflow              # experiment tracking UI → http://localhost:5000

make test                # unit tests + end-to-end pipeline + API
```

Change `config.yaml` to set the area, years, season, WorldCover classes counted as green (add `30`
for grassland), model and thresholds. For district statistics, put a GeoJSON of district boundaries at
`data/boundaries/districts.geojson` (with a `name` field).

Docker (API + map, outputs mounted read-only): `docker compose up --build`.

## Project structure

```
greenwatch/
  data/download.py    Sentinel-2 composites + WorldCover (STAC)
  data/synthetic.py   offline synthetic scenes with known ground truth
  data/tiles.py       tiling + spatial block split
  features.py         bands + spectral indices
  model.py            U-Net, checkpoints, sliding-window inference
  train.py            training, threshold tuning, U-Net vs NDVI evaluation
  infer.py            yearly probability maps + web overlays
  change.py           loss / gain detection and vectorisation
  aggregate.py        H3 / district / city statistics
  api/main.py         FastAPI
web/index.html        map UI (UZ / RU / EN)
tests/                pytest (incl. end-to-end run)
config.yaml           all parameters
```

## Limitations (honest)

- **10 m resolution.** The model sees green *areas* (parks, tree rows, yards), not individual trees.
  A single cut tree is invisible. This complements a per-tree registry; it does not replace it.
- **Labels are WorldCover, not ground truth.** WorldCover has about 75–77 % overall accuracy, with
  errors at edges and in mixed urban pixels. The model learns those errors too. Next step: hand-label
  a few hundred Tashkent points for the test set.
- **Phenology and dry summers** can look like loss. Hysteresis, the `confirmed` flag and summer-only
  composites reduce this, but don't eliminate it.
- **A loss is not a violation.** Permitted construction also removes green cover. To flag
  *illegal* removal, you would join the results with permits or cadastre data, which needs official access.

## Roadmap

- [ ] Hand-labelled Tashkent test set (stratified points, 2 annotators)
- [ ] Siamese / bi-temporal change model trained on detected events
- [ ] Super-resolution or higher-resolution imagery for street-tree rows
- [ ] Join with the tree registry: "registry says N trees here, canopy is gone"
- [ ] Scheduled seasonal runs + alerts per district

## Data & licences

Sentinel-2: © ESA Copernicus (free, open). ESA WorldCover: CC BY 4.0. Basemap: © OpenStreetMap contributors, © CARTO.
