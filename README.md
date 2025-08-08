# GeoMind – Global Image Geolocation (Director + Experts)

A two-stage deep learning system for approximate geolocation of street‑level (or similar) images:
1. Director – multi-class classifier (13 broad world regions).
2. Region Expert – per‑region regressor producing normalized (lat, lon) which are then denormalized into real geographic coordinates.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Regions & Coordinate Normalization](#regions--coordinate-normalization)
- [Data](#data)
- [Training](#training)
  - [Director v3](#director-v3)
  - [Experts (Regressors)](#experts-regressors)
- [Inference](#inference)
- [Repository Structure](#repository-structure)
- [Results & Metrics](#results--metrics)
  - [Classification Report](#classification-report)
  - [Confusion Matrix](#confusion-matrix)
- [How to Run](#how-to-run)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Example Commands](#example-commands)
- [Roadmap / Potential Improvements](#roadmap--potential-improvements)
- [FAQ](#faq)
- [License](#license)

---

## Project Overview

GeoMind tackles the problem of predicting an image’s approximate geographic location from visual cues.  
Instead of regressing latitude/longitude directly (highly multi‑modal), we:
1. Classify the image into one of 13 macro regions (Director).
2. Run a region‑specific regression model (Expert) constrained to that region’s bounding box.

This decomposition improves stability and reduces the search space.

---

## Key Features

- Two-stage pipeline (global classification + conditional regression).
- EfficientNet‑B0 backbone (easily swappable).
- Rich visual augmentation set.
- Class imbalance mitigation (WeightedRandomSampler + per‑class loss weights).
- Early stopping + ReduceLROnPlateau scheduler.
- Error analysis: classification report & confusion matrix.
- Region-normalized coordinate regression (Sigmoid outputs mapped to geographic ranges).
- Haversine distance evaluation + Google Maps links for qualitative inspection.

---

## Architecture

```
┌──────────┐        image        ┌────────────────┐    region_id
│  Image   │  ─────────────────▶ │   Director     │ ───────────┐
└──────────┘                     │ (13 classes)   │            │
                                 └────────────────┘            ▼
                                             ┌────────────────────────┐
                                             │  Region Expert (k)     │
                                             │  (lat/lon regression)  │
                                             └────────────────────────┘
                                                        │  (norm [0,1]^2)
                                                        ▼
                                            Denormalize → (latitude, longitude)
```

---

## Regions & Coordinate Normalization

Each region k has:
- lat_min, lat_max
- lon_min, lon_max

Expert output (y_lat, y_lon) ∈ [0, 1]^2 is denormalized:
```
lat = y_lat * (lat_max - lat_min) + lat_min
lon = y_lon * (lon_max - lon_min) + lon_min
```

---

## Data

- CSV includes at least: `filename`, `region_id` (other CSVs may contain `latitude`, `longitude`, `superRegion`).
- CSV validation checks:
  - Missing columns
  - Label range (0–12)
  - Missing files
- Split: 90% train / 10% validation (fixed seed).
- 13 region classes with imbalanced distribution (handled via weighting & sampling).

---

## Training

### Director v3
Design choices:
- Input resolution: 224×224 (native EfficientNet‑B0 size).
- Full fine‑tuning (all layers unfrozen).
- Loss: `CrossEntropyLoss` with per-class weights.
- Optimizer: `AdamW (lr=5e-4, weight_decay=1e-4)`.
- Scheduler: `ReduceLROnPlateau (mode='min', factor=0.5, patience=3)`.
- Early stopping: patience = 6 (monitors val_loss).
- Augmentations (train):
  - RandomResizedCrop(224, scale=(0.8,1.0), ratio=(1.0,1.0))
  - RandomHorizontalFlip / RandomVerticalFlip
  - ColorJitter (brightness, contrast, saturation, hue)
  - RandomRotation (±20°)
  - RandomGrayscale (p=0.08)
  - Normalization (ImageNet)
- Validation transform: CenterCrop or deterministic resize + normalization.
- Balanced batches: WeightedRandomSampler + class weights in loss.

### Experts (Regressors)
- Same backbone (EfficientNet‑B0).
- Head: Linear → 2 + Sigmoid.
- One model per region (trained with filtered dataset).
- Loss typically MSE or Smooth L1 (depending on implementation).
- Denormalization performed only at inference.

---

## Inference

Steps:
1. Load image → (optionally) center crop (e.g., 640×480) → resize 224×224 → normalize.
2. Director predicts region_id (0–12).
3. Load the corresponding region Expert.
4. Expert outputs (norm_lat, norm_lon).
5. Denormalize to real coordinates.
6. (Optional) compute Haversine distance if ground truth is available.
7. (Optional) produce Google Maps link: `https://www.google.com/maps?q=lat,lon`.

---

## Repository Structure

(Adjust to actual tree)
```
.
├── data/
│   ├── raw/
│   │   └── images/
│   └── metadata_final.csv
├── saved_models/
│   ├── director/
│   │   └── efficientnet_b0_director_v2.pth
│   └── experts/
│       ├── expert_regressor_0.pth
│       └── ...
├── src/
│   ├── training/
│   │   ├── train_director.py
│   │   └── train_expert.py
│   └── inference/
│       └── run_inference.py
├── notebooks/
│   ├── director_v3_training.ipynb
│   └── inference_documentation.ipynb
├── README.md          (Polish)
├── README_en.md       (English)
└── requirements.txt
```

---

## Results & Metrics

Best checkpoint saved by lowest validation loss:
- Best val loss: 1.3318
- Later val accuracy peaked at 0.6397 (loss-based early stopping did not update the checkpoint beyond best loss).

### Classification Report

```
              precision    recall  f1-score   support
0             0.7010      0.6194    0.6577    1298
1             0.5291      0.5644    0.5462    1336
2             0.5709      0.4495    0.5030     752
3             0.3496      0.3482    0.3489     247
4             0.4233      0.4858    0.4524     739
5             0.7531      0.7494    0.7512    1225
6             0.5157      0.5863    0.5487     365
7             0.5820      0.6688    0.6224     939
8             0.7591      0.6407    0.6949     551
9             0.7919      0.6429    0.7096     882
10            0.8173      0.8473    0.8320     491
11            0.7250      0.8372    0.7771    1118
12            0.5000      0.4211    0.4571      57

accuracy                                0.6397   10000
macro avg       0.6168      0.6047    0.6078   10000
weighted avg    0.6460      0.6397    0.6399   10000
```

### Confusion Matrix

Python code reproducing the matrix:

```python
import matplotlib.pyplot as plt
import numpy as np

cm = np.array([
    [804, 87, 66, 20,102, 73, 21, 28, 0,13, 4,78, 2],
    [ 60,754, 21, 26, 42, 39, 49,127,27,52,17,115, 7],
    [ 66, 40,338, 26,152, 44, 24, 18, 3, 6, 4, 30, 1],
    [  7, 37, 19, 86, 31,  6,  9,  9, 1, 3,15, 24, 0],
    [ 64, 55, 67, 34,359, 95, 10, 18, 2, 5, 9, 18, 3],
    [ 64, 47, 27,  5, 93,918,  6, 13, 5,13,14, 19, 1],
    [  8, 33,  9,  7, 21,  1,214, 56, 5, 0, 2,  6, 3],
    [  9,126,  7,  7, 19,  8, 55,628,41,23, 6,  8, 2],
    [  1, 24,  3,  4,  6,  4,  9,130,353,11, 4,  2, 0],
    [ 21,131,  8, 10,  9, 11,  1, 35, 17,567,16, 52, 4],
    [  0, 17,  0,  4,  8, 12,  6,  4,  5, 16,416,  3, 0],
    [ 43, 57, 26, 14,  6,  8, 10,  9,  2,  4,  2,936, 1],
    [  0, 17,  1,  3,  0,  0,  1,  4,  4,  3,  0,  0,24],
])

fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=np.arange(13),
    yticklabels=np.arange(13),
    title="Confusion Matrix",
    ylabel="True label",
    xlabel="Predicted label"
)
plt.xticks(rotation=45)
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.show()
```

---

## How to Run

### Requirements
- Python 3.10+
- GPU (CUDA recommended)
- Packages: `torch`, `torchvision`, `efficientnet_pytorch`, `pandas`, `tqdm`, `scikit-learn`, `matplotlib`, `Pillow`, `numpy`

### Installation

```bash
git clone https://github.com/<your_org_or_user>/geomind.git
cd geomind
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Example Commands

Train Director:
```bash
python src/training/train_director.py
```

Run inference over a folder:
```bash
python src/inference/run_inference.py \
  --images_dir data/raw/images \
  --csv metadata_inference.csv \
  --director_path saved_models/director/efficientnet_b0_director_v2.pth \
  --experts_dir saved_models/experts/
```

(Adjust arguments to your actual script interface.)

---

## Roadmap / Potential Improvements

| Area | Idea |
|------|------|
| Checkpointing | Separate best-by-loss & best-by-accuracy saves |
| Loss | Label smoothing / Focal Loss |
| Augmentations | MixUp / CutMix |
| Backbone | EfficientNet‑V2 / ConvNeXt / ViT |
| Ensemble | Multiple Directors + voting / logit averaging |
| Experts | Joint multi-head model / uncertainty output |
| Calibration | Temperature scaling of softmax |
| Performance | AMP (mixed precision) |
| Interpretability | Grad-CAM visualizations |
| Deployment | ONNX / TorchScript export |
| Sampling | Hard example mining from confusion matrix |

---

## FAQ

**Why a two-stage approach?**  
It decomposes a multi-modal global regression into simpler conditional regressions.

**Why 224×224?**  
Native size for EfficientNet‑B0, fast and sufficiently expressive.

**Why Sigmoid in experts?**  
To constrain outputs to [0,1] so denormalization is stable.

**Can more regions be added?**  
Yes: update region definitions, bounding boxes, retrain Director & new Experts.

**Why did accuracy improve while loss did not?**  
CrossEntropy can plateau or worsen slightly while accuracy still fluctuates. Consider dual checkpoint metrics if accuracy is critical.

---

## License

```
MIT License

Copyright (c) 2025 gulis-dev, piotr-kaptur
```
