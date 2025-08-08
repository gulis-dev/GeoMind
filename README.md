# GeoMind – GeoGuessr AI (Director + Experts)

> Two‑stage deep learning pipeline for predicting an image’s approximate geographic location: (1) Region classification (Director) → (2) Region‑conditioned coordinate regression (Experts).

---

## Created by:
### - Oskar Andrukiewicz (gulis-dev)
### - Piotr Kaptur (piotr-kaptur)

IMPORTANT NOTE! 
We created it together in Pycharma using the collaboration feature, so only gulis-dev committed

## Notebooks Overview (Quick Access)

| No | Notebook | Link | Purpose (Summary) |
|----|----------|------|-------------------|
| 01 | Data Collecting | [`notebooks/01_Data_Collecting.ipynb`](notebooks/01_Data_Collecting.ipynb) | Notes / scaffold for sourcing raw imagery & metadata |
| 02 | Data Preprocessing | [`notebooks/02_Data_Preprocessing.ipynb`](notebooks/02_Data_Preprocessing.ipynb) | Cleaning & preparing tabular metadata before modeling |
| 03 | Data Analysis & Imbalance Check | [`notebooks/03_data_analysis_and_Imbalance_check.ipynb`](notebooks/03_data_analysis_and_Imbalance_check.ipynb) | Exploratory analysis: class distribution, imbalance, visual stats (large outputs) |
| 04 | Director Training (earlier version) | [`notebooks/04_train_director.ipynb`](notebooks/04_train_director.ipynb) | Initial Director model training experiments |
| 05 | Experts Training | [`notebooks/05_train_experts.ipynb`](notebooks/05_train_experts.ipynb) | Per‑region regression (lat/lon normalization & training loops) |
| 06 | Inference Overview | [`notebooks/06_inference_overview.ipynb`](notebooks/06_inference_overview.ipynb) | End‑to‑end inference documentation |
| 07 | Director v3 Training | [`notebooks/07_director_v3_training.ipynb`](notebooks/07_director_v3_training.ipynb) | Improved Director: augmentation, class balancing, early stopping |

If you are browsing on GitHub and large notebooks (e.g. #03) load slowly, consider cloning and opening locally, or clearing heavy cell outputs before new commits.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Core Motivation](#core-motivation)
- [Architecture](#architecture)
- [Regions & Coordinate Normalization](#regions--coordinate-normalization)
- [Data Pipeline](#data-pipeline)
- [Modeling](#modeling)
  - [Director (Classification)](#director-classification)
  - [Experts (Conditional Regression)](#experts-conditional-regression)
- [Training Setup](#training-setup)
- [Results (Director v3)](#results-director-v3)
- [Inference Workflow](#inference-workflow)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Citation (Optional)](#citation-optional)

---

## Project Overview

GeoMind localizes an image approximately on the globe. Instead of regressing latitude & longitude directly (highly multimodal / discontinuous), it divides the task:

1. Director: classify the image into one of 13 macro geographic regions.
2. Region Expert: a dedicated regression model (one per region) outputs normalized (lat, lon) which are denormalized using that region’s bounding box.

This cascaded approach reduces ambiguity and improves stability vs. monolithic global regression.

---

## Core Motivation

- Global coordinate regression is non-convex and multi-peaked (coastal roads vs. inland desert look similar).
- Restricting the regression domain (after region classification) allows Experts to specialize.
- Enables modular improvement: swap backbone for Director or retrain a weak Expert independently.

---

## Architecture

```
Image
  │
  ▼
┌────────────────┐
│   Director      │  EfficientNet-B0 (13-way softmax)
└───────┬────────┘
        │ region_id (0..12)
        ▼
┌────────────────────────┐
│ Region Expert (id=k)   │  EfficientNet-B0 head → 2 sigmoid outputs
└──────────┬─────────────┘
           │ (norm_lat, norm_lon) ∈ [0,1]^2
           ▼
Denormalize via region bounding box → (latitude, longitude)
```

---

## Regions & Coordinate Normalization

Each region k defines:
- lat_min, lat_max
- lon_min, lon_max

Expert outputs y_lat, y_lon in [0,1]; convert:
```
lat = y_lat * (lat_max - lat_min) + lat_min
lon = y_lon * (lon_max - lon_min) + lon_min
```

Advantages:
- Uniform learning scale across regions.
- Avoids exploding coordinate variance across the globe.
- Facilitates per-region error analysis.

---

## Data Pipeline

(Ref. Notebooks 01–03)

1. Collection: Acquire raw images + metadata (filenames, region IDs, optional ground-truth coordinates).
2. Validation:
   - Check file existence.
   - Validate label range (0–12 for 13 regions).
   - Detect missing or malformed rows.
3. Split:
   - Train / Validation (90/10 stratified or random with fixed seed).
4. Analysis:
   - Class imbalance quantification.
   - Spatial dispersion per region.
   - Visual sanity checks (edge cases, ambiguous scenery).
5. (Optional) Future additions:
   - Hard example tagging (misclassified by Director).
   - Geo-visual clustering for region refinement.

---

## Modeling

### Director (Classification)

- Backbone: EfficientNet-B0 (can be swapped later).
- Input: 224×224 RGB (scaled & normalized).
- Output: 13 logits → softmax.
- Class imbalance mitigation:
  - WeightedRandomSampler
  - Per-class weight vector in CrossEntropyLoss

### Experts (Conditional Regression)

- One model per region (same backbone, lightweight head).
- Head: Global pooling → Linear → 2 units → Sigmoid.
- Loss: MSE / SmoothL1 on normalized coordinates.
- Denormalization only at evaluation / inference.

---

## Training Setup

Key configuration (Director v3 – Notebook 07):

| Component | Choice |
|-----------|--------|
| Resolution | 224×224 |
| Optimizer | AdamW (lr=5e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Early Stopping | Patience 6 (val_loss) |
| Loss | CrossEntropy (class weights) |
| Augmentations | RandomResizedCrop, flips (H/V), Rotation (~±20°), ColorJitter, RandomGrayscale |
| Normalization | ImageNet mean/std |
| Checkpoint | Best val_loss |

Potential improvements (see Roadmap): multi-metric checkpointing (accuracy + loss), label smoothing, MixUp / CutMix, partial freezing for speed, advanced backbones.

---

## Results (Director v3)

Best (by val_loss): 1.3318  
Peak validation accuracy reached later: 0.6397 (not saved under loss-only strategy).

Classification Report (summary):
- Strong classes: 5, 10, 11 (high precision & recall).
- Weaker / confused: 3, 12, partial confusion among adjacent region pairs (geographic similarity).
- Weighted F1 ≈ 0.64.

(Full matrix & report reproduced in notebook 07.)

Observations:
- Some systematic confusions might justify region boundary refinement or hierarchical sub-classes.
- Dual checkpointing (best_loss, best_accuracy) recommended to capture later accuracy gains.

---

## Inference Workflow

(Ref. Notebook 06)

1. Preprocess image (resize/crop → 224×224 → normalize).
2. Director predicts region_id.
3. Load corresponding Expert weights.
4. Expert outputs normalized (lat, lon).
5. Denormalize to real coordinates.
6. (Optional) Compute Haversine distance if ground-truth available.
7. Generate quick review link:
   ```
   https://www.google.com/maps?q=<lat>,<lon>
   ```
8. Aggregate metrics across a batch (mean distance, percentile thresholds).

---

## Repository Structure

(Representative – adjust as code evolves.)
```
.
├── notebooks/
│   ├── 01_Data_Collecting.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_data_analysis_and_Imbalance_check.ipynb
│   ├── 04_train_director.ipynb
│   ├── 05_train_experts.ipynb
│   ├── 06_inference_overview.ipynb
│   └── 07_director_v3_training.ipynb
├── src/
│   ├── data/                  # (planned) dataset & transforms modules
│   ├── models/                # (planned) director & expert definitions
│   ├── training/              # training scripts / loops
│   ├── inference/             # inference utilities / CLI
│   └── utils/                 # metrics, logging, geo utils
├── saved_models/
│   ├── director/
│   └── experts/
├── requirements.txt
├── README.md
└── LICENSE (pending choice: MIT / Apache-2.0)
```

Suggested next refactors:
- Move repeated notebook code → `src/`.
- Provide a CLI entry (e.g. `python -m geomind.inference ...`).

---

## License

(Choose one—placeholder below.)

Example (MIT):
```
This project is licensed under the MIT License.

```

---

## Quick Start Snippet

```bash
# Install
git clone https://github.com/<your-user-or-org>/GeoMind.git
cd GeoMind
pip install -r requirements.txt

# (Optional) Train director
python src/training/train_director.py --config configs/director_v3.yaml

# Inference (example)
python src/inference/run_inference.py \
    --images_dir data/raw/images \
    --director_path saved_models/director/director_v3_best.pt \
    --experts_dir saved_models/experts \
    --output predictions.csv
```

---

## Disclaimer

This system produces approximate geolocations and may be wrong or biased. Do not use for safety‑critical, surveillance, or privacy‑sensitive applications without thorough validation and ethical review.

---

Feel free to open issues / PRs for improvements, benchmarking results, or integration ideas. Contributions welcome!
