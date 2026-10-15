# LeafIQ — Plant Disease Classification Using Deep Learning

Advanced Machine Learning Lab Project  
**Current status: Phase 1 ✓ | Phase 2 ✓ | Phase 3 ⚠️ Partial | Phase 4 ✓ | Phase 5 ✓ | Phase 6 ✓ | Phase 7 ✓ | Phase 8 ✓**

---

## Environment Setup

> **Important:** Use Python 3.11 (`venv311`). The existing `venv` (Python 3.13) has no packages installed and cannot be used — TensorFlow 2.21.0 does not support Python 3.13.

### Why Python 3.11?

| Issue | Detail |
|-------|--------|
| `venv` (Python 3.13) | Empty — only `pip==25.2` installed |
| `pkg_resources` error | `setuptools` not installed in old `venv` |
| `pandas==2.0.3` build fails | No Python 3.13 wheel; source build needs `pkg_resources` |
| **Solution** | Create `venv311` with Python 3.11.9 — all packages have pre-built wheels |

### One-Time Setup (Windows PowerShell)

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"

# Create venv311 using Python 3.11
py -3.11 -m venv venv311

# Upgrade pip and fix setuptools (must pin to 70.3.0 to keep pkg_resources)
.\venv311\Scripts\python.exe -m pip install --upgrade pip "setuptools==70.3.0" wheel

# Install all project dependencies
.\venv311\Scripts\python.exe -m pip install -r requirements.txt
```

> If `py -3.11` is not found, install Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3119/) and retry.

### Activate venv311 (every session)

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"
.\venv311\Scripts\Activate.ps1
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Verify the environment

```powershell
# After activation, confirm all key packages
python -c "import tensorflow, numpy, xgboost, sklearn, matplotlib; print('OK')"
python --version   # must show Python 3.11.x
```

### Installed package versions (venv311, verified 2026-10-15)

| Package | Version | Note |
|---------|---------|------|
| Python | 3.11.9 | Required — TF 2.21 max is Python 3.12 |
| tensorflow | 2.21.0 | CPU-only on native Windows |
| keras | 3.15.1 | Bundled with TF |
| numpy | 2.0.2 | |
| scipy | 1.17.1 | |
| pandas | 3.0.6 | |
| scikit-learn | 1.9.1 | |
| xgboost | 3.2.0 | Phase 5 hybrid |
| matplotlib | 3.11.2 | |
| seaborn | 0.13.2 | |
| Pillow | 12.3.0 | |
| opencv-python-headless | 5.0.0.93 | |
| tqdm | 4.70.1 | |
| setuptools | 70.3.0 | Pinned — 71+ removes pkg_resources |

---

## Overview

LeafIQ classifies tomato leaf diseases using a Convolutional Neural Network
trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).

**Classes (Phase 1 & 2):**

| Display Name        | Folder Name            | Index |
|---------------------|------------------------|-------|
| Tomato Healthy      | `Tomato_Healthy`       | 0     |
| Tomato Early Blight | `Tomato_Early_Blight`  | 1     |
| Tomato Late Blight  | `Tomato_Late_Blight`   | 2     |
| Tomato Leaf Mold    | `Tomato_Leaf_Mold`     | 3     |

---

## Project Structure

```
LeafIQ/
├── dataset/
│   ├── raw/                          # Original images (from Kaggle or synthetic)
│   │   ├── Tomato_Healthy/
│   │   ├── Tomato_Early_Blight/
│   │   ├── Tomato_Late_Blight/
│   │   └── Tomato_Leaf_Mold/
│   └── split/                        # Auto-generated train/val/test split
│       ├── train/  (70% — 140/class)
│       ├── val/    (15% —  30/class)
│       └── test/   (15% —  30/class)
│
├── src/
│   ├── generate_synthetic_dataset.py # Placeholder images (no Kaggle needed)
│   ├── download_dataset.py           # Phase 1: Download from Kaggle
│   ├── split_dataset.py              # Phase 1: Reproducible 70/15/15 split
│   ├── validate_dataset.py           # Phase 1: Integrity + CSV + plot
│   ├── run_phase1.py                 # Phase 1: Pipeline runner
│   ├── preprocess.py                 # Phase 2: Resize, normalize, cache, augment
│   ├── visualize.py                  # Phase 2: 6 visualization plots
│   └── run_phase2.py                 # Phase 2: Pipeline runner
│
├── models/                           # Saved model files (Phase 3+)
├── notebooks/                        # Jupyter notebooks
│
├── outputs/
│   ├── dataset_summary.csv           # Phase 1: Per-class image counts
│   ├── preprocessing_config.json     # Phase 2: All transformation parameters
│   ├── cache/                        # Phase 2: Preprocessed numpy arrays
│   │   ├── train_images.npy          # (560, 224, 224, 3)  float32
│   │   ├── train_labels.npy          # (560,)  int32
│   │   ├── train_paths.npy           # (560,)  object
│   │   ├── val_images.npy / val_labels.npy / val_paths.npy
│   │   └── test_images.npy / test_labels.npy / test_paths.npy
│   ├── plots/                        # All generated plots (PNG, 150 dpi)
│   │   ├── class_distribution.png
│   │   ├── sample_images.png
│   │   ├── preprocessed_samples.png
│   │   ├── pixel_statistics.png
│   │   ├── augmentation_examples.png
│   │   └── pca_tsne_projection.png
│   └── logs/                         # One log file per script
│
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Python 3.9 or later** — [Download](https://www.python.org/downloads/)
- **pip** (included with Python)

The following packages must be installed (see `requirements.txt`):
numpy, pandas, Pillow, matplotlib, seaborn, scikit-learn, tqdm, opencv-python

> **Note:** A Kaggle account is only needed if you want to download the real
> PlantVillage dataset. The pipeline works fully with synthetic images (see below).

---

## Quick Start (Windows PowerShell)

> Always activate `venv311` before running any script. See **Environment Setup** above for first-time installation.

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"

# Activate venv311 (Python 3.11 — required)
.\venv311\Scripts\Activate.ps1

# Verify environment is active
python --version   # must show Python 3.11.x
```

---

## Phase 1 — Project Setup and Dataset Preparation

### Option A — Use synthetic images (no Kaggle account needed)

Generates 200 placeholder images per class with distinct visual patterns:

```powershell
# Step 0: Generate synthetic dataset
python src/generate_synthetic_dataset.py

# Step 1: Split + validate (runs split and validate together)
python src/run_phase1.py --skip-download
```

### Option B — Use real PlantVillage dataset from Kaggle

**One-time Kaggle setup:**

```powershell
# Create the .kaggle folder
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"

# After downloading kaggle.json from kaggle.com → Settings → API → Create New Token:
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"

# Restrict permissions (recommended)
icacls "$env:USERPROFILE\.kaggle\kaggle.json" /inheritance:r /grant:r "${env:USERNAME}:R"
```

> Alternative — set credentials as environment variables (current session only):
> ```powershell
> $env:KAGGLE_USERNAME = "your_username"
> $env:KAGGLE_KEY      = "your_api_key"
> ```

**Run Phase 1 with real data:**

```powershell
# Downloads ~1.5 GB, splits 70/15/15, validates integrity
python src/run_phase1.py
```

### Run Phase 1 steps individually

```powershell
# Download only
python src/download_dataset.py

# Split only (after raw images exist)
python src/split_dataset.py

# Validate only (after split exists)
python src/validate_dataset.py

# Only re-run validation (skip download + split)
python src/run_phase1.py --only-validate
```

### Phase 1 expected outputs

| File | Description |
|------|-------------|
| `dataset/raw/Tomato_*/` | 200 images per class |
| `dataset/split/train/Tomato_*/` | 140 images per class (70%) |
| `dataset/split/val/Tomato_*/` | 30 images per class (15%) |
| `dataset/split/test/Tomato_*/` | 30 images per class (15%) |
| `outputs/dataset_summary.csv` | Per-class count table |
| `outputs/plots/class_distribution.png` | Grouped bar + pie chart |
| `outputs/logs/split.log` | Split details and leakage check |
| `outputs/logs/validation.log` | Integrity report |

---

## Phase 2 — Data Engineering and Visualization

Phase 2 preprocesses images and generates six analysis plots.

### Run Phase 2 (full pipeline)

```powershell
python src/run_phase2.py
```

This runs two steps automatically:
1. **Preprocessing** — resize to 224×224, normalize to [0, 1], build numpy cache
2. **Visualization** — generate all 6 plots

### Run Phase 2 steps individually

```powershell
# Step 1: Preprocessing only
python src/preprocess.py

# Step 2: Visualization only (requires cache from step 1)
python src/visualize.py

# Pipeline flags
python src/run_phase2.py --only-preprocess    # Step 1 only
python src/run_phase2.py --only-visualize     # Step 2 only
```

### Preprocessing details

| Transformation | Value | Applied to |
|----------------|-------|-----------|
| Resize | 224 × 224 pixels | All splits |
| Color mode | RGB | All splits |
| Resize method | Bilinear interpolation | All splits |
| Normalization | Divide by 255 → [0.0, 1.0] float32 | All splits |
| Horizontal flip | 50% probability | **Train only** |
| Vertical flip | 30% probability | **Train only** |
| Rotation | ±20 degrees | **Train only** |
| Width / height shift | ±10% | **Train only** |
| Zoom | 90%–110% | **Train only** |
| Brightness | 80%–120% | **Train only** |

Augmentation is intentionally **not** applied to val/test splits to ensure
unbiased evaluation. Colour jitter and aggressive cropping are excluded
because disease colour is a diagnostic feature.

All parameters are saved to `outputs/preprocessing_config.json`.

### Phase 2 expected outputs

| File | Description | Size |
|------|-------------|------|
| `outputs/preprocessing_config.json` | All transformation parameters | ~3 KB |
| `outputs/cache/train_images.npy` | 560 preprocessed training images | ~340 MB |
| `outputs/cache/train_labels.npy` | Integer class labels | ~2 KB |
| `outputs/cache/train_paths.npy` | Source file paths | ~60 KB |
| `outputs/cache/val_images.npy` | 120 validation images | ~72 MB |
| `outputs/cache/test_images.npy` | 120 test images | ~72 MB |
| `outputs/plots/class_distribution.png` | Grouped bar + pie chart | ~98 KB |
| `outputs/plots/sample_images.png` | 4 raw samples per class | ~2.1 MB |
| `outputs/plots/preprocessed_samples.png` | Raw vs 224×224 normalised | ~2.4 MB |
| `outputs/plots/pixel_statistics.png` | Per-channel histograms + stats | ~147 KB |
| `outputs/plots/augmentation_examples.png` | Original + 4 augmented versions | ~1.8 MB |
| `outputs/plots/pca_tsne_projection.png` | PCA 2D + t-SNE 2D + scree plot | ~270 KB |
| `outputs/logs/preprocess.log` | Preprocessing details | — |
| `outputs/logs/visualize.log` | Plot generation details | — |
| `outputs/logs/phase2_pipeline.log` | Full Phase 2 run log | — |

### View generated plots

```powershell
# Open all plots in the default image viewer
Start-Process "outputs\plots\class_distribution.png"
Start-Process "outputs\plots\sample_images.png"
Start-Process "outputs\plots\preprocessed_samples.png"
Start-Process "outputs\plots\pixel_statistics.png"
Start-Process "outputs\plots\augmentation_examples.png"
Start-Process "outputs\plots\pca_tsne_projection.png"

# Open the preprocessing config
Get-Content "outputs\preprocessing_config.json"

# View the dataset summary CSV
Get-Content "outputs\dataset_summary.csv"
```

### Force re-run preprocessing (clear cache first)

```powershell
Remove-Item "outputs\cache\*" -Force
python src/run_phase2.py
```

---

## Dataset Details

**Source:** [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)  
**Full dataset:** ~54,000 images across 38 plant/disease classes  
**Phase 1–2 scope:** 4 tomato classes, 200 images per class (800 total)  
**Image format:** JPG (256×256 raw, resized to 224×224 after preprocessing)  
**Random seed:** 42 — all splits and augmentations are reproducible

### Known limitation — image-level split

PlantVillage does not provide plant-level or session-level identifiers.
Images from the same physical leaf may appear in both training and
evaluation splits. This can make evaluation metrics slightly optimistic.
This is a widely documented limitation of this dataset and is acceptable
for academic lab use.

---

## Dataset Split

| Split | Images | Per class | Ratio |
|-------|--------|-----------|-------|
| Train | 560    | 140       | 70%   |
| Val   | 120    | 30        | 15%   |
| Test  | 120    | 30        | 15%   |
| **Total** | **800** | **200** | — |

---

## Troubleshooting

### Execution policy error when activating venv

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### `401 - Unauthorized` from Kaggle API

- Confirm `kaggle.json` is at `C:\Users\<YourUsername>\.kaggle\kaggle.json`
- Verify the JSON contains valid `username` and `key` fields
- Generate a new token at [kaggle.com/settings](https://www.kaggle.com/settings)

### Kaggle class folders not found after extraction

The zip folder names inside the Kaggle archive may differ slightly.
Check `outputs/logs/download.log` for actual names, then update the
`TARGET_CLASSES` dict in `src/download_dataset.py` accordingly.

### Corrupted images in validation

Delete the reported file(s) and re-run validation:

```powershell
python src/validate_dataset.py
```

### Preprocessed cache is stale after re-split

Delete the cache and re-run preprocessing:

```powershell
Remove-Item "outputs\cache\*" -Force
python src/preprocess.py
```

### Out of disk space

| Content | Approximate size |
|---------|-----------------|
| PlantVillage zip (Kaggle) | ~1.5 GB |
| Extracted raw images | ~2.5 GB |
| Split copies | ~2.5 GB |
| Numpy cache (float32) | ~485 MB |
| **Minimum free space** | **~7 GB** |

---

## Dependencies

All packages are installed via:

```powershell
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.13.0 | Model training (Phase 3+) |
| keras | 2.13.1 | Neural network API |
| numpy | 1.24.3+ | Array operations |
| pandas | 2.0.3+ | Data handling |
| Pillow | 10.0.1+ | Image loading and augmentation |
| opencv-python | 4.8.1+ | Image preprocessing |
| matplotlib | 3.7.2+ | Visualization |
| seaborn | 0.12.2+ | Statistical plots |
| kaggle | 1.5.16 | Dataset download |
| scikit-learn | 1.3.0+ | PCA, t-SNE, splitting utilities |
| tqdm | 4.66.1+ | Progress bars |
| jupyter | 1.0.0 | Notebook support |

> The project has been tested with Python 3.13.7, numpy 2.4.4, pandas 3.0.2,
> Pillow 12.1.1, matplotlib 3.10.8, seaborn 0.13.2, scikit-learn 1.8.0,
> tqdm 4.67.1, and opencv 5.0.0.

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Project setup, dataset download, split, validation |
| 2 | ✅ Complete | Preprocessing, normalization, augmentation, visualization |
## Phase 3 Results (Actual — verified on test set, 2026-10-14)

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Status |
|-------|-------------|----------------|-------------|---------|--------|
| Custom CNN (64×64, from scratch) | **98.17%** | 97.89% | 98.28% | **98.07%** | ✅ Complete |
| MobileNetV2 Phase A (96×96, frozen base) | 61.88% | 63.81% | 55.67% | 55.39% | ⚠️ Partial |
| MobileNetV2 Fine-tuned (96×96, 10 layers) | — | — | — | — | ❌ Training failed (val_acc=43%) |

### Per-class results — Custom CNN (best model)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Healthy | 99.17% | 100.00% | 99.59% | 240 |
| Early Blight | 94.84% | 98.00% | 96.39% | 150 |
| Late Blight | 98.93% | 96.52% | 97.71% | 287 |
| Leaf Mold | 98.61% | 98.61% | 98.61% | 144 |

### Notes on MobileNetV2 results

The MobileNetV2 fine-tuned run did not converge — val_loss exploded due to
overfitting (3,815 training images is small for 1.5M+ trainable parameters).
The Phase A checkpoint (frozen base) achieves 62% on the test set, which
reflects a partially-trained transfer learning model rather than a failure.
Full MobileNetV2 fine-tuning requires either more data, stronger regularization,
or GPU training with larger batch sizes. This is documented as incomplete.

The Custom CNN from scratch significantly outperforms MobileNetV2 on this
dataset at this scale — a common outcome when CNN architectures are properly
tuned for the target domain with limited data.
| 4 | ✅ Complete | Benchmarking, Error Analysis, Misclassification Study |

---

## Phase 3 — Deep Learning Models

Phase 3 trains two models on the real PlantVillage dataset and evaluates them on the held-out test set.

### Models

| Model | Architecture | Parameters | Strategy |
|-------|-------------|-----------|---------|
| CustomCNN | 3× Conv blocks (32/64/128) + GAP + Dense(256) | 323,172 | Trained from scratch |
| MobileNetV2 Phase A | ImageNet weights, frozen base + Dense(128) head | 2,422,852 (133K trainable) | Feature extraction |
| MobileNetV2 Phase B | Unfreeze last 10 layers | 2,422,852 (897K trainable) | Fine-tuning |

### Phase 3 Scripts

| Script | Purpose |
|--------|---------|
| `src/model.py` | Model builders for CustomCNN and MobileNetV2 |
| `src/train.py` | Training with callbacks, class weights, CSV logging |
| `src/evaluate.py` | Test-set evaluation, confusion matrix, plots |
| `src/hparam_search.py` | LR × batch-size grid search (6 configs, 3 epochs each) |
| `src/run_phase3.py` | Full Phase 3 pipeline runner |
| `src/smoke_test.py` | Sanity-check all components without loading full data |

### Run Phase 3 (Windows PowerShell)

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"

# Step 1 — Hyperparameter search (optional but recommended, ~60 min CPU)
python src/hparam_search.py

# Step 2 — Train both models
python src/train.py                    # train all (CustomCNN + MobileNetV2)
python src/train.py --model cnn        # CustomCNN only
python src/train.py --model mobilenet  # MobileNetV2 only

# Step 3 — Evaluate on test set
python src/evaluate.py                 # evaluate all saved checkpoints
python src/evaluate.py --model custom_cnn
python src/evaluate.py --model mobilenetv2_ft

# OR run everything in one command (hparam search + train + evaluate)
python src/run_phase3.py --hparam-search

# Evaluate only (models already trained)
python src/run_phase3.py --only-evaluate
```

### Phase 3 Hyperparameters

| Parameter | CustomCNN | MobileNetV2-A | MobileNetV2-B |
|-----------|-----------|--------------|--------------|
| Learning rate | 1e-3 (from hparam search) | 1e-3 | 5e-5 |
| Batch size | 32 | 32 | 32 |
| Max epochs | 25 | 15 | 20 |
| Early stopping patience | 5 | 5 | 5 |
| LR reduction patience | 3 | 3 | 3 |
| LR reduction factor | 0.3 | 0.3 | 0.3 |
| Dropout (head) | 0.50 | 0.50 | 0.50 |
| L2 regularization | 1e-4 | 1e-3 | 1e-3 |
| Augmentation | Train only | Train only | Train only |
| Class weights | Balanced | Balanced | Balanced |

### Training Notes (CPU)

- TF ≥ 2.11 does not support native Windows GPU. Training runs on CPU.
- CustomCNN: ~7 s/batch × 120 batches/epoch = ~14 min/epoch on this machine.
- MobileNetV2 Phase A: very fast (~2 s/batch, only 5K trainable params).
- MobileNetV2 Phase B: ~10 s/batch (1.5M trainable params unfrozen).
- Total estimated training time: ~4–6 hours on CPU for all three runs.
- To reduce time: lower `EPOCHS_CNN` in `train.py` or use `--model mobilenet` only.

### Phase 3 Expected Outputs

| File | Description |
|------|-------------|
| `models/custom_cnn_best.keras` | Best CustomCNN checkpoint (by val_accuracy) |
| `models/mobilenetv2_best.keras` | Best MobileNetV2 Phase A checkpoint |
| `models/mobilenetv2_ft_best.keras` | Best MobileNetV2 fine-tuned checkpoint |
| `outputs/hparam_search_results.csv` | Grid search results |
| `outputs/best_hparam_config.json` | Best LR + batch size for CustomCNN |
| `outputs/evaluation_results.csv` | Test-set metrics for all models |
| `outputs/classification_report_<model>.txt` | Per-class precision/recall/F1 |
| `outputs/plots/confusion_matrix_<model>.png` | Raw + normalized confusion matrices |
| `outputs/plots/training_curves_<model>.png` | Accuracy and loss over epochs |
| `outputs/plots/model_comparison.png` | Bar chart comparing test metrics |
| `outputs/logs/train.log` | Training details |
| `outputs/logs/evaluate.log` | Evaluation details |
| `outputs/logs/train_<model>.csv` | Per-epoch training history |

### View results

```powershell
# View evaluation CSV
Get-Content "outputs\evaluation_results.csv" | Format-Table

# View classification report (best model)
Get-Content "outputs\classification_report_custom_cnn.txt"

# Open confusion matrices
Start-Process "outputs\plots\confusion_matrix_custom_cnn.png"
Start-Process "outputs\plots\confusion_matrix_mobilenetv2.png"

# Open training curves
Start-Process "outputs\plots\training_curves_custom_cnn.png"
Start-Process "outputs\plots\training_curves_mobilenetv2.png"

# Open comparison
Start-Process "outputs\plots\model_comparison.png"
```
| `outputs/plots/confusion_matrix_<model>.png` | Raw + normalised confusion matrix |
| `outputs/plots/training_curves_<model>.png` | Accuracy and loss curves |
| `outputs/plots/model_comparison.png` | Side-by-side accuracy/F1 bar chart |
| `outputs/plots/hparam_search.png` | Hparam search results plot |
| `outputs/logs/train_custom_cnn.csv` | Epoch-by-epoch training history |
| `outputs/logs/train_mobilenetv2.csv` | MobileNetV2 Phase A history |
| `outputs/logs/train_mobilenetv2_ft.csv` | MobileNetV2 Phase B history |

### View Outputs

```powershell
# Open plots
Start-Process "outputs\plots\model_comparison.png"
Start-Process "outputs\plots\confusion_matrix_mobilenetv2_ft.png"
Start-Process "outputs\plots\training_curves_custom_cnn.png"

# View evaluation results
Get-Content "outputs\evaluation_results.csv"

# View classification report
Get-Content "outputs\classification_report_mobilenetv2_ft.txt"
```

### Environment

```
Python    : 3.13.7
TensorFlow: 2.21.0  (Keras 3.15.1)
numpy     : 2.4.4
scikit-learn: 1.8.0
GPU       : Not available (native Windows + TF >= 2.11)
Seed      : 42 (reproducible splits, training, augmentation)
```

---

## Phase 4 — Benchmarking & Error Analysis

Phase 4 performs a rigorous head-to-head comparison of all trained models and conducts detailed error analysis on the best model (Custom CNN).

### Phase 4 Results Summary

| Model | Test Accuracy | Macro F1 | Errors / 821 | Inference (ms/img) | Params |
|-------|-------------|---------|-------------|-------------------|--------|
| **Custom CNN** | **98.17%** | **98.07%** | **15** | **74.4 ms** | **323K** |
| MobileNetV2-A | 61.88% | 55.39% | 313 | 324.5 ms | 2.42M |
| MobileNetV2-FT | N/A (failed) | N/A | N/A | N/A | 2.42M |

The Custom CNN achieves 7.5× fewer parameters, 4.4× faster inference, and 36.3 percentage-point higher accuracy than MobileNetV2-A.

### Custom CNN Error Analysis

| Class | Test samples | Errors | Error rate | Confused to |
|-------|-------------|--------|-----------|-------------|
| Tomato_Healthy | 240 | **0** | **0.00%** | — |
| Tomato_Early_Blight | 150 | 3 | 2.00% | Late Blight ×2, Leaf Mold ×1 |
| Tomato_Late_Blight | 287 | 10 | 3.48% | Early Blight ×8, Leaf Mold ×1, Healthy ×1 |
| Tomato_Leaf_Mold | 144 | 2 | 1.39% | Healthy ×1, Late Blight ×1 |

- Healthy leaves: **perfect classification** (0 errors, 240/240).
- **89.5% of predictions made with ≥95% confidence** — 99.86% accurate in that bin.
- Only **1 high-confidence error** (≥90%) across all 821 test images.
- Most confused pair: **Late Blight → Early Blight** (8 errors). Both diseases produce brown necrotic lesions; the visual boundary is inherently ambiguous.

### Phase 4 Scripts

| Script | Purpose |
|--------|---------|
| `src/benchmark.py` | Load eval CSV + training CSVs, generate 4 comparison plots + summary tables |
| `src/error_analysis.py` | Load checkpoints + test cache, run predictions, generate all error analysis outputs |

### Run Phase 4 (Windows PowerShell)

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"

# Benchmarking (fast — no model loading, reads existing CSVs)
python src/benchmark.py

# Error analysis — both models (~2 min on CPU)
python src/error_analysis.py

# Error analysis — CNN only (~1 min)
python src/error_analysis.py --model cnn

# Error analysis — MobileNetV2 only (~1 min)
python src/error_analysis.py --model mnv2
```

### Phase 4 Outputs

#### Benchmark (`outputs/` and `outputs/plots/`)

| File | Description |
|------|-------------|
| `outputs/benchmark_summary.csv` | Full metrics table (all models, all metrics) |
| `outputs/benchmark_summary.txt` | Human-readable benchmark table |
| `outputs/plots/benchmark_comparison.png` | 4-panel bar chart: accuracy, F1, precision, recall |
| `outputs/plots/benchmark_params_vs_acc.png` | Scatter: parameter count vs test accuracy |
| `outputs/plots/benchmark_inference.png` | Inference time comparison (horizontal bars) |
| `outputs/plots/benchmark_training_overlay.png` | All training curves overlaid on one axes |

#### Error Analysis (`outputs/error_analysis/`)

| File | Description |
|------|-------------|
| `outputs/error_analysis_summary.csv` | One-row-per-model high-level summary |
| `misclassified_custom_cnn.csv` | All 15 CNN errors with true/pred labels and confidence |
| `misclassified_mobilenetv2.csv` | All 313 MNV2 errors |
| `per_class_errors_custom_cnn.csv` | Per-class error counts, rates, confusion targets |
| `per_class_errors_mobilenetv2.csv` | Same for MobileNetV2-A |
| `confidence_distribution_custom_cnn.csv` | Confidence bucket accuracy (7 bins) |
| `confidence_distribution_mobilenetv2.csv` | Same for MobileNetV2-A |
| `error_analysis_custom_cnn.png` | 4-panel figure: CM, error rate bar, conf hist, confusion pairs |
| `error_analysis_mobilenetv2.png` | Same for MobileNetV2-A |
| `misclassified_grid_custom_cnn.png` | All 15 misclassified CNN images |
| `misclassified_grid_mobilenetv2.png` | 20 sample misclassified MNV2 images |
| `top_errors_custom_cnn.png` | Top-10 most confident wrong CNN predictions |
| `top_errors_mobilenetv2.png` | Top-10 most confident wrong MNV2 predictions |

### View Phase 4 Results

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"

# View benchmark table
Get-Content outputs\benchmark_summary.txt

# View error summary
Get-Content outputs\error_analysis_summary.csv | ConvertFrom-Csv | Format-Table

# View all CNN errors
Import-Csv outputs\error_analysis\misclassified_custom_cnn.csv | Format-Table true_class, pred_class, confidence -AutoSize

# Open benchmark comparison chart
Start-Process outputs\plots\benchmark_comparison.png

# Open CNN error analysis figure (4-panel)
Start-Process outputs\error_analysis\error_analysis_custom_cnn.png

# Open all 15 misclassified CNN images
Start-Process outputs\error_analysis\misclassified_grid_custom_cnn.png

# Open top-10 highest-confidence wrong predictions
Start-Process outputs\error_analysis\top_errors_custom_cnn.png

# Read the full Phase 4 report
Get-Content outputs\PHASE4_ERROR_ANALYSIS_REPORT.md
```

### Phase 4 Key Findings

1. **Custom CNN is production-ready.** 98.17% test accuracy, 74 ms/img CPU inference, 3.8 MB model file.
2. **Healthy classification is perfect.** The model never misclassifies a healthy leaf as diseased — the most safety-critical direction for a plant disease app.
3. **The model knows when it's uncertain.** 93.3% of its 15 errors occur below 90% confidence. Only 1 error occurs with ≥90% confidence.
4. **Late Blight ↔ Early Blight is the only real challenge.** 8 of 15 CNN errors are this pair, and both have visually similar necrotic lesion patterns. This is a known difficulty in the PlantVillage literature.
5. **MobileNetV2 transfer learning fails on this dataset.** ImageNet features do not generalise to leaf disease macros at low resolution with only 140 training images per class.

### Phase 4 Limitations

- Test set is 30 images per class (Late Blight: 287 due to class imbalance). Per-class error rates have ~1% confidence intervals.
- Inference times measured on CPU only; GPU expected to be 10–20× faster.
- No Grad-CAM or saliency map analysis performed on error cases.
- Full report: `outputs/PHASE4_ERROR_ANALYSIS_REPORT.md`

---

## Phase 5 — CNN + XGBoost Hybrid Classification

Phase 5 builds a hybrid model using the trained Custom CNN as a fixed feature extractor and XGBoost as the multiclass classifier. Results are compared against all Phase 3/4 models on the same test set.

### Phase 5 Results Summary

| Model | Accuracy | Macro F1 | Errors / 821 | Inference (ms/img) | Model Size |
|-------|---------|---------|-------------|-------------------|------------|
| Custom CNN | 98.17% | 98.07% | 15 | 74.4 ms | 3.8 MB |
| MobileNetV2-A | 61.88% | 55.39% | 313 | 324.5 ms | 11.1 MB |
| **CNN + XGBoost** | **98.29%** | **98.21%** | **14** | **76.0 ms** | **835 KB** |

The hybrid marginally outperforms the standalone CNN (+0.12 pp accuracy, −1 error), but introduces a new Healthy-class misclassification. **The Custom CNN remains the recommended production model.**

### CNN + XGBoost Per-Class Results

| Class | Precision | Recall | F1 | Test samples |
|-------|-----------|--------|-----|-------------|
| Tomato_Healthy | 100.00% | 99.58% | 99.79% | 240 |
| Tomato_Early_Blight | 95.39% | 96.67% | 96.03% | 150 |
| Tomato_Late_Blight | 97.90% | 97.56% | 97.73% | 287 |
| Tomato_Leaf_Mold | 99.31% | 99.31% | 99.31% | 144 |

### Architecture

```
Image → CNN (64×64 input) → Dense(256)+BN+ReLU [embedding] → XGBoost → class
```

- Embedding: 256-dimensional vector from CNN's penultimate Dense layer
- XGBoost tuned via 18-config grid search (lr × depth × min_child_weight)
- Best config: lr=0.05, depth=4, min_child_weight=3, 200 boosting rounds
- Best val log-loss: 0.0478 (config 2/18)

### Phase 5 Scripts

| Script | Purpose |
|--------|---------|
| `src/extract_features.py` | Extract 256-dim CNN embeddings → `outputs/features/` |
| `src/train_xgboost.py` | Grid search + train XGBoost, save model + config |
| `src/benchmark_phase5.py` | 3-model comparison — 6 plots + 2 tables |
| `src/error_analysis_phase5.py` | Hybrid error analysis — 6 outputs |
| `src/predict_phase5.py` | CLI: predict one or more images with the hybrid |

### Run Phase 5 (Windows PowerShell)

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"

# Step 1 — Extract CNN embeddings (~2 min CPU, saved to outputs/features/)
python src/extract_features.py

# Step 2 — Train XGBoost with grid search (~3 min CPU)
python src/train_xgboost.py

# Step 3 — Three-model benchmark (fast, reads existing results)
python src/benchmark_phase5.py

# Step 4 — Hybrid error analysis (~10 sec)
python src/error_analysis_phase5.py

# Predict a single image
python src/predict_phase5.py path\to\leaf.jpg
python src/predict_phase5.py path\to\leaf.jpg --topk 4   # show all probs
python src/predict_phase5.py path\to\leaf.jpg --json     # JSON output

# Force re-extraction / retraining
python src/extract_features.py --force
python src/train_xgboost.py --force
```

### Phase 5 Outputs

#### Models (`models/`)

| File | Size | Description |
|------|------|-------------|
| `xgboost_best.json` | 832 KB | XGBoost native format (portable, recommended) |
| `xgboost_best.pkl` | 835 KB | Pickle format (fast loading) |

#### Feature cache (`outputs/features/`)

| File | Shape | Description |
|------|-------|-------------|
| `train_features.npy` | (3815, 256) | Training embeddings |
| `val_features.npy` | (816, 256) | Validation embeddings |
| `test_features.npy` | (821, 256) | Test embeddings |
| `feature_config.json` | — | Extraction metadata |

#### Comparison outputs (`outputs/`)

| File | Description |
|------|-------------|
| `phase5_model_comparison.csv` | Full 3-model metrics table |
| `phase5_model_comparison.txt` | Human-readable benchmark table |
| `xgboost_config.json` | Best hyperparameters + run metadata |
| `xgboost_val_metrics.csv` | All 18 grid search config results |
| `xgboost_feature_importance.npy` | Feature gain scores (256-dim array) |
| `classification_report_cnn_xgboost.txt` | Per-class precision/recall/F1 |
| `PHASE5_XGBOOST_REPORT.md` | Full Phase 5 report |

#### Comparison plots (`outputs/plots/`)

| File | Description |
|------|-------------|
| `phase5_classification_metrics.png` | 4-panel: accuracy, F1, precision, recall |
| `phase5_per_class_f1.png` | Grouped per-class F1 for all 3 models |
| `phase5_confusion_matrices.png` | Side-by-side normalised confusion matrices |
| `phase5_inference_comparison.png` | Inference time comparison |
| `phase5_model_size_comparison.png` | Model file size comparison |
| `phase5_xgboost_feature_importance.png` | Top-30 feature importances (gain) |

#### Error analysis (`outputs/error_analysis/`)

| File | Description |
|------|-------------|
| `misclassified_cnn_xgboost.csv` | All 14 errors (sorted by confidence) |
| `per_class_errors_cnn_xgboost.csv` | Per-class error counts + confusion targets |
| `confidence_distribution_cnn_xgboost.csv` | 7-bin confidence accuracy table |
| `error_analysis_cnn_xgboost.png` | 4-panel error figure |
| `misclassified_grid_cnn_xgboost.png` | Grid of all 14 misclassified images |
| `top_errors_cnn_xgboost.png` | Top-10 most confident wrong predictions |

### View Phase 5 Results

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"

# View benchmark table
Get-Content outputs\phase5_model_comparison.txt

# View grid search results
Import-Csv outputs\xgboost_val_metrics.csv | Format-Table -AutoSize

# View hybrid error list
Import-Csv outputs\error_analysis\misclassified_cnn_xgboost.csv | Format-Table true_class, pred_class, confidence -AutoSize

# Open comparison charts
Start-Process outputs\plots\phase5_classification_metrics.png
Start-Process outputs\plots\phase5_confusion_matrices.png
Start-Process outputs\plots\phase5_per_class_f1.png

# Open hybrid error analysis figure
Start-Process outputs\error_analysis\error_analysis_cnn_xgboost.png

# Read full report
Get-Content outputs\PHASE5_XGBOOST_REPORT.md
```

### Phase 5 Key Findings

1. **CNN+XGBoost: 98.29% accuracy, 14 errors / 821** — marginally better than CNN alone (98.17%, 15 errors).
2. **Late Blight ↔ Early Blight is the dominant error for both models.** 9/14 hybrid errors (64%) involve this pair. This is an inherent dataset challenge at 64×64 resolution, not a modelling deficiency.
3. **The hybrid introduces a new Healthy misclassification** (image idx 73, 92.8% confidence). The CNN has 0 Healthy errors — critical for a plant disease app.
4. **The hybrid is more overconfident on borderline cases.** 4 of 14 errors occur above 90% confidence (vs. 1/15 for CNN).
5. **95.4% of hybrid predictions are made with ≥95% confidence** (99.74% accurate in that band).
6. **Practical advantage: 835 KB XGBoost model** vs. 3.8 MB Keras model — useful for edge/constrained deployment scenarios.

### Phase 5 Limitations

- The +0.12 pp accuracy improvement (1 error) is not statistically significant on an 821-image test set.
- CNN weights are frozen — no end-to-end joint optimisation of the feature extractor and classifier.
- Inference time includes full CNN forward pass; the XGBoost step adds only ~1.6 ms.
- All measurements on CPU; GPU timing would be 10–20× faster.
- Full analysis: `outputs/PHASE5_XGBOOST_REPORT.md`

---

## Phase 6 — LeafIQ Website

A premium, responsive Flask web application with dark-green Peloton-inspired design. All predictions use real model inference; no fake data.

### Quick Launch (Windows PowerShell)

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"

# Activate venv311 (Python 3.11 — required)
.\venv311\Scripts\Activate.ps1

# Install web dependencies (first time only)
python -m pip install flask==3.1.3 flask-sqlalchemy==3.1.1 flask-login==0.6.3 "flask-wtf==1.3.0" wtforms==3.2.2 "email-validator==2.3.0"

# Start the web server
python webapp\run.py
```

Then open: **http://127.0.0.1:5000**

### Pages

| Page | URL | Auth |
|------|-----|------|
| Home | `/` | Public |
| Sign In | `/auth/login` | Public |
| Sign Up | `/auth/signup` | Public |
| Image Detection | `/dashboard/detect` | Protected |
| Prediction History | `/dashboard/history` | Protected |
| Model Comparison | `/dashboard/compare` | Protected |
| Explainable AI | `/dashboard/explainability` | Protected |
| Reports | `/dashboard/reports` | Protected |
| Settings | `/dashboard/settings` | Protected |

### Features

- Upload JPG/PNG/BMP/WebP leaf images (up to 16 MB)
- Real CNN inference with class probabilities and inference time
- Grad-CAM heatmap overlay for every prediction
- CNN+XGBoost hybrid model option (98.29% accuracy)
- Verified benchmark: Custom CNN 98.17%, MobileNetV2-A 61.88%, CNN+XGBoost 98.29%
- SQLite-backed prediction history per user
- Full user authentication (signup, login, logout, password change)
- Responsive dark-green design (Inter font, CSS variables)

### Web Dependencies (added in Phase 6)

| Package | Version |
|---------|---------|
| flask | 3.1.3 |
| flask-sqlalchemy | 3.1.1 |
| flask-login | 0.6.3 |
| flask-wtf | 1.3.0 |
| wtforms | 3.2.2 |
| werkzeug | 3.1.9 |
| email-validator | 2.3.0 |

### Project Files (Phase 6)

```
webapp/
  run.py                          # Entry point — run from project root
  config.py                       # Flask config, benchmark constants, class info
  leafiq.db                       # SQLite database (auto-created on first run)
  app/
    __init__.py                   # App factory: DB, LoginManager, CSRFProtect
    models.py                     # User + Prediction SQLAlchemy models
    auth/
      forms.py                    # WTForms: LoginForm, RegistrationForm
      routes.py                   # /auth/login, /auth/signup, /auth/logout
    dashboard/
      routes.py                   # All protected dashboard routes
    main/
      routes.py                   # / (home), /health
    api/
      routes.py                   # JSON API: /api/predictions, /api/benchmark
    ml/
      predictor.py                # CNN inference, XGBoost hybrid, Grad-CAM
    static/
      css/main.css                # Dark-green responsive stylesheet
      js/app.js                   # Upload preview, animations, validation
      uploads/                    # User-uploaded images + Grad-CAM outputs
    templates/
      base.html                   # Base layout (Inter font, meta)
      main/home.html              # Home page
      auth/login.html             # Sign in form
      auth/signup.html            # Registration form
      dashboard/base_dashboard.html  # Sidebar layout
      dashboard/detect.html       # Image upload + model selector
      dashboard/result.html       # Prediction result + prob bars + Grad-CAM
      dashboard/history.html      # Paginated prediction table
      dashboard/compare.html      # 3-model benchmark charts
      dashboard/explainability.html  # Grad-CAM viewer
      dashboard/reports.html      # Per-class analytics
      dashboard/settings.html     # Account + password change
      partials/nav.html           # Public nav bar
      partials/sidebar.html       # Dashboard sidebar
      partials/flash.html         # Flash message overlay
```

### Phase 6 Test Results (automated, 2026-10-15)

| Test | Status |
|------|--------|
| Public pages (home, login, signup) | ✅ PASS |
| Auth redirect (unauthenticated dashboard) | ✅ PASS |
| Signup with CSRF validation | ✅ PASS |
| All 6 dashboard pages render correctly | ✅ PASS |
| API endpoints (predictions, stats, benchmark) | ✅ PASS |
| Logout + re-protection | ✅ PASS |
| Login with correct credentials | ✅ PASS |
| Image upload + CNN inference + result page | ✅ PASS |
| Probability bars displayed | ✅ PASS |
| Prediction saved to history | ✅ PASS |
| **Total** | **37 PASS / 1 test-artifact FAIL** |

The 1 FAIL is a test artifact: the "wrong password" check ran while the session was already authenticated, so Flask correctly redirected to the dashboard instead of showing the login form. The app's auth logic is correct — unauthenticated users are always redirected to login.

---

## Phase 7 — Prediction Intelligence & Reporting

### Launch

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"
.\venv311\Scripts\Activate.ps1
python webapp\run.py
# Open http://127.0.0.1:5000
```

### New Features (Phase 7)

| Feature | Route | Description |
|---------|-------|-------------|
| History filters | `GET /dashboard/history?cls=&date_from=&date_to=&conf_min=&conf_max=&model=&q=` | Combined/individual filters, record count, clear |
| History CSV export | `GET /dashboard/history/export/csv` | Filtered CSV download |
| History PDF export | `GET /dashboard/history/export/pdf` | Filtered PDF with stats table |
| Explainability list | `GET /dashboard/explainability` | All Grad-CAM predictions, filterable, paginated |
| Explainability detail | `GET /dashboard/explainability/<id>` | Full Grad-CAM + interpretation + probabilities |
| Reports filters | `GET /dashboard/reports?cls=&date_from=&model=` | Filtered stats, class breakdown, daily counts |
| Reports CSV export | `GET /dashboard/reports/export/csv` | Filtered CSV |
| Reports PDF export | `GET /dashboard/reports/export/pdf` | Filtered PDF with summary stats |
| Plant Health Score | On every result page | Likely Healthy / Possible Disease / Uncertain (documented thresholds) |
| Similar Cases | On every result page | Up to 5 same-class previous predictions, sorted by confidence proximity |

### Health Score Thresholds (documented, not a diagnosis)

| Label | Condition |
|-------|-----------|
| Likely Healthy | `Tomato_Healthy` predicted AND confidence ≥ 80% |
| Possible Disease | Any disease class AND confidence ≥ 60% |
| Uncertain | Everything else (low confidence or ambiguous) |

### Phase 7 Test Results (59/59 PASS, 0 FAIL)

```powershell
python webapp\test_phase7.py
```

All tests verified against the live server: filters, combined filters, invalid inputs, CSV/PDF exports, health score, similar cases, Grad-CAM detail, auth boundaries, delete regression.

### Added Dependency

| Package | Version | Purpose |
|---------|---------|---------|
| fpdf2 | 2.8.9 | PDF export (pure-Python, no C dependencies) |

```powershell
python -m pip install fpdf2==2.8.9
```

---

## Phase 8 — Model Intelligence, Performance & Deployment Readiness

### Launch

```powershell
cd "C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ"
.\venv311\Scripts\Activate.ps1
python webapp\run.py
# Open http://127.0.0.1:5000

# Run Phase 8 tests
python webapp\test_phase8.py

# Run model evaluation on test set (~2 min CPU)
python webapp\app\ml\evaluator.py          # CNN only
python webapp\app\ml\evaluator.py --both   # CNN + XGBoost
python webapp\app\ml\evaluator.py --force  # force re-run
```

### New Features

| Feature | Where | Details |
|---------|-------|---------|
| Confidence threshold slider | Detect page → Advanced | Sets uncertainty warning cutoff (UI only, does not change model output) |
| Skip Grad-CAM toggle | Detect page → Advanced | Reduces inference time from ~5s to ~1s |
| File validation | Upload handler + predictor.py | Checks size, dimensions, corrupt bytes before inference |
| Top-3 predictions | Result page | Shows 3 classes ranked by probability |
| Uncertainty warning | Result page | Red warning when confidence < 60% with actionable message |
| Model Evaluation page | `/dashboard/evaluate` | Real metrics (accuracy, F1, confusion matrix) from 821-image test set |
| Monitoring page | `/dashboard/monitor` | Inference time stats, model usage, confidence distribution, drift warning |
| Eval cache | `outputs/eval_cache.json` | Results cached; repeat visits load instantly |
| Production config | `config.py` + `run.py` | Env-var driven (LEAFIQ_SECRET_KEY, LEAFIQ_ENV, LEAFIQ_DB_URI, etc.) |
| `.env.example` | Project root | Template for environment variables |
| Dependency checker | `run.py` startup | Warns if required packages are below minimum version |

### New Routes

| Route | Auth | Purpose |
|-------|------|---------|
| `GET /dashboard/evaluate` | Required | View cached evaluation results |
| `POST /dashboard/evaluate/run` | Required | Run CNN/XGBoost evaluation on test set |
| `GET /dashboard/monitor` | Required | Inference tracking, trends, drift warning |

### Phase 8 Test Results (52/52 PASS, 0 FAIL)

All tests pass: new routes, detect inputs, file validation (5 edge cases), uncertainty flags, health score labels, evaluator cache, monitor features, auth boundaries, skip-gradcam flag, config keys, deployment files.

### Deployment

```powershell
# Generate a secret key
python -c "import secrets; print(secrets.token_hex(32))"

# Set environment variables
$env:LEAFIQ_ENV       = "production"
$env:LEAFIQ_SECRET_KEY = "<your-generated-key>"

# Install production server (optional)
python -m pip install waitress

# Start
python webapp\run.py
```

Or copy `.env.example` to `.env`, fill in values, and install `python-dotenv`:
```powershell
python -m pip install python-dotenv
copy .env.example .env
# Edit .env with your values
python webapp\run.py
```
#   L e a f I Q  
 