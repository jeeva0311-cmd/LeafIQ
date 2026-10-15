"""
LeafIQ - Image Preprocessing Module
=====================================
Phase 2: Data Engineering

Responsibilities
----------------
1. Resize all images to 224×224 (MobileNetV2 / VGG16 compatible)
2. Apply normalization  — pixels scaled to [0, 1] (divide by 255)
3. Define and document the augmentation pipeline (training images only)
4. Cache preprocessed numpy arrays to outputs/cache/ so downstream
   scripts (visualize, train) can load instantly without re-processing
5. Save a preprocessing config JSON documenting every transformation

Augmentation strategy (training only)
--------------------------------------
Augmentations are applied *only* to training images. Validation and test
images receive resize + normalize ONLY — no augmentation — to preserve
unbiased evaluation.

Augmentations chosen for tomato leaf disease:
  - Horizontal flip     : leaves can appear mirrored; no disease side-bias
  - Vertical flip       : images were taken from varying orientations
  - Rotation ±20°       : camera angle variation
  - Width/height shift 10%: slight positional variation
  - Zoom 10%            : distance variation during capture
  - Brightness ±20%     : lighting variation in field/lab photos
  - NOT: colour jitter  (disease colour is a diagnostic feature — preserve it)
  - NOT: aggressive crop (may cut out disease lesions)

Usage
-----
    python src/preprocess.py                # process all splits
    python src/preprocess.py --split train  # process one split only
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR    = PROJECT_ROOT / "dataset" / "split"
CACHE_DIR    = PROJECT_ROOT / "outputs"  / "cache"
LOG_DIR      = PROJECT_ROOT / "outputs"  / "logs"
CONFIG_PATH  = PROJECT_ROOT / "outputs"  / "preprocessing_config.json"

# ── Hyperparameters ────────────────────────────────────────────────────────────
TARGET_SIZE   = (224, 224)   # H × W
RESIZE_METHOD = Image.BILINEAR

CLASSES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ── Logging ────────────────────────────────────────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocess.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core image operations
# ─────────────────────────────────────────────────────────────────────────────

def load_and_resize(path: Path) -> np.ndarray:
    """Open image, convert to RGB, resize to TARGET_SIZE, return uint8 array."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((TARGET_SIZE[1], TARGET_SIZE[0]), RESIZE_METHOD)
    return np.array(img, dtype=np.uint8)


def normalize(arr: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] uint8 to [0.0, 1.0] float32."""
    return arr.astype(np.float32) / 255.0


def preprocess_image(path: Path) -> np.ndarray:
    """Full preprocessing pipeline: load → resize → normalize. Returns float32."""
    return normalize(load_and_resize(path))


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers  (pure NumPy / PIL — no Keras dependency yet)
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(arr_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply stochastic augmentations to a single uint8 image array.
    Returns augmented uint8 array (same shape).

    NOTE: During model training these will be reproduced using
    tf.keras.preprocessing.image.ImageDataGenerator with identical params.
    This implementation is used here purely for *visualisation* of what
    augmented images look like.
    """
    img = Image.fromarray(arr_uint8, "RGB")

    # 1. Horizontal flip (50% prob)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Vertical flip (30% prob)
    if rng.random() < 0.3:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 3. Rotation ±20°
    angle = float(rng.uniform(-20, 20))
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # 4. Width/height shift ≤10%
    w, h   = img.size
    dx     = int(rng.uniform(-0.1, 0.1) * w)
    dy     = int(rng.uniform(-0.1, 0.1) * h)
    img    = img.transform(img.size, Image.AFFINE,
                           (1, 0, dx, 0, 1, dy),
                           resample=Image.BILINEAR)

    # 5. Zoom ±10%  (crop then resize back)
    zoom   = float(rng.uniform(0.9, 1.1))
    new_w  = max(1, int(w * zoom))
    new_h  = max(1, int(h * zoom))
    left   = max(0, (new_w - w) // 2)
    top    = max(0, (new_h - h) // 2)
    img    = img.resize((new_w, new_h), Image.BILINEAR)
    img    = img.crop((left, top, left + w, top + h))
    # pad if crop goes out of bounds
    result = Image.new("RGB", (w, h), (0, 0, 0))
    result.paste(img, (0, 0))
    img    = result

    # 6. Brightness ±20%
    factor  = float(rng.uniform(0.8, 1.2))
    arr_out = np.clip(np.array(img, dtype=np.float32) * factor, 0, 255).astype(np.uint8)

    return arr_out


# ─────────────────────────────────────────────────────────────────────────────
# Cache builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cache(split: str) -> dict:
    """
    Preprocess all images in one split and save as .npy files.

    Saves:
      outputs/cache/<split>_images.npy   shape (N, 224, 224, 3)  float32
      outputs/cache/<split>_labels.npy   shape (N,)              int32
      outputs/cache/<split>_paths.npy    shape (N,)              str (object array)

    Returns a summary dict.
    """
    cache_images = CACHE_DIR / f"{split}_images.npy"
    cache_labels = CACHE_DIR / f"{split}_labels.npy"
    cache_paths  = CACHE_DIR / f"{split}_paths.npy"

    if cache_images.exists():
        log.info("  [SKIP] %s cache already exists (%s). Delete to regenerate.", split, cache_images)
        arr   = np.load(cache_images)
        lbls  = np.load(cache_labels)
        return {"split": split, "n": len(arr), "shape": arr.shape, "cached": True}

    images_list = []
    labels_list = []
    paths_list  = []

    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

    for cls in CLASSES:
        folder = SPLIT_DIR / split / cls
        if not folder.exists():
            log.warning("  Folder missing: %s — skipping", folder)
            continue

        files = sorted(
            f for f in folder.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        log.info("  Processing %s / %-28s  %d images ...", split, cls, len(files))

        for fpath in tqdm(files, desc=f"  {cls}", leave=False, unit="img"):
            try:
                arr = preprocess_image(fpath)        # float32 (224,224,3)
                images_list.append(arr)
                labels_list.append(class_to_idx[cls])
                paths_list.append(str(fpath))
            except Exception as exc:
                log.warning("  Skipping %s: %s", fpath.name, exc)

    if not images_list:
        log.error("No images processed for split '%s'", split)
        return {"split": split, "n": 0, "error": True}

    X = np.stack(images_list, axis=0)        # (N, 224, 224, 3)  float32
    y = np.array(labels_list, dtype=np.int32)
    p = np.array(paths_list,  dtype=object)

    np.save(cache_images, X)
    np.save(cache_labels, y)
    np.save(cache_paths,  p)

    log.info("  Saved %s cache: %s  shape=%s  dtype=%s", split, cache_images, X.shape, X.dtype)
    return {"split": split, "n": len(X), "shape": X.shape, "cached": False}


# ─────────────────────────────────────────────────────────────────────────────
# Config  (documents every transformation)
# ─────────────────────────────────────────────────────────────────────────────

def save_config(summaries: list) -> None:
    config = {
        "project": "LeafIQ",
        "phase": 2,
        "preprocessing": {
            "target_size": list(TARGET_SIZE),
            "channels": 3,
            "color_mode": "RGB",
            "resize_method": "BILINEAR",
            "normalization": "divide_by_255  →  [0.0, 1.0] float32",
        },
        "augmentation": {
            "applied_to": "train split ONLY",
            "horizontal_flip": {"prob": 0.5},
            "vertical_flip":   {"prob": 0.3},
            "rotation":        {"range_deg": [-20, 20]},
            "width_shift":     {"fraction": 0.10},
            "height_shift":    {"fraction": 0.10},
            "zoom":            {"range": [0.9, 1.1]},
            "brightness":      {"range": [0.8, 1.2]},
            "NOT_applied": [
                "colour_jitter  (disease colour is diagnostic)",
                "aggressive_crop  (may remove lesion regions)",
            ],
            "keras_equivalent": {
                "class": "tf.keras.preprocessing.image.ImageDataGenerator",
                "params": {
                    "rescale":             "1./255",
                    "horizontal_flip":     True,
                    "vertical_flip":       True,
                    "rotation_range":      20,
                    "width_shift_range":   0.1,
                    "height_shift_range":  0.1,
                    "zoom_range":          0.1,
                    "brightness_range":    [0.8, 1.2],
                    "fill_mode":           "nearest",
                },
            },
        },
        "class_to_index": {cls: i for i, cls in enumerate(CLASSES)},
        "index_to_class": {str(i): cls for i, cls in enumerate(CLASSES)},
        "cache_files": {
            s: {
                "images": f"outputs/cache/{s}_images.npy",
                "labels": f"outputs/cache/{s}_labels.npy",
                "paths":  f"outputs/cache/{s}_paths.npy",
            }
            for s in SPLITS
        },
        "split_summary": summaries,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    log.info("Preprocessing config saved: %s", CONFIG_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LeafIQ Phase 2 — Preprocessing")
    parser.add_argument("--split", choices=SPLITS,
                        help="Process a single split only (default: all)")
    return parser.parse_args()


def main():
    args  = parse_args()
    splits_to_run = [args.split] if args.split else SPLITS

    log.info("=" * 60)
    log.info("LeafIQ — Image Preprocessing")
    log.info("Target size: %dx%d  |  Normalization: /255 → [0,1]", *TARGET_SIZE)
    log.info("=" * 60)

    summaries = []
    for split in splits_to_run:
        log.info("\nProcessing split: %s", split.upper())
        result = build_cache(split)
        summaries.append(result)

    save_config(summaries)

    log.info("\n── Preprocessing Summary ─────────────────────────────────────")
    total = 0
    for r in summaries:
        n    = r.get("n", 0)
        shp  = r.get("shape", "n/a")
        flag = "[cached]" if r.get("cached") else "[new]"
        log.info("  %-6s  %5d images  shape=%s  %s", r["split"], n, shp, flag)
        total += n

    log.info("  Total: %d preprocessed images", total)
    log.info("\nPreprocessing complete.")
    log.info("Next step:  python src/visualize.py")


if __name__ == "__main__":
    main()
