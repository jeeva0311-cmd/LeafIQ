"""
LeafIQ - Training Script  (Phase 3)
=====================================
Trains three variants:
  Run 1 — CustomCNN       (64x64, from scratch, up to 25 epochs)
  Run 2 — MobileNetV2-A   (96x96, frozen base warmup, up to 15 epochs)
  Run 3 — MobileNetV2-FT  (96x96, fine-tune LAST 10 layers only, up to 20 epochs)

Overfitting fix for MobileNetV2
---------------------------------
3815 training images is small for 1.5M-param fine-tuning.
Solutions applied:
  - Unfreeze only the LAST 10 layers (not 30) -> ~120K trainable params
  - Add Dense(128)+BN+Dropout(0.5)+L2(1e-3) head in model.py
  - Use LR=5e-5 (very conservative)
  - Phase B starts from fresh ImageNet (not Phase A checkpoint)
  - Keep EarlyStopping patience=5

Usage
-----
    python src/train.py                    # all three runs
    python src/train.py --model cnn
    python src/train.py --model mobilenet
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "outputs" / "cache"
MODELS_DIR   = PROJECT_ROOT / "models"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

BATCH_SIZE    = 32
EPOCHS_CNN    = 25
EPOCHS_MNV2_A = 15
EPOCHS_MNV2_B = 20
SEED          = 42
CNN_LR        = 1e-3

_hparam_cfg = PROJECT_ROOT / "outputs" / "best_hparam_config.json"
if _hparam_cfg.exists():
    _cfg   = json.loads(_hparam_cfg.read_text())
    CNN_LR = float(_cfg.get("learning_rate", CNN_LR))


def load_split(split: str):
    X = np.load(CACHE_DIR / f"{split}_images.npy")
    y = np.load(CACHE_DIR / f"{split}_labels.npy")
    log.info("  Loaded %-6s  X=%s  dtype=%s", split, X.shape, X.dtype)
    return X, y


def compute_class_weights(y: np.ndarray) -> dict:
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    cw = {int(c): float(v) for c, v in zip(classes, w)}
    log.info("  Class weights: %s", {k: round(v, 4) for k, v in cw.items()})
    return cw


def make_dataset(X: np.ndarray, y: np.ndarray,
                 target_size: tuple,
                 augment: bool = False,
                 shuffle: bool = False):
    import tensorflow as tf
    th, tw = target_size[:2]

    def _train(image, label):
        image = tf.image.resize(image, [th, tw])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    def _val(image, label):
        image = tf.image.resize(image, [th, tw])
        return image, label

    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype("float32"), y.astype("int32"))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    ds = ds.map(_train if augment else _val,
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def get_callbacks(save_path: Path, csv_path: Path):
    import tensorflow as tf
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path), append=False),
    ]


def train_custom_cnn(X_train, y_train, X_val, y_val, class_weights) -> Path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from model import build_custom_cnn, CNN_INPUT_SIZE

    log.info("\n" + "=" * 60)
    log.info("Run 1: CustomCNN  input=%s  LR=%.0e  max_epochs=%d",
             CNN_INPUT_SIZE, CNN_LR, EPOCHS_CNN)
    log.info("=" * 60)

    model    = build_custom_cnn(learning_rate=CNN_LR)
    save_pth = MODELS_DIR / "custom_cnn_best.keras"
    csv_pth  = LOG_DIR    / "train_custom_cnn.csv"

    train_ds = make_dataset(X_train, y_train, CNN_INPUT_SIZE,
                            augment=True, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   CNN_INPUT_SIZE)

    log.info("Parameters: %s", f"{model.count_params():,}")
    t0 = time.perf_counter()
    h  = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS_CNN,
                   callbacks=get_callbacks(save_pth, csv_pth),
                   class_weight=class_weights, verbose=1)
    elapsed = time.perf_counter() - t0
    log.info("CustomCNN  best_val_acc=%.4f  %.0f min %.0f s",
             max(h.history["val_accuracy"]), elapsed // 60, elapsed % 60)
    return save_pth


def train_mobilenetv2(X_train, y_train, X_val, y_val, class_weights) -> Path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from model import build_mobilenetv2, MNV2_INPUT_SIZE

    train_ds = make_dataset(X_train, y_train, MNV2_INPUT_SIZE,
                            augment=True, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   MNV2_INPUT_SIZE)

    # ── Phase A: frozen base warmup ───────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Run 2: MobileNetV2 Phase A (frozen)  input=%s  "
             "LR=1e-3  max_epochs=%d", MNV2_INPUT_SIZE, EPOCHS_MNV2_A)
    log.info("=" * 60)

    model_a = build_mobilenetv2(fine_tune=False, learning_rate=1e-3)
    save_a  = MODELS_DIR / "mobilenetv2_best.keras"
    csv_a   = LOG_DIR    / "train_mobilenetv2.csv"

    log.info("Trainable params: %s",
             f"{sum(v.numpy().size for v in model_a.trainable_variables):,}")
    t0 = time.perf_counter()
    h_a = model_a.fit(train_ds, validation_data=val_ds,
                      epochs=EPOCHS_MNV2_A,
                      callbacks=get_callbacks(save_a, csv_a),
                      class_weight=class_weights, verbose=1)
    elapsed_a = time.perf_counter() - t0
    log.info("Phase A  best_val_acc=%.4f  %.0f min %.0f s",
             max(h_a.history["val_accuracy"]), elapsed_a // 60, elapsed_a % 60)

    # ── Phase B: fine-tune LAST 10 LAYERS only (fresh ImageNet, LR=5e-5) ─────
    # Unfreeze only last 10 MobileNetV2 layers => ~120K trainable params
    # (not 1.5M) — this prevents overfitting on the 3815-image dataset.
    # Start from fresh ImageNet weights for a clean optimisation landscape.
    log.info("\n" + "=" * 60)
    log.info("Run 3: MobileNetV2 Phase B (fine-tune last 10 layers, "
             "fresh ImageNet)  LR=5e-5  max_epochs=%d", EPOCHS_MNV2_B)
    log.info("=" * 60)

    model_ft = build_mobilenetv2(
        fine_tune=True,
        fine_tune_layers=10,
        fine_tune_lr=5e-5,
    )
    save_ft = MODELS_DIR / "mobilenetv2_ft_best.keras"
    csv_ft  = LOG_DIR    / "train_mobilenetv2_ft.csv"

    log.info("Trainable params: %s",
             f"{sum(v.numpy().size for v in model_ft.trainable_variables):,}")
    t0 = time.perf_counter()
    h_ft = model_ft.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS_MNV2_B,
                        callbacks=get_callbacks(save_ft, csv_ft),
                        class_weight=class_weights, verbose=1)
    elapsed_ft = time.perf_counter() - t0
    log.info("Phase B  best_val_acc=%.4f  %.0f min %.0f s",
             max(h_ft.history["val_accuracy"]), elapsed_ft // 60, elapsed_ft % 60)
    return save_ft


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "mobilenet", "all"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    log.info("=" * 60)
    log.info("LeafIQ -- Phase 3: Training  model=%s", args.model)
    log.info("=" * 60)

    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    class_weights    = compute_class_weights(y_train)

    if args.model in ("cnn", "all"):
        train_custom_cnn(X_train, y_train, X_val, y_val, class_weights)

    if args.model in ("mobilenet", "all"):
        train_mobilenetv2(X_train, y_train, X_val, y_val, class_weights)

    log.info("\nAll training done. Next: python src/evaluate.py")


if __name__ == "__main__":
    main()
