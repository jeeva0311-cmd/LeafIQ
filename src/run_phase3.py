"""
LeafIQ - Phase 3 Pipeline Runner
==================================
Orchestrates the full Phase 3 workflow:

  Step 0 (optional) — Hyperparameter search  --hparam-search
  Step 1             — Train CustomCNN
  Step 2             — Train MobileNetV2 (Phase A + B)
  Step 3             — Evaluate both models on test set

Usage
-----
    # Full pipeline (train all + evaluate)
    python src/run_phase3.py

    # Run hyperparameter search first, then train best config
    python src/run_phase3.py --hparam-search

    # Train only one model
    python src/run_phase3.py --model cnn
    python src/run_phase3.py --model mobilenet

    # Evaluate only (models already trained)
    python src/run_phase3.py --only-evaluate

    # Evaluate a specific model
    python src/run_phase3.py --only-evaluate --model custom_cnn
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase3_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search  (grid: LR × batch size, 3 epochs each)
# ─────────────────────────────────────────────────────────────────────────────

def run_hparam_search() -> dict:
    """
    Small grid search over LR and batch size using CustomCNN.
    Runs 3 epochs per config, picks best val_accuracy.
    Results saved to outputs/hparam_search_results.csv
    """
    import numpy as np
    import csv

    # Import TF here so env vars take effect
    import tensorflow as tf
    tf.random.set_seed(42)

    sys.path.insert(0, str(SRC_DIR))
    from model import build_custom_cnn
    from train import load_split, make_dataset, compute_class_weights

    log.info("=" * 60)
    log.info("Hyperparameter Search  (CustomCNN, 3 epochs per config)")
    log.info("=" * 60)

    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    class_weights    = compute_class_weights(y_train)

    LR_GRID    = [1e-2, 1e-3, 5e-4]
    BATCH_GRID = [32, 64]
    HPARAM_EPOCHS = 3

    results = []

    for lr in LR_GRID:
        for bs in BATCH_GRID:
            log.info("\n  Config: lr=%.0e  batch=%d", lr, bs)
            train_ds = make_dataset(X_train, y_train, augment=True,
                                    shuffle=True)
            val_ds   = make_dataset(X_val,   y_val,   augment=False)

            model = build_custom_cnn(learning_rate=lr)
            # Monkey-patch batch size via dataset (already fixed in make_dataset)
            # For hparam search use a local dataset with correct batch size
            train_ds = (
                tf.data.Dataset
                  .from_tensor_slices((X_train.astype("float32"),
                                       y_train.astype("int32")))
                  .shuffle(len(X_train), seed=42)
                  .map(lambda x, y: (tf.image.random_flip_left_right(x), y),
                       num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(bs)
                  .prefetch(tf.data.AUTOTUNE)
            )
            val_ds = (
                tf.data.Dataset
                  .from_tensor_slices((X_val.astype("float32"),
                                       y_val.astype("int32")))
                  .batch(bs)
                  .prefetch(tf.data.AUTOTUNE)
            )

            t0 = time.perf_counter()
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=HPARAM_EPOCHS,
                class_weight=class_weights,
                verbose=0,
            )
            elapsed = time.perf_counter() - t0

            best_val = max(history.history["val_accuracy"])
            log.info("  lr=%.0e  batch=%d  best_val_acc=%.4f  time=%.0fs",
                     lr, bs, best_val, elapsed)
            results.append({
                "lr": lr, "batch_size": bs,
                "best_val_accuracy": round(best_val, 4),
                "epochs": HPARAM_EPOCHS,
                "time_s": round(elapsed, 1),
            })

    # Sort by best val accuracy
    results.sort(key=lambda r: r["best_val_accuracy"], reverse=True)

    csv_path = OUTPUTS_DIR / "hparam_search_results.csv"
    fields   = ["lr", "batch_size", "best_val_accuracy", "epochs", "time_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    log.info("\nHparam search complete. Results saved: %s", csv_path)
    log.info("Top config: lr=%.0e  batch=%d  val_acc=%.4f",
             results[0]["lr"], results[0]["batch_size"],
             results[0]["best_val_accuracy"])

    # Persist best config so train.py can pick it up
    best_cfg = {"learning_rate": results[0]["lr"],
                "batch_size":    results[0]["batch_size"]}
    cfg_path = OUTPUTS_DIR / "best_hparam_config.json"
    cfg_path.write_text(json.dumps(best_cfg, indent=2))
    log.info("Best config saved: %s", cfg_path)

    return best_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ─────────────────────────────────────────────────────────────────────────────

def run_step(label: str, script: Path, extra_args: list = None) -> None:
    log.info("\n" + "=" * 60)
    log.info("PHASE 3 — %s", label)
    log.info("=" * 60)
    cmd = [sys.executable, str(script)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log.error("Step '%s' failed (exit %d). Fix errors and re-run.",
                  label, result.returncode)
        sys.exit(result.returncode)
    log.info("'%s' completed OK.", label)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LeafIQ Phase 3 Pipeline")
    p.add_argument("--hparam-search", action="store_true",
                   help="Run hyperparameter grid search before training")
    p.add_argument("--model", choices=["cnn", "mobilenet", "all"],
                   default="all",
                   help="Which model(s) to train (default: all)")
    p.add_argument("--only-evaluate", action="store_true",
                   help="Skip training, run evaluation only")
    p.add_argument("--eval-model",
                   choices=["custom_cnn", "mobilenetv2",
                             "mobilenetv2_ft", "all"],
                   default="all",
                   help="Which saved model to evaluate (default: all available)")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.perf_counter()

    log.info("=" * 60)
    log.info("LeafIQ — Phase 3 Pipeline")
    log.info("=" * 60)

    if not args.only_evaluate:
        # Optional hparam search
        if args.hparam_search:
            run_hparam_search()

        # Training
        train_args = ["--model", args.model]
        run_step("Training", SRC_DIR / "train.py", train_args)

    # Evaluation
    eval_args = ["--model", args.eval_model]
    run_step("Evaluation", SRC_DIR / "evaluate.py", eval_args)

    elapsed = time.perf_counter() - t_start
    log.info("\n" + "=" * 60)
    log.info("Phase 3 COMPLETE  (total time: %.0f min %.0f s)",
             elapsed // 60, elapsed % 60)
    log.info("=" * 60)
    log.info("Outputs:")
    log.info("  models/          — saved checkpoints (.keras)")
    log.info("  outputs/plots/   — confusion matrices, curves, comparison")
    log.info("  outputs/         — evaluation_results.csv, classification reports")
    log.info("  outputs/logs/    — train*.csv, evaluate.log, phase3_pipeline.log")
    log.info("\nReady for Phase 4: Benchmarking and Comparative Analysis.")


if __name__ == "__main__":
    main()
