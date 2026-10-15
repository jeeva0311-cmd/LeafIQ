"""
LeafIQ - Hyperparameter Search
================================
Grid search over learning rate and batch size using CustomCNN.
Runs 3 epochs per configuration on a 20% subsample of training data
(for speed on CPU), then reports the best configuration.

Grid:
  LR:         [1e-2, 1e-3, 5e-4]
  Batch size: [32, 64]
  Total runs: 6

Uses a fixed 20% stratified subsample (~763 images) so each 3-epoch
run is fast enough on CPU (~2–4 min per config, ~15 min total).

Outputs
-------
  outputs/hparam_search_results.csv
  outputs/best_hparam_config.json
  outputs/logs/hparam_search.log
  outputs/plots/hparam_search.png

Usage
-----
    python src/hparam_search.py
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import sys
import csv
import json
import time
import logging
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "outputs" / "cache"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
PLOTS_DIR    = PROJECT_ROOT / "outputs" / "plots"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "hparam_search.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SEED           = 42
HPARAM_EPOCHS  = 3
SUBSAMPLE_FRAC = 0.20     # use 20% of train data per run for speed

LR_GRID    = [1e-2, 1e-3, 5e-4]
BATCH_GRID = [32, 64]


def subsample(X, y, frac, seed=SEED):
    """Stratified subsample of fraction `frac` from (X, y)."""
    from sklearn.model_selection import train_test_split
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=frac, random_state=seed, stratify=y
    )
    return X_sub, y_sub


def main():
    import tensorflow as tf
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from model import build_custom_cnn
    from train import make_dataset, compute_class_weights

    log.info("=" * 60)
    log.info("LeafIQ — Hyperparameter Search")
    log.info("Grid: LR=%s  Batch=%s  Epochs=%d  Subsample=%.0f%%",
             LR_GRID, BATCH_GRID, HPARAM_EPOCHS, SUBSAMPLE_FRAC * 100)
    log.info("=" * 60)

    # Load data
    log.info("Loading train/val cache ...")
    X_train_full = np.load(CACHE_DIR / "train_images.npy")
    y_train_full = np.load(CACHE_DIR / "train_labels.npy")
    X_val        = np.load(CACHE_DIR / "val_images.npy")
    y_val        = np.load(CACHE_DIR / "val_labels.npy")
    log.info("  Full train: %s  Val: %s", X_train_full.shape, X_val.shape)

    # Subsample training data for speed
    X_train, y_train = subsample(X_train_full, y_train_full, SUBSAMPLE_FRAC)
    log.info("  Subsampled train: %s (%.0f%%)",
             X_train.shape, SUBSAMPLE_FRAC * 100)

    class_weights = compute_class_weights(y_train_full)  # use full labels for weights

    results = []
    total_configs = len(LR_GRID) * len(BATCH_GRID)
    config_num = 0

    for lr in LR_GRID:
        for bs in BATCH_GRID:
            config_num += 1
            log.info("\n── Config %d/%d: lr=%.0e  batch=%d ──────────────────────",
                     config_num, total_configs, lr, bs)

            # Build fresh model each run
            model = build_custom_cnn(learning_rate=lr)

            train_ds = (
                tf.data.Dataset
                  .from_tensor_slices(
                      (X_train.astype("float32"), y_train.astype("int32")))
                  .shuffle(len(X_train), seed=SEED)
                  .map(lambda x, y: (
                      tf.image.random_flip_left_right(x),
                      y), num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(bs)
                  .prefetch(tf.data.AUTOTUNE)
            )
            val_ds = (
                tf.data.Dataset
                  .from_tensor_slices(
                      (X_val.astype("float32"), y_val.astype("int32")))
                  .batch(bs)
                  .prefetch(tf.data.AUTOTUNE)
            )

            t0 = time.perf_counter()
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=HPARAM_EPOCHS,
                class_weight=class_weights,
                verbose=1,
            )
            elapsed = time.perf_counter() - t0

            val_accs    = history.history["val_accuracy"]
            best_val    = max(val_accs)
            final_loss  = history.history["loss"][-1]

            log.info("  lr=%.0e  batch=%d  best_val_acc=%.4f  "
                     "final_loss=%.4f  time=%.0fs",
                     lr, bs, best_val, final_loss, elapsed)

            results.append({
                "lr":               lr,
                "batch_size":       bs,
                "best_val_accuracy": round(float(best_val), 4),
                "final_train_loss":  round(float(final_loss), 4),
                "epochs":            HPARAM_EPOCHS,
                "time_s":            round(elapsed, 1),
            })

            # Free model memory
            del model
            tf.keras.backend.clear_session()

    # ── Sort and report ───────────────────────────────────────────────────────
    results.sort(key=lambda r: r["best_val_accuracy"], reverse=True)

    log.info("\n── Hyperparameter Search Results ─────────────────────────────")
    log.info("  %-8s  %-10s  %-18s  %-10s",
             "LR", "Batch", "Best Val Acc", "Time (s)")
    log.info("  " + "-" * 52)
    for r in results:
        log.info("  %-8.0e  %-10d  %-18.4f  %-10.1f",
                 r["lr"], r["batch_size"],
                 r["best_val_accuracy"], r["time_s"])

    best = results[0]
    log.info("\nBest config: lr=%.0e  batch=%d  val_acc=%.4f",
             best["lr"], best["batch_size"], best["best_val_accuracy"])

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUTS_DIR / "hparam_search_results.csv"
    fields = ["lr", "batch_size", "best_val_accuracy",
              "final_train_loss", "epochs", "time_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    log.info("Saved: %s", csv_path)

    # ── Save best config JSON ─────────────────────────────────────────────────
    best_cfg = {
        "learning_rate": best["lr"],
        "batch_size":    best["batch_size"],
        "source":        "hparam_search.py",
        "val_accuracy_at_3_epochs": best["best_val_accuracy"],
    }
    cfg_path = OUTPUTS_DIR / "best_hparam_config.json"
    cfg_path.write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
    log.info("Best config JSON: %s", cfg_path)

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        lrs     = sorted(set(r["lr"]         for r in results))
        batches = sorted(set(r["batch_size"] for r in results))

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(lrs))
        bw = 0.3
        colors = ["#3498db", "#e67e22"]

        for bi, bs in enumerate(batches):
            vals = [
                next(r["best_val_accuracy"]
                     for r in results
                     if r["lr"] == lr and r["batch_size"] == bs)
                for lr in lrs
            ]
            bars = ax.bar(x + bi * bw, [v * 100 for v in vals],
                          bw, label=f"batch={bs}", color=colors[bi], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + bw / 2)
        ax.set_xticklabels([f"lr={lr:.0e}" for lr in lrs])
        ax.set_ylabel("Val Accuracy (%) @ 3 epochs")
        ax.set_title("LeafIQ — Hyperparameter Search (CustomCNN, 20% subsample)")
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        sns.despine(ax=ax)
        plt.tight_layout()

        plot_path = PLOTS_DIR / "hparam_search.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Plot saved: %s", plot_path)
    except Exception as e:
        log.warning("Could not generate hparam plot: %s", e)

    log.info("\nHparam search complete.")
    log.info("Next: python src/train.py  (will use best config automatically)")
    return best_cfg


if __name__ == "__main__":
    main()
