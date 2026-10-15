"""
LeafIQ - Dataset Validation Script
====================================
Performs a thorough quality check on the split dataset:

  1. Image integrity  — tries to open every file; flags corrupted ones
  2. Image count      — counts images per class per split
  3. Class imbalance  — reports imbalance ratio and flags severe skew (>3×)
  4. Missing files    — reports any empty class folders
  5. Summary CSV      — saves dataset_summary.csv to outputs/
  6. Distribution plot— saves class_distribution.png to outputs/plots/

Usage
-----
    python src/validate_dataset.py
"""

import sys
import csv
import logging
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR    = PROJECT_ROOT / "dataset" / "split"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
PLOTS_DIR    = OUTPUTS_DIR  / "plots"
LOG_DIR      = OUTPUTS_DIR  / "logs"

# ── Configuration ──────────────────────────────────────────────────────────────
SPLITS  = ["train", "val", "test"]
CLASSES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
IMBALANCE_WARN_RATIO = 3.0   # warn if max_class / min_class > this

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "validation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Image integrity check
# ─────────────────────────────────────────────────────────────────────────────

def check_image_integrity(folder: Path) -> tuple[int, list[Path]]:
    """
    Try to open every image in folder with Pillow.
    Returns (valid_count, corrupted_paths).
    """
    from PIL import Image, UnidentifiedImageError

    valid, corrupted = 0, []
    for f in sorted(folder.iterdir()):
        if f.suffix not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(f) as img:
                img.verify()   # checks headers without fully decoding
            valid += 1
        except (UnidentifiedImageError, OSError, SyntaxError):
            corrupted.append(f)

    return valid, corrupted


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build counts table
# ─────────────────────────────────────────────────────────────────────────────

def build_counts_table() -> dict:
    """
    Returns a nested dict: counts[split][class] = valid_image_count
    Also returns a flat list of all corrupted file paths.
    """
    counts:    dict[str, dict[str, int]]  = {s: {} for s in SPLITS}
    corrupted: list[Path]                 = []
    missing:   list[str]                  = []

    log.info("\n── Image Integrity Check ─────────────────────────────────────")

    for split in SPLITS:
        for cls in CLASSES:
            folder = SPLIT_DIR / split / cls
            if not folder.exists():
                log.error("  MISSING  %s/%s", split, cls)
                missing.append(f"{split}/{cls}")
                counts[split][cls] = 0
                continue

            valid, bad = check_image_integrity(folder)
            counts[split][cls] = valid
            corrupted.extend(bad)

            status = "OK" if not bad else f"CORRUPTED:{len(bad)}"
            log.info("  [%-14s]  %-8s / %-30s  valid=%d", status, split, cls, valid)

    if missing:
        log.error("\n%d folder(s) are missing. Run split_dataset.py first.", len(missing))

    if corrupted:
        log.warning("\n%d corrupted file(s) detected:", len(corrupted))
        for p in corrupted:
            log.warning("  %s", p)
    else:
        log.info("\nNo corrupted images found.")

    return counts, corrupted, missing


# ─────────────────────────────────────────────────────────────────────────────
# 3. Class imbalance report
# ─────────────────────────────────────────────────────────────────────────────

def check_imbalance(counts: dict) -> None:
    """Report per-split class imbalance."""
    log.info("\n── Class Imbalance Report ────────────────────────────────────")

    for split in SPLITS:
        split_counts = counts[split]
        if not split_counts:
            continue
        values = [v for v in split_counts.values() if v > 0]
        if not values:
            continue

        max_count = max(values)
        min_count = min(values)
        ratio = max_count / min_count if min_count > 0 else float("inf")

        log.info("\n  Split: %s", split.upper())
        for cls in CLASSES:
            n = split_counts.get(cls, 0)
            pct = n / max_count * 100 if max_count > 0 else 0
            bar = "█" * int(pct / 5)   # 1 block = 5%
            log.info("  %-30s  %5d  %s", cls, n, bar)

        if ratio > IMBALANCE_WARN_RATIO:
            log.warning(
                "  ⚠  Imbalance ratio %.1f× (max=%d, min=%d) exceeds %.0f× threshold.",
                ratio, max_count, min_count, IMBALANCE_WARN_RATIO,
            )
            log.warning(
                "     Consider using class weights or oversampling during training."
            )
        else:
            log.info("  Imbalance ratio: %.1f×  (within acceptable range)", ratio)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_csv(counts: dict, corrupted: list, missing: list) -> Path:
    """Write dataset_summary.csv to outputs/."""
    csv_path = OUTPUTS_DIR / "dataset_summary.csv"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls in CLASSES:
        row = {"class": cls}
        total = 0
        for split in SPLITS:
            n = counts[split].get(cls, 0)
            row[split] = n
            total += n
        row["total"] = total
        row["corrupted"] = sum(1 for p in corrupted if cls in str(p))
        row["missing_splits"] = sum(1 for m in missing if cls in m)
        rows.append(row)

    # Grand total row
    grand = {"class": "TOTAL"}
    for split in SPLITS:
        grand[split] = sum(r[split] for r in rows)
    grand["total"]          = sum(r["total"]           for r in rows)
    grand["corrupted"]      = sum(r["corrupted"]        for r in rows)
    grand["missing_splits"] = sum(r["missing_splits"]   for r in rows)
    rows.append(grand)

    fieldnames = ["class"] + SPLITS + ["total", "corrupted", "missing_splits"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("\nDataset summary CSV saved: %s", csv_path)
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Class distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def save_distribution_plot(counts: dict) -> Path:
    """
    Create a grouped bar chart of image counts per class per split.
    Saves to outputs/plots/class_distribution.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (safe on headless/Windows)
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        log.warning("Cannot create plot: %s  (install matplotlib)", e)
        return None

    # Short display names for readability
    short_names = {
        "Tomato_Healthy":       "Healthy",
        "Tomato_Early_Blight":  "Early Blight",
        "Tomato_Late_Blight":   "Late Blight",
        "Tomato_Leaf_Mold":     "Leaf Mold",
    }

    x          = np.arange(len(CLASSES))
    bar_width  = 0.25
    colors     = {"train": "#2ecc71", "val": "#3498db", "test": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: grouped bar per split ───────────────────────────────────────────
    ax = axes[0]
    for i, split in enumerate(SPLITS):
        values = [counts[split].get(cls, 0) for cls in CLASSES]
        bars = ax.bar(
            x + i * bar_width, values, bar_width,
            label=split.capitalize(), color=colors[split], alpha=0.85,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 10,
                    str(val), ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([short_names[c] for c in CLASSES], rotation=15, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Image Count per Class per Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── Right: stacked bar (total composition) ────────────────────────────────
    ax2 = axes[1]
    bottoms = np.zeros(len(CLASSES))
    for split in SPLITS:
        values = np.array([counts[split].get(cls, 0) for cls in CLASSES])
        ax2.bar(
            x, values, bar_width * 2,
            bottom=bottoms, label=split.capitalize(),
            color=colors[split], alpha=0.85,
        )
        bottoms += values

    ax2.set_xticks(x)
    ax2.set_xticklabels([short_names[c] for c in CLASSES], rotation=15, ha="right")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Number of Images")
    ax2.set_title("Total Images per Class (Stacked by Split)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("LeafIQ — PlantVillage Dataset Distribution", fontsize=14, y=1.01)
    plt.tight_layout()

    plot_path = PLOTS_DIR / "class_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Distribution plot saved:  %s", plot_path)
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("LeafIQ — Dataset Validation")
    log.info("=" * 60)

    # Check split directory exists
    if not SPLIT_DIR.exists():
        log.error("Split directory not found: %s", SPLIT_DIR)
        log.error("Run  python src/split_dataset.py  first.")
        sys.exit(1)

    # 1 & 2. Integrity check + count
    counts, corrupted, missing = build_counts_table()

    # 3. Imbalance
    check_imbalance(counts)

    # 4. CSV
    csv_path = save_summary_csv(counts, corrupted, missing)

    # 5. Plot
    plot_path = save_distribution_plot(counts)

    # ── Final verdict ─────────────────────────────────────────────────────────
    log.info("\n── Validation Result ─────────────────────────────────────────")
    issues = []
    if missing:
        issues.append(f"{len(missing)} missing folder(s)")
    if corrupted:
        issues.append(f"{len(corrupted)} corrupted image(s)")
    total_images = sum(
        counts[s].get(c, 0) for s in SPLITS for c in CLASSES
    )
    if total_images == 0:
        issues.append("no images found — dataset appears empty")

    if issues:
        log.error("Validation FAILED: %s", ", ".join(issues))
        log.error("Fix the issues above before proceeding to model training.")
        sys.exit(1)
    else:
        log.info("Validation PASSED.")
        log.info("Total images across all splits: %d", total_images)
        log.info("Outputs:")
        log.info("  CSV  : %s", csv_path)
        if plot_path:
            log.info("  Plot : %s", plot_path)
        log.info("\nPhase 1 complete. Ready for Phase 2: Model Training.")


if __name__ == "__main__":
    main()
