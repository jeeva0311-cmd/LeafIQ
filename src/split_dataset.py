"""
LeafIQ - Dataset Split Script
==============================
Splits raw images for each class into train / val / test sets
using a reproducible 70 : 15 : 15 ratio.

Strategy
--------
- Splitting is done at the IMAGE level (not patient/plant level), because
  PlantVillage does not provide plant-level IDs.
- Image-level splitting LIMITATION: images from the same physical leaf or
  the same imaging session may appear in different splits, which can cause
  overly optimistic evaluation metrics. This is a known limitation of the
  PlantVillage dataset and is documented here for transparency.
- A fixed random seed (SEED = 42) guarantees reproducibility across runs.
- Each run wipes and re-creates the split folders to ensure a clean,
  consistent split — no stale data from partial runs.

Output structure
----------------
dataset/split/
    train/
        Tomato_Healthy/          *.jpg / *.jpeg / *.png
        Tomato_Early_Blight/
        Tomato_Late_Blight/
        Tomato_Leaf_Mold/
    val/   <same four sub-folders>
    test/  <same four sub-folders>

Usage
-----
    python src/split_dataset.py
"""

import sys
import shutil
import logging
import random
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "dataset" / "raw"
SPLIT_DIR    = PROJECT_ROOT / "dataset" / "split"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"

# ── Configuration ──────────────────────────────────────────────────────────────
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO is the remainder: 0.15

CLASSES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "split.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def get_images(class_dir: Path) -> list:
    """Return sorted list of image paths in a directory."""
    return sorted(
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix in IMAGE_EXTENSIONS
    )


def split_list(items: list, train_r: float, val_r: float, seed: int):
    """
    Deterministically split a list into (train, val, test).
    test gets the remainder after train + val to avoid rounding loss.
    """
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    n_test  = n - n_train - n_val   # remainder

    return (shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:],
            n_train, n_val, n_test)


def copy_files(file_list: list, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in file_list:
        shutil.copy2(src, dest_dir / src.name)


def reset_split_dirs() -> None:
    """
    Remove all existing split folders and recreate them empty.
    Uses file-by-file deletion to handle Windows file locks gracefully.
    """
    log.info("Resetting split directories (clean slate)...")
    for split in ["train", "val", "test"]:
        split_path = SPLIT_DIR / split
        if split_path.exists():
            # Walk and delete every file first
            for f in split_path.rglob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass  # skip locked files
            # Remove empty directories bottom-up
            for d in sorted(split_path.rglob("*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except Exception:
                        pass
            try:
                split_path.rmdir()
            except Exception:
                pass
        split_path.mkdir(parents=True, exist_ok=True)
    log.info("Split directories reset.")


def split_class(class_name: str) -> dict:
    """Split one class. Returns a summary dict."""
    src_dir = RAW_DIR / class_name
    if not src_dir.exists():
        log.error("Raw class folder not found: %s", src_dir)
        log.error("Run  python src/download_dataset.py  first.")
        sys.exit(1)

    images = get_images(src_dir)
    if not images:
        log.error("No images found in %s", src_dir)
        sys.exit(1)

    train_imgs, val_imgs, test_imgs, n_train, n_val, n_test = split_list(
        images, TRAIN_RATIO, VAL_RATIO, SEED
    )

    copy_files(train_imgs, SPLIT_DIR / "train" / class_name)
    copy_files(val_imgs,   SPLIT_DIR / "val"   / class_name)
    copy_files(test_imgs,  SPLIT_DIR / "test"  / class_name)

    log.info(
        "  %-30s  total=%-5d  train=%-5d  val=%-5d  test=%-5d",
        class_name, len(images), n_train, n_val, n_test,
    )
    return {
        "class": class_name,
        "total": len(images),
        "train": n_train,
        "val":   n_val,
        "test":  n_test,
    }


def verify_no_leakage(summary: list) -> None:
    """Verify zero filename overlap across splits."""
    log.info("\n── Data Leakage Check ────────────────────────────────────────")
    leakage_found = False

    for row in summary:
        cls  = row["class"]
        sets = {
            split: set(
                f.name for f in (SPLIT_DIR / split / cls).glob("*")
                if f.is_file()
            )
            for split in ["train", "val", "test"]
        }
        tv = sets["train"] & sets["val"]
        tt = sets["train"] & sets["test"]
        vt = sets["val"]   & sets["test"]

        if tv or tt or vt:
            log.error(
                "  LEAKAGE in %s: train∩val=%d  train∩test=%d  val∩test=%d",
                cls, len(tv), len(tt), len(vt),
            )
            leakage_found = True
        else:
            log.info("  [OK] %-30s  no overlap between splits", cls)

    if leakage_found:
        log.error("Data leakage detected! Delete dataset/split/ and re-run.")
        sys.exit(1)
    else:
        log.info("No data leakage detected.")


def print_summary(summary: list) -> None:
    log.info("\n── Split Summary ─────────────────────────────────────────────")
    log.info("  %-30s  %6s  %6s  %6s  %6s", "Class", "Total", "Train", "Val", "Test")
    log.info("  " + "-" * 62)

    totals = {"total": 0, "train": 0, "val": 0, "test": 0}
    for row in summary:
        log.info("  %-30s  %6d  %6d  %6d  %6d",
                 row["class"], row["total"], row["train"], row["val"], row["test"])
        for k in totals:
            totals[k] += row[k]

    log.info("  " + "-" * 62)
    log.info("  %-30s  %6d  %6d  %6d  %6d",
             "TOTAL", totals["total"], totals["train"], totals["val"], totals["test"])

    t = totals["total"]
    log.info(
        "\n  Effective split:  train=%.1f%%  val=%.1f%%  test=%.1f%%",
        totals["train"] / t * 100,
        totals["val"]   / t * 100,
        totals["test"]  / t * 100,
    )
    log.info(
        "\n  LIMITATION: Image-level split — PlantVillage has no plant-level IDs,\n"
        "  so images from the same leaf may appear in different splits.\n"
        "  Acceptable for academic/lab use."
    )


def main() -> None:
    log.info("=" * 60)
    log.info("LeafIQ — Dataset Split  (seed=%d)", SEED)
    log.info("Ratio: train=%.0f%%  val=%.0f%%  test=%.0f%%",
             TRAIN_RATIO * 100, VAL_RATIO * 100, (1 - TRAIN_RATIO - VAL_RATIO) * 100)
    log.info("=" * 60)

    reset_split_dirs()   # always start clean

    summary = []
    for cls in CLASSES:
        summary.append(split_class(cls))

    verify_no_leakage(summary)
    print_summary(summary)

    log.info("\nDataset split complete.")
    log.info("Next step:  python src/validate_dataset.py")


if __name__ == "__main__":
    main()
