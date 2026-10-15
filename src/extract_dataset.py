"""
LeafIQ - Real Dataset Extraction Script
=========================================
Extracts the four target tomato classes from the downloaded archive.zip
into dataset/raw/, using only the colour images.

Mapping (zip path -> project folder name):
  plantvillage dataset/color/Tomato___healthy      -> Tomato_Healthy
  plantvillage dataset/color/Tomato___Early_blight -> Tomato_Early_Blight
  plantvillage dataset/color/Tomato___Late_blight  -> Tomato_Late_Blight
  plantvillage dataset/color/Tomato___Leaf_Mold    -> Tomato_Leaf_Mold

Usage:
    python src/extract_dataset.py
"""

import sys
import zipfile
import logging
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ZIP_PATH     = Path(r"C:\Users\JEEVA PRAKASINI G\Downloads\archive.zip")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "dataset" / "raw"
LOG_DIR      = PROJECT_ROOT / "outputs"  / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "extract.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Class mapping: zip prefix -> clean project folder name ────────────────────
CLASS_MAP = {
    "plantvillage dataset/color/Tomato___healthy":      "Tomato_Healthy",
    "plantvillage dataset/color/Tomato___Early_blight": "Tomato_Early_Blight",
    "plantvillage dataset/color/Tomato___Late_blight":  "Tomato_Late_Blight",
    "plantvillage dataset/color/Tomato___Leaf_Mold":    "Tomato_Leaf_Mold",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def wipe_raw_dirs() -> None:
    """Remove all existing files inside each class raw folder (file-by-file)."""
    for clean_name in CLASS_MAP.values():
        dest_dir = RAW_DIR / clean_name
        if not dest_dir.exists():
            continue
        files = [f for f in dest_dir.iterdir() if f.is_file()]
        removed = 0
        failed  = 0
        for f in files:
            try:
                f.unlink()
                removed += 1
            except Exception as exc:
                log.warning("Could not delete %s: %s", f.name, exc)
                failed += 1
        log.info("  Wiped %-30s  removed=%d  locked=%d", clean_name, removed, failed)


def extract_and_organize() -> None:
    if not ZIP_PATH.exists():
        log.error("Zip not found: %s", ZIP_PATH)
        sys.exit(1)

    log.info("Opening zip: %s  (%.1f GB)", ZIP_PATH,
             ZIP_PATH.stat().st_size / 1e9)

    # Step 1 — clear old files
    log.info("Step 1: Clearing existing raw class folders ...")
    wipe_raw_dirs()

    # Step 2 — extract from zip
    log.info("Step 2: Extracting real images from zip ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = zf.namelist()
        log.info("Zip has %d total entries.", len(members))

        for zip_prefix, clean_name in CLASS_MAP.items():
            dest_dir = RAW_DIR / clean_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            prefix  = zip_prefix + "/"
            to_copy = [
                m for m in members
                if m.startswith(prefix)
                and not m.endswith("/")
                and Path(m).suffix.lower() in IMAGE_EXTS
            ]

            if not to_copy:
                log.error("No images found for: %s", prefix)
                continue

            log.info("  Extracting %-30s  %d images ...", clean_name, len(to_copy))
            written = 0
            for member in to_copy:
                filename = Path(member).name
                dest_file = dest_dir / filename
                dest_file.write_bytes(zf.read(member))
                written += 1

            log.info("  Done:      %-30s  %d images saved", clean_name, written)

    # Step 3 — summary
    log.info("\n── Extraction Summary ──────────────────────────────────────")
    total = 0
    all_ok = True
    for clean_name in CLASS_MAP.values():
        d = RAW_DIR / clean_name
        if d.exists():
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            log.info("  %-30s  %d images", clean_name, n)
            total += n
            if n == 0:
                all_ok = False
        else:
            log.error("  %-30s  MISSING", clean_name)
            all_ok = False

    log.info("  Total: %d real images across 4 classes", total)

    if all_ok:
        log.info("\nExtraction complete.")
        log.info("Next: python src/run_phase1.py --skip-download")
    else:
        log.error("\nSome classes are empty. Check outputs/logs/extract.log")
        sys.exit(1)


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("LeafIQ -- Real Dataset Extraction")
    log.info("=" * 60)
    extract_and_organize()
