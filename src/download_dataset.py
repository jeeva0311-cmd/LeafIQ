"""
LeafIQ - Dataset Download Script
=================================
Downloads the PlantVillage dataset from Kaggle and organizes
the four tomato classes needed for Phase 1.

IMPORTANT: Never hardcode Kaggle credentials.
Set them via environment variables or the kaggle.json file.

Setup (one-time):
    Option A - kaggle.json (recommended):
        1. Go to https://www.kaggle.com/settings → API → Create New Token
        2. Save the downloaded kaggle.json to:
           Windows: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json
        3. Run this script — it will find the credentials automatically.

    Option B - environment variables:
        $env:KAGGLE_USERNAME = "your_username"
        $env:KAGGLE_KEY      = "your_api_key"
        Then run this script.

Usage:
    python src/download_dataset.py
"""

import os
import sys
import zipfile
import shutil
import logging
from pathlib import Path

# ── Project root (one level up from src/) ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "dataset" / "raw"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "download.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Kaggle dataset identifier ──────────────────────────────────────────────────
KAGGLE_DATASET = "emmarex/plantdisease"   # ~1.5 GB zip

# ── The four tomato classes we need (folder names inside PlantVillage) ─────────
TARGET_CLASSES = {
    "Tomato_healthy":              "Tomato_Healthy",
    "Tomato__Early_blight":        "Tomato_Early_Blight",
    "Tomato__Late_blight":         "Tomato_Late_Blight",
    "Tomato__Leaf_Mold":           "Tomato_Leaf_Mold",
}
# Keys   = folder names as they appear inside the Kaggle zip
# Values = clean names we rename them to inside dataset/raw/


def check_kaggle_credentials() -> bool:
    """Return True if Kaggle credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_json = kaggle_json.exists()
    has_env  = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if has_json:
        log.info("Kaggle credentials found: %s", kaggle_json)
        return True
    if has_env:
        log.info("Kaggle credentials found via environment variables.")
        return True

    log.error(
        "Kaggle credentials NOT found.\n"
        "  Option A: Place kaggle.json at  %s\n"
        "  Option B: Set environment variables:\n"
        "            $env:KAGGLE_USERNAME = 'your_username'\n"
        "            $env:KAGGLE_KEY      = 'your_api_key'",
        kaggle_json,
    )
    return False


def download_dataset(dest_dir: Path) -> Path:
    """Download the Kaggle dataset zip into dest_dir and return the zip path."""
    try:
        import kaggle  # noqa: F401  (triggers credential check)
    except ImportError:
        log.error("kaggle package not installed. Run:  pip install kaggle==1.5.16")
        sys.exit(1)
    except Exception as exc:
        log.error("Kaggle import error: %s", exc)
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset '%s' to '%s' …", KAGGLE_DATASET, dest_dir)

    import kaggle as kg
    kg.api.authenticate()
    kg.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(dest_dir),
        unzip=False,
        quiet=False,
    )

    # Find the downloaded zip
    zips = list(dest_dir.glob("*.zip"))
    if not zips:
        log.error("Download finished but no .zip file found in %s", dest_dir)
        sys.exit(1)

    log.info("Download complete: %s", zips[0])
    return zips[0]


def extract_and_organize(zip_path: Path, raw_dir: Path) -> None:
    """
    Extract only the four target tomato class folders from the zip,
    rename them to our clean names, and place them in dataset/raw/.
    """
    log.info("Extracting zip: %s", zip_path)

    extract_tmp = raw_dir / "_tmp_extract"
    extract_tmp.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_members = zf.namelist()
        log.info("Zip contains %d entries total.", len(all_members))

        # Identify members belonging to our target classes
        to_extract = []
        for member in all_members:
            for kaggle_name in TARGET_CLASSES:
                if kaggle_name in member:
                    to_extract.append(member)
                    break

        if not to_extract:
            log.warning(
                "None of the target class folders were found in the zip.\n"
                "Expected folder names: %s\n"
                "Actual top-level folders (first 30):\n  %s",
                list(TARGET_CLASSES.keys()),
                "\n  ".join(sorted({m.split("/")[0] for m in all_members})[:30]),
            )
            log.info("Extracting full zip so you can inspect structure manually.")
            zf.extractall(extract_tmp)
        else:
            log.info("Extracting %d files for target classes …", len(to_extract))
            for member in to_extract:
                zf.extract(member, extract_tmp)

    # Walk extracted tree and move class folders to raw_dir
    moved = 0
    for kaggle_name, clean_name in TARGET_CLASSES.items():
        dest = raw_dir / clean_name
        if dest.exists():
            log.info("Class folder already exists, skipping: %s", dest)
            moved += 1
            continue

        # Search recursively for the class folder
        matches = list(extract_tmp.rglob(kaggle_name))
        if not matches:
            log.warning("Class folder not found in extracted content: %s", kaggle_name)
            continue

        src_folder = matches[0]
        shutil.move(str(src_folder), str(dest))
        count = len(list(dest.glob("*.jpg")) + list(dest.glob("*.JPG"))
                    + list(dest.glob("*.jpeg")) + list(dest.glob("*.png")))
        log.info("  %-35s → %s  (%d images)", kaggle_name, clean_name, count)
        moved += 1

    # Clean up temp folder and zip
    shutil.rmtree(extract_tmp, ignore_errors=True)
    log.info("Removed temporary extraction folder.")

    if moved == len(TARGET_CLASSES):
        log.info("All %d class folders successfully organized in %s", moved, raw_dir)
    else:
        log.warning(
            "%d/%d class folders found. Check the log above for details.",
            moved, len(TARGET_CLASSES),
        )


def verify_raw_dataset(raw_dir: Path) -> bool:
    """Quick sanity check: all four class folders exist and are non-empty."""
    log.info("\n── Raw Dataset Verification ──────────────────────────────────")
    all_ok = True
    for clean_name in TARGET_CLASSES.values():
        folder = raw_dir / clean_name
        if not folder.exists():
            log.error("MISSING folder: %s", folder)
            all_ok = False
            continue
        images = [
            f for f in folder.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        status = "OK" if images else "EMPTY"
        log.info("  [%s] %-30s  %d images", status, clean_name, len(images))
        if not images:
            all_ok = False

    return all_ok


def main() -> None:
    log.info("=" * 60)
    log.info("LeafIQ — Dataset Download")
    log.info("=" * 60)

    # 1. Credential check
    if not check_kaggle_credentials():
        sys.exit(1)

    # 2. Check if dataset already downloaded
    already_done = all(
        (RAW_DIR / clean_name).exists()
        for clean_name in TARGET_CLASSES.values()
    )
    if already_done:
        log.info("All class folders already present in %s", RAW_DIR)
        log.info("Delete dataset/raw/ and re-run to force re-download.")
        verify_raw_dataset(RAW_DIR)
        return

    # 3. Download
    zip_path = download_dataset(RAW_DIR)

    # 4. Extract and organize
    extract_and_organize(zip_path, RAW_DIR)

    # 5. Verify
    ok = verify_raw_dataset(RAW_DIR)
    if ok:
        log.info("\nDataset ready. Next step:  python src/split_dataset.py")
    else:
        log.error(
            "\nSome classes are missing. Possible causes:\n"
            "  1. The folder names inside the zip differ from expected.\n"
            "     Check outputs/logs/download.log for actual zip contents.\n"
            "  2. The download was interrupted — delete the zip and retry.\n"
            "  3. The dataset structure on Kaggle changed.\n"
            "     Expected class names: %s",
            list(TARGET_CLASSES.keys()),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
