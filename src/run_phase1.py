"""
LeafIQ - Phase 1 Pipeline Runner
==================================
Runs all Phase 1 steps in order:
  Step 1 — Download dataset from Kaggle
  Step 2 — Split into train / val / test
  Step 3 — Validate dataset and generate outputs

Usage
-----
    python src/run_phase1.py              # run all steps
    python src/run_phase1.py --skip-download  # if dataset already downloaded
    python src/run_phase1.py --only-validate  # jump straight to validation
"""

import sys
import logging
import argparse
import subprocess
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def run_step(label: str, script: Path) -> None:
    """Run a Python script as a subprocess and exit if it fails."""
    log.info("\n" + "=" * 60)
    log.info("STEP: %s", label)
    log.info("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        log.error(
            "Step '%s' failed with exit code %d.",
            label, result.returncode,
        )
        log.error("Fix the error above and re-run:  python src/run_phase1.py")
        sys.exit(result.returncode)

    log.info("Step '%s' completed successfully.", label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LeafIQ Phase 1 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/run_phase1.py                 Run all three steps
  python src/run_phase1.py --skip-download Dataset already in dataset/raw/
  python src/run_phase1.py --only-validate Jump to validation only
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Step 1 (download). Use when dataset/raw/ already exists.",
    )
    group.add_argument(
        "--only-validate",
        action="store_true",
        help="Run Step 3 (validate) only. Use when split already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("LeafIQ — Phase 1 Pipeline")
    log.info("Project root: %s", PROJECT_ROOT)
    log.info("=" * 60)

    steps = [
        ("Step 1: Download Dataset",    SRC_DIR / "download_dataset.py"),
        ("Step 2: Split Dataset",       SRC_DIR / "split_dataset.py"),
        ("Step 3: Validate Dataset",    SRC_DIR / "validate_dataset.py"),
    ]

    if args.only_validate:
        steps = steps[2:]   # only validation
    elif args.skip_download:
        steps = steps[1:]   # split + validate

    for label, script in steps:
        if not script.exists():
            log.error("Script not found: %s", script)
            sys.exit(1)
        run_step(label, script)

    log.info("\n" + "=" * 60)
    log.info("Phase 1 COMPLETE")
    log.info("=" * 60)
    log.info("Outputs generated:")
    log.info("  outputs/dataset_summary.csv")
    log.info("  outputs/plots/class_distribution.png")
    log.info("  outputs/logs/download.log")
    log.info("  outputs/logs/split.log")
    log.info("  outputs/logs/validation.log")
    log.info("  outputs/logs/pipeline.log")
    log.info("\nReady for Phase 2: Model Training.")


if __name__ == "__main__":
    main()
