"""
LeafIQ - Phase 2 Pipeline Runner
==================================
Runs all Phase 2 steps in order:
  Step 1 — Preprocessing  (resize 224×224, normalize, build numpy cache)
  Step 2 — Visualization  (6 plots saved to outputs/plots/)

Usage
-----
    python src/run_phase2.py                    # run both steps
    python src/run_phase2.py --only-preprocess  # Step 1 only
    python src/run_phase2.py --only-visualize   # Step 2 only (needs cache)
"""

import sys
import logging
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase2_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

STEPS = [
    ("Step 1: Preprocessing",  SRC_DIR / "preprocess.py"),
    ("Step 2: Visualization",  SRC_DIR / "visualize.py"),
]


def run_step(label: str, script: Path) -> None:
    log.info("\n" + "=" * 60)
    log.info("PHASE 2 — %s", label)
    log.info("=" * 60)
    result = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log.error("'%s' failed (exit %d). Fix errors and re-run.", label, result.returncode)
        sys.exit(result.returncode)
    log.info("'%s' completed OK.", label)


def parse_args():
    p = argparse.ArgumentParser(description="LeafIQ Phase 2 Pipeline")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-preprocess", action="store_true",
                   help="Run preprocessing step only")
    g.add_argument("--only-visualize",  action="store_true",
                   help="Run visualization step only (needs preprocess cache)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.only_preprocess:
        steps = STEPS[:1]
    elif args.only_visualize:
        steps = STEPS[1:]
    else:
        steps = STEPS

    log.info("=" * 60)
    log.info("LeafIQ — Phase 2 Pipeline  (%d step(s))", len(steps))
    log.info("=" * 60)

    for label, script in steps:
        if not script.exists():
            log.error("Script not found: %s", script)
            sys.exit(1)
        run_step(label, script)

    log.info("\n" + "=" * 60)
    log.info("Phase 2 COMPLETE")
    log.info("=" * 60)
    log.info("Outputs:")
    log.info("  Preprocessing config : outputs/preprocessing_config.json")
    log.info("  Numpy cache          : outputs/cache/{train,val,test}_*.npy")
    log.info("  Plots (6)            : outputs/plots/*.png")
    log.info("  Logs                 : outputs/logs/preprocess.log")
    log.info("                       : outputs/logs/visualize.log")
    log.info("                       : outputs/logs/phase2_pipeline.log")
    log.info("\nReady for Phase 3: Model Training.")


if __name__ == "__main__":
    main()
