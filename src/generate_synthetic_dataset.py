"""
LeafIQ - Synthetic Dataset Generator
======================================
Generates realistic synthetic tomato leaf images for each of the four
classes when the real Kaggle PlantVillage dataset is not available.

Each class has a distinct visual signature based on real disease appearance:
  Tomato_Healthy      — uniform green leaf texture
  Tomato_Early_Blight — dark concentric brown spots on green background
  Tomato_Late_Blight  — irregular dark/water-soaked patches, lighter margins
  Tomato_Leaf_Mold    — yellowish-green with diffuse grey-brown patches

Images are 256×256 RGB JPEGs saved to dataset/raw/<ClassName>/.

NOTE: These are placeholder images for pipeline testing only.
      Replace with real PlantVillage images by running:
          python src/download_dataset.py
      after placing your kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json
"""

import sys
import random
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "dataset" / "raw"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "generate_synthetic.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# How many images to generate per class (realistic for Phase 2 visualisations)
N_PER_CLASS = 200
IMG_SIZE    = 256
SEED        = 42


# ── Per-class image generators ────────────────────────────────────────────────

def _green_base(rng: random.Random, np_rng: np.random.Generator,
                size: int = IMG_SIZE) -> np.ndarray:
    """Green leaf base with subtle texture variation."""
    base_g = rng.randint(110, 160)
    base_r = rng.randint(30, 70)
    base_b = rng.randint(20, 55)
    noise  = np_rng.integers(-18, 18, (size, size, 3), dtype=np.int16)
    arr    = np.clip(
        np.array([[[base_r, base_g, base_b]]], dtype=np.int16) + noise,
        0, 255,
    ).astype(np.uint8)
    return arr


def make_healthy(rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = _green_base(rng, np_rng)
    # Add slight leaf vein lines
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    for angle in range(0, 180, 20):
        rad = np.radians(angle)
        x2  = int(cx + np.cos(rad) * IMG_SIZE * 0.45)
        y2  = int(cy + np.sin(rad) * IMG_SIZE * 0.45)
        draw.line([(cx, cy), (x2, y2)], fill=(30, 100, 25), width=1)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def make_early_blight(rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = _green_base(rng, np_rng)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    # Concentric ring spots — characteristic of early blight
    n_spots = rng.randint(3, 7)
    for _ in range(n_spots):
        cx = rng.randint(30, IMG_SIZE - 30)
        cy = rng.randint(30, IMG_SIZE - 30)
        for ring in range(4, 0, -1):
            r     = ring * rng.randint(6, 11)
            shade = rng.randint(50, 110)
            color = (shade, shade // 3, 5)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        # Yellow halo
        draw.ellipse([cx - 16, cy - 16, cx + 16, cy + 16],
                     outline=(200, 190, 20), width=2)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def make_late_blight(rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = _green_base(rng, np_rng)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    # Irregular water-soaked dark patches
    n_patches = rng.randint(2, 5)
    for _ in range(n_patches):
        cx = rng.randint(20, IMG_SIZE - 20)
        cy = rng.randint(20, IMG_SIZE - 20)
        # Irregular polygon
        pts = []
        for angle in range(0, 360, rng.randint(25, 45)):
            rad = np.radians(angle)
            r   = rng.randint(20, 55)
            pts.append((int(cx + np.cos(rad) * r),
                        int(cy + np.sin(rad) * r)))
        if len(pts) >= 3:
            dark = (rng.randint(20, 60), rng.randint(20, 50), rng.randint(10, 35))
            draw.polygon(pts, fill=dark)
            # Light grey margin (typical late blight border)
            draw.polygon(pts, outline=(180, 175, 160), width=2)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    return img


def make_leaf_mold(rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    # Yellowish green base (chlorosis)
    base_g = rng.randint(140, 180)
    base_r = rng.randint(100, 150)
    base_b = rng.randint(20, 50)
    noise  = np_rng.integers(-15, 15, (IMG_SIZE, IMG_SIZE, 3), dtype=np.int16)
    arr    = np.clip(
        np.array([[[base_r, base_g, base_b]]], dtype=np.int16) + noise,
        0, 255,
    ).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    # Diffuse grey-brown mold patches on underside simulation
    n_patches = rng.randint(4, 10)
    for _ in range(n_patches):
        cx = rng.randint(10, IMG_SIZE - 10)
        cy = rng.randint(10, IMG_SIZE - 10)
        rx = rng.randint(10, 35)
        ry = rng.randint(8, 25)
        shade_r = rng.randint(130, 170)
        shade_g = rng.randint(110, 145)
        shade_b = rng.randint(80, 115)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                     fill=(shade_r, shade_g, shade_b))
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


# ── Per-class dispatch ─────────────────────────────────────────────────────────

GENERATORS = {
    "Tomato_Healthy":      make_healthy,
    "Tomato_Early_Blight": make_early_blight,
    "Tomato_Late_Blight":  make_late_blight,
    "Tomato_Leaf_Mold":    make_leaf_mold,
}


def generate_class(class_name: str, n: int, base_seed: int) -> None:
    dest = RAW_DIR / class_name
    dest.mkdir(parents=True, exist_ok=True)

    existing = list(dest.glob("*.jpg"))
    if len(existing) >= n:
        log.info("  [SKIP] %s — already has %d images", class_name, len(existing))
        return

    rng    = random.Random(base_seed)
    np_rng = np.random.default_rng(base_seed)
    gen_fn = GENERATORS[class_name]

    for i in range(n):
        img = gen_fn(rng, np_rng)
        img.save(dest / f"{class_name}_{i:04d}.jpg", quality=90)

    log.info("  Generated %-30s  %d images → %s", class_name, n, dest)


def main() -> None:
    log.info("=" * 60)
    log.info("LeafIQ — Synthetic Dataset Generator")
    log.info("Images per class: %d  |  Size: %dx%d", N_PER_CLASS, IMG_SIZE, IMG_SIZE)
    log.info("=" * 60)

    for idx, class_name in enumerate(GENERATORS):
        generate_class(class_name, N_PER_CLASS, base_seed=SEED + idx * 1000)

    total = N_PER_CLASS * len(GENERATORS)
    log.info("\nDone. %d synthetic images saved to %s", total, RAW_DIR)
    log.info("NOTE: Replace with real Kaggle data when available.")
    log.info("Next: python src/run_phase1.py --skip-download")


if __name__ == "__main__":
    main()
