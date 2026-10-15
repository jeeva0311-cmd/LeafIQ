"""
LeafIQ - Data Visualization Script
=====================================
Phase 2: Data Engineering — Visualization

Generates and saves 6 plots to outputs/plots/:

  Plot 1 — class_distribution.png         (already from Phase 1; regenerated here
                                            in higher quality with extra stats)
  Plot 2 — sample_images.png              Representative images per class (raw)
  Plot 3 — preprocessed_samples.png       Same images after resize + normalize
  Plot 4 — pixel_statistics.png           Per-channel mean/std/histogram
  Plot 5 — augmentation_examples.png      Grid of augmented training images
  Plot 6 — pca_tsne_projection.png        PCA + t-SNE of flattened image features

All plots are saved at 150 dpi as PNG.

Usage
-----
    python src/visualize.py
"""

import sys
import logging
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")     # non-interactive backend — safe on Windows without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR    = PROJECT_ROOT / "dataset" / "split"
CACHE_DIR    = PROJECT_ROOT / "outputs"  / "cache"
PLOTS_DIR    = PROJECT_ROOT / "outputs"  / "plots"
LOG_DIR      = PROJECT_ROOT / "outputs"  / "logs"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
CLASSES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]
DISPLAY_NAMES = {
    "Tomato_Healthy":      "Healthy",
    "Tomato_Early_Blight": "Early Blight",
    "Tomato_Late_Blight":  "Late Blight",
    "Tomato_Leaf_Mold":    "Leaf Mold",
}
SPLITS = ["train", "val", "test"]
CLASS_COLORS = ["#27ae60", "#e67e22", "#c0392b", "#8e44ad"]
SEED = 42

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "visualize.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, name: str, tight: bool = True) -> Path:
    path = PLOTS_DIR / name
    if tight:
        plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", path)
    return path


def _pick_sample_images(split: str = "train", n_per_class: int = 4,
                        seed: int = SEED) -> dict:
    """
    Returns {class_name: [PIL.Image, ...]} from the split folder.
    Uses the raw (un-preprocessed) images.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for cls in CLASSES:
        folder = SPLIT_DIR / split / cls
        if not folder.exists():
            log.warning("Missing folder: %s", folder)
            samples[cls] = []
            continue
        files = sorted(f for f in folder.iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        chosen = rng.choice(len(files), size=min(n_per_class, len(files)),
                            replace=False)
        samples[cls] = [Image.open(files[i]).convert("RGB") for i in chosen]
    return samples


def _load_cache(split: str):
    """Load preprocessed numpy cache. Returns (X, y) or (None, None)."""
    xi = CACHE_DIR / f"{split}_images.npy"
    yi = CACHE_DIR / f"{split}_labels.npy"
    if not xi.exists() or not yi.exists():
        return None, None
    return np.load(xi), np.load(yi)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Enhanced class distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution() -> Path:
    log.info("Plot 1: Class distribution ...")

    counts = {s: {} for s in SPLITS}
    for split in SPLITS:
        for cls in CLASSES:
            folder = SPLIT_DIR / split / cls
            if folder.exists():
                counts[split][cls] = len([
                    f for f in folder.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ])
            else:
                counts[split][cls] = 0

    x          = np.arange(len(CLASSES))
    bar_width  = 0.22
    split_cols = {"train": "#2ecc71", "val": "#3498db", "test": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LeafIQ — Dataset Class Distribution (Phase 2)", fontsize=14, y=1.01)

    # ── Grouped bar ───────────────────────────────────────────────────────────
    ax = axes[0]
    for i, split in enumerate(SPLITS):
        vals = [counts[split][c] for c in CLASSES]
        bars = ax.bar(x + i * bar_width, vals, bar_width,
                      label=split.capitalize(), color=split_cols[split], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        str(v), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([DISPLAY_NAMES[c] for c in CLASSES], rotation=12, ha="right")
    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Image Count")
    ax.set_title("Images per Class per Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    # ── Pie — total per class ─────────────────────────────────────────────────
    ax2 = axes[1]
    totals = [sum(counts[s][c] for s in SPLITS) for c in CLASSES]
    wedge_props = {"edgecolor": "white", "linewidth": 2}
    wedges, texts, autotexts = ax2.pie(
        totals,
        labels=[DISPLAY_NAMES[c] for c in CLASSES],
        colors=CLASS_COLORS,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=wedge_props,
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax2.set_title("Total Image Proportion per Class")

    return _save(fig, "class_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Representative raw images
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_images() -> Path:
    log.info("Plot 2: Sample images (raw) ...")

    n_per_class = 4
    samples = _pick_sample_images(split="train", n_per_class=n_per_class)

    fig, axes = plt.subplots(
        len(CLASSES), n_per_class,
        figsize=(n_per_class * 2.8, len(CLASSES) * 2.8),
    )
    fig.suptitle("LeafIQ — Representative Raw Images (Training Set)", fontsize=13)

    for row_i, cls in enumerate(CLASSES):
        imgs = samples.get(cls, [])
        for col_j in range(n_per_class):
            ax = axes[row_i][col_j]
            if col_j < len(imgs):
                ax.imshow(imgs[col_j])
            else:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            ax.axis("off")
            if col_j == 0:
                ax.set_title(DISPLAY_NAMES[cls], fontsize=10, pad=4,
                             color=CLASS_COLORS[row_i], fontweight="bold")

    return _save(fig, "sample_images.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Preprocessed images side-by-side with raw
# ─────────────────────────────────────────────────────────────────────────────

def plot_preprocessed_samples() -> Path:
    log.info("Plot 3: Preprocessed vs raw comparison ...")

    from preprocess import load_and_resize, normalize

    n_show = 3  # images per class
    samples = _pick_sample_images(split="train", n_per_class=n_show)

    n_classes = len(CLASSES)
    # Columns: raw_0, prep_0, raw_1, prep_1, raw_2, prep_2
    n_cols = n_show * 2
    fig, axes = plt.subplots(n_classes, n_cols,
                             figsize=(n_cols * 2.2, n_classes * 2.5))
    fig.suptitle("LeafIQ — Raw vs Preprocessed (224×224, /255 normalised)",
                 fontsize=12)

    TARGET_SIZE = (224, 224)

    for row_i, cls in enumerate(CLASSES):
        imgs = samples.get(cls, [])
        for idx in range(n_show):
            raw_ax  = axes[row_i][idx * 2]
            prep_ax = axes[row_i][idx * 2 + 1]

            if idx < len(imgs):
                raw_img = imgs[idx]
                raw_ax.imshow(raw_img)
                raw_ax.set_title(f"Raw\n{raw_img.size[0]}×{raw_img.size[1]}",
                                 fontsize=7)

                # Preprocess via PIL (no dependency on cache existing)
                resized  = raw_img.resize((TARGET_SIZE[1], TARGET_SIZE[0]),
                                          Image.BILINEAR)
                norm_arr = np.array(resized, dtype=np.float32) / 255.0
                prep_ax.imshow(np.clip(norm_arr, 0, 1))
                prep_ax.set_title("224×224\n/255", fontsize=7,
                                  color="#3498db")
            else:
                for a in (raw_ax, prep_ax):
                    a.set_facecolor("#f0f0f0")

            for a in (raw_ax, prep_ax):
                a.axis("off")

            if idx == 0:
                raw_ax.set_ylabel(DISPLAY_NAMES[cls], fontsize=9,
                                  color=CLASS_COLORS[row_i], fontweight="bold")

    return _save(fig, "preprocessed_samples.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Pixel statistics (per-channel histograms + mean/std)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pixel_statistics() -> Path:
    log.info("Plot 4: Pixel statistics (per-channel) ...")

    X_train, y_train = _load_cache("train")
    if X_train is None:
        log.warning("  Train cache missing — building on-the-fly from raw images ...")
        imgs, labels = [], []
        for i, cls in enumerate(CLASSES):
            folder = SPLIT_DIR / "train" / cls
            if not folder.exists():
                continue
            files = sorted(f for f in folder.iterdir()
                           if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            for fp in files:
                with Image.open(fp) as im:
                    im = im.convert("RGB").resize((224, 224), Image.BILINEAR)
                arr = np.array(im, dtype=np.float32) / 255.0
                imgs.append(arr)
                labels.append(i)
        X_train = np.stack(imgs)
        y_train = np.array(labels, dtype=np.int32)

    channel_names = ["Red", "Green", "Blue"]
    channel_cols  = ["#e74c3c", "#27ae60", "#3498db"]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("LeafIQ — Pixel Statistics (Training Set, 224×224 normalised)",
                 fontsize=13)

    # Row 0: per-channel histogram (all classes combined)
    for ch, (cname, ccol) in enumerate(zip(channel_names, channel_cols)):
        ax = fig.add_subplot(gs[0, ch])
        channel_data = X_train[:, :, :, ch].ravel()
        ax.hist(channel_data, bins=80, color=ccol, alpha=0.75, edgecolor="none")
        ax.set_title(f"{cname} Channel\nμ={channel_data.mean():.3f}  σ={channel_data.std():.3f}",
                     fontsize=9)
        ax.set_xlabel("Pixel value [0–1]")
        ax.set_ylabel("Frequency")
        ax.axvline(channel_data.mean(), color="black", lw=1.2, linestyle="--",
                   label="mean")
        ax.legend(fontsize=7)
        sns.despine(ax=ax)

    # Row 1: per-class mean pixel value (R, G, B)
    ax_means = fig.add_subplot(gs[1, :2])
    class_means = []
    for i, cls in enumerate(CLASSES):
        mask  = y_train == i
        imgs_ = X_train[mask]
        if len(imgs_) == 0:
            class_means.append([0, 0, 0])
            continue
        class_means.append(imgs_.mean(axis=(0, 1, 2)).tolist())

    class_means = np.array(class_means)
    x_pos       = np.arange(len(CLASSES))
    bw          = 0.25
    for ch, (cname, ccol) in enumerate(zip(channel_names, channel_cols)):
        ax_means.bar(x_pos + ch * bw, class_means[:, ch], bw,
                     label=cname, color=ccol, alpha=0.8)
    ax_means.set_xticks(x_pos + bw)
    ax_means.set_xticklabels([DISPLAY_NAMES[c] for c in CLASSES],
                              rotation=12, ha="right")
    ax_means.set_title("Mean Pixel Value per Class per Channel")
    ax_means.set_ylabel("Mean [0–1]")
    ax_means.legend()
    ax_means.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax_means)

    # Row 1 right: per-class std dev
    ax_std = fig.add_subplot(gs[1, 2])
    class_stds = []
    for i in range(len(CLASSES)):
        mask = y_train == i
        imgs_ = X_train[mask]
        if len(imgs_) == 0:
            class_stds.append(0.0)
            continue
        class_stds.append(float(imgs_.std()))
    ax_std.bar([DISPLAY_NAMES[c] for c in CLASSES], class_stds,
               color=CLASS_COLORS, alpha=0.85, edgecolor="white")
    ax_std.set_title("Pixel Std Dev per Class")
    ax_std.set_ylabel("Std Dev [0–1]")
    ax_std.tick_params(axis="x", rotation=15)
    sns.despine(ax=ax_std)

    # Row 2: overall intensity histogram per class (all channels, overlay)
    ax_ov = fig.add_subplot(gs[2, :])
    for i, cls in enumerate(CLASSES):
        mask  = y_train == i
        imgs_ = X_train[mask]
        if len(imgs_) == 0:
            continue
        flat = imgs_.ravel()
        ax_ov.hist(flat, bins=100, alpha=0.45, label=DISPLAY_NAMES[cls],
                   color=CLASS_COLORS[i], density=True)
    ax_ov.set_title("Intensity Distribution per Class (all channels, density)")
    ax_ov.set_xlabel("Pixel value [0–1]")
    ax_ov.set_ylabel("Density")
    ax_ov.legend()
    sns.despine(ax=ax_ov)

    return _save(fig, "pixel_statistics.png", tight=False)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Augmentation examples
# ─────────────────────────────────────────────────────────────────────────────

def plot_augmentation_examples() -> Path:
    log.info("Plot 5: Augmentation examples ...")

    # Import augmentation function from preprocess module
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from preprocess import load_and_resize, augment_image

    n_aug_per_class = 5   # original + 4 augmented versions
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(len(CLASSES), n_aug_per_class,
                             figsize=(n_aug_per_class * 2.5, len(CLASSES) * 2.5))
    fig.suptitle("LeafIQ — Augmentation Examples (Training Images)\n"
                 "Col 1: Original  |  Cols 2–5: Augmented",
                 fontsize=12)

    col_labels = ["Original", "Aug ×1", "Aug ×2", "Aug ×3", "Aug ×4"]

    for row_i, cls in enumerate(CLASSES):
        folder = SPLIT_DIR / "train" / cls
        if not folder.exists():
            for j in range(n_aug_per_class):
                axes[row_i][j].axis("off")
            continue

        files = sorted(f for f in folder.iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        img_path = files[rng.integers(0, len(files))]
        orig_arr = load_and_resize(img_path)   # uint8 (224, 224, 3)

        for col_j in range(n_aug_per_class):
            ax = axes[row_i][col_j]
            if col_j == 0:
                ax.imshow(orig_arr)
            else:
                aug_arr = augment_image(orig_arr, np.random.default_rng(SEED + row_i * 10 + col_j))
                ax.imshow(aug_arr)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(col_labels[col_j], fontsize=9, pad=3)
            if col_j == 0:
                ax.set_ylabel(DISPLAY_NAMES[cls], fontsize=9,
                              color=CLASS_COLORS[row_i], fontweight="bold")

    return _save(fig, "augmentation_examples.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — PCA + t-SNE projections
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_tsne() -> Path:
    log.info("Plot 6: PCA + t-SNE projection ...")

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    X_train, y_train = _load_cache("train")

    if X_train is None:
        log.warning("  Cache missing — loading images on-the-fly ...")
        imgs, labels = [], []
        for i, cls in enumerate(CLASSES):
            folder = SPLIT_DIR / "train" / cls
            if not folder.exists():
                continue
            files = sorted(f for f in folder.iterdir()
                           if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            for fp in files:
                with Image.open(fp) as im:
                    im = im.convert("RGB").resize((64, 64), Image.BILINEAR)
                arr = np.array(im, dtype=np.float32) / 255.0
                imgs.append(arr.ravel())
                labels.append(i)
        X_flat   = np.stack(imgs)
        y_train  = np.array(labels, dtype=np.int32)
    else:
        # Downsample to 64×64 for speed (flatten 224→64)
        n = len(X_train)
        small = []
        for i in range(n):
            pil = Image.fromarray((X_train[i] * 255).astype(np.uint8))
            pil = pil.resize((64, 64), Image.BILINEAR)
            small.append(np.array(pil, dtype=np.float32).ravel() / 255.0)
        X_flat = np.stack(small)

    log.info("  Feature matrix: %s — running PCA ...", X_flat.shape)

    # ── PCA (reduce to 50 dims first, then t-SNE on those) ───────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    pca      = PCA(n_components=50, random_state=SEED)
    X_pca50  = pca.fit_transform(X_scaled)

    # PCA 2-component projection for Plot
    pca2    = PCA(n_components=2, random_state=SEED)
    X_pca2  = pca2.fit_transform(X_scaled)

    # ── t-SNE (on 50 PCA dims for speed) ─────────────────────────────────────
    log.info("  Running t-SNE (perplexity=30, n_iter=500) ...")
    tsne   = TSNE(n_components=2, perplexity=30, max_iter=500,
                  random_state=SEED, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_pca50)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LeafIQ — PCA & t-SNE Image Feature Projections (Training Set)",
                 fontsize=13)

    # Left: PCA 2-D
    ax = axes[0]
    for i, cls in enumerate(CLASSES):
        mask = y_train == i
        ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                   c=CLASS_COLORS[i], label=DISPLAY_NAMES[cls],
                   alpha=0.6, s=18, edgecolors="none")
    ax.set_title(f"PCA — 2 Components\n(var explained: "
                 f"{pca2.explained_variance_ratio_.sum() * 100:.1f}%)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=8, markerscale=1.5)
    sns.despine(ax=ax)

    # Middle: t-SNE 2-D
    ax2 = axes[1]
    for i, cls in enumerate(CLASSES):
        mask = y_train == i
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=CLASS_COLORS[i], label=DISPLAY_NAMES[cls],
                    alpha=0.6, s=18, edgecolors="none")
    ax2.set_title("t-SNE — 2 Components\n(PCA 50-dim input)")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend(fontsize=8, markerscale=1.5)
    sns.despine(ax=ax2)

    # Right: PCA explained variance (scree plot — first 20 components)
    ax3   = axes[2]
    pca20 = PCA(n_components=20, random_state=SEED).fit(X_scaled)
    cumvar = np.cumsum(pca20.explained_variance_ratio_) * 100
    ax3.bar(range(1, 21), pca20.explained_variance_ratio_ * 100,
            color="#3498db", alpha=0.7, edgecolor="white")
    ax3.plot(range(1, 21), cumvar, "r-o", markersize=4, label="Cumulative")
    ax3.axhline(80, color="grey", linestyle="--", lw=1, label="80% threshold")
    ax3.set_title("PCA Scree Plot (top 20 components)")
    ax3.set_xlabel("Principal Component")
    ax3.set_ylabel("Explained Variance (%)")
    ax3.legend(fontsize=8)
    sns.despine(ax=ax3)

    return _save(fig, "pca_tsne_projection.png", tight=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("LeafIQ — Phase 2 Visualizations")
    log.info("Output directory: %s", PLOTS_DIR)
    log.info("=" * 60)

    saved = []

    log.info("\n[1/6] Class distribution plot")
    saved.append(plot_class_distribution())

    log.info("\n[2/6] Sample images (raw)")
    saved.append(plot_sample_images())

    log.info("\n[3/6] Preprocessed vs raw comparison")
    saved.append(plot_preprocessed_samples())

    log.info("\n[4/6] Pixel statistics")
    saved.append(plot_pixel_statistics())

    log.info("\n[5/6] Augmentation examples")
    saved.append(plot_augmentation_examples())

    log.info("\n[6/6] PCA + t-SNE projection")
    saved.append(plot_pca_tsne())

    log.info("\n── Visualization Summary ─────────────────────────────────────")
    for p in saved:
        size_kb = p.stat().st_size // 1024
        log.info("  %-45s  %d KB", p.name, size_kb)

    log.info("\nAll %d plots saved to:  %s", len(saved), PLOTS_DIR)
    log.info("Phase 2 visualizations complete.")


if __name__ == "__main__":
    main()
