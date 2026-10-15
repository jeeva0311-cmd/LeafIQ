"""
LeafIQ - Phase 5: CNN Feature Extraction
==========================================
Loads the trained Custom CNN checkpoint and extracts fixed-length embeddings
from the penultimate Dense(256) layer for train / val / test splits.

The cached images are 224x224 float32.  The CNN was trained on 64x64 input,
so we resize on-the-fly (same approach as evaluate.py).

Embedding layer: the Dense(256)->BN->ReLU block just before the softmax head.
This gives a 256-dimensional feature vector per image — richer than the raw
GAP output (128-dim) because it includes learned non-linear combinations.

No test data is touched until evaluation.  Features are extracted once and
cached so train_xgboost.py never needs to reload the Keras model.

Outputs (outputs/features/):
  train_features.npy   (3815, 256)  float32
  train_labels.npy     (3815,)      int64
  val_features.npy     (816,  256)  float32
  val_labels.npy       (816,)       int64
  test_features.npy    (821,  256)  float32
  test_labels.npy      (821,)       int64
  feature_config.json              metadata

Usage:
    python src/extract_features.py
    python src/extract_features.py --force   # overwrite existing features
"""

import os, sys, json, time, argparse, logging, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "outputs" / "cache"
FEAT_DIR     = PROJECT_ROOT / "outputs" / "features"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"
MODELS_DIR   = PROJECT_ROOT / "models"

FEAT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "extract_features.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

CNN_CKPT    = MODELS_DIR / "custom_cnn_best.keras"
CNN_INPUT_H = 64   # model was trained on 64x64
BATCH_SIZE  = 128  # safe for 2 GB RAM; images are 64x64x3 float32

CLASS_NAMES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]


# ── build extractor model ─────────────────────────────────────────────────

def build_extractor():
    """
    Load the full CNN, then create a sub-model that outputs the Dense(256)
    activation (after BN + ReLU, before Dropout + final softmax).

    Layer walk strategy: find the last Dense layer with units==256 and take
    its post-activation output.  If not found by name, fall back to the layer
    just before 'predictions'.
    """
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    log.info("Loading checkpoint: %s", CNN_CKPT)
    full_model = tf.keras.models.load_model(str(CNN_CKPT))
    log.info("  Input  : %s", full_model.input_shape)
    log.info("  Params : %d", full_model.count_params())

    # Print layer names for transparency
    log.info("  Layers :")
    for i, layer in enumerate(full_model.layers):
        log.info("    [%2d] %-40s  out=%s", i, layer.name,
                 getattr(layer, "output_shape", "?"))

    # Strategy: find the Activation layer that follows the Dense(256)+BN block.
    # In our architecture the sequence is:
    #   Dense(256) -> BN -> Activation("relu") -> Dropout(0.5) -> Dense(4,softmax)
    # We want the output of the Activation("relu") — index just before Dropout.
    embedding_tensor = None
    embedding_dim    = None

    for i, layer in enumerate(full_model.layers):
        # Look for an Activation layer whose input comes from a BN that follows Dense(256)
        if layer.__class__.__name__ == "Activation":
            # Check the layer just before this activation
            prev = full_model.layers[i - 1] if i > 0 else None
            if prev and prev.__class__.__name__ == "BatchNormalization":
                # Check the layer before BN
                prev2 = full_model.layers[i - 2] if i > 1 else None
                if prev2 and prev2.__class__.__name__ == "Dense":
                    cfg = prev2.get_config()
                    if cfg.get("units") == 256:
                        embedding_tensor = layer.output
                        embedding_dim    = 256
                        log.info("  Embedding layer: %s (256-dim)", layer.name)
                        break

    # Fallback: use the layer just before 'predictions'
    if embedding_tensor is None:
        for i, layer in enumerate(full_model.layers):
            if layer.name == "predictions":
                prev = full_model.layers[i - 1]
                embedding_tensor = prev.output
                embedding_dim    = prev.output_shape[-1]
                log.info("  Embedding layer (fallback): %s (%d-dim)",
                         prev.name, embedding_dim)
                break

    if embedding_tensor is None:
        raise RuntimeError("Could not locate embedding layer in CNN.")

    extractor = tf.keras.Model(
        inputs=full_model.input,
        outputs=embedding_tensor,
        name="CNN_Extractor",
    )
    log.info("  Extractor output shape: %s", extractor.output_shape)
    return extractor, embedding_dim


# ── extraction ────────────────────────────────────────────────────────────

def extract_split(extractor, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load cached images for one split, resize to 64x64, extract features."""
    import tensorflow as tf

    img_path = CACHE_DIR / f"{split}_images.npy"
    lbl_path = CACHE_DIR / f"{split}_labels.npy"

    log.info("  Loading %s cache ...", split)
    images = np.load(img_path, mmap_mode="r")   # (N, 224, 224, 3)
    labels = np.load(lbl_path)                   # (N,)
    n = len(images)
    log.info("  %s: %d images, %d labels", split, n, len(labels))

    # Build tf.data pipeline: resize -> batch -> extract
    ds = (
        tf.data.Dataset.from_tensor_slices(images)
        .map(lambda x: tf.image.resize(x, [CNN_INPUT_H, CNN_INPUT_H]),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(2)
    )

    log.info("  Extracting features for %s (batch=%d) ...", split, BATCH_SIZE)
    t0 = time.perf_counter()
    feats_list = []
    for batch in ds:
        feats_list.append(extractor(batch, training=False).numpy())
    features = np.concatenate(feats_list, axis=0).astype(np.float32)
    elapsed  = time.perf_counter() - t0

    log.info("  Done: features=%s  time=%.1fs  (%.1f ms/img)",
             features.shape, elapsed, elapsed / n * 1000)
    assert features.shape[0] == n, f"Feature count mismatch: {features.shape[0]} != {n}"
    return features, labels.astype(np.int64)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeafIQ Phase 5 Feature Extraction")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if feature files already exist")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("LeafIQ -- Phase 5: CNN Feature Extraction")
    log.info("=" * 60)

    # Check if already done
    already_done = all(
        (FEAT_DIR / f"{s}_features.npy").exists() and
        (FEAT_DIR / f"{s}_labels.npy").exists()
        for s in ["train", "val", "test"]
    )
    if already_done and not args.force:
        log.info("Feature files already exist. Use --force to re-extract.")
        # Still print shapes for verification
        for s in ["train", "val", "test"]:
            f = np.load(FEAT_DIR / f"{s}_features.npy")
            l = np.load(FEAT_DIR / f"{s}_labels.npy")
            log.info("  %s: features=%s labels=%s", s, f.shape, l.shape)
        return

    extractor, emb_dim = build_extractor()

    config = {
        "cnn_checkpoint":    str(CNN_CKPT),
        "cnn_input_hw":      CNN_INPUT_H,
        "embedding_dim":     emb_dim,
        "class_names":       CLASS_NAMES,
        "batch_size":        BATCH_SIZE,
        "splits": {}
    }

    for split in ["train", "val", "test"]:
        feats, labels = extract_split(extractor, split)

        feat_path = FEAT_DIR / f"{split}_features.npy"
        lbl_path  = FEAT_DIR / f"{split}_labels.npy"
        np.save(feat_path, feats)
        np.save(lbl_path,  labels)
        log.info("  Saved: %s", feat_path)
        log.info("  Saved: %s", lbl_path)

        config["splits"][split] = {
            "n_samples":    int(feats.shape[0]),
            "feature_shape": list(feats.shape),
            "label_counts": {
                CLASS_NAMES[i]: int((labels == i).sum())
                for i in range(len(CLASS_NAMES))
            }
        }

    cfg_path = FEAT_DIR / "feature_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    log.info("Feature config saved: %s", cfg_path)

    log.info("\nExtraction complete.")
    log.info("  Output dir : %s", FEAT_DIR)
    log.info("  Embedding  : %d-dim per image", emb_dim)
    for s, info in config["splits"].items():
        log.info("  %s: %s", s, info["feature_shape"])


if __name__ == "__main__":
    main()
