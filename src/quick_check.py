"""Quick check: new model sizes + dataset resize pipeline."""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tensorflow as tf
from model import build_custom_cnn, build_mobilenetv2, CNN_INPUT_SIZE, MNV2_INPUT_SIZE
from train  import make_dataset, compute_class_weights

print("TF", tf.__version__)

# --- Build models ---
cnn  = build_custom_cnn()
mnv2 = build_mobilenetv2(fine_tune=False)
print(f"CNN  input={CNN_INPUT_SIZE}   params={cnn.count_params():,}")
print(f"MNV2 input={MNV2_INPUT_SIZE}  trainable={sum(v.numpy().size for v in mnv2.trainable_variables):,}")

# --- Tiny dataset (224x224 source, will be resized in pipeline) ---
X = np.random.rand(64, 224, 224, 3).astype("float32")
y = np.array([i % 4 for i in range(64)], dtype="int32")

# --- CNN pipeline ---
t0 = time.perf_counter()
ds_cnn = make_dataset(X, y, CNN_INPUT_SIZE, augment=True, shuffle=True)
for xb, yb in ds_cnn.take(1):
    print(f"CNN  batch shape: {tuple(xb.shape)}  range=[{xb.numpy().min():.2f},{xb.numpy().max():.2f}]")
    out = cnn(xb, training=False)
    print(f"CNN  output shape: {out.shape}  sums={out.numpy().sum(axis=1).round(3)}")
print(f"CNN  step time: {(time.perf_counter()-t0)*1000:.0f} ms")

# --- MNV2 pipeline ---
t0 = time.perf_counter()
ds_mnv2 = make_dataset(X, y, MNV2_INPUT_SIZE, augment=False)
for xb, yb in ds_mnv2.take(1):
    print(f"MNV2 batch shape: {tuple(xb.shape)}")
    out2 = mnv2(xb, training=False)
    print(f"MNV2 output shape: {out2.shape}  sums={out2.numpy().sum(axis=1).round(3)}")
print(f"MNV2 step time: {(time.perf_counter()-t0)*1000:.0f} ms")

# --- Class weights ---
cw = compute_class_weights(y)
print(f"Class weights: {cw}")

print("\nQUICK CHECK PASSED")
