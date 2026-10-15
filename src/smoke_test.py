"""
LeafIQ - Phase 3 Smoke Test
Verifies model builds, forward pass, dataset pipeline, and callbacks
without loading the full training cache or running real training.
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

print("=" * 60)
print("LeafIQ — Phase 3 Smoke Test")
print("=" * 60)

# ── 1. TF import ──────────────────────────────────────────────────────────────
print("\n[1] Importing TensorFlow ...")
t0 = time.perf_counter()
import tensorflow as tf
print(f"    TF {tf.__version__}  Keras {tf.keras.__version__}  "
      f"(imported in {time.perf_counter()-t0:.1f}s)")

# ── 2. CustomCNN build ────────────────────────────────────────────────────────
print("\n[2] Building CustomCNN ...")
from model import build_custom_cnn, build_mobilenetv2
cnn = build_custom_cnn()
n_params = cnn.count_params()
print(f"    Parameters: {n_params:,}")
assert n_params > 0, "Model has no parameters"

# ── 3. MobileNetV2 frozen build ───────────────────────────────────────────────
print("\n[3] Building MobileNetV2 (frozen base) ...")
mnv2 = build_mobilenetv2(fine_tune=False)
trainable = sum(v.numpy().size for v in mnv2.trainable_variables)
total     = mnv2.count_params()
print(f"    Total params: {total:,}  Trainable: {trainable:,}")
assert trainable < total, "Frozen base should have fewer trainable params than total"

# ── 4. MobileNetV2 fine-tune build ────────────────────────────────────────────
print("\n[4] Building MobileNetV2 (fine-tune last 30) ...")
mnv2_ft = build_mobilenetv2(fine_tune=True, fine_tune_layers=30)
trainable_ft = sum(v.numpy().size for v in mnv2_ft.trainable_variables)
print(f"    Trainable params (fine-tune): {trainable_ft:,}")
assert trainable_ft > trainable, "Fine-tune should unlock more params"

# ── 5. Forward pass — both models ─────────────────────────────────────────────
print("\n[5] Forward pass (batch=4) ...")
dummy = np.random.rand(4, 224, 224, 3).astype("float32")

out_cnn  = cnn.predict(dummy, verbose=0)
out_mnv2 = mnv2.predict(dummy, verbose=0)

assert out_cnn.shape  == (4, 4), f"CNN output shape wrong: {out_cnn.shape}"
assert out_mnv2.shape == (4, 4), f"MNV2 output shape wrong: {out_mnv2.shape}"
# Probabilities should sum to ~1
np.testing.assert_allclose(out_cnn.sum(axis=1),  np.ones(4), atol=1e-5)
np.testing.assert_allclose(out_mnv2.sum(axis=1), np.ones(4), atol=1e-5)
print(f"    CNN output:  {out_cnn.shape}  sums={out_cnn.sum(axis=1).round(4)}")
print(f"    MNV2 output: {out_mnv2.shape}  sums={out_mnv2.sum(axis=1).round(4)}")

# ── 6. Dataset pipeline ───────────────────────────────────────────────────────
print("\n[6] Dataset pipeline (100 dummy samples) ...")
from train import make_dataset, compute_class_weights, _augment

X_fake = np.random.rand(100, 224, 224, 3).astype("float32")
y_fake = np.array([i % 4 for i in range(100)], dtype="int32")

train_ds = make_dataset(X_fake, y_fake, augment=True,  shuffle=True)
val_ds   = make_dataset(X_fake, y_fake, augment=False, shuffle=False)

for X_batch, y_batch in train_ds.take(1):
    print(f"    Train batch: X={tuple(X_batch.shape)}  y={tuple(y_batch.shape)}"
          f"  range=[{X_batch.numpy().min():.3f},{X_batch.numpy().max():.3f}]")
for X_batch, y_batch in val_ds.take(1):
    print(f"    Val   batch: X={tuple(X_batch.shape)}  y={tuple(y_batch.shape)}")

# ── 7. Class weights ──────────────────────────────────────────────────────────
print("\n[7] Class weight computation ...")
cw = compute_class_weights(y_fake)
assert len(cw) == 4, "Should have weights for 4 classes"
print(f"    Class weights: {cw}")

# ── 8. Callbacks instantiation ────────────────────────────────────────────────
print("\n[8] Callbacks instantiation ...")
from train import get_callbacks
from pathlib import Path
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    cbs = get_callbacks(Path(tmp)/"test.keras", Path(tmp)/"test.csv")
    assert len(cbs) == 4, f"Expected 4 callbacks, got {len(cbs)}"
    print(f"    {len(cbs)} callbacks: {[type(c).__name__ for c in cbs]}")

# ── 9. Mini training run (2 epochs, tiny data) ────────────────────────────────
print("\n[9] Mini training run (2 epochs, 100 samples) ...")
tiny_ds  = make_dataset(X_fake, y_fake, augment=True,  shuffle=True)
tiny_val = make_dataset(X_fake, y_fake, augment=False, shuffle=False)

tiny_cnn = build_custom_cnn()
hist = tiny_cnn.fit(tiny_ds, validation_data=tiny_val, epochs=2, verbose=0)
assert "accuracy"     in hist.history
assert "val_accuracy" in hist.history
final_loss = hist.history["loss"][-1]
print(f"    2-epoch train loss: {final_loss:.4f}  "
      f"val_acc: {hist.history['val_accuracy'][-1]:.4f}")

# ── 10. evaluate.py import check ─────────────────────────────────────────────
print("\n[10] evaluate.py import check ...")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "evaluate", ROOT / "src" / "evaluate.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)
assert hasattr(eval_mod, "compute_metrics")
assert hasattr(eval_mod, "plot_confusion_matrix")
assert hasattr(eval_mod, "plot_training_curves")
assert hasattr(eval_mod, "plot_model_comparison")
print("    All evaluate.py functions present.")

print("\n" + "=" * 60)
print("SMOKE TEST PASSED — all 10 checks OK")
print("=" * 60)
