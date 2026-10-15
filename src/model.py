"""
LeafIQ - Model Definitions
============================
Phase 3: Model Training

Two model builders:
  1. build_custom_cnn()   — lightweight CNN from scratch (64×64 input)
  2. build_mobilenetv2()  — MobileNetV2 + ImageNet weights (96×96 input)

CPU training note
-----------------
224×224 is the MobileNetV2 design size, but it makes CPU steps ~7 s each.
Resizing to 64×64 (CNN) / 96×96 (MobileNetV2) inside the tf.data pipeline
cuts step time to <1 s while preserving enough spatial information for
4-class disease discrimination. The caches stay as-is (224×224 float32);
resizing happens on-the-fly.

Architecture: CustomCNN  (input 64×64)
---------------------------------------
Input (64,64,3)
  └─ Block 1: Conv2D(32,3) → BN → ReLU → Conv2D(32,3) → BN → ReLU → MaxPool → Dropout(0.25)
  └─ Block 2: Conv2D(64,3) → BN → ReLU → Conv2D(64,3) → BN → ReLU → MaxPool → Dropout(0.25)
  └─ Block 3: Conv2D(128,3) → BN → ReLU → Conv2D(128,3) → BN → ReLU → MaxPool → Dropout(0.25)
  └─ GlobalAveragePooling2D
  └─ Dense(256) → BN → ReLU → Dropout(0.5)
  └─ Dense(4, softmax)

Architecture: MobileNetV2  (input 96×96, minimum supported by MobileNetV2)
---------------------------------------------------------------------------
MobileNetV2(include_top=False, input_shape=(96,96,3)) → GAP → Dropout → Dense(4)
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ── Size constants ─────────────────────────────────────────────────────────────
CNN_INPUT_SIZE  = (64, 64, 3)    # fast on CPU, enough for 4-class leaf disease
MNV2_INPUT_SIZE = (96, 96, 3)    # MobileNetV2 minimum recommended size

NUM_CLASSES = 4
CLASS_NAMES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]


def _conv_bn_relu(x, filters: int, kernel_size: int = 3, l2: float = 1e-4):
    x = layers.Conv2D(filters, kernel_size, padding="same",
                      kernel_regularizer=regularizers.l2(l2),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Custom CNN  (64×64 input)
# ─────────────────────────────────────────────────────────────────────────────

def build_custom_cnn(
    num_classes:   int   = NUM_CLASSES,
    input_shape:   tuple = CNN_INPUT_SIZE,
    dropout_conv:  float = 0.25,
    dropout_fc:    float = 0.50,
    l2:            float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Lightweight CNN trained from scratch on 64×64 images.
    ~323K parameters, <1 s/step on CPU.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs
    for filters in [32, 64, 128]:
        x = _conv_bn_relu(x, filters, l2=l2)
        x = _conv_bn_relu(x, filters, l2=l2)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(dropout_conv)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False,
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_fc)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = keras.Model(inputs, outputs, name="LeafIQ_CustomCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — MobileNetV2  (96×96 input)
# ─────────────────────────────────────────────────────────────────────────────

def build_mobilenetv2(
    num_classes:      int   = NUM_CLASSES,
    input_shape:      tuple = MNV2_INPUT_SIZE,
    dropout_fc:       float = 0.50,   # higher dropout to fight overfitting
    learning_rate:    float = 1e-3,
    fine_tune:        bool  = False,
    fine_tune_layers: int   = 10,     # only unfreeze last 10 layers (not 30)
    fine_tune_lr:     float = 5e-5,
) -> keras.Model:
    """
    MobileNetV2 transfer learning on 96×96 images.

    Phase A (fine_tune=False): base frozen — only the head is trained.
    Phase B (fine_tune=True):  last `fine_tune_layers` unfrozen.

    The preprocess_input call (scaling [0,1]→[−1,1]) is baked into the model
    so the cached float32 arrays are fed directly.
    """
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    if fine_tune:
        base.trainable = True
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False

    inputs = keras.Input(shape=input_shape, name="input_image")
    # Scale [0,1] → [−1,1] as MobileNetV2 expects
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-3),
                     use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_fc)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = keras.Model(
        inputs, outputs,
        name="LeafIQ_MobileNetV2_FT" if fine_tune else "LeafIQ_MobileNetV2",
    )
    lr = fine_tune_lr if fine_tune else learning_rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    print("=" * 60)
    print("LeafIQ — Model Smoke Test")
    print("=" * 60)

    dummy_cnn  = np.random.rand(2, 64, 64, 3).astype("float32")
    dummy_mnv2 = np.random.rand(2, 96, 96, 3).astype("float32")

    print("\n[1] CustomCNN (64×64)")
    cnn = build_custom_cnn()
    cnn.summary(line_length=72)
    out = cnn.predict(dummy_cnn, verbose=0)
    print(f"  Output: {out.shape}  sums≈1: {out.sum(axis=1).round(4)}")

    print("\n[2] MobileNetV2 frozen (96×96)")
    m2 = build_mobilenetv2(fine_tune=False)
    out2 = m2.predict(dummy_mnv2, verbose=0)
    print(f"  Output: {out2.shape}  sums≈1: {out2.sum(axis=1).round(4)}")
    print(f"  Trainable params: {sum(v.numpy().size for v in m2.trainable_variables):,}")

    print("\n[3] MobileNetV2 fine-tune (96×96)")
    m3 = build_mobilenetv2(fine_tune=True)
    print(f"  Trainable params: {sum(v.numpy().size for v in m3.trainable_variables):,}")

    print("\nAll OK.")
