"""
Apple Leaf Disease Classifier — Training Script
Classes: Apple black_spot | Apple Brown_spot | Apple Normal
Architecture: MobileNetV2 with transfer learning + fine-tuning
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_HEAD = 10   # train only the head first
EPOCHS_FINE = 20   # then fine-tune top layers of base
SEED        = 42

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, "leafs", "Apple", "train")
TEST_DIR   = os.path.join(BASE_DIR, "leafs", "Apple", "test")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data generators ───────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
    validation_split=0.15,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True,
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False,
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

num_classes = len(train_gen.class_indices)
print(f"\nClasses: {train_gen.class_indices}")
print(f"Train samples: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}\n")

# Save class mapping
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)

# ── Model: MobileNetV2 frozen base + custom head ──────────────────────────────
base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Phase 1: Train head only ──────────────────────────────────────────────────
print("\n=== Phase 1: Training head ===")
callbacks_head = [
    EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    CSVLogger(os.path.join(MODEL_DIR, "history_head.csv")),
]
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_head,
)

# ── Phase 2: Fine-tune top layers of base ─────────────────────────────────────
print("\n=== Phase 2: Fine-tuning ===")
base.trainable = True
# Freeze everything except the last 30 layers
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_fine = [
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.h5"),
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1,
    ),
    EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    CSVLogger(os.path.join(MODEL_DIR, "history_fine.csv")),
]
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_fine,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n=== Test Evaluation ===")
loss, acc = model.evaluate(test_gen, verbose=1)
print(f"Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

# Save final model
model.save(os.path.join(MODEL_DIR, "apple_leaf_model.h5"))
print(f"\nModel saved to: {MODEL_DIR}")
