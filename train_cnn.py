import os
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D,
    Flatten, Dense, Dropout, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# PATHS + CONFIG
train_dir = "dataset/Train"
test_dir = "dataset/Test"

out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
run_id = time.strftime("%Y%m%d_%H%M%S")

IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 30

# DATA GENERATORS
# Validation comes from an 80/20 split of Train only.
# Test is a separate folder, never seen during training.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Build sorted class list from Train directory
train_gen_tmp = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)
classes = sorted(train_gen_tmp.class_indices.keys())

# Training generator (from Train, subset="training")
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=classes,
    subset="training",
    shuffle=True
)

# Validation generator (from Train, subset="validation")
val_gen = val_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=classes,
    subset="validation",
    shuffle=False
)

# Test generator (separate Test folder, no subset)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=classes,
    shuffle=False
)

print("Class indices (train):", train_gen.class_indices)
print("Class indices (val):  ", val_gen.class_indices)
print("Class indices (test): ", test_gen.class_indices)

# Safety check: class order must match across all generators
assert train_gen.class_indices == val_gen.class_indices == test_gen.class_indices, \
    "Class index mismatch!"

# MODEL: Baseline CNN from scratch
# No pretrained weights. No MobileNetV2. No transfer learning.
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# CALLBACKS
callbacks = [
    EarlyStopping(
        monitor="val_accuracy", mode="max",
        patience=5, restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=2, min_lr=1e-6
    )
]

# TRAIN — single phase, no freezing, no fine-tuning
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# EVALUATE ON TEST (only once, at the end)
test_gen.reset()
pred = model.predict(test_gen, verbose=0)
y_pred = np.argmax(pred, axis=1)
y_true = test_gen.classes

all_labels = list(range(len(classes)))
report_text = classification_report(
    y_true, y_pred,
    labels=all_labels,
    target_names=classes,
    zero_division=0
)
cm = confusion_matrix(y_true, y_pred, labels=all_labels)

print("\nClassification report (TEST)")
print(report_text)
print("Confusion matrix (TEST)")
print(cm)

# SAVE FIGURES
cm_png = os.path.join(out_dir, f"baseline_confusion_matrix_{run_id}.png")
acc_png = os.path.join(out_dir, f"baseline_train_acc_{run_id}.png")
loss_png = os.path.join(out_dir, f"baseline_train_loss_{run_id}.png")

# Confusion matrix
plt.figure(figsize=(9, 7))
if HAS_SEABORN:
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes
    )
else:
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Baseline CNN (Test Set)")
plt.tight_layout()
plt.savefig(cm_png, dpi=200)
plt.close()
print("Saved confusion matrix to", cm_png)

# Accuracy curve
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Curve - Accuracy (Baseline CNN)")
plt.legend()
plt.tight_layout()
plt.savefig(acc_png, dpi=200)
plt.close()
print("Saved accuracy curve to", acc_png)

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve - Loss (Baseline CNN)")
plt.legend()
plt.tight_layout()
plt.savefig(loss_png, dpi=200)
plt.close()
print("Saved loss curve to", loss_png)

# SAVE REPORT + META + MODEL
with open(os.path.join(out_dir, f"baseline_report_{run_id}.txt"), "w", encoding="utf-8") as f:
    f.write(f"Run ID: {run_id}\n")
    f.write("Architecture: Baseline CNN\n\n")
    f.write(f"Classes: {', '.join(classes)}\n\n")
    f.write("Classification report (TEST)\n")
    f.write(report_text + "\n\n")
    f.write("Confusion matrix (TEST)\n")
    f.write(str(cm) + "\n")

with open(os.path.join(out_dir, f"baseline_meta_{run_id}.json"), "w", encoding="utf-8") as f:
    json.dump({
        "run_id": run_id,
        "architecture": "Baseline_CNN",
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_samples": int(train_gen.samples),
        "val_samples": int(val_gen.samples),
        "test_samples": int(test_gen.samples),
        "num_classes": len(classes),
        "class_indices": test_gen.class_indices,
        "stopped_epoch": int(history.epoch[-1]) if hasattr(history, "epoch") else None,
        "saved_confusion_matrix_png": os.path.basename(cm_png),
        "saved_train_acc_png": os.path.basename(acc_png),
        "saved_train_loss_png": os.path.basename(loss_png),
    }, f, indent=2)

model_path = os.path.join(out_dir, f"baseline_cnn_{run_id}.keras")
model.save(model_path)
print(f"\nSaved model to {model_path}")
print(f"Saved results to {out_dir}")
