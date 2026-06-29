import os, glob, json, datetime
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#  Config 
WEBCAM_DIR  = os.path.join("dataset", "WebcamTrain")
ORIG_DIR    = os.path.join("dataset", "Train")
RESULTS     = "results"
IMG_SIZE    = 224
BATCH       = 16
EPOCHS      = 20
UNFREEZE_N  = 30      # unfreeze last N layers of MobileNetV2 backbone
LR_FINE     = 1e-4    # low LR for fine-tuning

CLASSES = [
    "apple","banana","carrot","cucumber","eggplant",
    "ginger","lemon","orange","peach","pear"
]

#  Find existing model 
def find_model():
    m = sorted(glob.glob(os.path.join(RESULTS, "mobilenet_*.keras")), reverse=True)
    if not m:
        print("[ERROR] No mobilenet_*.keras in results/"); exit(1)
    return m[0]

#  Check webcam data exists 
def check_webcam_data():
    if not os.path.exists(WEBCAM_DIR):
        print(f"[ERROR] {WEBCAM_DIR} not found.")
        print("  Run collect_webcam_data.py first.")
        exit(1)
    total = sum(
        len(glob.glob(os.path.join(WEBCAM_DIR, c, "*.jpg")))
        for c in CLASSES
    )
    if total < 10:
        print(f"[ERROR] Only {total} webcam images found. Collect at least 10 total.")
        exit(1)
    print(f"[INFO] Found {total} webcam images in {WEBCAM_DIR}")
    return total

#  Main 
def main():
    mp = find_model()
    print(f"\n[INFO] Loading base model: {mp}")
    base_model = tf.keras.models.load_model(mp)
    print("[INFO] Base model loaded.")

    check_webcam_data()

    #  Data generators 
    aug = dict(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=[0.6, 1.4],   # simulate different lighting
        validation_split=0.15
    )

    # Webcam images (your real photos)
    webcam_gen = ImageDataGenerator(**aug)
    webcam_train = webcam_gen.flow_from_directory(
        WEBCAM_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH, classes=CLASSES,
        subset="training", shuffle=True
    )
    webcam_val = webcam_gen.flow_from_directory(
        WEBCAM_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH, classes=CLASSES,
        subset="validation", shuffle=False
    )

    #  Rebuild model with unfrozen top layers 
    # Find the MobileNetV2 backbone sub-layer
    backbone = None
    for layer in base_model.layers:
        if "mobilenetv2" in layer.name.lower():
            backbone = layer
            break

    if backbone is not None:
        # Freeze all first, then unfreeze last N
        backbone.trainable = True
        for layer in backbone.layers[:-UNFREEZE_N]:
            layer.trainable = False
        unfrozen = sum(1 for l in backbone.layers if l.trainable)
        print(f"[INFO] Unfrozen {unfrozen} backbone layers for fine-tuning.")
    else:
        # Fallback: unfreeze whole model but use very low LR
        base_model.trainable = True
        print("[INFO] Backbone not isolated — full model unfrozen with low LR.")

    # Recompile with low learning rate
    base_model.compile(
        optimizer=optimizers.Adam(learning_rate=LR_FINE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    base_model.summary(print_fn=lambda x: None)  # suppress long summary

    #  Callbacks 
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = os.path.join(RESULTS, f"mobilenet_finetuned_{ts}.keras")

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(best_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1)
    ]

    #  Train 
    print(f"\n[INFO] Fine-tuning on {webcam_train.samples} webcam images...")
    print(f"[INFO] Validation: {webcam_val.samples} images")
    print(f"[INFO] Epochs: {EPOCHS}  |  Batch: {BATCH}  |  LR: {LR_FINE}\n")

    history = base_model.fit(
        webcam_train,
        validation_data=webcam_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    #  Save & report 
    final_val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\n[INFO] Best val accuracy: {final_val_acc:.4f}")
    print(f"[INFO] Fine-tuned model saved: {best_path}")
    print("[INFO] Update camera_demo.py to load this model, or it will auto-load")
    print("       as the newest mobilenet_*.keras file.\n")

    # Save metadata
    meta = {
        "base_model": mp,
        "finetuned_model": best_path,
        "timestamp": ts,
        "webcam_train_samples": webcam_train.samples,
        "webcam_val_samples": webcam_val.samples,
        "best_val_accuracy": round(final_val_acc, 4),
        "unfrozen_layers": UNFREEZE_N,
        "learning_rate": LR_FINE,
        "epochs_run": len(history.history["accuracy"])
    }
    meta_path = os.path.join(RESULTS, f"finetune_meta_{ts}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata saved: {meta_path}")

if __name__ == "__main__":
    main()