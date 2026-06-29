import os
import json
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Configuration
RESULTS_DIR = "results"
TEST_DIR = "dataset/Test"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "error_analysis")

# Find latest model
def find_latest_model():
    import glob
    models = sorted(glob.glob(os.path.join(RESULTS_DIR, "mobilenet_*.keras")), reverse=True)
    if not models:
        models = sorted(glob.glob(os.path.join(RESULTS_DIR, "baseline_cnn_*.keras")), reverse=True)
    if not models:
        print("[ERROR] No model found in results/")
        return None
    return models[0]


def analyze_errors():
    model_path = find_latest_model()
    if not model_path:
        return

    print(f"\n[INFO] Loading model: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(model_path)

    # Detect model type
    is_mobilenet = "mobilenet" in model_path.lower()
    img_size = 224 if is_mobilenet else 100
    preprocess = preprocess_input if is_mobilenet else lambda x: x / 255.0

    # Get class names from model metadata
    meta_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')], reverse=True)
    class_names = None
    for meta_file in meta_files:
        with open(os.path.join(RESULTS_DIR, meta_file), 'r') as f:
            meta = json.load(f)
            if 'class_indices' in meta:
                class_names = [k for k, v in sorted(meta['class_indices'].items(), key=lambda x: x[1])]
                break

    if not class_names:
        # Fallback
        class_names = ["apple", "banana", "carrot", "cucumber", "eggplant",
                       "ginger", "lemon", "orange", "peach", "pear"]

    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Image size: {img_size}x{img_size}")

    # Prepare data generator
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(img_size, img_size),
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
        classes=class_names
    )

    # Predict
    print("\n[INFO] Running predictions on test set...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Find errors
    errors = []
    for i, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred)):
        if true_idx != pred_idx:
            errors.append({
                'index': i,
                'true_class': class_names[true_idx],
                'pred_class': class_names[pred_idx],
                'confidence': float(predictions[i][pred_idx]),
                'true_confidence': float(predictions[i][true_idx]),
                'filename': test_gen.filenames[i]
            })

    print(f"\n[INFO] Found {len(errors)} misclassifications out of {len(y_true)} test images")
    print(f"[INFO] Accuracy: {(len(y_true) - len(errors)) / len(y_true) * 100:.2f}%")

    if len(errors) == 0:
        print("\n[✓] Perfect accuracy! No errors to analyze.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Copy error images
    print("\n[INFO] Copying misclassified images...")
    for error in errors:
        src_path = os.path.join(TEST_DIR, error['filename'])
        if os.path.exists(src_path):
            # Create class folder
            error_folder = os.path.join(OUTPUT_DIR, f"{error['true_class']}_as_{error['pred_class']}")
            os.makedirs(error_folder, exist_ok=True)

            # Copy with confidence in filename
            basename = os.path.basename(error['filename'])
            new_name = f"conf_{error['confidence']:.3f}_{basename}"
            dst_path = os.path.join(error_folder, new_name)
            shutil.copy2(src_path, dst_path)

    # Generate error analysis report
    report_path = os.path.join(OUTPUT_DIR, "error_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Total test images: {len(y_true)}\n")
        f.write(f"Misclassifications: {len(errors)}\n")
        f.write(f"Accuracy: {(len(y_true) - len(errors)) / len(y_true) * 100:.2f}%\n\n")

        # Confusion pairs
        f.write("-" * 60 + "\n")
        f.write("CONFUSION PAIRS (True -> Predicted)\n")
        f.write("-" * 60 + "\n\n")

        confusion_pairs = {}
        for error in errors:
            pair = (error['true_class'], error['pred_class'])
            if pair not in confusion_pairs:
                confusion_pairs[pair] = []
            confusion_pairs[pair].append(error)

        for (true_cls, pred_cls), errs in sorted(confusion_pairs.items(), key=lambda x: -len(x[1])):
            f.write(f"{true_cls} -> {pred_cls}: {len(errs)} cases\n")
            avg_conf = np.mean([e['confidence'] for e in errs])
            f.write(f"  Average confidence: {avg_conf:.3f}\n")
            for e in errs[:3]:  # Show first 3 examples
                f.write(f"    - {os.path.basename(e['filename'])} (conf: {e['confidence']:.3f})\n")
            f.write("\n")

        # Per-class error rates
        f.write("-" * 60 + "\n")
        f.write("PER-CLASS ERROR ANALYSIS\n")
        f.write("-" * 60 + "\n\n")

        class_counts = {cls: {'total': 0, 'errors': 0} for cls in class_names}
        for i, true_idx in enumerate(y_true):
            cls = class_names[true_idx]
            class_counts[cls]['total'] += 1
            if y_pred[i] != true_idx:
                class_counts[cls]['errors'] += 1

        for cls in class_names:
            total = class_counts[cls]['total']
            errs = class_counts[cls]['errors']
            if total > 0:
                error_rate = errs / total * 100
                f.write(f"{cls:12s}: {errs}/{total} errors ({error_rate:.1f}%)\n")

    print(f"\n[INFO] Error report saved: {report_path}")
    print(f"[INFO] Error images saved to: {OUTPUT_DIR}")

    # Create confusion matrix visualization
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[INFO] Confusion matrix saved: {cm_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    analyze_errors()
