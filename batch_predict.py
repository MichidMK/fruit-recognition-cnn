import os
import sys
import glob
import argparse
import csv
import json
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration
RESULTS_DIR = "results"
DEFAULT_MODEL_PATTERN = "mobilenet_*.keras"

CLASS_NAMES = [
    "apple", "banana", "carrot", "cucumber", "eggplant",
    "ginger", "lemon", "orange", "peach", "pear"
]


def find_latest_model():
    """Find the most recent model file."""
    models = sorted(glob.glob(os.path.join(RESULTS_DIR, DEFAULT_MODEL_PATTERN)), reverse=True)
    if not models:
        # Fallback to baseline
        models = sorted(glob.glob(os.path.join(RESULTS_DIR, "baseline_cnn_*.keras")), reverse=True)
    if not models:
        print("[ERROR] No model found in results/")
        return None
    return models[0]


def load_model_and_config(model_path):
    """Load model and detect its configuration."""
    print(f"[INFO] Loading model: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(model_path)

    # Detect model type from filename
    is_mobilenet = "mobilenet" in model_path.lower()
    img_size = 224 if is_mobilenet else 100
    preprocess = preprocess_input if is_mobilenet else lambda x: x / 255.0

    print(f"[INFO] Architecture: {'MobileNetV2' if is_mobilenet else 'Baseline CNN'}")
    print(f"[INFO] Input size: {img_size}x{img_size}")

    return model, img_size, preprocess


def preprocess_image(image_path, img_size, preprocess_func):
    """Load and preprocess an image for prediction."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    img = cv2.resize(img, (img_size, img_size))
    # Apply preprocessing
    img = preprocess_func(img.astype(np.float32))
    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def predict_single(model, image_path, img_size, preprocess_func):
    """Predict a single image and return results."""
    img = preprocess_image(image_path, img_size, preprocess_func)
    if img is None:
        return None

    predictions = model.predict(img, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])

    return {
        'predicted_class': CLASS_NAMES[predicted_idx],
        'confidence': confidence,
        'all_probabilities': {cls: float(prob) for cls, prob in zip(CLASS_NAMES, predictions)}
    }


def batch_predict(model, input_dir, img_size, preprocess_func, recursive=False):
    """Predict all images in a directory."""
    # Find all images
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []

    for pattern in patterns:
        if recursive:
            image_files.extend(glob.glob(os.path.join(input_dir, '**', pattern), recursive=True))
        else:
            image_files.extend(glob.glob(os.path.join(input_dir, pattern)))

    if not image_files:
        print(f"[ERROR] No images found in {input_dir}")
        return []

    print(f"[INFO] Found {len(image_files)} images")
    print("[INFO] Processing...\n")

    results = []
    for i, img_path in enumerate(image_files, 1):
        result = predict_single(model, img_path, img_size, preprocess_func)
        if result:
            result['image_path'] = img_path
            result['filename'] = os.path.basename(img_path)
            results.append(result)

        if i % 10 == 0 or i == len(image_files):
            print(f"  Processed {i}/{len(image_files)} images")

    return results


def save_results_csv(results, output_path):
    """Save prediction results to CSV."""
    if not results:
        print("[WARNING] No results to save")
        return

    # Prepare CSV headers
    headers = ['filename', 'predicted_class', 'confidence']
    headers.extend([f"prob_{cls}" for cls in CLASS_NAMES])
    headers.append('image_path')

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for result in results:
            row = {
                'filename': result['filename'],
                'predicted_class': result['predicted_class'],
                'confidence': f"{result['confidence']:.4f}",
                'image_path': result['image_path']
            }
            # Add all class probabilities
            for cls in CLASS_NAMES:
                row[f"prob_{cls}"] = f"{result['all_probabilities'][cls]:.4f}"

            writer.writerow(row)

    print(f"\n[INFO] Results saved to: {output_path}")


def generate_summary(results, output_path):
    """Generate a summary report of predictions."""
    if not results:
        return

    summary_path = output_path.replace('.csv', '_summary.txt')

    # Calculate statistics
    total = len(results)
    class_counts = {cls: 0 for cls in CLASS_NAMES}
    high_conf = 0  # > 0.9
    med_conf = 0   # 0.7 - 0.9
    low_conf = 0   # < 0.7

    for r in results:
        class_counts[r['predicted_class']] += 1
        conf = r['confidence']
        if conf > 0.9:
            high_conf += 1
        elif conf > 0.7:
            med_conf += 1
        else:
            low_conf += 1

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("BATCH PREDICTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images: {total}\n\n")

        f.write("-" * 60 + "\n")
        f.write("PREDICTION DISTRIBUTION\n")
        f.write("-" * 60 + "\n")
        for cls in CLASS_NAMES:
            count = class_counts[cls]
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)
            f.write(f"{cls:12s}: {count:4d} ({pct:5.1f}%) {bar}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("CONFIDENCE DISTRIBUTION\n")
        f.write("-" * 60 + "\n")
        f.write(f"High confidence (>90%): {high_conf} ({high_conf/total*100:.1f}%)\n")
        f.write(f"Medium confidence (70-90%): {med_conf} ({med_conf/total*100:.1f}%)\n")
        f.write(f"Low confidence (<70%): {low_conf} ({low_conf/total*100:.1f}%)\n")

        # Top uncertain predictions
        f.write("\n" + "-" * 60 + "\n")
        f.write("MOST UNCERTAIN PREDICTIONS\n")
        f.write("-" * 60 + "\n")
        sorted_results = sorted(results, key=lambda x: x['confidence'])
        for r in sorted_results[:10]:
            f.write(f"{r['filename']:30s} -> {r['predicted_class']:12s} ({r['confidence']:.3f})\n")

    print(f"[INFO] Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch prediction for fruit recognition')
    parser.add_argument('--input', '-i', required=True,
                        help='Input directory or image file')
    parser.add_argument('--output', '-o', default='results/batch_predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--model', '-m', default=None,
                        help='Specific model file (default: latest)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search subdirectories for images')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BATCH PREDICTION")
    print("=" * 60 + "\n")

    # Validate input
    if not os.path.exists(args.input):
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)

    # Find model
    model_path = args.model if args.model else find_latest_model()
    if not model_path:
        sys.exit(1)

    # Load model
    model, img_size, preprocess_func = load_model_and_config(model_path)

    # Run predictions
    if os.path.isfile(args.input):
        # Single file
        print(f"[INFO] Processing single image: {args.input}")
        result = predict_single(model, args.input, img_size, preprocess_func)
        if result:
            print(f"\nPrediction: {result['predicted_class']} ({result['confidence']*100:.2f}%)")
            print("\nAll probabilities:")
            for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
                print(f"  {cls:12s}: {prob*100:6.2f}%")
    else:
        # Directory
        results = batch_predict(model, args.input, img_size, preprocess_func, args.recursive)
        if results:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            save_results_csv(results, args.output)
            generate_summary(results, args.output)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
