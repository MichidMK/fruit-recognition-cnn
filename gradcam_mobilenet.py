import os
import glob

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

#Auto-find newest MobileNetV2 model
def find_model():
    # Prefer base mobilenet, exclude finetuned variants
    models = sorted(glob.glob("results/mobilenet_*.keras"), reverse=True)
    clean = [m for m in models if "finetuned" not in m and "cnn" not in m]
    if not clean:
        clean = models
    if not clean:
        raise FileNotFoundError("No mobilenet_*.keras found in results/")
    print(f"[INFO] Loading: {clean[0]}")
    return clean[0]

IMG_SIZE   = 224
TEST_DIR   = "dataset/Test"
OUTPUT_DIR = "results/gradcam_mobilenet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(find_model())

# Class list must match the sorted order used during training
classes = sorted(os.listdir(TEST_DIR))

#Find the MobileNetV2 backbone sub-model
backbone = None
for layer in model.layers:
    if hasattr(layer, 'layers') and len(layer.layers) > 10:
        backbone = layer
        break

if backbone is None:
    raise RuntimeError("Cannot find MobileNetV2 backbone in model")

print(f"[INFO] Backbone: {backbone.name}")

# Find last Conv2D in the backbone
last_conv_name = None
for layer in backbone.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_name = layer.name

if last_conv_name is None:
    raise RuntimeError("No Conv2D found in backbone")

print(f"[INFO] Last conv layer: {last_conv_name}")

# Build feature extractor: backbone_input -> (last_conv_output, backbone_output)
feat_extractor = tf.keras.Model(
    inputs=backbone.inputs,
    outputs=[backbone.get_layer(last_conv_name).output, backbone.output]
)

# Build head model: backbone_output -> final prediction
head_input = tf.keras.Input(shape=backbone.output_shape[1:])
x = head_input
head_layers = [
    l for l in model.layers
    if l.name not in [ll.name for ll in backbone.layers]
    + [backbone.name, model.layers[0].name]
]
for l in head_layers:
    x = l(x)
head_model = tf.keras.Model(head_input, x)

print(f"[INFO] Head layers: {[l.name for l in head_layers]}")


def make_gradcam(img_array, target_class_idx):
    """Compute Grad-CAM heatmap for a given target class index."""
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_out, backbone_out = feat_extractor(img_tensor, training=False)
        tape.watch(conv_out)
        preds = head_model(backbone_out, training=False)
        loss = preds[:, target_class_idx]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return None, preds.numpy()[0]

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), preds.numpy()[0]


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))


failed = 0
total = 0

for true_cls in classes:
    cls_dir = os.path.join(TEST_DIR, true_cls)
    if not os.path.isdir(cls_dir):
        print(f"[WARN] Folder not found: {cls_dir}")
        continue

    true_idx = classes.index(true_cls)
    imgs = list_images(cls_dir)[:10]

    for img_name in imgs:
        total += 1
        img_path = os.path.join(cls_dir, img_name)

        # Preprocess using MobileNetV2's preprocess_input (NOT rescale 1/255)
        pil_img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = img_to_array(pil_img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        heatmap, probs = make_gradcam(arr, true_idx)
        pred_idx = int(np.argmax(probs))

        if heatmap is None:
            failed += 1
            continue

        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = cv2.imread(img_path)
        original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        out_name = f"true_{true_cls}_pred_{classes[pred_idx]}_{img_name}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), overlay)

print(f"\nGrad-CAM saved to {OUTPUT_DIR}")
print(f"Processed: {total} images")
print(f"Failed images (no grads): {failed}")
