import os
import glob

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#Auto-find newest baseline CNN model
def find_model():
    models = sorted(glob.glob("results/baseline_cnn_*.keras"), reverse=True)
    if not models:
        raise FileNotFoundError("No baseline_cnn_*.keras found in results/")
    print(f"[INFO] Loading: {models[0]}")
    return models[0]

IMG_SIZE   = 100
TEST_DIR   = "dataset/Test"
OUTPUT_DIR = "results/gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(find_model())

# Force build so all layer shapes are populated
_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32), training=False)

# Class list must match the sorted order used during training
classes = sorted(os.listdir(TEST_DIR))

#Split model into conv_model + classifier# Find the last Conv2D layer
last_conv = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv = layer
        break

if last_conv is None:
    raise RuntimeError("No Conv2D layer found in model")

print(f"[INFO] Last Conv2D layer: {last_conv.name}")

# conv_model: input -> last Conv2D output
conv_model = tf.keras.Model(inputs=model.inputs, outputs=last_conv.output)

# classifier: last Conv2D output -> final prediction
last_conv_index = next(
    i for i, l in enumerate(model.layers) if l.name == last_conv.name
)
x_in = tf.keras.Input(shape=last_conv.output.shape[1:])
x = x_in
for layer in model.layers[last_conv_index + 1:]:
    x = layer(x)
classifier = tf.keras.Model(x_in, x)


def make_gradcam(img_array, target_class_idx):
    """Compute Grad-CAM heatmap for a given target class index."""
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out = conv_model(img_tensor, training=False)
        tape.watch(conv_out)
        preds = classifier(conv_out, training=False)
        loss = preds[:, target_class_idx]
    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return None, preds.numpy()[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)
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

        # Preprocess: rescale to [0,1] to match train_cnn.py
        pil_img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = img_to_array(pil_img) / 255.0
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
