import os, glob, tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

RESULTS = "results"

def find_model():
    # Prefer finetuned, else use base retrained
    m = sorted(glob.glob(os.path.join(RESULTS, "mobilenet_finetuned_*.keras")), reverse=True)
    if not m:
        m = sorted(glob.glob(os.path.join(RESULTS, "mobilenet_2*.keras")), reverse=True)
        m = [x for x in m if "finetuned" not in x]
    if not m:
        raise FileNotFoundError("No MobileNetV2 model found in results/")
    return m[0]

model_path = find_model()
print(f"[INFO] Loading: {model_path}")
model = tf.keras.models.load_model(model_path)

# Dynamic range quantization — reduces model ~75%, faster on CPU
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

out_path = os.path.join(RESULTS, "mobilenet_quantized.tflite")
with open(out_path, "wb") as f:
    f.write(tflite_model)

orig_mb  = os.path.getsize(model_path) / 1e6
quant_mb = len(tflite_model) / 1e6
print(f"\n[INFO] Original:   {orig_mb:.1f} MB")
print(f"[INFO] Quantized:  {quant_mb:.1f} MB  ({(1-quant_mb/orig_mb)*100:.0f}% smaller)")
print(f"[INFO] Saved:      {out_path}")
print("\n[INFO] Now run camera_demo.py — it will use the quantized model automatically.")
