import json
import numpy as np
from PIL import Image
import tensorflow as tf


def load_artifacts(model_path="model_cnn.keras",
                   class_path="class_names.json",
                   config_path="model_config.json"):
    # Panggil SEKALI saat aplikasi start
    model = tf.keras.models.load_model(model_path, compile=False)
    with open(class_path) as f:
        class_names = json.load(f)
    with open(config_path) as f:
        img_size = tuple(json.load(f)["img_size"])
    return model, class_names, img_size


def predict_image(image_file, model, class_names, img_size, top_k=3):
    # image_file: path string ATAU file-like object
    img = Image.open(image_file).convert("RGB").resize(img_size, Image.BILINEAR)
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    probs = model.predict(arr, verbose=0)[0]
    top = probs.argsort()[::-1][:top_k]
    return {
        "label": class_names[int(top[0])],
        "confidence": float(probs[top[0]]),
        "top_k": [{"label": class_names[int(i)], "confidence": float(probs[i])} for i in top],
    }
