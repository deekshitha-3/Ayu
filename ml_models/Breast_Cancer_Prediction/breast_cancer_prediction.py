import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the trained model once
model_path = os.path.join(os.path.dirname(__file__), "bcd_model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)
IMAGE_SIZE = (150, 150)  # Image size used during training

def preprocess_image(img: Image.Image, target_size=IMAGE_SIZE) -> np.ndarray:
    """Preprocess PIL image for prediction."""
    img = img.resize(target_size)
    img_array = np.array(img)
    if img_array.ndim == 2:  # grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_breast_cancer(img: Image.Image) -> str:
    """Predict class for a given PIL image."""
    processed = preprocess_image(img)
    prediction = model.predict(processed, verbose=0)[0][0]
    return "Malignant (Suffering from Breast Cancer)" if prediction > 0.5 else "Benign (Not Suffering from Breast Cancer)"
