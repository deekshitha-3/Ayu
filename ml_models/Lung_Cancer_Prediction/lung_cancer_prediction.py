import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image  # 👈 this is needed to use Image.Image type hints

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Image size used during training
IMAGE_SIZE = (150, 150)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "LCD.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

def preprocess_image(img: Image.Image, target_size=IMAGE_SIZE) -> np.ndarray:
    """Preprocess the input image for prediction."""
    img = img.resize(target_size)
    img = img.convert("RGB")  # 👈 Forcefully convert to RGB (drops alpha)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_lung_cancer(img: Image.Image) -> str:
    """Predict if the image shows signs of lung cancer."""
    processed = preprocess_image(img)
    prediction = model.predict(processed, verbose=0)[0][0]
    return "Positive (Suffering from Lung Cancer)" if prediction > 0.5 else "Negative (Not Suffering from Lung Cancer)"
