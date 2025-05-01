import streamlit as st
import numpy as np
from PIL import Image
from ml_models.Lung_Cancer_Prediction.lung_cancer_prediction import predict_lung_cancer
from utils.db import predictions

st.title("🫁 Lung Cancer Prediction")
img = st.file_uploader("Upload Lung Scan Image", type=["jpg", "jpeg", "png"])

if img:
    image = Image.open(img)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        result = predict_lung_cancer(image)
        st.success(result)
        predictions.insert_one({"type": "lung", "result": result})
