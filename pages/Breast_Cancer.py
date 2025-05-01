import streamlit as st
from PIL import Image
from ml_models.Breast_Cancer_Prediction.breast_cancer_prediction import predict_breast_cancer
from utils.db import predictions

st.title("🎗️ Breast Cancer Prediction")
img = st.file_uploader("Upload Mammogram Image", type=["jpg", "png", "jpeg"])

if img:
    image = Image.open(img)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        result = predict_breast_cancer(image)
        st.success(result)
        predictions.insert_one({"type": "breast", "result": result})
