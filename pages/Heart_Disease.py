import streamlit as st
import json
from ml_models.Heart_Disease_Prediction.heartpredict import predict_heart_disease
from utils.db import predictions
import subprocess

st.title("❤️ Heart Disease Prediction")
pdf = st.file_uploader("Upload Heart Report (PDF)", type=["pdf"])

if pdf:
    with open("uploads/temp_heart.pdf", "wb") as f:
        f.write(pdf.read())

    try:
        output = subprocess.check_output(["python", "utils/scrapHeart.py", "uploads/temp_heart.pdf"])
        parsed_data = json.loads(output.decode("utf-8"))
        st.json(parsed_data)

        if st.button("Predict"):
            result = predict_heart_disease(parsed_data)
            st.success(result)
            predictions.insert_one({"type": "heart", "input": parsed_data, "result": result})
    except subprocess.CalledProcessError as e:
        st.error("Error extracting data from PDF.")
        st.code(e.output.decode())

    missing_keys = [k for k, v in parsed_data.items() if v is None]
    if missing_keys:
        st.warning(f"Some values are missing: {', '.join(missing_keys)}. Prediction may be less accurate.")

