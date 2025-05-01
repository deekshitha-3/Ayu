import streamlit as st
import json
import subprocess
from ml_models.Diabetes_Prediction.diabetespredict import predict_diabetes
from utils.db import predictions

st.title("🩸 Diabetes Prediction")
pdf = st.file_uploader("Upload Diabetes Report (PDF)", type=["pdf"])

if pdf:
    with open("uploads/temp_diabetes.pdf", "wb") as f:
        f.write(pdf.read())

    try:
        output = subprocess.check_output(["python", "utils/scrapDiabetes.py", "uploads/temp_diabetes.pdf"])
        parsed_data = json.loads(output.decode("utf-8"))
        st.json(parsed_data)

        if st.button("Predict"):
            result = predict_diabetes(parsed_data)
            st.success(result)
            predictions.insert_one({"type": "diabetes", "input": parsed_data, "result": result})
    except subprocess.CalledProcessError as e:
        st.error("Error extracting data from PDF.")
        st.code(e.output.decode())
