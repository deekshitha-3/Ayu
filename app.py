import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="PredictiX", page_icon="🩺", layout="centered")

# Title and Intro
st.title("🩺 PredictiX")
st.subheader("AI-Based Medical Diagnosis System")

st.markdown("""
PredictiX is a unified and intelligent diagnosis system designed to assist in the early detection of critical diseases using Machine Learning and Deep Learning.
""")

# Key diseases list
st.markdown("### 🧬 Diseases Covered")
st.markdown("""
- ❤️ **Heart Disease**
- 🩸 **Diabetes**
- 🎗️ **Breast Cancer**
- 🫁 **Lung Cancer**
""")

# Model Accuracy Table
st.markdown("---")
st.markdown("### 📊 Model Accuracies")

accuracy_data = {
    "Disease": ["Diabetes", "Breast Cancer", "Lung Cancer", "Heart Disease"],
    "Accuracy": ["98.25%", "98.25%", "96%", "85.25%"]
}
df = pd.DataFrame(accuracy_data)
st.table(df)

# Features
st.markdown("---")
st.markdown("### 🚀 Key Features")
st.markdown("""
- 📄 Upload medical **PDF reports** for auto-filled predictions (via Regex + ML)
- 🖼️ Upload **medical images** for CNN-based detection (Breast & Lung Cancer)
- 🧠 Integrated ML models with high accuracy
- 🗃️ Results stored securely in **MongoDB**
""")

# Navigation
st.markdown("---")
st.markdown("### 👉 Get Started")
st.markdown("Use the **sidebar** to select a disease and begin your prediction.")
