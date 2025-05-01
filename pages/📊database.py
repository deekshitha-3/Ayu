import streamlit as st
import pandas as pd
from utils.db import predictions

st.title("📊 Prediction History")

# Optional filter dropdown
types = ["All", "heart", "diabetes", "breast", "lung"]
selected_type = st.selectbox("Filter by Prediction Type", types)

# Build query
query = {} if selected_type == "All" else {"type": selected_type}

# Fetch records from MongoDB
records = list(predictions.find(query))

if not records:
    st.info("No prediction records found.")
else:
    # Flatten input for PDF-based records
    def flatten_record(rec):
        base = {
            "Type": rec.get("type", "").capitalize(),
            "Result": rec.get("result", "")
        }
        input_data = rec.get("input", {})

        # Handle missing input fields gracefully
        if isinstance(input_data, dict):
            for key in input_data:
                base[key] = input_data.get(key, "N/A")  # Default to "N/A" for missing values
        
        return base


    flat_data = [flatten_record(r) for r in records]
    df = pd.DataFrame(flat_data)

    # Optional: Remove MongoDB _id
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    st.dataframe(df, use_container_width=True)
