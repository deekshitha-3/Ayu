import joblib
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def predict_diabetes(data: dict) -> str:
    try:
        # Make sure all expected keys exist and are valid
        values = [
            float(data.get("Pregnancies", 0)),
            float(data.get("Glucose", 0)),
            float(data.get("BloodPressure", 0)),
            float(data.get("SkinThickness", 0)),
            float(data.get("Insulin", 0)),
            float(data.get("BMI", 0.0)),
            float(data.get("DiabetesPedigreeFunction", 0.0)),
            float(data.get("Age", 0)),
        ]

        input_data = np.array(values).reshape(1, -1)
        std_data = scaler.transform(input_data)
        prediction = model.predict(std_data)[0]

        return "Suffering from Diabetes" if prediction == 1 else "Not Suffering from Diabetes"

    except Exception as e:
        return f"Error during prediction: {str(e)}"
