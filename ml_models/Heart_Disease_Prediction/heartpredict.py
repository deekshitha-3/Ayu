import joblib
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), "heart_disease.pkl")
model = joblib.load(model_path)

def safe_int(val, default=0):
    try:
        return int(val)
    except:
        return default

def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default

def predict_heart_disease(data: dict) -> str:
    input_data = np.array([
        safe_int(data.get('Age')),
        safe_int(data.get('Sex')),
        safe_int(data.get('Chest pain type')),
        safe_int(data.get('Resting blood pressure')),
        safe_int(data.get('Serum cholesterol in mg/dl')),
        1 if str(data.get('Fasting blood sugar > 120 mg/dl')).lower() in ['positive', 'yes', 'true', '1'] else 0,
        safe_int(data.get('Resting Electrocardiographic Results')),
        safe_int(data.get('Maximum Heart Rate Achieved')),
        safe_int(data.get('Exercise Induced Angina')),
        safe_float(data.get('Old peak')),
        safe_int(data.get('Slope of the peak exercise ST Segment')),
        safe_int(data.get('Number of major vessels (0-3) colored by fluoroscopy')),
        safe_int(data.get('Thal (Thallium Stress Test Result)')),
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    return "Suffering from Heart Disease" if prediction == 1 else "Not Suffering from Heart Disease"
