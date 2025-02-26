# Ayu-Medical Diagnostic System

## Brief Summary

The Medical Diagnostic System is an AI-driven application designed to predict various diseases with high accuracy. Using machine learning models, the system helps diagnose conditions like Diabetes, Breast Cancer, Kidney Disease, and Heart Disease. The models have been trained on relevant datasets and saved for deployment, enabling automated, efficient, and reliable medical predictions.

### **Technologies Used:**

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Jupyter Notebooks  

## Dataset Links

All the datasets were used from Kaggle:

- [Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  
- [Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)  
- [Heart Disease Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)  
- [Kidney Disease Dataset](https://www.kaggle.com/mansoordaku/ckdisease)  

## Project Structure

```
Medical_Diagnostic_Project/
│── model/               # Contains trained .pkl model files
│── notebook/            # Jupyter notebooks for training models
│── README.md            # Project documentation
│── requirements.txt     # Dependencies
```

## Installation

### Prerequisites

Ensure you have Python installed (>=3.8). Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

To retrain any model, navigate to the `notebook/` directory and execute the corresponding Jupyter notebook.

### Predicting Diseases

Load the trained models from the `model/` directory and use them to predict diseases:

```python
import pickle
import numpy as np

# Load model
with open('model/diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input features (replace with actual values)
input_data = np.array([[5, 116, 74, 0, 0, 25.6, 0.201, 30]])
prediction = model.predict(input_data)
print("Diabetes Prediction:", prediction)
```

## References and Appendices

- Public medical datasets from Kaggle and UCI Machine Learning Repository  
- Research papers on AI in healthcare diagnostics  

