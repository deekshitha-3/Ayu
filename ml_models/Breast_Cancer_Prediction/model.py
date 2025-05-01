from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Path to save the dummy model
model_dir = os.path.join("ml_models", "Breast_Cancer_Prediction")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "bcd_model.h5")

# Define a simple CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the dummy model
model.save(model_path)

print(f"Dummy model saved at: {model_path}")
