# src/model.py

import numpy as np
import cv2
from keras.models import load_model

# Load your trained deepfake detector model
def load_deepfake_model(model_path="models/deepfake_detector.h5"):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")

# Predict Real or Deepfake from face image
def predict_deepfake(model, face_path):
    try:
        img = cv2.imread(face_path)
        img = cv2.resize(img, (224, 224))  # Adjust to your model's input size
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]
        label = "Real" if prediction < 0.5 else "Deepfake"
        confidence = 1 - prediction if label == "Real" else prediction
        return label, float(confidence)
    except Exception as e:
        return "Error", 0.0
