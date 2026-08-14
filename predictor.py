"""
predictor.py
------------
Loads the trained ANN model, StandardScaler, and LabelEncoder, and exposes
a single `predict_emotion()` function used by app.py (Streamlit) or any
other client script.
"""

import os
import numpy as np
import joblib
import tensorflow as tf

from preprocessing import (
    build_face_landmarker, resize_image, to_rgb, extract_blendshape_features,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_ann_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


class EmotionPredictor:
    """Wraps model + scaler + label encoder + MediaPipe landmarker."""

    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH,
                 encoder_path=ENCODER_PATH, image_size=(224, 224)):
        missing = [p for p in [model_path, scaler_path, encoder_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Missing trained artifact(s): {missing}. "
                f"Run train_model.py first to generate models/."
            )

        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.image_size = image_size
        self.landmarker = build_face_landmarker(running_mode="IMAGE")

    def predict_from_bgr(self, image_bgr: np.ndarray):
        """
        image_bgr: an OpenCV-style BGR numpy array (as read by cv2.imread
        or captured from a webcam / st.camera_input).

        Returns:
            dict with keys: label, confidence, probabilities (dict per class)
            or None if no face was detected in the frame.
        """
        image = resize_image(image_bgr, self.image_size)
        image_rgb = to_rgb(image)
        features = extract_blendshape_features(image_rgb, self.landmarker)

        if features is None:
            return None

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probs = self.model.predict(features_scaled, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

        prob_dict = {
            cls: float(p) for cls, p in zip(self.label_encoder.classes_, probs)
        }

        return {"label": label, "confidence": confidence, "probabilities": prob_dict}


# Convenience singleton loader (so Streamlit can cache it easily)
_predictor_instance = None


def get_predictor() -> EmotionPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EmotionPredictor()
    return _predictor_instance
