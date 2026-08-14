"""
preprocessing.py
-----------------
Facial Emotion Recognition (ANN + MediaPipe) - Capstone Project

Handles:
    - Image resizing / normalization
    - MediaPipe FaceLandmarker (Tasks API) setup
    - Blendshape feature extraction (52-D vector per face)

Why blendshapes instead of raw pixels / raw landmark coordinates?
    MediaPipe's Face Landmarker can output 52 "blendshape" scores
    (e.g. mouthSmileLeft, browDownRight, jawOpen, eyeSquintLeft ...).
    These already encode *facial muscle movement intensity*, which is
    exactly what drives an expression. Feeding these 52 numbers into a
    small ANN is far more robust and lightweight than feeding
    48x48 / 224x224 raw pixels, and avoids needing a CNN.
"""

import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Model asset handling
# ---------------------------------------------------------------------------
FACE_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task"
)
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def download_face_landmarker_model(dest_path: str = FACE_LANDMARKER_MODEL_PATH) -> str:
    """Download MediaPipe's face_landmarker.task asset if not already present."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"[preprocessing] Downloading face_landmarker.task -> {dest_path}")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, dest_path)
        print("[preprocessing] Download complete.")
    else:
        print(f"[preprocessing] Found existing model asset at {dest_path}")
    return dest_path


# ---------------------------------------------------------------------------
# FaceLandmarker builder (Tasks API - IMAGE mode, used for static images)
# ---------------------------------------------------------------------------
def build_face_landmarker(model_path: str = FACE_LANDMARKER_MODEL_PATH,
                           running_mode: str = "IMAGE") -> mp_vision.FaceLandmarker:
    """
    Build a MediaPipe FaceLandmarker in IMAGE mode (for dataset processing)
    or VIDEO/LIVE_STREAM mode (for webcam use in the Streamlit app).
    """
    if not os.path.exists(model_path):
        download_face_landmarker_model(model_path)

    mode_map = {
        "IMAGE": mp_vision.RunningMode.IMAGE,
        "VIDEO": mp_vision.RunningMode.VIDEO,
        "LIVE_STREAM": mp_vision.RunningMode.LIVE_STREAM,
    }

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mode_map[running_mode],
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def resize_image(image: np.ndarray, size=(224, 224)) -> np.ndarray:
    """Resize image. Upscaling small dataset images (e.g. FER2013's 48x48)
    materially improves MediaPipe's face-detection success rate."""
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1]."""
    return image.astype("float32") / 255.0


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure 3-channel RGB (MediaPipe requires RGB, many face datasets are grayscale)."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def augment_image(image: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Light augmentation: horizontal flip + small brightness/contrast jitter.
    Kept simple/dependency-free (cv2 + numpy only)."""
    rng = rng or np.random.default_rng()
    aug = image.copy()

    if rng.random() < 0.5:
        aug = cv2.flip(aug, 1)

    alpha = rng.uniform(0.9, 1.1)   # contrast
    beta = rng.uniform(-15, 15)     # brightness
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    return aug


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------
BLENDSHAPE_DIM = 52  # fixed output size of MediaPipe's blendshape model


def extract_blendshape_features(image_rgb: np.ndarray, landmarker: mp_vision.FaceLandmarker):
    """
    Run FaceLandmarker on an RGB image and return a (52,) numpy vector of
    blendshape scores, or None if no face was detected.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_blendshapes:
        return None

    scores = [c.score for c in result.face_blendshapes[0]]
    return np.array(scores, dtype="float32")


def preprocess_and_extract(image_path: str, landmarker: mp_vision.FaceLandmarker,
                            target_size=(224, 224)):
    """End-to-end: load -> resize -> RGB -> extract blendshapes."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = resize_image(image, target_size)
    image_rgb = to_rgb(image)
    return extract_blendshape_features(image_rgb, landmarker)
