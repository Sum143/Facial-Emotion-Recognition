"""
app.py
------
Streamlit application for the Facial Emotion Recognition (ANN + MediaPipe)
capstone project.

Pages:
    - Home
    - Project Description
    - Prediction (Upload Image / Webcam)

Run:
    streamlit run app.py
"""

import numpy as np
import cv2
import streamlit as st
import pandas as pd

from predictor import get_predictor
from train_model import (
    PROJECT_TITLE, PROBLEM_STATEMENT, BUSINESS_OBJECTIVE,
    EXPECTED_OUTPUT, TARGET_VARIABLE, INPUT_FEATURES, EMOTION_CLASSES,
)

# ---------------------------------------------------------------------------
# Page config + dark theme styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🙂",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1b1f2b 0%, #0d0f16 65%);
        color: #e8e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #12141c;
        border-right: 1px solid #2a2d3a;
    }
    h1, h2, h3 { color: #f5f5ff; }
    .metric-card {
        background: linear-gradient(145deg, #1c2030, #14161f);
        border: 1px solid #2c2f40;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .emotion-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        background: linear-gradient(135deg, #6a5cff, #ff5ca8);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6a5cff, #ff5ca8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1.2em;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] { color: #f5f5ff; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

EMOTION_EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🙂 Emotion Recognition")
page = st.sidebar.radio("Navigate", ["Home", "Project Description", "Prediction"])

st.sidebar.markdown("---")
st.sidebar.caption("Innomatics Research Labs — ANN Capstone Project")


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("Facial Emotion Recognition")
    st.subheader("ANN + MediaPipe Blendshape Features")
    st.write(
        "This application detects a face in an uploaded image or a webcam "
        "snapshot, extracts 52 MediaPipe facial-blendshape scores, and "
        "classifies the expression into one of seven emotions using a "
        "trained Artificial Neural Network."
    )

    cols = st.columns(len(EMOTION_CLASSES))
    for c, emotion in zip(cols, EMOTION_CLASSES):
        with c:
            st.markdown(
                f"<div class='metric-card' style='text-align:center;'>"
                f"<div style='font-size:2rem'>{EMOTION_EMOJI.get(emotion,'')}</div>"
                f"<div>{emotion.capitalize()}</div></div>",
                unsafe_allow_html=True,
            )

    st.info("Use the sidebar to explore the project description or try a live prediction.")


# ---------------------------------------------------------------------------
# PROJECT DESCRIPTION
# ---------------------------------------------------------------------------
elif page == "Project Description":
    st.title("Project Description")

    st.markdown("### Problem Statement")
    st.write(PROBLEM_STATEMENT)

    st.markdown("### Business Objective")
    st.write(BUSINESS_OBJECTIVE)

    st.markdown("### Expected Output")
    st.write(EXPECTED_OUTPUT)

    st.markdown("### Target Variable")
    st.write(TARGET_VARIABLE)

    st.markdown("### Input Features")
    st.write(INPUT_FEATURES)

    st.markdown("### Pipeline Summary")
    st.markdown(
        """
        1. **Dataset** — FER-style folder dataset (`train/<emotion>/*.jpg`)
        2. **Preprocessing** — resize, RGB conversion, optional augmentation
        3. **Feature Extraction** — MediaPipe FaceLandmarker → 52 blendshape scores
        4. **Model** — Artificial Neural Network (Dense + BatchNorm + Dropout)
        5. **Tuning** — Optuna hyperparameter search over depth, width, dropout, LR
        6. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
        7. **Deployment** — This Streamlit app, served on Render / Streamlit Cloud
        """
    )

    st.markdown("### Technologies Used")
    st.write("Python, TensorFlow/Keras, MediaPipe Tasks API, OpenCV, scikit-learn, Optuna, Streamlit")


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
elif page == "Prediction":
    st.title("Predict Emotion")
    st.caption("Upload an image or capture one from your webcam.")

    tab_upload, tab_webcam = st.tabs(["📁 Upload Image", "📷 Webcam"])

    image_bgr = None

    with tab_upload:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=350)

    with tab_webcam:
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is not None:
        if st.button("🔮 Predict Emotion"):
            with st.spinner("Extracting facial features and running the model..."):
                try:
                    predictor = get_predictor()
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()

                result = predictor.predict_from_bgr(image_bgr)

            if result is None:
                st.warning("No face detected. Please try a clearer, front-facing photo.")
            else:
                emoji = EMOTION_EMOJI.get(result["label"], "")
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<span class='emotion-pill'>{emoji} {result['label'].capitalize()}</span>"
                    f"</div>", unsafe_allow_html=True,
                )
                st.metric("Confidence Score", f"{result['confidence']*100:.1f}%")

                st.markdown("#### Class Probabilities")
                prob_df = pd.DataFrame(
                    sorted(result["probabilities"].items(), key=lambda x: -x[1]),
                    columns=["Emotion", "Probability"],
                )
                st.bar_chart(prob_df.set_index("Emotion"))

    st.markdown("---")
    st.markdown("#### User Instructions")
    st.markdown(
        """
        - Use a **well-lit, front-facing** photo for best accuracy.
        - Only **one face** is analyzed per image.
        - If "No face detected" appears, move closer to the camera or improve lighting.
        """
    )
