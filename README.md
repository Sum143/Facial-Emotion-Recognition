# Facial Emotion Recognition — ANN + MediaPipe

**Innomatics Research Labs — Image Data Based ANN Capstone Project**

Classifies a person's facial emotion (angry, disgust, fear, happy, neutral, sad, surprise)
from an image or webcam frame, using an Artificial Neural Network trained on
**MediaPipe FaceLandmarker blendshape scores** — a 52-dimensional feature vector
that directly encodes facial-muscle movement, instead of raw pixels or a CNN.

---

## 1. Problem Statement

| Item | Description |
|---|---|
| **Project Title** | Facial Emotion Recognition using ANN + MediaPipe |
| **Problem Statement** | Automatically classify a person's facial emotion from an image or webcam frame. |
| **Business Objective** | Power emotion-aware applications — engagement tracking, sentiment kiosks, driver monitoring — that react to a user's real-time emotional state. |
| **Expected Output** | Predicted emotion label + confidence score. |
| **Target Variable** | `emotion` (7 classes) |
| **Input Features** | 52 MediaPipe face-blendshape scores |

## 2. Dataset

Use any FER-style dataset in **folder-per-class** layout, e.g. the Kaggle
"Face Expression Recognition" / FER2013 folder version:

```
dataset/
├── train/
│   ├── angry/*.jpg
│   ├── disgust/*.jpg
│   ├── fear/*.jpg
│   ├── happy/*.jpg
│   ├── neutral/*.jpg
│   ├── sad/*.jpg
│   └── surprise/*.jpg
└── test/
    └── ... (same structure)
```

> **Note on image size:** FER2013 images are only 48×48 grayscale. The
> pipeline upscales every image to 224×224 and converts to RGB before
> running MediaPipe, which substantially improves face-detection success.
> Some low-quality/cropped images will still be dropped automatically if no
> face is detected — this is logged during Step 5/6.

## 3. Pipeline (matches the capstone's 20 steps)

| Step | What happens | Where |
|---|---|---|
| 1–2 | Problem statement, dataset sourcing | `train_model.py` (docstring/constants) |
| 3 | Dataset import, shape/dtype/missing/duplicate/class-balance checks | `load_dataset_index()` |
| 4 | EDA — count plot, pie chart, observations | `run_eda()` |
| 5 | Resize, RGB conversion, (optional) augmentation | `preprocessing.py` |
| 6 | Feature extraction via MediaPipe blendshapes | `build_feature_matrix()` |
| 7 | X / y separation + label encoding | `separate_input_output()` |
| 8 / 8a | 80:20 stratified split + `StandardScaler` | `split_and_scale()` |
| 9 | 3 baseline ANN architectures compared | `train_baseline_models()` |
| 10 | Optuna hyperparameter search (layers, units, dropout, LR, activation) | `tune_with_optuna()` |
| 11 | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix | `evaluate_model()` |
| 12 | Save model (`.keras`), scaler, label encoder | `save_artifacts()` |
| 12a | MediaPipe FaceLandmarker Tasks-API integration | `preprocessing.py` |
| 13 | Streamlit app — Home / Description / Prediction (Upload + Webcam) | `app.py` |
| 14 | Deployment (Render / Streamlit Cloud / Hugging Face) | see below |

## 4. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Train the model

```bash
python train_model.py --dataset_dir dataset --epochs 40 --trials 25
```

This downloads MediaPipe's `face_landmarker.task` asset automatically into
`models/` the first time it runs, extracts features (cached to
`dataset/features_cache.npz` so re-runs are fast), trains baseline ANNs,
runs Optuna tuning, evaluates the best model, and saves:

- `models/emotion_ann_model.keras`
- `models/scaler.pkl`
- `models/label_encoder.pkl`
- `models/metrics.json`
- `screenshots/eda_class_distribution.png`, `eda_class_pie.png`, `confusion_matrix.png`

## 6. Run the app

```bash
streamlit run app.py
```

## 7. Folder Structure

```
Facial_Emotion_Recognition_ANN_MediaPipe/
├── app.py                 # Streamlit application
├── train_model.py         # Full training pipeline (steps 1-12)
├── preprocessing.py        # Image + MediaPipe feature extraction utilities
├── predictor.py            # Inference wrapper used by app.py
├── requirements.txt
├── README.md
├── models/                 # Saved model, scaler, label encoder (generated)
├── dataset/                # Place train/ and test/ folders here
├── screenshots/             # EDA plots, confusion matrix, app screenshots
└── notebook/
    └── Facial_Emotion_Recognition.ipynb   # Exploratory / step-wise notebook
```

## 8. Deployment

Recommended: **Render** (Docker or native Python web service running
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`).
Also works on Streamlit Community Cloud or Hugging Face Spaces.

Before deploying:
- [ ] Fix any deployment-environment errors
- [ ] Verify the saved model loads correctly at startup
- [ ] Test a real prediction end-to-end
- [ ] Share the deployment URL

## 9. Learning Outcomes

- Building an ANN classifier on top of MediaPipe's Tasks API blendshape output
  instead of raw pixels/CNNs
- Handling face-detection failures gracefully in a data pipeline
- Hyperparameter optimization with Optuna
- Packaging a full ML project into a deployable Streamlit app

## 10. Challenges Faced / Future Scope

- **Challenges:** low-resolution dataset images reduce MediaPipe's face-detection
  success rate; class imbalance (e.g. `disgust` is typically underrepresented).
- **Future Scope:** add class-weighting or SMOTE for imbalance, support
  multi-face detection, add real-time video-stream inference (`LIVE_STREAM`
  running mode) instead of single-frame webcam capture.
