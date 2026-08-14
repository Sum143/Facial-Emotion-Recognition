"""
train_model.py
---------------
Facial Emotion Recognition using ANN + MediaPipe Blendshape Features
Innomatics Research Labs - Image Data Based ANN Capstone Project

Follows the capstone steps 1 - 12:
    1  Problem Statement
    2  Dataset Collection
    3  Dataset Import
    4  Exploratory Data Analysis
    5  Data Preprocessing
    6  Feature Extraction (MediaPipe blendshapes)
    7  Input / Output Separation
    8  Train Test Split
    8a Feature Scaling
    9  Model Building (baseline ANNs)
    10 Hyperparameter Tuning (Optuna)
    11 Model Evaluation
    12 Model Saving

Run:
    python train_model.py --dataset_dir dataset --epochs 40 --trials 25
"""

import os
import argparse
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import optuna

from preprocessing import (
    build_face_landmarker, preprocess_and_extract, BLENDSHAPE_DIM,
)

# ===========================================================================
# STEP 1 : PROBLEM STATEMENT
# ===========================================================================
PROJECT_TITLE = "Facial Emotion Recognition using ANN + MediaPipe"
PROBLEM_STATEMENT = (
    "Automatically classify a person's facial emotion from an image or "
    "webcam frame into one of several emotion classes."
)
BUSINESS_OBJECTIVE = (
    "Enable emotion-aware applications (e.g. customer sentiment kiosks, "
    "e-learning engagement tracking, driver monitoring) to react to a "
    "user's real-time emotional state."
)
EXPECTED_OUTPUT = "Predicted emotion label with a confidence score."
TARGET_VARIABLE = "emotion (categorical: angry, disgust, fear, happy, neutral, sad, surprise)"
INPUT_FEATURES = "52 MediaPipe face-blendshape scores extracted from the input image."

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ===========================================================================
# STEP 3 : DATASET IMPORT
# ===========================================================================
def load_dataset_index(dataset_dir: str, split: str = "train") -> pd.DataFrame:
    """
    Expects a folder-per-class layout (standard FER2013 Kaggle "folder" format):

        dataset/
            train/
                angry/*.jpg
                disgust/*.jpg
                ...
            test/
                angry/*.jpg
                ...
    """
    split_dir = os.path.join(dataset_dir, split)
    rows = []
    for emotion in sorted(os.listdir(split_dir)):
        class_dir = os.path.join(split_dir, emotion)
        if not os.path.isdir(class_dir):
            continue
        for img_path in glob.glob(os.path.join(class_dir, "*")):
            rows.append({"filepath": img_path, "emotion": emotion})
    df = pd.DataFrame(rows)
    print(f"[Step 3] Loaded {split} index: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[Step 3] Columns: {list(df.columns)}")
    print(f"[Step 3] Missing values:\n{df.isnull().sum()}")
    print(f"[Step 3] Duplicate rows: {df.duplicated().sum()}")
    print(f"[Step 3] Class distribution:\n{df['emotion'].value_counts()}")
    return df


# ===========================================================================
# STEP 4 : EXPLORATORY DATA ANALYSIS
# ===========================================================================
def run_eda(df: pd.DataFrame, out_dir: str = "screenshots"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="emotion", order=df["emotion"].value_counts().index)
    plt.title("Class Distribution - Emotion Counts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_class_distribution.png"))
    plt.close()

    dist = df["emotion"].value_counts(normalize=True) * 100
    plt.figure(figsize=(6, 6))
    plt.pie(dist.values, labels=dist.index, autopct="%1.1f%%")
    plt.title("Emotion Class Proportion")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_class_pie.png"))
    plt.close()

    print(f"[Step 4] EDA plots saved to {out_dir}/")
    print("[Step 4] Observations: class distribution is typically imbalanced for "
          "FER-style datasets (e.g. 'disgust' is usually the rarest class) — "
          "consider class_weight or stratified sampling during training.")


# ===========================================================================
# STEPS 5 & 6 : PREPROCESSING + FEATURE EXTRACTION
# ===========================================================================
def build_feature_matrix(df: pd.DataFrame, target_size=(224, 224), cache_path: str = None):
    """
    For every image: resize -> RGB -> MediaPipe blendshape extraction.
    Images where MediaPipe fails to detect a face are dropped (logged).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[Step 5/6] Loading cached features from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        return cached["X"], cached["y"]

    landmarker = build_face_landmarker(running_mode="IMAGE")

    features, labels = [], []
    dropped = 0
    for i, row in df.iterrows():
        vec = preprocess_and_extract(row["filepath"], landmarker, target_size)
        if vec is None:
            dropped += 1
            continue
        features.append(vec)
        labels.append(row["emotion"])
        if i % 500 == 0:
            print(f"[Step 5/6] Processed {i}/{len(df)} images...")

    print(f"[Step 5/6] Feature extraction complete. "
          f"{len(features)} usable samples, {dropped} dropped (no face detected).")

    X = np.vstack(features) if features else np.empty((0, BLENDSHAPE_DIM))
    y = np.array(labels)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"[Step 5/6] Cached features to {cache_path}")

    return X, y


# ===========================================================================
# STEP 7 : INPUT / OUTPUT SEPARATION
# ===========================================================================
def separate_input_output(X: np.ndarray, y: np.ndarray):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"[Step 7] X shape: {X.shape}, y shape: {y_encoded.shape}")
    print(f"[Step 7] Classes: {list(label_encoder.classes_)}")
    return X, y_encoded, label_encoder


# ===========================================================================
# STEP 8 / 8a : TRAIN-TEST SPLIT + SCALING
# ===========================================================================
def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"[Step 8] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ===========================================================================
# STEP 9 : MODEL BUILDING (baseline ANNs)
# ===========================================================================
def build_ann(input_dim: int, num_classes: int, hidden_layers=(128, 64),
              dropout=0.3, learning_rate=1e-3, activation="relu"):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_baseline_models(X_train, y_train, X_test, y_test, num_classes, epochs=30):
    """Train a couple of baseline ANN architectures and compare validation accuracy."""
    architectures = {
        "small_ann":  dict(hidden_layers=(64, 32), dropout=0.2, learning_rate=1e-3),
        "medium_ann": dict(hidden_layers=(128, 64), dropout=0.3, learning_rate=1e-3),
        "deep_ann":   dict(hidden_layers=(256, 128, 64), dropout=0.4, learning_rate=5e-4),
    }

    results = {}
    early_stop = callbacks.EarlyStopping(patience=6, restore_best_weights=True)

    for name, params in architectures.items():
        print(f"[Step 9] Training baseline: {name} ({params})")
        model = build_ann(X_train.shape[1], num_classes, **params)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs, batch_size=32, verbose=0,
            callbacks=[early_stop],
        )
        val_acc = max(history.history["val_accuracy"])
        results[name] = val_acc
        print(f"[Step 9] {name} best val_accuracy = {val_acc:.4f}")

    best_name = max(results, key=results.get)
    print(f"[Step 9] Baseline comparison: {results}")
    print(f"[Step 9] Best baseline architecture: {best_name}")
    return best_name, results


# ===========================================================================
# STEP 10 : HYPERPARAMETER TUNING (OPTUNA)
# ===========================================================================
def tune_with_optuna(X_train, y_train, X_test, y_test, num_classes, n_trials=25, epochs=40):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = tuple(
            trial.suggest_categorical(f"units_l{i}", [32, 64, 128, 256])
            for i in range(n_layers)
        )
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        activation = trial.suggest_categorical("activation", ["relu", "elu", "tanh"])

        model = build_ann(
            X_train.shape[1], num_classes,
            hidden_layers=hidden_layers, dropout=dropout,
            learning_rate=learning_rate, activation=activation,
        )
        early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs, batch_size=32, verbose=0,
            callbacks=[early_stop],
        )
        return max(history.history["val_accuracy"])

    study = optuna.create_study(direction="maximize", study_name="fer_ann_tuning")
    study.optimize(objective, n_trials=n_trials)

    print(f"[Step 10] Best trial value (val_accuracy): {study.best_value:.4f}")
    print(f"[Step 10] Best params: {study.best_params}")
    return study.best_params, study.best_value


# ===========================================================================
# STEP 11 : MODEL EVALUATION
# ===========================================================================
def evaluate_model(model, X_test, y_test, label_encoder, out_dir="screenshots"):
    os.makedirs(out_dir, exist_ok=True)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")

    print(f"[Step 11] Accuracy : {acc:.4f}")
    print(f"[Step 11] Precision: {prec:.4f}")
    print(f"[Step 11] Recall   : {rec:.4f}")
    print(f"[Step 11] F1 Score : {f1:.4f}")
    print(f"[Step 11] ROC-AUC  : {auc:.4f}")
    print("[Step 11] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    return metrics


# ===========================================================================
# STEP 12 : MODEL SAVING
# ===========================================================================
def save_artifacts(model, scaler, label_encoder, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "emotion_ann_model.keras"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(out_dir, "label_encoder.pkl"))
    print(f"[Step 12] Saved model, scaler, and label encoder to {out_dir}/")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description=PROJECT_TITLE)
    parser.add_argument("--dataset_dir", type=str, default="dataset",
                         help="Folder containing train/ and test/ subfolders per class")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--trials", type=int, default=25, help="Optuna trials")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    print(f"=== {PROJECT_TITLE} ===")
    print(f"Problem Statement : {PROBLEM_STATEMENT}")
    print(f"Business Objective: {BUSINESS_OBJECTIVE}")
    print(f"Target Variable   : {TARGET_VARIABLE}")
    print(f"Input Features    : {INPUT_FEATURES}\n")

    # Step 3
    train_df = load_dataset_index(args.dataset_dir, split="train")
    test_df = load_dataset_index(args.dataset_dir, split="test")
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Step 4
    run_eda(full_df)

    # Steps 5 & 6
    size = (args.image_size, args.image_size)
    X, y = build_feature_matrix(full_df, target_size=size,
                                 cache_path="dataset/features_cache.npz")

    # Step 7
    X, y_encoded, label_encoder = separate_input_output(X, y)

    # Step 8 / 8a
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y_encoded)
    num_classes = len(label_encoder.classes_)

    # Step 9
    best_name, baseline_results = train_baseline_models(
        X_train, y_train, X_test, y_test, num_classes, epochs=args.epochs
    )

    # Step 10
    best_params, best_val_acc = tune_with_optuna(
        X_train, y_train, X_test, y_test, num_classes,
        n_trials=args.trials, epochs=args.epochs,
    )

    # Rebuild best model with tuned params
    n_layers = best_params["n_layers"]
    hidden_layers = tuple(best_params[f"units_l{i}"] for i in range(n_layers))
    final_model = build_ann(
        X_train.shape[1], num_classes,
        hidden_layers=hidden_layers,
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        activation=best_params["activation"],
    )
    early_stop = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    final_model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=args.epochs, batch_size=32, callbacks=[early_stop], verbose=1,
    )

    # Step 11
    metrics = evaluate_model(final_model, X_test, y_test, label_encoder)

    # Step 12
    save_artifacts(final_model, scaler, label_encoder)

    with open("models/metrics.json", "w") as f:
        json.dump({"baseline_results": baseline_results,
                    "best_params": best_params,
                    "final_metrics": metrics}, f, indent=2)

    print("\n[Pipeline] Training complete. Artifacts saved in models/.")


if __name__ == "__main__":
    main()
