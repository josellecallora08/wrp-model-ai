import os, random, io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils.features import load_fixed, extract_features, augment_once

# -------------------------------
# Config (mirrors main.py)
# -------------------------------
CSV_PATH = "labels.csv"
AUDIO_DIR = "data/audio"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "audio_clf_svm.joblib")

SR = 16000
DURATION = 10.0
AUG_PER_TRAIN_CLIP = 5

LABEL_ORDER = ["very bad", "bad", "good", "very good"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_ORDER)}

# -------------------------------
# Helper Functions
# -------------------------------


def resolve_audio_path(audio_dir: str, file_name: str):
    """Try exact name; if not found, try by stem with common extensions."""
    import pathlib
    from urllib.parse import unquote
    exts = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    name = unquote(str(file_name).strip().strip('"').strip("'"))
    exact = os.path.join(audio_dir, name)
    if os.path.isfile(exact):
        return exact
    stem = pathlib.Path(name).stem
    for ext in exts:
        cand = os.path.join(audio_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None

def predict_audio_files(uploaded_files):
    """Run prediction on uploaded audio files using saved model."""
    if not os.path.isfile(MODEL_PATH):
        st.error("❌ Missing trained model. Please train first in the Training tab.")
        return pd.DataFrame()

    # Load trained model bundle
    bundle = load(MODEL_PATH)
    model = bundle["model"]
    sr = bundle["sr"]
    duration = bundle["duration"]
    id2 = bundle["id_to_label"]

    results = []
    os.makedirs(AUDIO_DIR, exist_ok=True)

    for f in uploaded_files:
        tmp_path = os.path.join(AUDIO_DIR, f.name)
        with open(tmp_path, "wb") as out_f:
            out_f.write(f.getbuffer())

        predicted_label, pred_prob = None, None

        try:
            ysig = load_fixed(tmp_path, sr=sr, duration=duration)
            x = extract_features(ysig, sr=sr).reshape(1, -1)

            # Predict label
            pred_id = int(model.predict(x)[0])
            predicted_label = id2[pred_id]

            # Confidence
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(x)[0]
                pred_prob = float(np.max(prob))

        except Exception as e:
            st.warning(f"⚠️ Failed processing {f.name}: {e}")

        results.append({
            "file_name": f.name,
            "predicted_label": predicted_label,
            "confidence": pred_prob,
            "audio_path": tmp_path
        })

    df = pd.DataFrame(results)
    return df


def save_training_data(audio_files_dict):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    rows = []
    for label, files in audio_files_dict.items():
        if files:
            for f in files:
                file_path = os.path.join(AUDIO_DIR, f.name)
                with open(file_path, "wb") as out_f:
                    out_f.write(f.getbuffer())
                rows.append({"file_name": f.name, "label": label.lower()})
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    return df

from sklearn.model_selection import train_test_split

def train_pipeline():
    df = pd.read_csv(CSV_PATH)
    rows = []
    for _, r in df.iterrows():
        fpath = os.path.join(AUDIO_DIR, r["file_name"])
        if not os.path.isfile(fpath):
            continue
        try:
            # Original audio
            ysig = load_fixed(fpath, sr=SR, duration=DURATION)
            x = extract_features(ysig, sr=SR)
            rows.append({"x": x, "y": LABEL_TO_ID[r["label"].strip().lower()], "path": fpath})

            # Augmentations
            for _ in range(AUG_PER_TRAIN_CLIP):
                y_aug = augment_once(ysig, sr=SR)
                x_aug = extract_features(y_aug, sr=SR)
                rows.append({"x": x_aug, "y": LABEL_TO_ID[r["label"].strip().lower()], "path": fpath})

        except Exception as e:
            print(f"[WARN] Failed {fpath}: {e}")

    if len(rows) < 10: # Need enough data to split
        st.error(f"❌ Not enough data to train and validate. Found {len(rows)} usable feature sets. Need at least 10.")
        return None, None, None

    X = np.stack([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])

    # Split data for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, C=10, gamma="scale"))
    ])

    # Train on the training set
    model = pipe.fit(X_train, y_train)

    # Evaluate on the validation set
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=LABEL_ORDER, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)

    # Re-train on all data for the final model
    model_final = pipe.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump({
        "model": model_final,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "sr": SR,
        "duration": DURATION
    }, MODEL_PATH)

    return acc, report, cm, y_val, y_pred

def predict_audio(files):
    # Placeholder until real model is loaded
    results = []
    labels = ["Very Good", "Good", "Bad", "Very Bad"]
    for f in files:
        acc = np.round(np.random.uniform(0.7, 0.99), 2)
        pred_label = random.choice(labels)
        results.append({"file": f.name, "predicted_label": pred_label, "accuracy": acc})
    return results



# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Audio Model UI", layout="wide")
st.title("🎙️ Audio Classification UI")

tab1, tab2, tab3 = st.tabs(["🔍 Test Model (Predict)", "🛠️ Train Model", "📊 Dashboard"])

# -------------------------------
# Tab 1: Test Model
# -------------------------------
with tab1:
    st.header("Test Model (Predict)")
    uploaded_files = st.file_uploader(
        "Upload one or multiple audio files",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        accept_multiple_files=True,
        key="pred_uploader"
    )

    col1, col2 = st.columns(2)
    with col1:
        run_prediction = st.button("Run Prediction")

    if run_prediction:
        if uploaded_files:
            df_results = predict_audio_files(uploaded_files)

            if not df_results.empty:
                # st.subheader("Prediction Results")
                # st.dataframe(df_results)

                # --- Dashboard ---
                st.subheader("📊 Prediction Dashboard")

                total_processed = len(df_results)
                total_predicted = df_results["predicted_label"].notna().sum()
                avg_conf = 0
                if "confidence" in df_results and df_results["confidence"].notna().any():
                    avg_conf = df_results["confidence"].dropna().mean()
                
                label_counts = df_results["predicted_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)

                # Add custom CSS for borders around metrics
                st.markdown("""
                <style>
                div[data-testid="metric-container"] {
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 10px;
                    padding: 15px;
                    margin: 5px 0;
                }
                </style>
                """, unsafe_allow_html=True)

                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files Processed", total_processed)
                with col2:
                    st.metric("Successfully Predicted", f"{total_predicted}/{total_processed}")
                with col3:
                    st.metric("Average Confidence", f"{avg_conf:.2%}")

                st.subheader("Prediction Counts")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Very Good", label_counts.get("very good", 0))
                with col2:
                    st.metric("Good", label_counts.get("good", 0))
                with col3:
                    st.metric("Bad", label_counts.get("bad", 0))
                with col4:
                    st.metric("Very Bad", label_counts.get("very bad", 0))

                if total_predicted > 0:
                    st.subheader("Detailed Predictions")
                    for _, row in df_results.iterrows():
                        st.markdown("---")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**File:** `{row['file_name']}`")
                            st.write(f"**Prediction:** {row['predicted_label']}")
                            if pd.notna(row['confidence']):
                                st.write(f"**Confidence:** {row['confidence']:.2%}")
                        with col2:
                            audio_path = row.get("audio_path")
                            if audio_path and os.path.isfile(audio_path):
                                st.audio(audio_path)
                            else:
                                st.warning("Audio file not found.")
        else:
            st.warning("Please upload audio files first.")


# -------------------------------
# Tab 2: Train Model
# -------------------------------
with tab2:
    st.header("Train a Model")

    audio_files_dict = {
        "Very Good": st.file_uploader("Upload 'Very Good' audios", type=["wav","mp3"], accept_multiple_files=True, key="vg"),
        "Good": st.file_uploader("Upload 'Good' audios", type=["wav","mp3"], accept_multiple_files=True, key="g"),
        "Bad": st.file_uploader("Upload 'Bad' audios", type=["wav","mp3"], accept_multiple_files=True, key="b"),
        "Very Bad": st.file_uploader("Upload 'Very Bad' audios", type=["wav","mp3"], accept_multiple_files=True, key="vb"),
    }

    if st.button("🚀 Train Model"):
        df = save_training_data(audio_files_dict)
        if df.empty:
            st.warning("No audio files were uploaded. Please upload files for at least one category.")
            st.stop()

        st.success("✅ Audio files saved and labels.csv generated!")
        st.dataframe(df)

        with st.spinner("Training model... This may take a moment."):
            results = train_pipeline()

        if results and results[0] is not None:
            acc, report, cm, y_val, y_pred = results
            st.subheader("📊 Validation Results")
            st.metric("Validation Accuracy", f"{acc:.2%}")
            
            st.text("Classification Report:")
            st.json(report)
            
            st.text("Confusion Matrix (rows=true, cols=pred):")
            st.dataframe(pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER))

            # Display True vs. Predicted Labels
            st.subheader("Validation Set Predictions")
            val_labels = [ID_TO_LABEL[i] for i in y_val]
            pred_labels = [ID_TO_LABEL[i] for i in y_pred]
            df_preds = pd.DataFrame({
                "True Label": val_labels,
                "Predicted Label": pred_labels
            })
            st.dataframe(df_preds)

            st.success("✅ Model trained and saved!")

            # --- Explanation Section ---
            with st.expander("How to Interpret These Results"):
                st.markdown("""
                Here’s a quick guide to understanding the model's performance metrics:

                - **Validation Accuracy**: This is the main score. It tells you the overall percentage of audio files in the validation set that the model labeled correctly. A higher number is better.

                - **Classification Report**: This gives you a detailed breakdown of performance for each category.
                    - **Precision**: Of all the files the model *predicted* as "Good", how many were actually "Good"? High precision means the model is trustworthy when it makes a prediction for that category.
                    - **Recall**: Of all the files that were *actually* "Good", how many did the model correctly identify? High recall means the model is good at finding all instances of a category.
                    - **F1-Score**: A combined score of Precision and Recall. It's useful for comparing the overall performance of different categories.

                - **Confusion Matrix**: This table shows you exactly where the model is getting confused.
                    - The **rows** represent the *true* labels.
                    - The **columns** represent the *predicted* labels.
                    - For example, if the row for "Good" has a `5` in the "Very Good" column, it means the model incorrectly labeled 5 "Good" recordings as "Very Good". The numbers on the diagonal (top-left to bottom-right) are correct predictions.

                - **Validation Set Predictions Table**: This table shows the raw, file-by-file comparison of the true label versus what the model predicted for the unseen validation data. It's a direct way to see individual successes and failures.
                """)
        else:
            st.error("❌ Model training failed.")

# -------------------------------
# Tab 3: Dashboard
# -------------------------------
with tab3:
    st.header("Model Dashboard")

    categories = ["Very Good", "Good", "Bad", "Very Bad"]
    counts = [np.random.randint(5, 20) for _ in categories]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Audio Files per Category")
        df_counts = pd.DataFrame({"Category": categories, "Count": counts})
        st.bar_chart(df_counts.set_index("Category"))

    with col2:
        st.subheader("Accuracy Trend (Dummy)")
        epochs = list(range(1, 11))
        acc = np.random.uniform(0.7, 0.95, size=10)
        fig, ax = plt.subplots()
        ax.plot(epochs, acc, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training Accuracy Trend")
        st.pyplot(fig)

    st.subheader("Confusion Matrix (Dummy)")
    confusion = pd.DataFrame(
        np.random.randint(0, 50, size=(4, 4)),
        index=categories,
        columns=categories
    )
    st.dataframe(confusion)
