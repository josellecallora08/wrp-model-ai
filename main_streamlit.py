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
    saved_count = 0

    for label, files in audio_files_dict.items():
        if files:
            st.write(f"📁 Saving {len(files)} files for label: **{label}**")
            for f in files:
                file_path = os.path.join(AUDIO_DIR, f.name)
                try:
                    with open(file_path, "wb") as out_f:
                        out_f.write(f.getbuffer())
                    rows.append({"file_name": f.name, "label": label.lower()})
                    saved_count += 1
                except Exception as e:
                    st.warning(f"⚠️ Failed to save {f.name}: {e}")

    st.success(f"✅ Saved {saved_count} audio files to {AUDIO_DIR}")
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    return df


def append_training_data(corrections_list):
    """
    Append corrected predictions to the existing training data.
    corrections_list: List of dicts with 'file_name', 'label', 'audio_bytes'
    Returns: number of files added
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Load existing labels or create new DataFrame
    if os.path.isfile(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH)
    else:
        existing_df = pd.DataFrame(columns=["file_name", "label"])

    new_rows = []
    saved_count = 0

    for item in corrections_list:
        file_name = item["file_name"]
        label = item["label"].lower()
        audio_bytes = item["audio_bytes"]

        # Save audio file
        file_path = os.path.join(AUDIO_DIR, file_name)
        try:
            with open(file_path, "wb") as out_f:
                out_f.write(audio_bytes)

            # Check if file already exists in labels
            if file_name not in existing_df["file_name"].values:
                new_rows.append({"file_name": file_name, "label": label})
            else:
                # Update existing label
                existing_df.loc[existing_df["file_name"] == file_name, "label"] = label

            saved_count += 1
        except Exception as e:
            st.warning(f"⚠️ Failed to save {file_name}: {e}")

    # Append new rows
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Save updated labels.csv
    existing_df.to_csv(CSV_PATH, index=False)

    return saved_count, len(existing_df)

from sklearn.model_selection import train_test_split

def train_pipeline():
    df = pd.read_csv(CSV_PATH)
    st.info(f"📋 Found {len(df)} entries in labels.csv")

    rows = []
    files_not_found = 0
    files_failed = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, r in df.iterrows():
        fpath = os.path.join(AUDIO_DIR, r["file_name"])
        if not os.path.isfile(fpath):
            files_not_found += 1
            st.warning(f"⚠️ File not found: {fpath}")
            continue

        # Update progress
        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(df)}: {r['file_name']}")
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
            files_failed += 1
            st.warning(f"⚠️ Failed processing {fpath}: {e}")

    progress_bar.empty()
    status_text.empty()

    # Show summary
    st.info(f"📊 Summary: {len(rows)} feature sets created | {files_not_found} files not found | {files_failed} files failed to process")

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
    report = classification_report(y_val, y_pred, target_names=LABEL_ORDER, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    # Re-train on all data for the final model
    model_final = pipe.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump({
        "model": model_final,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "sr": SR,
        "duration": DURATION,
        # Save metrics for dashboard
        "validation_accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "training_samples": len(df),
        "feature_samples": len(rows),
        "trained_at": pd.Timestamp.now().isoformat()
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


def evaluate_model_accuracy(test_files_dict):
    """
    Evaluate model accuracy on labeled test data.
    test_files_dict: {label: [uploaded_files]}
    Returns accuracy, classification report, confusion matrix, and detailed results.
    """
    if not os.path.isfile(MODEL_PATH):
        st.error("❌ No trained model found. Please train a model first in the Training tab.")
        return None

    # Load trained model
    bundle = load(MODEL_PATH)
    model = bundle["model"]
    sr = bundle["sr"]
    duration = bundle["duration"]
    id2 = bundle["id_to_label"]

    y_true = []
    y_pred = []
    detailed_results = []

    total_files = sum(len(files) for files in test_files_dict.values() if files)
    if total_files == 0:
        st.warning("No test files uploaded.")
        return None

    progress_bar = st.progress(0)
    status_text = st.empty()
    processed = 0

    for true_label, files in test_files_dict.items():
        if not files:
            continue

        true_label_lower = true_label.lower()
        true_id = LABEL_TO_ID.get(true_label_lower)

        if true_id is None:
            st.warning(f"⚠️ Unknown label: {true_label}")
            continue

        for f in files:
            processed += 1
            progress_bar.progress(processed / total_files)
            status_text.text(f"Processing {processed}/{total_files}: {f.name}")

            try:
                # Save temporarily
                tmp_path = os.path.join(AUDIO_DIR, f"test_{f.name}")
                with open(tmp_path, "wb") as out_f:
                    out_f.write(f.getbuffer())

                # Extract features and predict
                ysig = load_fixed(tmp_path, sr=sr, duration=duration)
                x = extract_features(ysig, sr=sr).reshape(1, -1)
                pred_id = int(model.predict(x)[0])
                predicted_label = id2[pred_id]

                # Get confidence
                confidence = None
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(x)[0]
                    confidence = float(np.max(prob))

                y_true.append(true_id)
                y_pred.append(pred_id)

                # Store the audio bytes for playback
                f.seek(0)
                audio_bytes = f.read()

                detailed_results.append({
                    "file_name": f.name,
                    "true_label": true_label_lower,
                    "predicted_label": predicted_label,
                    "correct": true_label_lower == predicted_label,
                    "confidence": confidence,
                    "audio_bytes": audio_bytes
                })

                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            except Exception as e:
                st.warning(f"⚠️ Failed to process {f.name}: {e}")
                # Try to store audio bytes even on error for playback
                try:
                    f.seek(0)
                    audio_bytes = f.read()
                except:
                    audio_bytes = None
                detailed_results.append({
                    "file_name": f.name,
                    "true_label": true_label_lower,
                    "predicted_label": "ERROR",
                    "correct": False,
                    "confidence": None,
                    "audio_bytes": audio_bytes
                })

    progress_bar.empty()
    status_text.empty()

    if len(y_true) == 0:
        st.error("❌ No files were successfully processed.")
        return None

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)

    # Get unique labels present in the data
    present_labels = sorted(set(y_true + y_pred))
    present_label_names = [LABEL_ORDER[i] for i in present_labels]

    report = classification_report(y_true, y_pred, labels=present_labels,
                                   target_names=present_label_names, output_dict=True,
                                   zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "labels": present_label_names,
        "detailed_results": pd.DataFrame(detailed_results),
        "y_true": y_true,
        "y_pred": y_pred
    }



# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Audio Model UI", layout="wide")
st.title("🎙️ Audio Classification UI")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Test Model (Predict)", "🛠️ Train Model", "📊 Dashboard", "🎯 Evaluate Accuracy", "🔄 Improve Model"])

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

    # Check for training data and model
    has_training_data = os.path.isfile(CSV_PATH)
    has_model = os.path.isfile(MODEL_PATH)

    if not has_training_data and not has_model:
        st.info("No training data or model found. Train a model in the **Train Model** tab to see statistics here.")
    else:
        # Model Status Section
        st.subheader("Model Status")
        col1, col2, col3 = st.columns(3)

        with col1:
            if has_model:
                st.metric("Model", "Trained")
                # Get model file size
                model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
                st.caption(f"Size: {model_size:.2f} MB")
            else:
                st.metric("Model", "Not Trained")

        with col2:
            if has_training_data:
                train_df = pd.read_csv(CSV_PATH)
                st.metric("Training Files", len(train_df))
            else:
                st.metric("Training Files", 0)

        with col3:
            st.metric("Categories", len(LABEL_ORDER))

        st.markdown("---")

        # Training Data Distribution
        if has_training_data:
            train_df = pd.read_csv(CSV_PATH)

            st.subheader("Training Data Distribution")

            col1, col2 = st.columns(2)

            with col1:
                # Count per category
                label_counts = train_df["label"].value_counts()

                # Create ordered counts for all labels
                ordered_counts = []
                for label in LABEL_ORDER:
                    ordered_counts.append(label_counts.get(label, 0))

                # Display metrics
                metric_cols = st.columns(4)
                for i, label in enumerate(LABEL_ORDER):
                    with metric_cols[i]:
                        count = label_counts.get(label, 0)
                        st.metric(label.title(), count)

                # Bar chart
                df_counts = pd.DataFrame({
                    "Category": LABEL_ORDER,
                    "Count": ordered_counts
                })
                st.bar_chart(df_counts.set_index("Category"))

            with col2:
                # Pie chart for distribution
                fig, ax = plt.subplots(figsize=(6, 6))
                colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']
                non_zero_labels = [LABEL_ORDER[i] for i, c in enumerate(ordered_counts) if c > 0]
                non_zero_counts = [c for c in ordered_counts if c > 0]
                non_zero_colors = [colors[i] for i, c in enumerate(ordered_counts) if c > 0]

                if non_zero_counts:
                    ax.pie(non_zero_counts, labels=non_zero_labels, autopct='%1.1f%%',
                           colors=non_zero_colors, startangle=90)
                    ax.set_title("Label Distribution")
                    st.pyplot(fig)
                else:
                    st.info("No data to display")

            # Data balance warning
            if ordered_counts:
                max_count = max(ordered_counts)
                min_count = min(c for c in ordered_counts if c > 0) if any(c > 0 for c in ordered_counts) else 0
                if max_count > 0 and min_count > 0 and max_count / min_count > 3:
                    st.warning("Your dataset is imbalanced. Consider adding more samples to underrepresented categories for better model performance.")

            st.markdown("---")

            # Recent training files
            st.subheader("Training Files")
            st.dataframe(train_df, use_container_width=True)

        else:
            st.info("No training data found. Upload and train data in the **Train Model** tab.")

        # Model info section
        if has_model:
            st.markdown("---")
            st.subheader("Model Performance")

            try:
                bundle = load(MODEL_PATH)

                # Main metrics row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    val_acc = bundle.get("validation_accuracy")
                    if val_acc is not None:
                        st.metric("Validation Accuracy", f"{val_acc:.1%}")
                    else:
                        st.metric("Validation Accuracy", "N/A")

                with col2:
                    report = bundle.get("classification_report", {})
                    if report and "weighted avg" in report:
                        f1 = report["weighted avg"].get("f1-score", 0)
                        st.metric("F1 Score (Weighted)", f"{f1:.1%}")
                    else:
                        st.metric("F1 Score", "N/A")

                with col3:
                    training_samples = bundle.get("training_samples")
                    if training_samples:
                        st.metric("Training Files", training_samples)
                    else:
                        st.metric("Training Files", "N/A")

                with col4:
                    feature_samples = bundle.get("feature_samples")
                    if feature_samples:
                        st.metric("Feature Samples", feature_samples)
                        st.caption("(with augmentation)")
                    else:
                        st.metric("Feature Samples", "N/A")

                # Per-class performance
                report = bundle.get("classification_report", {})
                if report:
                    st.markdown("**Per-Category Performance:**")
                    perf_cols = st.columns(4)
                    for i, label in enumerate(LABEL_ORDER):
                        if label in report:
                            with perf_cols[i]:
                                precision = report[label].get("precision", 0)
                                recall = report[label].get("recall", 0)
                                f1 = report[label].get("f1-score", 0)
                                st.write(f"**{label.title()}**")
                                st.write(f"Precision: {precision:.0%}")
                                st.write(f"Recall: {recall:.0%}")
                                st.write(f"F1: {f1:.0%}")

                # Confusion Matrix
                cm = bundle.get("confusion_matrix")
                if cm:
                    st.markdown("**Confusion Matrix (Validation):**")
                    cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(cm_df, use_container_width=True)
                    with col2:
                        # Heatmap
                        fig, ax = plt.subplots(figsize=(6, 5))
                        im = ax.imshow(cm, cmap='Blues')
                        ax.set_xticks(range(len(LABEL_ORDER)))
                        ax.set_yticks(range(len(LABEL_ORDER)))
                        ax.set_xticklabels(LABEL_ORDER, rotation=45, ha='right', fontsize=8)
                        ax.set_yticklabels(LABEL_ORDER, fontsize=8)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('True')
                        for i in range(len(LABEL_ORDER)):
                            for j in range(len(LABEL_ORDER)):
                                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=10)
                        fig.colorbar(im)
                        plt.tight_layout()
                        st.pyplot(fig)

                st.markdown("---")
                st.subheader("Model Details")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.write("**Model Type:**")
                    st.code("SVM (RBF)")

                with col2:
                    st.write("**Sample Rate:**")
                    st.code(f"{bundle.get('sr', 'N/A')} Hz")

                with col3:
                    st.write("**Audio Duration:**")
                    st.code(f"{bundle.get('duration', 'N/A')}s")

                with col4:
                    trained_at = bundle.get("trained_at")
                    if trained_at:
                        st.write("**Trained At:**")
                        try:
                            dt = pd.Timestamp(trained_at)
                            st.code(dt.strftime("%Y-%m-%d %H:%M"))
                        except:
                            st.code(trained_at[:16])
                    else:
                        st.write("**Trained At:**")
                        st.code("N/A")

                # Labels the model knows
                st.write("**Trained Labels:**")
                id_to_label = bundle.get("id_to_label", {})
                if id_to_label:
                    label_chips = " | ".join([f"`{label}`" for label in id_to_label.values()])
                    st.markdown(label_chips)

            except Exception as e:
                st.warning(f"Could not load model details: {e}")

# -------------------------------
# Tab 4: Evaluate Accuracy
# -------------------------------
with tab4:
    st.header("Evaluate Model Accuracy")
    st.markdown("""
    Upload **labeled test audio files** to evaluate how accurate your trained model is.
    This uses files that the model has **never seen before** to give you an unbiased accuracy score.
    """)

    # Check if model exists
    if not os.path.isfile(MODEL_PATH):
        st.warning("⚠️ No trained model found. Please train a model first in the **Train Model** tab.")
    else:
        st.success("✅ Trained model found. Ready to evaluate.")

        st.subheader("Upload Test Files by Category")
        st.markdown("Upload audio files for each category. These should be **different** from your training files.")

        test_files_dict = {
            "Very Good": st.file_uploader("Upload 'Very Good' test audios", type=["wav", "mp3", "m4a", "flac", "ogg"], accept_multiple_files=True, key="test_vg"),
            "Good": st.file_uploader("Upload 'Good' test audios", type=["wav", "mp3", "m4a", "flac", "ogg"], accept_multiple_files=True, key="test_g"),
            "Bad": st.file_uploader("Upload 'Bad' test audios", type=["wav", "mp3", "m4a", "flac", "ogg"], accept_multiple_files=True, key="test_b"),
            "Very Bad": st.file_uploader("Upload 'Very Bad' test audios", type=["wav", "mp3", "m4a", "flac", "ogg"], accept_multiple_files=True, key="test_vb"),
        }

        # Show upload counts
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Very Good", len(test_files_dict["Very Good"]) if test_files_dict["Very Good"] else 0)
        with col2:
            st.metric("Good", len(test_files_dict["Good"]) if test_files_dict["Good"] else 0)
        with col3:
            st.metric("Bad", len(test_files_dict["Bad"]) if test_files_dict["Bad"] else 0)
        with col4:
            st.metric("Very Bad", len(test_files_dict["Very Bad"]) if test_files_dict["Very Bad"] else 0)

        total_test_files = sum(len(files) for files in test_files_dict.values() if files)
        st.info(f"📁 Total test files: {total_test_files}")

        if st.button("🎯 Evaluate Model Accuracy", disabled=total_test_files == 0):
            with st.spinner("Evaluating model on test data..."):
                results = evaluate_model_accuracy(test_files_dict)

            if results:
                st.subheader("📊 Evaluation Results")

                # Main accuracy metric
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Accuracy", f"{results['accuracy']:.2%}")
                with col2:
                    correct = results['detailed_results']['correct'].sum()
                    total = len(results['detailed_results'])
                    st.metric("Correct Predictions", f"{correct}/{total}")
                with col3:
                    avg_conf = results['detailed_results']['confidence'].dropna().mean()
                    st.metric("Average Confidence", f"{avg_conf:.2%}" if not pd.isna(avg_conf) else "N/A")

                # Per-category accuracy
                st.subheader("Per-Category Performance")
                report_df = pd.DataFrame(results['report']).T
                # Filter out summary rows
                category_rows = [label for label in report_df.index if label in LABEL_ORDER]
                if category_rows:
                    st.dataframe(report_df.loc[category_rows, ['precision', 'recall', 'f1-score', 'support']].round(3))

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                st.markdown("**Rows** = True Labels, **Columns** = Predicted Labels")
                cm_df = pd.DataFrame(
                    results['confusion_matrix'],
                    index=results['labels'],
                    columns=results['labels']
                )
                st.dataframe(cm_df)

                # Confusion matrix heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(results['confusion_matrix'], cmap='Blues')
                ax.set_xticks(range(len(results['labels'])))
                ax.set_yticks(range(len(results['labels'])))
                ax.set_xticklabels(results['labels'], rotation=45, ha='right')
                ax.set_yticklabels(results['labels'])
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix Heatmap')

                # Add text annotations
                for i in range(len(results['labels'])):
                    for j in range(len(results['labels'])):
                        text = ax.text(j, i, results['confusion_matrix'][i, j],
                                       ha="center", va="center", color="black")

                fig.colorbar(im)
                plt.tight_layout()
                st.pyplot(fig)

                # Detailed results table with audio playback
                st.subheader("Detailed Results")
                st.markdown("Listen to each audio file to verify the prediction:")

                for idx, row in results['detailed_results'].iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1, 2])
                        with col1:
                            st.write(f"**{row['file_name']}**")
                        with col2:
                            st.write(f"True: **{row['true_label']}**")
                        with col3:
                            st.write(f"Pred: **{row['predicted_label']}**")
                        with col4:
                            status = '✅' if row['correct'] else '❌'
                            conf = f"{row['confidence']:.0%}" if pd.notna(row['confidence']) else "N/A"
                            st.write(f"{status} ({conf})")
                        with col5:
                            if row.get('audio_bytes'):
                                st.audio(row['audio_bytes'])
                            else:
                                st.write("No audio")
                        st.markdown("---")

                # Misclassified files with audio playback
                misclassified = results['detailed_results'][results['detailed_results']['correct'] == False]
                if len(misclassified) > 0:
                    st.subheader(f"❌ Misclassified Files ({len(misclassified)})")
                    st.markdown("Listen to misclassified files to understand where the model went wrong:")

                    for idx, row in misclassified.iterrows():
                        with st.container():
                            col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
                            with col1:
                                st.write(f"**{row['file_name']}**")
                            with col2:
                                st.write(f"True: **{row['true_label']}** → Pred: **{row['predicted_label']}**")
                            with col3:
                                conf = f"{row['confidence']:.0%}" if pd.notna(row['confidence']) else "N/A"
                                st.write(f"Conf: {conf}")
                            with col4:
                                if row.get('audio_bytes'):
                                    st.audio(row['audio_bytes'])
                                else:
                                    st.write("No audio")
                            st.markdown("---")

                # Explanation
                with st.expander("How to Interpret These Results"):
                    st.markdown("""
                    **Understanding Your Model's Accuracy:**

                    - **Overall Accuracy**: The percentage of test files the model classified correctly.
                      A good model typically has 80%+ accuracy, but this depends on your use case.

                    - **Precision**: When the model predicts a category, how often is it right?
                      - High precision = Few false positives
                      - Example: If precision for "Very Good" is 0.90, then 90% of files predicted as "Very Good" actually are.

                    - **Recall**: Of all files in a category, how many did the model find?
                      - High recall = Few false negatives
                      - Example: If recall for "Bad" is 0.85, the model correctly identified 85% of all "Bad" files.

                    - **F1-Score**: The harmonic mean of precision and recall. Good for comparing overall category performance.

                    - **Confusion Matrix**: Shows where mistakes happen.
                      - Diagonal (top-left to bottom-right) = correct predictions
                      - Off-diagonal = errors
                      - Example: A value of 5 in row "Good", column "Very Good" means 5 "Good" files were misclassified as "Very Good"

                    **Tips for Improvement:**
                    - If accuracy is low, try adding more diverse training data
                    - If certain categories are confused, those audio types may be similar - consider combining them
                    - Check the misclassified files to understand patterns in errors
                    """)

# -------------------------------
# Tab 5: Improve Model
# -------------------------------
with tab5:
    st.header("Improve Model with Feedback")
    st.markdown("""
    Use this tab to **improve your model** through an iterative feedback loop:
    1. Upload audio files to get predictions
    2. Review predictions and correct any wrong labels
    3. Add the corrected data to your training set
    4. Optionally retrain the model with the new data
    """)

    # Check if model exists
    if not os.path.isfile(MODEL_PATH):
        st.warning("⚠️ No trained model found. Please train a model first in the **Train Model** tab.")
    else:
        st.success("✅ Trained model found. Ready to improve.")

        # Initialize session state for storing predictions
        if "improve_predictions" not in st.session_state:
            st.session_state.improve_predictions = None
        if "improve_corrections" not in st.session_state:
            st.session_state.improve_corrections = {}

        # Step 1: Upload files
        st.subheader("Step 1: Upload Audio Files")
        improve_files = st.file_uploader(
            "Upload audio files to get predictions",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            accept_multiple_files=True,
            key="improve_uploader"
        )

        if st.button("🔍 Get Predictions", disabled=not improve_files):
            with st.spinner("Running predictions..."):
                # Load model
                bundle = load(MODEL_PATH)
                model = bundle["model"]
                sr = bundle["sr"]
                duration = bundle["duration"]
                id2 = bundle["id_to_label"]

                predictions = []
                for f in improve_files:
                    try:
                        # Read audio bytes
                        f.seek(0)
                        audio_bytes = f.read()

                        # Save temporarily for processing
                        tmp_path = os.path.join(AUDIO_DIR, f"improve_{f.name}")
                        os.makedirs(AUDIO_DIR, exist_ok=True)
                        with open(tmp_path, "wb") as out_f:
                            out_f.write(audio_bytes)

                        # Extract features and predict
                        ysig = load_fixed(tmp_path, sr=sr, duration=duration)
                        x = extract_features(ysig, sr=sr).reshape(1, -1)
                        pred_id = int(model.predict(x)[0])
                        predicted_label = id2[pred_id]

                        # Get confidence
                        confidence = None
                        if hasattr(model, "predict_proba"):
                            prob = model.predict_proba(x)[0]
                            confidence = float(np.max(prob))

                        predictions.append({
                            "file_name": f.name,
                            "predicted_label": predicted_label,
                            "confidence": confidence,
                            "audio_bytes": audio_bytes
                        })

                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                    except Exception as e:
                        st.warning(f"⚠️ Failed to process {f.name}: {e}")
                        f.seek(0)
                        predictions.append({
                            "file_name": f.name,
                            "predicted_label": None,
                            "confidence": None,
                            "audio_bytes": f.read()
                        })

                st.session_state.improve_predictions = predictions
                # Initialize corrections with predicted labels
                st.session_state.improve_corrections = {
                    p["file_name"]: p["predicted_label"] or "good" for p in predictions
                }

            st.success(f"✅ Processed {len(predictions)} files")

        # Step 2: Review and correct predictions
        if st.session_state.improve_predictions:
            st.subheader("Step 2: Review & Correct Predictions")
            st.markdown("Review each prediction below. If the AI got it wrong, select the correct label.")

            predictions = st.session_state.improve_predictions

            # Summary stats
            total = len(predictions)
            successful = sum(1 for p in predictions if p["predicted_label"] is not None)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Files", total)
            with col2:
                st.metric("Successfully Predicted", successful)

            st.markdown("---")

            # Display each prediction with correction option
            for idx, pred in enumerate(predictions):
                with st.container():
                    col1, col2, col3, col4 = st.columns([2.5, 1.5, 2, 2])

                    with col1:
                        st.write(f"**{pred['file_name']}**")
                        if pred["audio_bytes"]:
                            st.audio(pred["audio_bytes"])

                    with col2:
                        if pred["predicted_label"]:
                            conf_str = f"{pred['confidence']:.0%}" if pred['confidence'] else "N/A"
                            st.write(f"**AI Prediction:**")
                            st.write(f"{pred['predicted_label']}")
                            st.write(f"Confidence: {conf_str}")
                        else:
                            st.write("**AI Prediction:**")
                            st.write("Failed to predict")

                    with col3:
                        st.write("**Correct Label:**")
                        # Get current correction value
                        current_value = st.session_state.improve_corrections.get(
                            pred["file_name"], pred["predicted_label"] or "good"
                        )
                        # Find index of current value
                        try:
                            default_idx = LABEL_ORDER.index(current_value)
                        except ValueError:
                            default_idx = 2  # default to "good"

                        corrected_label = st.selectbox(
                            "Select correct label",
                            options=LABEL_ORDER,
                            index=default_idx,
                            key=f"correct_{idx}",
                            label_visibility="collapsed"
                        )
                        st.session_state.improve_corrections[pred["file_name"]] = corrected_label

                    with col4:
                        # Show if correction differs from prediction
                        if pred["predicted_label"] and corrected_label != pred["predicted_label"]:
                            st.write("**Status:**")
                            st.warning(f"Corrected")
                        elif pred["predicted_label"]:
                            st.write("**Status:**")
                            st.success("Confirmed")
                        else:
                            st.write("**Status:**")
                            st.info("New label")

                    st.markdown("---")

            # Step 3: Save corrections
            st.subheader("Step 3: Add to Training Data")

            # Show summary of corrections
            corrections_summary = {}
            for pred in predictions:
                corrected = st.session_state.improve_corrections.get(pred["file_name"])
                if corrected:
                    if corrected not in corrections_summary:
                        corrections_summary[corrected] = 0
                    corrections_summary[corrected] += 1

            st.write("**Files to add by category:**")
            summary_cols = st.columns(4)
            for i, label in enumerate(LABEL_ORDER):
                with summary_cols[i]:
                    count = corrections_summary.get(label, 0)
                    st.metric(label.title(), count)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 Save to Training Data"):
                    # Prepare corrections list
                    corrections_list = []
                    for pred in predictions:
                        corrected_label = st.session_state.improve_corrections.get(pred["file_name"])
                        if corrected_label and pred["audio_bytes"]:
                            corrections_list.append({
                                "file_name": pred["file_name"],
                                "label": corrected_label,
                                "audio_bytes": pred["audio_bytes"]
                            })

                    if corrections_list:
                        saved_count, total_in_dataset = append_training_data(corrections_list)
                        st.success(f"✅ Saved {saved_count} files to training data!")
                        st.info(f"📊 Total files in training dataset: {total_in_dataset}")
                    else:
                        st.warning("No files to save.")

            with col2:
                retrain_after_save = st.checkbox("Retrain model after saving", value=False)

            # Step 4: Retrain (optional)
            st.subheader("Step 4: Retrain Model (Optional)")
            st.markdown("After adding new data, you can retrain the model to incorporate the feedback.")

            if st.button("🚀 Retrain Model Now"):
                # Check if we have training data
                if not os.path.isfile(CSV_PATH):
                    st.error("❌ No training data found. Please save some corrections first.")
                else:
                    with st.spinner("Retraining model with updated data... This may take a moment."):
                        results = train_pipeline()

                    if results and results[0] is not None:
                        acc, report, cm, y_val, y_pred = results
                        st.success("✅ Model retrained successfully!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("New Validation Accuracy", f"{acc:.2%}")
                        with col2:
                            # Count training samples
                            train_df = pd.read_csv(CSV_PATH)
                            st.metric("Training Samples", len(train_df))
                        with col3:
                            st.metric("Categories", len(LABEL_ORDER))

                        with st.expander("View Detailed Results"):
                            st.text("Classification Report:")
                            st.json(report)

                            st.text("Confusion Matrix:")
                            st.dataframe(pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER))
                    else:
                        st.error("❌ Retraining failed. Check if you have enough diverse training data.")

            # Clear session button
            st.markdown("---")
            if st.button("🗑️ Clear & Start Over"):
                st.session_state.improve_predictions = None
                st.session_state.improve_corrections = {}
                st.rerun()

        # Show current training data stats
        st.markdown("---")
        st.subheader("📊 Current Training Data")
        if os.path.isfile(CSV_PATH):
            current_df = pd.read_csv(CSV_PATH)
            st.write(f"**Total files in training set:** {len(current_df)}")

            # Show distribution
            if not current_df.empty:
                label_dist = current_df["label"].value_counts()
                dist_cols = st.columns(4)
                for i, label in enumerate(LABEL_ORDER):
                    with dist_cols[i]:
                        count = label_dist.get(label, 0)
                        st.metric(label.title(), count)
        else:
            st.info("No training data yet. Upload and label some files to get started.")
