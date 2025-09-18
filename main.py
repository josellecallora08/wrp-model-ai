# import os, warnings, random
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# from joblib import dump
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import LeaveOneOut

# from utils.features import load_fixed, extract_features, augment_once

# # ----------------- Config -----------------
# CSV_PATH = "labels.csv"
# AUDIO_DIR = "data/audio"
# MODEL_DIR = "models"
# MODEL_PATH = os.path.join(MODEL_DIR, "fewshot_audio_clf_main.joblib")

# SR = 16000
# DURATION = 10.0
# RANDOM_STATE = 42
# AUG_PER_TRAIN_CLIP = 3   # 0 to disable augmentation
# USE_KNN = True           # kNN (1-NN) is strong for few-shot; else logistic

# LABEL_ORDER = ["very bad", "bad", "good", "very good"]
# LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_ORDER)}
# ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_ORDER)}

# random.seed(RANDOM_STATE)
# np.random.seed(RANDOM_STATE)

# # -------------- Load CSV ---------------
# if not os.path.isfile(CSV_PATH):
#     raise FileNotFoundError(f"Missing {CSV_PATH}")

# df = pd.read_csv(CSV_PATH)
# for col in ["file_name", "label"]:
#     if col not in df.columns:
#         raise ValueError(f"labels.csv must have columns: file_name,label (missing {col})")

# df["label_norm"] = df["label"].astype(str).str.strip().str.lower()
# df = df[df["label_norm"].isin(LABEL_TO_ID.keys())].copy()
# if len(df) < 5:
#     raise ValueError("Too few rows after filtering labels. Need at least 5.")

# # -------------- Build dataset ----------
# rows = []
# for _, r in df.iterrows():
#     fpath = os.path.join(AUDIO_DIR, str(r["file_name"]))
#     if not os.path.isfile(fpath):
#         print(f"[WARN] Missing audio file: {fpath}")
#         continue
#     try:
#         ysig = load_fixed(fpath, sr=SR, duration=DURATION)
#         x = extract_features(ysig, sr=SR)
#         rows.append({"x": x, "y": LABEL_TO_ID[r["label_norm"]], "path": fpath})
#     except Exception as e:
#         print(f"[WARN] Failed {fpath}: {e}")

# if len(rows) < 5:
#     raise ValueError("Too few usable audio files after loading.")

# X = np.stack([r["x"] for r in rows])
# y = np.array([r["y"] for r in rows])

# # -------------- Classifier -------------
# if USE_KNN:
#     base_clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
# else:
#     base_clf = LogisticRegression(multi_class="multinomial", max_iter=2000)

# pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("clf", base_clf)
# ])

# # -------------- LOOCV (honest at N=10) -
# loo = LeaveOneOut()
# y_true, y_pred = [], []

# for train_idx, test_idx in loo.split(X):
#     X_tr, y_tr = X[train_idx], y[train_idx]

#     # Optional: augment only training fold from raw audio, then re-extract features
#     if AUG_PER_TRAIN_CLIP > 0:
#         aug_feats, aug_labels = [], []
#         for idx in train_idx:
#             path = rows[idx]["path"]
#             raw = load_fixed(path, sr=SR, duration=DURATION)
#             for _ in range(AUG_PER_TRAIN_CLIP):
#                 y_aug = augment_once(raw, SR)
#                 x_aug = extract_features(y_aug, SR)
#                 aug_feats.append(x_aug)
#                 aug_labels.append(y[idx])
#         if aug_feats:
#             X_tr = np.vstack([X_tr, np.stack(aug_feats)])
#             y_tr = np.concatenate([y_tr, np.array(aug_labels)])

#     model = pipe.fit(X_tr, y_tr)
#     pred = model.predict(X[test_idx])[0]
#     y_true.append(int(y[test_idx]))
#     y_pred.append(int(pred))

# print("\n=== Leave-One-Out CV report ===")
# print(classification_report(y_true, y_pred, target_names=LABEL_ORDER))
# def id2label(i: int) -> str:
#     return ID_TO_LABEL.get(i, f"<unknown:{i}>")

# print("\n=== Leave-One-Out CV report ===")
# # Optional: restrict to actually-seen labels to avoid mismatch warnings
# labels_present = sorted(set(y_true) | set(y_pred))
# target_names_present = [id2label(i) for i in labels_present]
# print(classification_report(y_true, y_pred, labels=labels_present,
#                             target_names=target_names_present, zero_division=0))

# print("\n=== Detailed predictions ===")
# for true_id, pred_id, row in zip(y_true, y_pred, rows):
#     print(f"File: {os.path.basename(row['path'])}")
#     print(f"  True: {id2label(true_id)}")
#     print(f"  Pred: {id2label(pred_id)}")

# print("Confusion matrix (rows=true, cols=pred):")
# print(confusion_matrix(y_true, y_pred, labels=labels_present))


# print("Confusion matrix (rows=true, cols=pred):")
# print(confusion_matrix(y_true, y_pred))

# # -------------- Final train on ALL -----
# X_final, y_final = X.copy(), y.copy()
# if AUG_PER_TRAIN_CLIP > 0:
#     aug_feats, aug_labels = [], []
#     for idx in range(len(rows)):
#         path = rows[idx]["path"]
#         raw = load_fixed(path, sr=SR, duration=DURATION)
#         for _ in range(AUG_PER_TRAIN_CLIP):
#             y_aug = augment_once(raw, SR)
#             x_aug = extract_features(y_aug, SR)
#             aug_feats.append(x_aug)
#             aug_labels.append(y[idx])
#     if aug_feats:
#         X_final = np.vstack([X_final, np.stack(aug_feats)])
#         y_final = np.concatenate([y_final, np.array(aug_labels)])

# final_model = pipe.fit(X_final, y_final)

# os.makedirs(MODEL_DIR, exist_ok=True)
# dump({
#     "model": final_model,
#     "label_to_id": LABEL_TO_ID,
#     "id_to_label": ID_TO_LABEL,
#     "sr": SR,
#     "duration": DURATION
# }, MODEL_PATH)

# print(f"\nSaved model to: {MODEL_PATH}")

import os, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from utils.features import load_fixed, extract_features, augment_once

# ----------------- Config -----------------
CSV_PATH = "labels.csv"
AUDIO_DIR = "data/audio"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "audio_clf_svm.joblib")

SR = 16000
DURATION = 10.0
RANDOM_STATE = 42
AUG_PER_TRAIN_CLIP = 5   # more augmentation for tiny datasets

LABEL_ORDER = ["very bad", "bad", "good", "very good"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_ORDER)}

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# -------------- Load CSV ---------------
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"Missing {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
for col in ["file_name", "label"]:
    if col not in df.columns:
        raise ValueError("labels.csv must have columns: file_name,label")

df["label_norm"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label_norm"].isin(LABEL_TO_ID.keys())].copy()
if len(df) < 5:
    raise ValueError("Too few rows after filtering labels. Need at least 5.")

# -------------- Build dataset ----------
rows = []
for _, r in df.iterrows():
    fpath = os.path.join(AUDIO_DIR, str(r["file_name"]))
    if not os.path.isfile(fpath):
        print(f"[WARN] Missing audio file: {fpath}")
        continue
    try:
        ysig = load_fixed(fpath, sr=SR, duration=DURATION)
        x = extract_features(ysig, sr=SR)
        rows.append({"x": x, "y": LABEL_TO_ID[r["label_norm"]], "path": fpath})
    except Exception as e:
        print(f"[WARN] Failed {fpath}: {e}")

if len(rows) < 5:
    raise ValueError("Too few usable audio files after loading.")

X = np.stack([r["x"] for r in rows])
y = np.array([r["y"] for r in rows])

# -------------- Classifier -------------
base_clf = SVC(kernel="rbf", probability=True, C=10, gamma="scale")
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", base_clf)
])

# -------------- Stratified K-Fold CV ---
kf = StratifiedKFold(n_splits=min(5, len(y)), shuffle=True, random_state=RANDOM_STATE)
y_true, y_pred = [], []

for train_idx, test_idx in kf.split(X, y):
    X_tr, y_tr = X[train_idx], y[train_idx]

    # Augment training data
    if AUG_PER_TRAIN_CLIP > 0:
        aug_feats, aug_labels = [], []
        for idx in train_idx:
            path = rows[idx]["path"]
            raw = load_fixed(path, sr=SR, duration=DURATION)
            for _ in range(AUG_PER_TRAIN_CLIP):
                y_aug = augment_once(raw, SR)
                x_aug = extract_features(y_aug, SR)
                aug_feats.append(x_aug)
                aug_labels.append(y[idx])
        if aug_feats:
            X_tr = np.vstack([X_tr, np.stack(aug_feats)])
            y_tr = np.concatenate([y_tr, np.array(aug_labels)])

    model = pipe.fit(X_tr, y_tr)
    pred = model.predict(X[test_idx])
    y_true.extend(y[test_idx])
    y_pred.extend(pred)

print("\n=== Stratified K-Fold CV report ===")
print(classification_report(y_true, y_pred, target_names=LABEL_ORDER))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred))

# -------------- Final train on ALL -----
X_final, y_final = X.copy(), y.copy()
if AUG_PER_TRAIN_CLIP > 0:
    aug_feats, aug_labels = [], []
    for idx in range(len(rows)):
        path = rows[idx]["path"]
        raw = load_fixed(path, sr=SR, duration=DURATION)
        for _ in range(AUG_PER_TRAIN_CLIP):
            y_aug = augment_once(raw, SR)
            x_aug = extract_features(y_aug, SR)
            aug_feats.append(x_aug)
            aug_labels.append(y[idx])
    if aug_feats:
        X_final = np.vstack([X_final, np.stack(aug_feats)])
        y_final = np.concatenate([y_final, np.array(aug_labels)])

final_model = pipe.fit(X_final, y_final)

os.makedirs(MODEL_DIR, exist_ok=True)
dump({
    "model": final_model,
    "label_to_id": LABEL_TO_ID,
    "id_to_label": ID_TO_LABEL,
    "sr": SR,
    "duration": DURATION
}, MODEL_PATH)

print(f"\nSaved model to: {MODEL_PATH}")
import time
from sklearn.metrics import accuracy_score

# ...existing code...

# -------------- Final train on ALL -----
X_final, y_final = X.copy(), y.copy()
if AUG_PER_TRAIN_CLIP > 0:
    aug_feats, aug_labels = [], []
    for idx in range(len(rows)):
        path = rows[idx]["path"]
        raw = load_fixed(path, sr=SR, duration=DURATION)
        for _ in range(AUG_PER_TRAIN_CLIP):
            y_aug = augment_once(raw, SR)
            x_aug = extract_features(y_aug, SR)
            aug_feats.append(x_aug)
            aug_labels.append(y[idx])
    if aug_feats:
        X_final = np.vstack([X_final, np.stack(aug_feats)])
        y_final = np.concatenate([y_final, np.array(aug_labels)])

start_time = time.time()
final_model = pipe.fit(X_final, y_final)
train_time = time.time() - start_time

# Accuracy on training set
y_train_pred = final_model.predict(X_final)
train_acc = accuracy_score(y_final, y_train_pred)

os.makedirs(MODEL_DIR, exist_ok=True)
dump({
    "model": final_model,
    "label_to_id": LABEL_TO_ID,
    "id_to_label": ID_TO_LABEL,
    "sr": SR,
    "duration": DURATION
}, MODEL_PATH)

print(f"\nSaved model to: {MODEL_PATH}")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Training time: {train_time:.2f}) seconds")