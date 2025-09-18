import os, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from utils.features import load_fixed, extract_features, augment_once
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------- Config -----------------
CSV_PATH = "labels.csv"
AUDIO_DIR = "data/audio"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fewshot_audio_clf.joblib")

SR = 16000
DURATION = 10.0
RANDOM_STATE = 42

# Choose classifier
USE_KNN = True           # True: kNN (few-shot friendly). False: Logistic Regression
K_FOR_KNN = 3            # k=1/3/5 are typical
AUG_PER_CLIP = 3         # 0 = no augmentation; 3–5 light augments per file recommended for tiny data

# Label set (4 classes)
LABEL_ORDER = ["very bad", "bad", "good", "very good"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_ORDER)}

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ----------------- Load CSV -----------------
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"Missing {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
for col in ["file_name", "label"]:
    if col not in df.columns:
        raise ValueError("labels.csv must have columns: file_name,label")

df["label_norm"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label_norm"].isin(LABEL_TO_ID.keys())].copy()
if len(df) == 0:
    raise ValueError("No rows left after filtering labels to known set.")

# ----------------- Resolve & featurize -----------------
def resolve_audio_path(audio_dir: str, csv_name: str):
    """Try exact name; if not found, try by stem with common extensions."""
    import pathlib
    from urllib.parse import unquote
    exts = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    name = unquote(str(csv_name).strip().strip('"').strip("'"))
    exact = os.path.join(audio_dir, name)
    if os.path.isfile(exact):
        return exact
    stem = pathlib.Path(name).stem
    for ext in exts:
        cand = os.path.join(audio_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None

rows = []
missing = 0
for _, r in df.iterrows():
    fpath = resolve_audio_path(AUDIO_DIR, r["file_name"])
    if not fpath:
        print(f"[WARN] Missing audio for CSV entry: {r['file_name']}")
        missing += 1
        continue
    try:
        raw = load_fixed(fpath, sr=SR, duration=DURATION)
        # light loudness normalization
        rms = float(np.sqrt(np.mean(raw**2) + 1e-12))
        if rms > 0:
            raw = raw / rms
        # base features
        x = extract_features(raw, sr=SR)
        rows.append({"x": x, "y": LABEL_TO_ID[r["label_norm"]], "path": fpath})

        # optional augmentation
        for _ in range(AUG_PER_CLIP):
            y_aug = augment_once(raw, SR)
            x_aug = extract_features(y_aug, SR)
            rows.append({"x": x_aug, "y": LABEL_TO_ID[r["label_norm"]], "path": fpath})

    except Exception as e:
        print(f"[WARN] Failed {fpath}: {e}")

if len(rows) < 2:
    raise ValueError("Too few usable audio rows to fit a model.")

X = np.stack([r["x"] for r in rows])
y = np.array([r["y"] for r in rows])

# ----------------- Split data for evaluation -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ----------------- Build & fit -----------------
if USE_KNN:
    base_clf = KNeighborsClassifier(n_neighbors=K_FOR_KNN, metric="euclidean")
else:
    base_clf = LogisticRegression(multi_class="multinomial", max_iter=2000)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", base_clf),
])
pipe.fit(X_train, y_train)

# ----------------- Evaluate -----------------
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Hold-out accuracy: {accuracy:.4f}")

# ----------------- Re-fit on all data & save -----------------
pipe.fit(X, y)  # Re-fit on the entire dataset for the final model
os.makedirs(MODEL_DIR, exist_ok=True)
dump({
    "model": pipe,
    "label_to_id": LABEL_TO_ID,
    "id_to_label": ID_TO_LABEL,
    "sr": SR,
    "duration": DURATION
}, MODEL_PATH)

print(f"Model trained on {len(rows)} feature rows "
      f"(from {len(df) - missing} audio files; missing={missing})")
print(f"Saved: {MODEL_PATH}")
print("Done.")
