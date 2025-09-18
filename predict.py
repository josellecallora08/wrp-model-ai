# import sys, os
# import numpy as np
# import pandas as pd
# from joblib import load
# from utils.features import load_fixed, extract_features

# MODEL_PATH = "models/fewshot_audio_clf.joblib"
# AUDIO_DIR = "data/audio"
# OUTPUT_CSV_PATH = "predictions.csv"

# def resolve_audio_path(audio_dir: str, csv_name: str):
#     """Try exact name; if not found, try by stem with common extensions."""
#     import pathlib
#     from urllib.parse import unquote
#     exts = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
#     name = unquote(str(csv_name).strip().strip('"').strip("'"))
#     exact = os.path.join(audio_dir, name)
#     if os.path.isfile(exact):
#         return exact
#     stem = pathlib.Path(name).stem
#     for ext in exts:
#         cand = os.path.join(audio_dir, stem + ext)
#         if os.path.isfile(cand):
#             return cand
#     return None

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python predict.py path/to/your.csv")
#         sys.exit(1)

#     csv_path = sys.argv[1]
#     if not os.path.isfile(csv_path):
#         print(f"File not found: {csv_path}")
#         sys.exit(1)

#     if not os.path.isfile(MODEL_PATH):
#         print(f"Missing model: {MODEL_PATH}. Train first with: python main.py")
#         sys.exit(1)

#     bundle = load(MODEL_PATH)
#     model = bundle["model"]
#     sr = bundle["sr"]
#     duration = bundle["duration"]
#     id2 = bundle["id_to_label"]

#     df = pd.read_csv(csv_path)
#     if "file_name" not in df.columns:
#         print("CSV must have a 'file_name' column.")
#         sys.exit(1)

#     predictions = []
#     for _, row in df.iterrows():
#         file_name = row["file_name"]
#         fpath = resolve_audio_path(AUDIO_DIR, file_name)
#         predicted_label = None  # Default to None

#         if not fpath:
#             print(f"\n[WARN] Could not find audio for: {file_name}")
#         else:
#             try:
#                 ysig = load_fixed(fpath, sr=sr, duration=duration)
#                 x = extract_features(ysig, sr=sr).reshape(1, -1)
#                 pred_id = int(model.predict(x)[0])
#                 predicted_label = id2[pred_id]

#                 print(f"Processed: {os.path.basename(fpath)} -> {predicted_label}")

#             except Exception as e:
#                 print(f"\n[ERROR] Failed processing {file_name}: {e}")
        
#         predictions.append(predicted_label)

#     df["predicted_label"] = predictions
#     df.to_csv(OUTPUT_CSV_PATH, index=False)

#     print(f"\nPredictions saved to: {OUTPUT_CSV_PATH}")

import sys, os
import numpy as np
import pandas as pd
from joblib import load
from utils.features import load_fixed, extract_features   # updated import

MODEL_PATH = "models/audio_clf_svm.joblib"  # updated filename
AUDIO_DIR = "data/audio"
OUTPUT_CSV_PATH = "predictions.csv"



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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/your.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    if not os.path.isfile(MODEL_PATH):
        print(f"Missing model: {MODEL_PATH}. Train first with: python main.py")
        sys.exit(1)

    # Load trained model bundle
    bundle = load(MODEL_PATH)
    model = bundle["model"]
    sr = bundle["sr"]
    duration = bundle["duration"]
    id2 = bundle["id_to_label"]

    df = pd.read_csv(csv_path)
    if "file_name" not in df.columns:
        print("CSV must have a 'file_name' column.")
        sys.exit(1)

    predictions, probs = [], []

    for _, row in df.iterrows():
        file_name = row["file_name"]
        fpath = resolve_audio_path(AUDIO_DIR, file_name)
        predicted_label, pred_prob = None, None

        if not fpath:
            print(f"\n[WARN] Could not find audio for: {file_name}")
        else:
            try:
                ysig = load_fixed(fpath, sr=sr, duration=duration)
                x = extract_features(ysig, sr=sr).reshape(1, -1)

                # Predict label
                pred_id = int(model.predict(x)[0])
                predicted_label = id2[pred_id]

                # Optional: prediction probability
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(x)[0]
                    pred_prob = float(np.max(prob))

                msg = f"Processed: {os.path.basename(fpath)} -> {predicted_label}"
                if pred_prob is not None:
                    msg += f" (conf={pred_prob:.2f})"
                print(msg)

            except Exception as e:
                print(f"\n[ERROR] Failed processing {file_name}: {e}")

        predictions.append(predicted_label)
        probs.append(pred_prob)

    # Save results
    df["predicted_label"] = predictions
    if any(p is not None for p in probs):
        df["confidence"] = probs
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\nPredictions saved to: {OUTPUT_CSV_PATH}")
