import sys, os
import numpy as np
from joblib import load
from utils.features import load_fixed, extract_features

MODEL_PATH = "models/fewshot_audio_clf.joblib"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/audio.wav")
        sys.exit(1)

    if not os.path.isfile(MODEL_PATH):
        print(f"Missing model: {MODEL_PATH}. Train first with: python main.py")
        sys.exit(1)

    bundle = load(MODEL_PATH)
    model = bundle["model"]
    sr = bundle["sr"]
    duration = bundle["duration"]
    id2 = bundle["id_to_label"]

    path = sys.argv[1]
    ysig = load_fixed(path, sr=sr, duration=duration)
    x = extract_features(ysig, sr=sr).reshape(1, -1)

    proba = getattr(model, "predict_proba", None)
    pred = int(model.predict(x)[0])
    print(f"Prediction: {id2[pred]}")
    if proba is not None:
        p = model.predict_proba(x)[0]
        for i in range(len(p)):
            print(f"{id2[i]:<10} -> {p[i]:.3f}")
