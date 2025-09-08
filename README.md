.

🎙️ wrp-model-ai

A minimal pipeline to train and use a machine learning model for classifying audio recordings into categories:

very bad

bad

good

very good

This project uses Python, librosa for audio feature extraction, and scikit-learn for classification.

📂 Project Structure
wrp-model-ai/
│── main.py               # Train + evaluate (cross-validation)
│── main_fit_only.py      # Train on all data, no evaluation
│── predict.py            # Predict the label of a new audio file
│── labels.csv            # File-to-label mapping
│── requirements.txt      # Dependencies
│
├── data/
│   └── audio/            # Audio files go here
│
├── models/
│   └── fewshot_audio_clf.joblib   # Saved model after training
│
└── utils/
    └── features.py       # Audio I/O, augmentation, feature extraction

⚙️ Setup

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows


Install dependencies:

pip install -r requirements.txt

🗂 Data Preparation

Your labels.csv must have two columns:

file_name,label
audio1.mp3,very good
audio2.mp3,good
audio3.mp3,bad
audio4.mp3,very bad


file_name → must exactly match the audio filename in data/audio/

label → one of: very bad, bad, good, very good

🚀 Usage
1. Train with Cross-Validation

Run Leave-One-Out CV (LOOCV) to get evaluation metrics:

python main.py


You’ll see:

Precision / Recall / F1 per class

Confusion matrix

Per-file predictions

2. Train Without Evaluation

If you only want to fit a model and save it:

python main_fit_only.py


The trained model is saved to:

models/fewshot_audio_clf.joblib

3. Predict on New Audio

Classify a new file:

python predict.py data/audio/your_file.mp3


Output example:

Prediction: good
very bad   -> 0.05
bad        -> 0.10
good       -> 0.70
very good  -> 0.15

🔍 Notes

With very few samples, accuracy will be poor.

Collect at least 20–50 recordings per class for meaningful results.

Consider merging into 2 classes (positive/negative) if dataset is too small.

For better results, replace MFCCs with pretrained embeddings (e.g., YAMNet, Wav2Vec2).

📝 License

Prototype project for testing and learning — no license specified yet.