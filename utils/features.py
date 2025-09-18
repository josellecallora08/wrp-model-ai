# import numpy as np
# import soundfile as sf
# import librosa

# # Audio I/O --------------------------------------------------------------

# def load_fixed(path: str, sr: int = 16000, duration: float = 10.0, mono: bool = True) -> np.ndarray:
#     """Load audio, resample, and pad/trim to fixed length."""
#     y, file_sr = sf.read(path, always_2d=False)
#     y = y.astype(np.float32)
#     if y.ndim > 1 and mono:
#         y = np.mean(y, axis=1)
#     if file_sr != sr:
#         y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
#     target_len = int(sr * duration)
#     if len(y) < target_len:
#         y = np.pad(y, (0, target_len - len(y)))
#     else:
#         y = y[:target_len]
#     return y

# # Light Augmentations (label-preserving) --------------------------------

# def augment_once(y: np.ndarray, sr: int) -> np.ndarray:
#     import random
#     choice = random.choice(["gain", "noise", "pitch", "stretch"])
#     if choice == "gain":
#         g_db = np.random.uniform(-6, 6)
#         return y * (10 ** (g_db / 20))
#     elif choice == "noise":
#         snr_db = np.random.uniform(10, 25)
#         rms = np.sqrt(np.mean(y**2) + 1e-12)
#         noise_rms = rms / (10 ** (snr_db / 20))
#         n = np.random.normal(0, noise_rms, size=y.shape)
#         return y + n
#     elif choice == "pitch":
#         steps = np.random.uniform(-1.0, 1.0)
#         return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
#     elif choice == "stretch":
#         rate = np.random.uniform(0.95, 1.05)
#         z = librosa.effects.time_stretch(y, rate=rate)
#         L = len(y)
#         return np.pad(z, (0, max(0, L-len(z))))[:L]

# # Feature extraction -----------------------------------------------------

# def _stats(mat: np.ndarray) -> np.ndarray:
#     return np.hstack([
#         np.mean(mat, axis=1), np.std(mat, axis=1),
#         np.percentile(mat, 10, axis=1), np.percentile(mat, 90, axis=1)
#     ])

# def extract_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
#     S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))**2
#     mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=64)
#     mel_db = librosa.power_to_db(mel + 1e-10)

#     mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel+1e-10), sr=sr, n_mfcc=20)
#     zcr = librosa.feature.zero_crossing_rate(y)
#     rms = librosa.feature.rms(y=y)
#     centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

#     return np.hstack([
#         _stats(mel_db),
#         _stats(mfcc),
#         _stats(zcr),
#         _stats(rms),
#         _stats(centroid),
#         _stats(rolloff),
#         _stats(contrast),
#     ])

import numpy as np
import soundfile as sf
import librosa

# ------------------------------------------------------------
# Audio Loader
# ------------------------------------------------------------
def load_fixed(path: str, sr: int = 16000, duration: float = 10.0, mono: bool = True) -> np.ndarray:
    """Load audio, resample, and pad/trim to fixed length."""
    y, file_sr = sf.read(path, always_2d=False)
    y = y.astype(np.float32)
    if y.ndim > 1 and mono:
        y = np.mean(y, axis=1)  # convert to mono
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

# ------------------------------------------------------------
# Light Augmentations (label-preserving)
# ------------------------------------------------------------
def augment_once(y: np.ndarray, sr: int) -> np.ndarray:
    import random
    choice = random.choice(["gain", "noise", "pitch", "stretch"])
    if choice == "gain":
        g_db = np.random.uniform(-6, 6)
        return y * (10 ** (g_db / 20))
    elif choice == "noise":
        snr_db = np.random.uniform(10, 25)
        rms = np.sqrt(np.mean(y**2) + 1e-12)
        noise_rms = rms / (10 ** (snr_db / 20))
        n = np.random.normal(0, noise_rms, size=y.shape)
        return y + n
    elif choice == "pitch":
        steps = np.random.uniform(-1.0, 1.0)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    elif choice == "stretch":
        rate = np.random.uniform(0.95, 1.05)
        z = librosa.effects.time_stretch(y, rate=rate)
        L = len(y)
        return np.pad(z, (0, max(0, L-len(z))))[:L]

# ------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------
def _stats(mat: np.ndarray) -> np.ndarray:
    """Aggregate time series into summary statistics."""
    return np.hstack([
        np.mean(mat, axis=1), np.std(mat, axis=1),
        np.percentile(mat, 10, axis=1), np.percentile(mat, 90, axis=1)
    ])

def extract_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract MFCC + delta features with statistical pooling."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.hstack([
        _stats(mfcc),
        _stats(delta),
        _stats(delta2),
    ])

