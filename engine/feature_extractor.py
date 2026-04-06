"""
Feature extractor for SYNAESTHESIA emotion pipeline.
Extracts a rich 128-dim feature vector from any audio clip:
  - 40 MFCCs (mean + std)
  - 12 Chroma features (mean + std)
  - 7 Spectral contrast bands (mean + std)
  - Tempo, ZCR, RMS energy, spectral centroid, spectral rolloff
"""

import numpy as np
import librosa

FEATURE_DIM = 128  


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a fixed-size feature vector from a raw audio signal.
    Works on any length clip — uses mean+std pooling so it's length-invariant.
    """
    features = []
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.append(np.mean(mfccs, axis=1))
    features.append(np.std(mfccs, axis=1))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma, axis=1))
    features.append(np.std(chroma, axis=1))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.append(np.mean(contrast, axis=1))
    features.append(np.std(contrast, axis=1))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    features.append(np.array([
        float(np.squeeze(tempo)),
        float(np.mean(zcr)), float(np.std(zcr)),
        float(np.mean(rms)), float(np.std(rms)),
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(rolloff)), float(np.std(rolloff)),
        float(np.mean(bandwidth)),
    ]))

    vec = np.concatenate(features).astype(np.float32)
    return vec


def extract_features_from_file(file_path: str, duration: float = 3.0) -> np.ndarray:
    """Load a file and extract features. Clips to `duration` seconds."""
    y, sr = librosa.load(file_path, duration=duration, mono=True)
    if len(y) == 0:
        raise ValueError(f"Empty audio file: {file_path}")
    return extract_features(y, sr)


if __name__ == "__main__":
    path = librosa.ex('trumpet')
    vec = extract_features_from_file(path)
    print(f"Feature vector shape: {vec.shape}")
    print(f"First 10 values: {vec[:10]}")
