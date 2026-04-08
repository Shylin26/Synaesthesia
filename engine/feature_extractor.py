"""
Feature extractor for SYNAESTHESIA emotion pipeline.
Extracts a rich 176-dim feature vector from any audio clip:
  - 40 MFCCs (mean + std) = 80
  - 12 Chroma features (mean + std) = 24
  - 7 Spectral contrast bands (mean + std) = 14
  - 10 scalar features (tempo, ZCR, RMS, centroid, rolloff, bandwidth)
  - 6 Tonnetz harmonic features (mean)
  - 2 Spectral flux (mean + std)
  - 20 Mel spectrogram bands (mean + std) = 40
"""

import numpy as np
import librosa

FEATURE_DIM = 176


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
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

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features.append(np.mean(tonnetz, axis=1))

    flux = librosa.onset.onset_strength(y=y, sr=sr)
    features.append(np.array([float(np.mean(flux)), float(np.std(flux))]))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.append(np.mean(mel_db, axis=1))
    features.append(np.std(mel_db, axis=1))

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
