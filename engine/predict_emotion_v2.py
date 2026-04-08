import os
import sys
import torch
import pickle
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.emotion_regressor import EmotionRegressor
from engine.feature_extractor import extract_features_from_file

MODEL_PATH = os.path.join(os.path.dirname(__file__),'..','models','emotion_regressor.pt')
SCALER_PATH = os.path.join(os.path.dirname(__file__),'..','models','deam_scaler.pkl')

EMOTION_DESCRIPTORS = [
    (0.7, 0.7, "euphoric"),
    (0.6, 0.4, "happy"),
    (0.4, 0.2, "content"),
    (0.2, -0.3, "peaceful"),
    (0.5, -0.5, "calm"),
    (0.1, -0.6, "serene"),
    (-0.1, -0.4, "melancholic"),
    (-0.4, -0.5, "sad"),
    (-0.6, -0.3, "depressed"),
    (-0.3, 0.3, "anxious"),
    (-0.5, 0.6, "angry"),
    (-0.7, 0.8, "furious"),
    (0.0, 0.5, "excited"),
    (0.3, 0.6, "energetic"),
    (-0.2, 0.1, "tense"),
    (0.0, 0.0, "neutral"),
]

def va_to_descriptor(v_norm: float, a_norm: float) -> str:
    best = min(EMOTION_DESCRIPTORS, key=lambda d: (d[0]-v_norm)**2 + (d[1]-a_norm)**2)
    return best[2]

def va_to_emotion(valence, arousal):
    v = (valence - 5) / 4
    a = (arousal - 5) / 4
    if v >= 0 and a >= 0:  return "HAPPY"
    if v < 0  and a >= 0:  return "ANGRY"
    if v >= 0 and a < 0:   return "CALM"
    return "SAD"

def va_to_musical_params(v_norm: float, a_norm: float) -> dict:
    tempo = int(80 + a_norm * 60)
    tempo = max(50, min(180, tempo))
    note_density = round(0.5 + a_norm * 0.4, 2)
    note_density = max(0.1, min(0.9, note_density))
    if v_norm >= 0.3:
        mode = "major"
    elif v_norm >= -0.1:
        mode = "dorian"
    elif v_norm >= -0.5:
        mode = "minor"
    else:
        mode = "phrygian"
    reverb_wet = round(0.2 + (-a_norm) * 0.3, 2)
    reverb_wet = max(0.1, min(0.6, reverb_wet))
    leap_prob = round(0.1 + abs(a_norm) * 0.4, 2)
    leap_prob = max(0.05, min(0.6, leap_prob))
    return {
        "tempo": tempo,
        "mode": mode,
        "note_density": note_density,
        "reverb_wet": reverb_wet,
        "leap_prob": leap_prob,
    }

def load_regressor():
    model = EmotionRegressor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_emotion_v2(file_path: str) -> dict:
    model, scaler = load_regressor()
    features = extract_features_from_file(file_path)
    features = scaler.transform([features])
    tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)[0]
        valence = float(output[0])
        arousal = float(output[1])

    v_norm = round((valence - 5) / 4, 3)
    a_norm = round((arousal - 5) / 4, 3)
    emotion = va_to_emotion(valence, arousal)
    descriptor = va_to_descriptor(v_norm, a_norm)
    musical_params = va_to_musical_params(v_norm, a_norm)

    return {
        "emotion": emotion,
        "descriptor": descriptor,
        "valence": round(valence, 2),
        "arousal": round(arousal, 2),
        "valence_norm": v_norm,
        "arousal_norm": a_norm,
        "musical_params": musical_params,
        "confidence": round(abs(v_norm) * 50 + abs(a_norm) * 50, 1),
        "secondary": None,
        "blend": descriptor,
        "scores": {
            "HAPPY": round(max(0, v_norm) * max(0, a_norm) * 100, 1),
            "SAD":   round(max(0, -v_norm) * max(0, -a_norm) * 100, 1),
            "ANGRY": round(max(0, -v_norm) * max(0, a_norm) * 100, 1),
            "CALM":  round(max(0, v_norm) * max(0, -a_norm) * 100, 1),
        }
    }

if __name__ == "__main__":
    import librosa
    result = predict_emotion_v2(librosa.ex('trumpet'))
    print(f"Emotion:     {result['emotion']}")
    print(f"Descriptor:  {result['descriptor']}")
    print(f"Valence:     {result['valence']:.2f} ({result['valence_norm']:+.3f})")
    print(f"Arousal:     {result['arousal']:.2f} ({result['arousal_norm']:+.3f})")
    print(f"Music params:{result['musical_params']}")


