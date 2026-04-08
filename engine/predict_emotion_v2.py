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
    (0.8, 0.8, "euphoric pop"),
    (0.7, 0.6, "neon nostalgia"),
    (0.5, 0.4, "happy"),
    (0.2, 0.2, "content"),
    (0.1, -0.3, "peaceful"),
    (0.4, -0.5, "dreamy"),
    (0.1, -0.8, "serene"),
    (-0.2, -0.5, "melancholic"),
    (-0.4, -0.6, "sad"),
    (-0.7, -0.4, "depressed"),
    (-0.3, 0.3, "anxious"),
    (-0.6, 0.5, "angst"),
    (-0.5, -0.1, "existential dread"),
    (-0.8, 0.8, "furious"),
    (0.0, 0.5, "excited"),
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
    
    if v_norm >= 0.4:
        mode = "major" if a_norm < 0.5 else "mixolydian"
    elif v_norm >= 0.0:
        mode = "dorian" if a_norm < 0.4 else "lydian"
    elif v_norm >= -0.4:
        mode = "minor"
    else:
        mode = "phrygian" if a_norm < 0.4 else "harmonic_minor"
        
    reverb_wet = round(0.3 + (-v_norm) * 0.4, 2)
    reverb_wet = max(0.1, min(0.8, reverb_wet))
    
    leap_prob = round(0.1 + abs(a_norm) * 0.5, 2)
    leap_prob = max(0.05, min(0.7, leap_prob))

    groove = round(max(0.1, a_norm * 0.8 + v_norm * 0.2 + 0.3), 2)
    syncopation = round(max(0.1, abs(v_norm)*0.5 + a_norm*0.5), 2)
    dissonance = round(max(0.0, -v_norm * 0.7 + a_norm * 0.3), 2)
    arpeggiation = round(max(0.1, 0.6 - abs(a_norm) * 0.4), 2)

    return {
        "tempo": tempo,
        "mode": mode,
        "note_density": note_density,
        "reverb_wet": reverb_wet,
        "leap_prob": leap_prob,
        "groove": min(1.0, groove),
        "syncopation": min(1.0, syncopation),
        "dissonance": min(1.0, dissonance),
        "arpeggiation": min(1.0, arpeggiation)
    }

def load_regressor():
    model = EmotionRegressor(feature_dim=176, d_model=64, nhead=4, num_layers=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'), strict=False)
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


