import os
import sys
import torch
import pickle
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.emotion_regressor import EmotionRegressor
from engine.feature_extractor import extract_features_from_file
MODEL_PATH=os.path.join(os.path.dirname(__file__),'..','models','emotion_regressor.pt')
SCALER_PATH=os.path.join(os.path.dirname(__file__),'..','models','deam_scaler.pkl')
def va_to_emotion(valence, arousal):
    v = (valence - 5) / 4   
    a = (arousal - 5) / 4
    if v >= 0 and a >= 0:  return "HAPPY"
    if v < 0  and a >= 0:  return "ANGRY"
    if v >= 0 and a < 0:   return "CALM"
    return "SAD"

def lod_regressor():
    model=EmotionRegressor()
    model.state_dict(torch.load(MODEL_PATH,map_location='cpu'))
    model.eval()
    with open(SCALER_PATH,'rb') as f:
        scaler=pickle.load(f)
    return model,scaler

def predict_emotio_v2(file_path:str)->dixt:
    model,scaler=load_regressor()
    features=extract_features_from_file(file_path)
    features=scaler.transorm([features])
    tensor=torch.tensor(features,dtype=torch.float32)

    with torch.no_grad():
        output=model(tensor)[0]
        valence=float(output[0])
        arousal=float(output[1])
    
    emotion=va_to_emotion(valence,arousal)
    v_norm=round((valence-5)/4,3)
    a_norm=round((arousal-5)/4,3)
    return {
        "emotion": emotion,
        "valence": valence,
        "arousal": arousal,
        "valence_norm": v_norm,
        "arousal_norm": a_norm,
        "confidence": None,
        "secondary": None,
        "blend": emotion,
        "scores": {
            "HAPPY": round(max(0, v_norm) * max(0, a_norm) * 100, 1),
            "SAD":   round(max(0, -v_norm) * max(0, -a_norm) * 100, 1),
            "ANGRY": round(max(0, -v_norm) * max(0, a_norm) * 100, 1),
            "CALM":  round(max(0, v_norm) * max(0, -a_norm) * 100, 1),
        }
    }

if __name__=="__main__":
    import librosa
    result=predict_emotio_v2(librosa.ex('trumpet'))
    print(f"Emotion:  {result['emotion']}")
    print(f"Valence:  {result['valence']:.2f}  ({result['valence_norm']:+.3f})")
    print(f"Arousal:  {result['arousal']:.2f}  ({result['arousal_norm']:+.3f})")

