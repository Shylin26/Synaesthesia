import os
import sys
import torch
import numpy as np
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.emotion_transformer import EmotionTransformer
from engine.feature_extractor import extract_features_from_file

MODEL_PATH = "/Users/parishachauhan/SYNAESTHESIA/models/emotion_transformer.pt"
SCALER_PATH = "/Users/parishachauhan/SYNAESTHESIA/models/scaler.pkl"

EMOTION_LABELS={
    0:"HAPPY",
    1:"SAD",
    2: "ANGRY",
    3:"CALM"
}
def load_model():
    model=EmotionTransformer()
    model.load_state_dict(torch.load(MODEL_PATH,map_location='cpu'))
    model.eval()
    with open(SCALER_PATH,'rb') as f:
        scaler=pickle.load(f)
    return model,scaler

CONFIDENCE_THRESHOLD = 55.0
SECONDARY_THRESHOLD = 15.0

def predict_emotion(file_path: str) -> dict:
    model, scaler = load_model()
    features = extract_features_from_file(file_path)
    features = scaler.transform([features])
    tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    all_scores = {
        EMOTION_LABELS[i]: round(probabilities[i].item() * 100, 1)
        for i in range(4)
    }
    sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    primary_label = sorted_emotions[0][0]
    primary_score = sorted_emotions[0][1]

    if primary_score < CONFIDENCE_THRESHOLD:
        return {
            "emotion": "UNCERTAIN",
            "confidence": primary_score,
            "secondary": None,
            "blend": "UNCERTAIN",
            "scores": all_scores
        }

    secondary = None
    if sorted_emotions[1][1] >= SECONDARY_THRESHOLD:
        secondary = {"emotion": sorted_emotions[1][0], "confidence": sorted_emotions[1][1]}

    blend = f"{primary_label} · {secondary['emotion']}" if secondary else primary_label

    return {
        "emotion": primary_label,
        "confidence": primary_score,
        "secondary": secondary,
        "blend": blend,
        "scores": all_scores
    }

if __name__=="__main__":
    import librosa
    test_file=librosa.ex('trumpet')
    result=predict_emotion(test_file)
    print(f"Emotion:{result['emotion']}({result['confidence']}%confidence)")
    print(f"All scores: {result['scores']}")
