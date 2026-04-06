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

def predict_emotion(file_path:str)->dict:
    model,scaler=load_model()
    features=extract_features_from_file(file_path)
    features=scaler.transform([features])
    tensor=torch.tensor(features,dtype=torch.float32)

    with torch.no_grad():
        output=model(tensor)
        probabilities=torch.softmax(output,dim=1)[0]
        predicted_class=torch.argmax(probabilities).item()
    return{
        "emotion":EMOTION_LABELS[predicted_class],
        "confidence":round(probabilities[predicted_class].item()*100,1),
        "scores":{
            EMOTION_LABELS[i]: round(probabilities[i].item() * 100, 1)
            for i in range(4)
        }

    }
if __name__=="__main__":
    import librosa
    test_file=librosa.ex('trumpet')
    result=predict_emotion(test_file)
    print(f"Emotion:{result['emotion']}({result['confidence']}%confidence)")
    print(f"All scores: {result['scores']}")
