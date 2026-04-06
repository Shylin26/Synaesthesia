import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
def extract_features(file_path):
    y,sr=librosa.load(file_path,duration=3.0)
    mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40)
    mfccs_mean=np.mean(mfccs,axis=1)
    return mfccs_mean

class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier,self).__init__()
        self.fc1=nn.Linear(40,128)
        self.fc2=nn.Linear(128,64)

        self.fc3=nn.Linear(64,8)
        self.dropout=nn.Dropout(0.3)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        
        x=F.relu(self.fc2(x))
        x=self.dropout(x)

        x=self.fc3(x)
        return x

if __name__=="__main__":
    print("Initialising Emotion AI model...")
    model=EmotionClassifier()
    print("Extracting 40 real features fom trumpet...")
    trumpet_path=librosa.ex('trumpet')
    real_features=extract_features(trumpet_path)
    tensor_features=torch.tensor(real_features,dtype=torch.float32).unsqueeze(0)
    print(f"Tensor shape:{tensor_features.shape}")
    
    predictions=model(tensor_features)

    print("\nModel Architecture:")
    print(model)
    print("\nRaw Output Predictions (8 values representing emotion scores):")
    print(predictions)

