import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.feature_extractor import extract_features_from_file
RAVDESS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ravdess', 'Ravdess')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.npz')
EMOTION_MAP={
    1:3,# key is ravdess code value is our class index
    2:3,
    3:0,
    4:1,
    5:2,
    6:2,
    7:1,
    8:0,
}
def build_dataset():
    X,y=[],[]
    wav_files=[]
    for root,dirs,files in os.walk(RAVDESS_DIR):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(os.path.join(root,f))
    print(f"Found{len(wav_files)} audio files.")

    for filepath in tqdm(wav_files,desc="Extracting features"):
        try:
            filename=os.path.basename(filepath)
            emotion_code=int(filename.split('-')[2])
            label=EMOTION_MAP.get(emotion_code)
            if label is None:
                continue
            features=extract_features_from_file(filepath)
            X.append(features)
            y.append(label)
        
        except Exception as e:
            print(f"Skipping {filepath}:{e}")
    
    X=np.array(X)
    y=np.array(y)
    np.savez(OUTPUT_PATH,X=X,y=y)
    print(f"Dataset saved:{X.shape[0]}samples,{X.shape[1]} features each.")
    print(f"Saved to {OUTPUT_PATH}")

if __name__=="__main__":
    build_dataset()
