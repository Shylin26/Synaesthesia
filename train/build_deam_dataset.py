import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.feature_extractor import extract_features_from_file


DEAM_DIR="/Users/parishachauhan/Downloads/DEAM"
AUDIO_DIR=os.path.join(DEAM_DIR,"audio","clips_45seconds")
ANNOT_PATH = os.path.join(DEAM_DIR, "annotations", "static_annotations_averaged_songs_1_2000.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'deam_dataset.npz')


def build():
    df=pd.read_csv(ANNOT_PATH)
    df.columns=df.columns.str.strip()
    print(f"Loaded {len(df)} annotations.")
    print(f"Columns: {list(df.columns)}")
    X,valence,arousal=[],[],[]
    for _, row in tqdm(df.iterrows(),total=len(df),desc="Extracting features"):
        song_id=int(row['song_id'])
        v=float(row['valence_mean'])
        a=float(row['arousal_mean'])

        audio_path=os.path.join(AUDIO_DIR,f"{song_id}.mp3")
        if not os.path.exists(audio_path):
            continue
        try:
            features=extract_features_from_file(audio_path,duration=30.0)
            X.append(features)
            valence.append(v)
            arousal.append(a)
        except Exception as e:
            print(f"Skipping {song_id}: {e}")
    
    X=np.array(X)
    valence=np.array(valence)
    arousal=np.array(arousal)
    np.savez(OUTPUT_PATH,X=X,valence=valence,arousal=arousal)
    print(f"\nSaved {X.shape[0]} samples → {OUTPUT_PATH}")
    print(f"Valence range: {valence.min():.2f} – {valence.max():.2f}")
    print(f"Arousal range: {arousal.min():.2f} – {arousal.max():.2f}")

if __name__=="__main__":
    build()



