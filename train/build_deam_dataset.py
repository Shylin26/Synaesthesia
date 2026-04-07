import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.feature_extractor import extract_features_from_file


DEAM_DIR = "/Users/parishachauhan/Downloads"
AUDIO_DIR = "/Users/parishachauhan/Downloads/MEMD_audio"
ANNOT_PATH = "/Users/parishachauhan/Downloads/annotations/annotations per each rater/song_level/static_annotations_songs_1_2000.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'deam_dataset.npz')


def build():
    df = pd.read_csv(ANNOT_PATH)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} annotations.")
    print(f"Columns: {list(df.columns)}")

    # Average valence and arousal per song
    df_avg = df.groupby('SongId').agg(
        valence_mean=('Valence', 'mean'),
        arousal_mean=('Arousal', 'mean')
    ).reset_index()
    df_avg.columns = ['song_id', 'valence_mean', 'arousal_mean']
    print(f"Unique songs: {len(df_avg)}")

    X, valence, arousal = [], [], []

    for _, row in tqdm(df_avg.iterrows(), total=len(df_avg), desc="Extracting features"):
        song_id = int(row['song_id'])
        v = float(row['valence_mean'])
        a = float(row['arousal_mean'])

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



