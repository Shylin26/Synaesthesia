import os
import sys
import librosa
import numpy as np
import scipy.ndimage as ndimage
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.fingerprinter import generate_hashes
from engine.db import store_song ,get_connection
SUPPORTED = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
def get_existing_songs():
    conn=get_connection()
    cursor=conn.cursor()
    cursor.execute("SELECT name FROM songs")
    names = {row[0] for row in cursor.fetchall()}
    conn.close()
    return names

def fingerprint_file(file_path:str)->list:
    y,sr=librosa.load(file_path,mono=True)
    D=librosa.stft(y)
    S_db=librosa.amplitude_to_db(np.abs(D),ref=np.max)
    neighbourhood=ndimage.maximum_filter(S_db,size=10)
    local_maxima=(S_db==neighbourhood)
    loud_peaks=(S_db>-50)
    peak_freqs,peak_times=np.where(local_maxima&loud_peaks)
    return generate_hashes(peak_freqs,peak_times)

def index_folder(folder_path:str):
    existing=get_existing_songs()
    audio_files=[]
    for root,dirs,files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(SUPPORTED):
                audio_files.append(os.path.join(root,f))
    print(f"Found{len(audio_files)} audio files")
    skipped=0
    indexed=0
    failed=0
    for file_path in tqdm(audio_files,desc="Indexing"):
        song_name = os.path.splitext(os.path.basename(file_path))[0]
        if song_name in existing:
            skipped+=1
            continue
        try:
            hashes=fingerprint_file(file_path)
            store_song(song_name,hashes)
            existing.add(song_name)
            indexed+=1
        
        except Exception as e:
            print(f"\nFailed: {song_name} — {e}")
            failed += 1

    print(f"\nDone. Indexed: {indexed} | Skipped: {skipped} | Failed: {failed}")

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("Usage: python indexer.py <folder_path>")
        sys.exit(1)
    index_folder(sys.argv[1])






