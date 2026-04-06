import librosa
import numpy as np
import scipy.ndimage as ndimage

# Because test_mic.py is in the engine/ folder, we can just import directly
from fingerprinter import generate_hashes
from db import recognize_audio

print("Simulating Microphone Recording...")

# Load the trumpet
y, sr = librosa.load(librosa.ex('trumpet'))

# CHOP OUT A TINY 1-SECOND SNIPPET FROM THE MIDDLE OF THE SONG
# From 1.0 seconds to 2.0 seconds
y_mic = y[sr : sr * 2] 

print("Processing fuzzy microphone audio...")
# Run our pipeline
D = librosa.stft(y_mic)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Extract Peaks
neighborhood = ndimage.maximum_filter(S_db, size=10)
local_maxima = (S_db == neighborhood)
loud_peaks = (S_db > -50)
peak_frequencies, peak_times = np.where(local_maxima & loud_peaks)

# Generate Mic Hashes
mic_hashes = generate_hashes(peak_frequencies, peak_times)

print(f"Extracted {len(mic_hashes)} hashes from 1-second mic snippet.")
print("Querying Database...")

# THE SHAZAM MOMENT
result = recognize_audio(mic_hashes)
print("\n====================")
print(result)
print("====================\n")