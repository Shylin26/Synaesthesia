import numpy as np
import scipy.ndimage as ndimage
import hashlib
try:
    from db import store_song
except ModuleNotFoundError:
    from engine.db import store_song

def generate_hashes(peak_frequencies, peak_times):
    """
    Takes constellation peaks and generates (hash, time_offset) pairs.
    """
    # Combine them into a list of (time, frequency) tuples and sort by time
    peaks = list(zip(peak_times, peak_frequencies))
    peaks.sort(key=lambda x: x[0])
    
    hashes = []
    
    # We use a fan-out of 15. This means every anchor point gets paired 
    # with the next 15 points that occur after it in time.
    FAN_VALUE = 15
    
    for i in range(len(peaks)):
        anchor = peaks[i]
        
        # Loop through the next 15 target points
        for j in range(1, FAN_VALUE + 1):
            if (i + j) < len(peaks):
                target = peaks[i + j]
                
                anchor_freq = anchor[1]
                target_freq = target[1]
                
                # The time difference between target and anchor
                delta_time = target[0] - anchor[0]
                
                # We only want target points that are actually in the future!
                if delta_time > 0:
                    # Create a string combining the two frequencies and the time difference
                    hash_string = f"{anchor_freq}|{target_freq}|{delta_time}"
                    
                    # Convert that string into a strict SHA-1 hash
                    hash_obj = hashlib.sha1(hash_string.encode('utf-8'))
                    hex_hash = hash_obj.hexdigest()[:20]
                    
                    # Save the hash and the absolute time the anchor occurred
                    hashes.append((hex_hash, int(anchor[0])))
                    
    return hashes

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import numpy as np
    print("Loading real audio sample...")
    y,sr=librosa.load(librosa.ex('trumpet'))
    print("Computing STFT to generate Spectrogram...")
    D=librosa.stft(y)
    S=np.abs(D)
    S_db=librosa.amplitude_to_db(S,ref=np.max)
    print("Drawing spectrogram...")
    plt.figure(figsize=(10,5))
    librosa.display.specshow(S_db,sr=sr,x_axis='time',y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    # plt.show()
    sr=44100
    t=np.linspace(0,1,sr,endpoint=False)
    wave_100hz=np.sin(2*np.pi*100*t)
    wave_500hz=np.sin(2*np.pi*500*t)
    wave_1000hz=np.sin(2*np.pi*1000*t)
    signal=wave_100hz+wave_500hz+wave_1000hz

    print("Computing FFT...")
    X=np.fft.fft(signal)
    magnitudes=np.abs(X)
    half_sr=sr//2
    frequencies=np.linspace(0,half_sr,half_sr)
    #plt.plot(frequencies, magnitudes[:half_sr])
    #plt.xlabel("Frequency (Hz)")
    #plt.ylabel("Magnitude")
    #plt.xlim(0, 1200) 
    #plt.show()

    print("Extracting MFCCS for AI training ...")
    mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
    print(f"Spectrogram shape:{S_db.shape}")
    print(f"MFCCs shape:{mfccs.shape}")

    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfccs,x_axis='time',sr=sr)
    plt.colorbar()
    plt.title('MFCC (Mel-Frequency Cepstral Coefficients)')
    plt.tight_layout()
    #plt.show()

    print("Generating constellations map...")
    neighbourhood=ndimage.maximum_filter(S_db,size=10)
    local_maxima=(S_db==neighbourhood)

    background_threshold=-50
    loud_peaks=(S_db>background_threshold)
    constellation_map=local_maxima&loud_peaks
    peak_frequencies,peak_times=np.where(constellation_map)
    print(f"Extracted{len(peak_times)}constellation peaks !")

    plt.figure(figsize=(10,5))
    plt.scatter(peak_times,peak_frequencies,color='red',s=10)

    plt.gca().invert_yaxis()
    plt.title('Constellation Map (Audio Fingerprint)')
    plt.xlabel("Time Frame")
    plt.ylabel('Frequency bin')
    plt.tight_layout()
    plt.show()
    print("Hashing peaks...")
    song_hashes = generate_hashes(peak_frequencies, peak_times)
    
    store_song("Trumpet Solo", song_hashes)



    
