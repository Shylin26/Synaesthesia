import numpy as np
import torch
import os
CHORD_TONES = {
    "HAPPY": {
        "chords": [[60,64,67], [65,69,72], [67,71,74], [60,64,67]],  # I IV V I C major
        "passing": [62, 64, 65, 67, 69, 71],
        "direction": 1,    # ascending
        "velocity_range": (80, 110),
    },
    "SAD": {
        "chords": [[57,60,64], [62,65,69], [64,67,71], [57,60,64]],  # i iv v i A minor
        "passing": [57, 59, 60, 62, 64, 65, 67],
        "direction": -1,   # descending
        "velocity_range": (50, 75),
    },
    "ANGRY": {
        "chords": [[60,63,66], [65,68,71], [59,62,65], [60,63,66]],  # diminished
        "passing": [60, 61, 63, 65, 66, 68],
        "direction": 0,    # unpredictable
        "velocity_range": (90, 127),
    },
    "CALM": {
        "chords": [[60,64,67], [62,65,69], [64,67,71], [60,64,67]],  # pentatonic
        "passing": [60, 62, 64, 67, 69],
        "direction": 0,    # circular
        "velocity_range": (55, 80),
    },
}
def generate_sequence(emotion_id, seq_len=32, transpose=0):
    emotions=["HAPPY","SAD","ANGRY","CALM"]
    emotion=emotions[emotion_id]
    profile=CHORD_TONES[emotion]
    sequence=[]
    velocities=[]
    chord_idx=0
    chord=profile["chords"][chord_idx]
    current=chord[0]
    for step in range(seq_len+1):
        sequence.append(current)
        v_min,v_max=profile["velocity_range"]
        vel=np.random.randint(v_min,v_max)
        velocities.append(vel)
        if step%8==7:
            chord_idx=(chord_idx+1)%len(profile["chords"])
            chord=profile["chords"][chord_idx]
        
        if np.random.random() < 0.65:
            candidates = chord
        else:
            candidates = profile["passing"]

        distances = [abs(n - current) for n in candidates]
        # Prefer notes at distance 2-5 semitones (melodic movement, not static)
        weights = np.array([1.0 / (abs(d - 3) + 0.5) for d in distances])
        weights /= weights.sum()
        next_note = int(np.random.choice(candidates, p=weights))

        if profile["direction"] == 1 and next_note < current:
            next_note = min(candidates, key=lambda n: abs(n - (current + 2)))
        elif profile["direction"] == -1 and next_note > current:
            next_note = min(candidates, key=lambda n: abs(n - (current - 2)))

        next_note = max(48, min(84, next_note + transpose))
        current = next_note
    return sequence, velocities

def build(num_per_emotion=3000, seq_len=64):
    all_inputs, all_targets, all_emotions = [], [], []

    for emotion_id in range(4):
        for _ in range(num_per_emotion):
            # Randomly transpose by -6 to +6 semitones for variety
            transpose = np.random.randint(-6, 7)
            seq, _ = generate_sequence(emotion_id, seq_len, transpose=transpose)
            all_inputs.append(seq[:-1])
            all_targets.append(seq[1:])
            all_emotions.append(emotion_id)

    X = np.array(all_inputs)
    y = np.array(all_targets)
    emotions = np.array(all_emotions)
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'music_dataset.npz')
    np.savez(out, X=X, y=y, emotions=emotions)
    print(f"Saved {len(all_inputs)} sequences → {out}")

if __name__=="__main__":
    build()
