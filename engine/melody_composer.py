import random
import numpy as np

SCALE_INTERVALS = {
    "major":      [0,2,4,5,7,9,11],
    "minor":      [0,2,3,5,7,8,10],
    "pentatonic": [0,2,4,7,9],
    "chromatic":  [0,1,2,3,4,5,6,7,8,9,10,11],
}

EMOTION_PARAMS = {
    "HAPPY":  {"mode":"major",     "octave":5, "leap_prob":0.3, "rest_prob":0.05, "direction": 0.6},
    "SAD":    {"mode":"minor",     "octave":4, "leap_prob":0.1, "rest_prob":0.15, "direction":-0.7},
    "ANGRY":  {"mode":"chromatic", "octave":4, "leap_prob":0.5, "rest_prob":0.02, "direction": 0.0},
    "CALM":   {"mode":"pentatonic","octave":4, "leap_prob":0.1, "rest_prob":0.2,  "direction": 0.2},
    "UNCERTAIN":{"mode":"minor",   "octave":4, "leap_prob":0.2, "rest_prob":0.1,  "direction": 0.0},
}

def _build_scale(root_midi: int, mode: str) -> list:
    intervals = SCALE_INTERVALS[mode]
    scale = []
    for octave_offset in [-12, 0, 12]:
        for interval in intervals:
            note = root_midi + interval + octave_offset
            if 48 <= note <= 84:
                scale.append(note)
    return sorted(set(scale))

def compose_melody(emotion: str, root_midi: int = 60, length: int = 20, bpm: float = 120.0) -> list:
    params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["CALM"])
    scale = _build_scale(root_midi, params["mode"])

    if not scale:
        scale = _build_scale(60, "major")

    direction = params["direction"]
    leap_prob = params["leap_prob"]
    notes = []
    current_idx = len(scale) // 2

    for step in range(length):
        if random.random() < params["rest_prob"]:
            notes.append(notes[-1] if notes else scale[current_idx])
            continue

        if random.random() < leap_prob:
            jump = random.choice([-4,-3,3,4])
        else:
            drift = 1 if random.random() < (0.5 + direction * 0.3) else -1
            jump = drift

        current_idx = max(0, min(len(scale)-1, current_idx + jump))

        if step % 4 == 3:
            phrase_target = int(len(scale) * (0.5 + direction * 0.2))
            phrase_target = max(0, min(len(scale)-1, phrase_target))
            if abs(current_idx - phrase_target) > 3:
                current_idx += 1 if current_idx < phrase_target else -1

        notes.append(scale[current_idx])

    return notes
