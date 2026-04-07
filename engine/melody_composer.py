import random
import numpy as np

SCALE_INTERVALS = {
    "major":      [0,2,4,5,7,9,11],
    "minor":      [0,2,3,5,7,8,10],
    "dorian":     [0,2,3,5,7,9,10],
    "pentatonic": [0,2,4,7,9],
    "chromatic":  [0,1,2,3,4,5,6,7,8,9,10,11],
}

CHORD_PROGRESSIONS = {
    "HAPPY":  [[0,4,7],[5,9,12],[7,11,14],[0,4,7]],
    "SAD":    [[0,3,7],[5,8,12],[7,10,14],[0,3,7]],
    "ANGRY":  [[0,3,6],[5,8,11],[10,13,17],[0,3,6]],
    "CALM":   [[0,4,7],[2,5,9],[4,7,11],[0,4,7]],
    "UNCERTAIN":[[0,3,7],[5,8,12],[7,10,14],[0,3,7]],
}

EMOTION_PARAMS = {
    "HAPPY":    {"mode":"major",      "leap_prob":0.35, "rest_prob":0.04, "direction": 0.6,  "tension":0.1},
    "SAD":      {"mode":"minor",      "leap_prob":0.12, "rest_prob":0.18, "direction":-0.65, "tension":0.4},
    "ANGRY":    {"mode":"chromatic",  "leap_prob":0.55, "rest_prob":0.02, "direction": 0.0,  "tension":0.7},
    "CALM":     {"mode":"pentatonic", "leap_prob":0.08, "rest_prob":0.22, "direction": 0.15, "tension":0.05},
    "UNCERTAIN":{"mode":"dorian",     "leap_prob":0.2,  "rest_prob":0.12, "direction": 0.0,  "tension":0.3},
}

def _build_scale(root_midi: int, mode: str) -> list:
    intervals = SCALE_INTERVALS[mode]
    scale = []
    for octave_offset in [-12, 0, 12]:
        for interval in intervals:
            note = root_midi + interval + octave_offset
            if 36 <= note <= 96:
                scale.append(note)
    return sorted(set(scale))

def _get_chord_notes(root_midi: int, chord_intervals: list) -> list:
    return [root_midi + i for i in chord_intervals if 36 <= root_midi + i <= 96]

def _compose_melody_voice(scale: list, length: int, params: dict, start_idx: int = None) -> list:
    direction = params["direction"]
    leap_prob = params["leap_prob"]
    tension = params["tension"]
    notes = []
    current_idx = start_idx if start_idx is not None else len(scale) // 2 + 2

    for step in range(length):
        if random.random() < params["rest_prob"]:
            notes.append(notes[-1] if notes else scale[current_idx])
            continue

        if random.random() < leap_prob:
            jump = random.choice([-5,-4,-3,3,4,5])
        elif random.random() < tension:
            jump = random.choice([-2,-1,1,2])
        else:
            drift = 1 if random.random() < (0.5 + direction * 0.35) else -1
            jump = drift

        current_idx = max(0, min(len(scale)-1, current_idx + jump))

        if step % 4 == 3:
            phrase_target = int(len(scale) * (0.5 + direction * 0.25))
            phrase_target = max(0, min(len(scale)-1, phrase_target))
            if abs(current_idx - phrase_target) > 4:
                current_idx += 1 if current_idx < phrase_target else -1

        notes.append(scale[current_idx])

    return notes

def _compose_bass_voice(root_midi: int, chord_prog: list, length: int, params: dict) -> list:
    bass_root = root_midi - 24
    notes = []
    for step in range(length):
        chord_idx = (step // 4) % len(chord_prog)
        chord = chord_prog[chord_idx]
        bass_note = bass_root + chord[0]
        bass_note = max(36, min(60, bass_note))
        if step % 4 == 2 and random.random() < 0.4:
            bass_note += chord[1] - chord[0]
        if step % 4 == 3 and random.random() < 0.3:
            next_chord_idx = (chord_idx + 1) % len(chord_prog)
            next_root = bass_root + chord_prog[next_chord_idx][0]
            bass_note = bass_note + (1 if next_root > bass_note else -1)
        notes.append(max(36, min(60, bass_note)))
    return notes

def _compose_inner_voice(root_midi: int, chord_prog: list, scale: list, length: int, params: dict) -> list:
    notes = []
    for step in range(length):
        chord_idx = (step // 4) % len(chord_prog)
        chord = chord_prog[chord_idx]
        chord_notes = _get_chord_notes(root_midi, chord)
        inner_candidates = [n for n in chord_notes if root_midi - 5 <= n <= root_midi + 7]
        if not inner_candidates:
            inner_candidates = chord_notes
        if step % 2 == 0:
            note = random.choice(inner_candidates[1:]) if len(inner_candidates) > 1 else inner_candidates[0]
        else:
            if random.random() < params["tension"]:
                scale_candidates = [n for n in scale if root_midi - 3 <= n <= root_midi + 9]
                note = random.choice(scale_candidates) if scale_candidates else inner_candidates[0]
            else:
                note = random.choice(inner_candidates)
        notes.append(note)
    return notes

def compose_melody(emotion: str, root_midi: int = 60, length: int = 20, bpm: float = 120.0) -> list:
    params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["CALM"])
    scale = _build_scale(root_midi, params["mode"])
    if not scale:
        scale = _build_scale(60, "major")
    melody_start = int(len(scale) * 0.6)
    return _compose_melody_voice(scale, length, params, start_idx=melody_start)

def compose_arrangement(emotion: str, root_midi: int = 60, length: int = 20, bpm: float = 120.0) -> dict:
    params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["CALM"])
    scale = _build_scale(root_midi, params["mode"])
    if not scale:
        scale = _build_scale(60, "major")

    chord_prog = CHORD_PROGRESSIONS.get(emotion, CHORD_PROGRESSIONS["CALM"])
    melody_start = int(len(scale) * 0.6)

    melody = _compose_melody_voice(scale, length, params, start_idx=melody_start)
    bass = _compose_bass_voice(root_midi, chord_prog, length, params)
    inner = _compose_inner_voice(root_midi, chord_prog, scale, length, params)

    return {
        "melody": melody,
        "bass": bass,
        "inner": inner,
        "all_voices": melody + bass + inner,
    }
