import random
from pathlib import Path

VA_MAP = {
    "HAPPY":     (0.7,  0.6),
    "SAD":       (-0.6, -0.5),
    "ANGRY":     (-0.4, 0.75),
    "CALM":      (0.55, -0.6),
    "UNCERTAIN": (0.0,  0.0),
}

BORROWED_CHORDS = {
    "SAD":  [[53,57,60], [56,60,63], [51,55,58]],
    "CALM": [[53,57,60], [56,60,63]],
}

def get_borrowed_chords(emotion: str) -> list:
    return BORROWED_CHORDS.get(emotion, [])

def _get_tension_notes(emotion: str, key: str) -> list:
    root = KEYS.index(key) if key in KEYS else 0
    if emotion in ("SAD", "CALM"):
        return [(root+3)%12+60, (root+8)%12+60, (root+10)%12+60]
    if emotion == "ANGRY":
        return [(root+6)%12+60, (root+1)%12+60]
    return [(root+4)%12+60, (root+7)%12+60, (root+11)%12+60]
PROGRESSIONS = {
    "HAPPY": {
        "progressions": [["I","IV","V","I"], ["I","V","vi","IV"], ["I","IV","I","V"]],
        "mode": "major",
        "instruments": ["piano","acoustic guitar","bass","drums"],
        "description": "Bright and energetic — major key with strong forward motion"
    },
    "SAD": {
        "progressions": [["i","VI","III","VII"], ["i","iv","i","V"], ["i","VI","VII","i"]],
        "mode": "minor",
        "instruments": ["piano","strings","cello"],
        "description": "Melancholic and introspective — minor key with descending motion"
    },
    "ANGRY": {
        "progressions": [["i","VII","VI","VII"], ["i","v","VI","VII"], ["i","III","VII","i"]],
        "mode": "minor",
        "instruments": ["electric guitar","bass","drums","synth"],
        "description": "Tense and aggressive — dissonant intervals with driving rhythm"
    },
    "CALM": {
        "progressions": [["I","V","vi","IV"], ["I","ii","IV","I"], ["I","IV","ii","V"]],
        "mode": "major",
        "instruments": ["piano","acoustic guitar","ambient pad","soft bass"],
        "description": "Peaceful and grounded — pentatonic movement with open voicing"
    },
    "UNCERTAIN": {
        "progressions": [["I","vi","IV","V"], ["i","VI","III","VII"]],
        "mode": "major",
        "instruments": ["piano","pad"],
        "description": "Ambiguous emotional state — neutral harmonic movement"
    }
}

KEYS = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

MAJOR_INTERVALS = {"I":0,"II":2,"III":4,"IV":5,"V":7,"VI":9,"VII":11,"ii":2,"iii":4,"vi":9,"vii":11}
MINOR_INTERVALS = {"i":0,"ii":2,"III":3,"iv":5,"v":7,"VI":8,"VII":10}

def roman_to_chord(roman:str,root:str,mode:str)->str:
    intervals={**MAJOR_INTERVALS,**MINOR_INTERVALS}
    semitone=intervals.get(roman,0)
    root_idx=KEYS.index(root) if root in KEYS else 0
    chord_root=KEYS[(root_idx+semitone)%12]
    if roman.islower():
        return chord_root+"m"
    return chord_root

def generate_chords(emotion:str,valence: float=None,arousal:float=None,key:str=None)->dict:
    if emotion not in PROGRESSIONS:
        emotion="UNCERTAIN"
    if key is None:
        if valence is not None and valence >0:
            key = random.choice(["C","G","D","F","Bb"])
        else:
            key = random.choice(["A","E","D","G"])
    
    data=PROGRESSIONS[emotion]
    progression=random.choice(data["progressions"])
    chords=[roman_to_chord(r,key,data["mode"])for r in progression]
    if arousal is not None:
        tempo=int(60+(arousal+1)*50)
    else:
        tempo=120
    return {
        "key": key,
        "mode": data["mode"],
        "tempo": tempo,
        "progression": chords,
        "roman_numerals": progression,
        "instruments": data["instruments"],
        "mood_description": data["description"],
        "borrowed_chords": get_borrowed_chords(emotion),
        "tension_notes": _get_tension_notes(emotion, key)
    }

def generate_chords_from_pipeline(pipeline_result:dict)->dict:
    emotion=pipeline_result.get("emotion","UNCERTAIN")
    va=VA_MAP.get(emotion,(0.0,0.0))
    valence,arousal=va
    return generate_chords(emotion,valence=valence,arousal=arousal)


if __name__ == "__main__":
    for emotion in ["HAPPY", "SAD", "ANGRY", "CALM"]:
        result = generate_chords(emotion)
        print(f"\n{emotion}:")
        print(f"  Key: {result['key']} {result['mode']}")
        print(f"  Chords: {' - '.join(result['progression'])}")
        print(f"  Tempo: {result['tempo']} BPM")
        print(f"  Instruments: {', '.join(result['instruments'])}")






