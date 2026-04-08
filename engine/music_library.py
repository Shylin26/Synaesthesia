import os
import json
import uuid
import numpy as np
import soundfile as sf
from datetime import datetime
from pathlib import Path

LIBRARY_DIR = Path(__file__).parent.parent / "data" / "library"
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = LIBRARY_DIR / "index.json"

def _load_index() -> list:
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            return json.load(f)
    return []

def _save_index(index: list):
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

def save_melody_to_library(
    notes: list,
    bass: list,
    inner: list,
    emotion: str,
    descriptor: str,
    valence: float,
    arousal: float,
    bpm: float,
    session_id: str = None,
    sample_rate: int = 44100,
) -> dict:
    track_id = str(uuid.uuid4())[:8]
    filename = f"{track_id}_{emotion.lower()}_{descriptor.replace(' ','_')}.wav"
    filepath = LIBRARY_DIR / filename

    duration_per_note = 60.0 / bpm * 0.5
    total_duration = len(notes) * duration_per_note
    num_samples = int(total_duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    def midi_to_freq(midi): return 440.0 * (2 ** ((midi - 69) / 12))

    def add_note(pitch, start_time, duration, amplitude=0.3, wave="sine"):
        start = int(start_time * sample_rate)
        end = int((start_time + duration) * sample_rate)
        end = min(end, num_samples)
        t = np.linspace(0, duration, end - start, endpoint=False)
        freq = midi_to_freq(pitch)
        if wave == "sine":
            wave_data = np.sin(2 * np.pi * freq * t)
        else:
            wave_data = np.sign(np.sin(2 * np.pi * freq * t)) * 0.5
        envelope = np.ones(len(t))
        attack = int(0.05 * sample_rate)
        release = int(0.2 * sample_rate)
        if attack < len(envelope):
            envelope[:attack] = np.linspace(0, 1, attack)
        if release < len(envelope):
            envelope[-release:] = np.linspace(1, 0, release)
        audio[start:end] += wave_data * envelope * amplitude

    for i, note in enumerate(notes):
        add_note(note, i * duration_per_note, duration_per_note * 0.9, amplitude=0.35)

    for i, note in enumerate(bass):
        if i % 2 == 0:
            add_note(note, i * duration_per_note, duration_per_note * 1.8, amplitude=0.25)

    for i, note in enumerate(inner):
        if i % 2 == 1:
            add_note(note, i * duration_per_note, duration_per_note * 0.8, amplitude=0.15)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.85

    sf.write(str(filepath), audio, sample_rate)

    entry = {
        "id": track_id,
        "filename": filename,
        "emotion": emotion,
        "descriptor": descriptor,
        "valence": round(valence, 2),
        "arousal": round(arousal, 2),
        "bpm": round(bpm, 1),
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "duration": round(total_duration, 1),
    }

    index = _load_index()
    index.insert(0, entry)
    if len(index) > 100:
        old = index.pop()
        old_path = LIBRARY_DIR / old["filename"]
        if old_path.exists():
            old_path.unlink()
    _save_index(index)

    return entry

def get_library() -> list:
    return _load_index()

def get_track_path(track_id: str) -> Path:
    index = _load_index()
    for entry in index:
        if entry["id"] == track_id:
            return LIBRARY_DIR / entry["filename"]
    return None
