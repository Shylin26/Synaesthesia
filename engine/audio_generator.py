import os
import sys
import torch
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

MODEL_ID = "facebook/musicgen-small"
_model = None
_processor = None

EMOTION_PROMPTS = {
    "euphoric":    "euphoric upbeat electronic dance music, bright synths, energetic, 128 bpm",
    "happy":       "happy acoustic guitar pop, bright major key, uplifting, warm",
    "content":     "content mellow indie pop, acoustic guitar, gentle, warm",
    "peaceful":    "peaceful ambient piano, soft, gentle, slow, major key",
    "calm":        "calm ambient piano with soft strings, slow, spacious, reverb",
    "serene":      "serene ambient music, soft piano, nature sounds, very slow, peaceful",
    "melancholic": "melancholic piano, minor key, slow, emotional, reverb, introspective",
    "sad":         "sad piano ballad, minor key, slow tempo, emotional, strings",
    "depressed":   "dark ambient, very slow, minor key, heavy reverb, sparse piano",
    "anxious":     "tense ambient, dissonant strings, unsettling, chromatic, building",
    "angry":       "aggressive rock, distorted guitar, fast, intense, minor key",
    "furious":     "heavy metal, very fast, distorted, aggressive, chromatic",
    "excited":     "excited pop, fast tempo, bright, energetic, synths",
    "energetic":   "energetic indie rock, guitar, drums, fast, major key",
    "tense":       "tense cinematic, strings, building tension, minor key, slow",
    "neutral":     "neutral ambient piano, moderate tempo, balanced, gentle",
}

def _load_model():
    global _model, _processor
    if _model is None:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        print("Loading MusicGen small (~300MB, first time only)...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
        _model.eval()
        print("MusicGen loaded.")
    return _model, _processor

def generate_audio(descriptor: str, duration: float = 8.0, musical_params: dict = None) -> str:
    model, processor = _load_model()

    base_prompt = EMOTION_PROMPTS.get(descriptor, EMOTION_PROMPTS["neutral"])

    if musical_params:
        mode = musical_params.get("mode", "minor")
        tempo = musical_params.get("tempo", 90)
        mode_str = {"major": "major key", "minor": "minor key", "dorian": "dorian mode", "phrygian": "phrygian mode"}.get(mode, "minor key")
        base_prompt = f"{base_prompt}, {mode_str}, {tempo} bpm"

    inputs = processor(text=[base_prompt], padding=True, return_tensors="pt")

    max_new_tokens = int(duration * 50)

    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_data = audio_values[0, 0].numpy()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/tmp")
    sf.write(tmp.name, audio_data, sampling_rate)
    return tmp.name


if __name__ == "__main__":
    path = generate_audio("melancholic", duration=5.0)
    print(f"Generated: {path}")
