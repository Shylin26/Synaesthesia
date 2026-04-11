import os
import sys
import shutil
import tempfile
import math
import random
import pretty_midi
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Initialize DB on startup
try:
    from engine.db import setup_db
    from engine.emotion_tracker import setup_tracker
    setup_db()
    setup_tracker()
except Exception as e:
    print(f"DB init warning: {e}")
from engine.pipeline import run_pipeline
from engine.emotion_tracker import get_emotional_arc, new_session_id
from engine.performance_analyzer import analyze_performance
from engine.transition_engine import get_transition_path
from engine.audio_generator import generate_audio
from engine.music_library import get_library, get_track_path
from engine.spotify_recommender import recommend_songs, SpotifyNotConfiguredError
from api.ws_stream import stream_emotion
class MidiRequest(BaseModel):
    notes: List[int]
    emotion: str
    bpm: int = 120
    bass: List[int] = []
    inner: List[int] = []

class IsoRequest(BaseModel):
    current_valence: float
    current_arousal: float
    target_valence: float
    target_arousal: float
    step: int = 0
    total_steps: int = 10

class MusicFeedback(BaseModel):
    session_id: str
    mp: dict
    start_v: float
    start_a: float
    end_v: float
    end_a: float

app = FastAPI(title="SYNAESTHESIA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_frontend():
    response = FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), session_id: Optional[str] = None):
    if not file.filename.endswith((".wav", ".mp3", ".ogg", ".flac", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_pipeline(tmp_path, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return result


@app.get("/session/{session_id}")
def get_session(session_id: str):
    return get_emotional_arc(session_id)


@app.get("/session/new")
def create_session():
    return {"session_id": new_session_id()}


@app.get("/library")
def library():
    return get_library()


@app.get("/library/{track_id}/play")
def play_track(track_id: str):
    path = get_track_path(track_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Track not found")
    return FileResponse(str(path), media_type="audio/wav",
                        filename=f"synaesthesia_{track_id}.wav")


@app.post("/library/save-audio")
async def save_audio_to_library(
    file: UploadFile = File(...),
    emotion: str = "CALM",
    descriptor: str = "calm",
    valence: float = 5.0,
    arousal: float = 5.0,
    bpm: float = 90.0,
    session_id: str = None,
    source: str = "analyse"
):
    from engine.music_library import LIBRARY_DIR, _load_index, _save_index
    import uuid
    from datetime import datetime
    import soundfile as sf
    import numpy as np

    track_id = str(uuid.uuid4())[:8]
    filename = f"{track_id}_{emotion.lower()}_{descriptor.replace(' ','_')[:20]}.wav"
    filepath = LIBRARY_DIR / filename

    audio_bytes = await file.read()
    with open(str(filepath), 'wb') as f:
        f.write(audio_bytes)

    try:
        data, sr = sf.read(str(filepath))
        duration = round(len(data) / sr, 1)
    except Exception:
        duration = 0.0

    entry = {
        "id": track_id,
        "filename": filename,
        "emotion": emotion,
        "descriptor": descriptor,
        "valence": round(valence, 2),
        "arousal": round(arousal, 2),
        "bpm": round(bpm, 1),
        "session_id": session_id,
        "source": source,
        "created_at": datetime.utcnow().isoformat(),
        "duration": duration,
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


@app.post("/analyze-performance")
async def analyze_performance_endpoint(
    file: UploadFile = File(...),
    target_emotion: str = "HAPPY"
):
    if not file.filename.endswith((".wav", ".mp3", ".ogg", ".flac", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = analyze_performance(tmp_path, target_emotion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return result


@app.get("/health")
def health():
    return {"status": "ok", "service": "SYNAESTHESIA"}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await stream_emotion(websocket)


@app.get("/transition/{current_emotion}")
def transition(current_emotion: str, target: str = "CALM"):
    return get_transition_path(current_emotion, target)


@app.get("/spotify/recommend")
def spotify_recommend(valence: float = 5.0, arousal: float = 5.0, limit: int = 5):
    try:
        tracks = recommend_songs(valence, arousal, limit)
        return {"tracks": tracks}
    except SpotifyNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/iso-step")
def iso_step(req: IsoRequest):
    progress = req.step / max(req.total_steps, 1)
    v = req.current_valence + (req.target_valence - req.current_valence) * progress
    a = req.current_arousal + (req.target_arousal - req.current_arousal) * progress
    tempo = int(max(50, min(180, 80 + a * 60)))
    if v >= 0.3:   mode = "major"
    elif v >= -0.1: mode = "dorian"
    elif v >= -0.5: mode = "minor"
    else:           mode = "phrygian"
    reverb_wet = round(max(0.05, min(0.6, 0.15 + (-a) * 0.3)), 2)
    note_density = round(max(0.1, min(0.9, 0.3 + a * 0.4)), 2)
    return {
        "step": req.step,
        "progress": round(progress, 3),
        "current_v": round(v, 3),
        "current_a": round(a, 3),
        "tempo": tempo,
        "mode": mode,
        "reverb_wet": reverb_wet,
        "note_density": note_density,
        "description": f"Step {req.step}/{req.total_steps} — {mode} at {tempo} BPM"
    }


@app.post("/generate-audio")
async def generate_audio_endpoint(
    file: UploadFile = File(...),
    duration: float = 8.0
):
    if not file.filename.endswith((".wav", ".mp3", ".ogg", ".flac", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        from engine.predict_emotion_v2 import predict_emotion_v2
        emotion_result = predict_emotion_v2(tmp_path)
        descriptor = emotion_result.get("descriptor", "neutral")
        musical_params = emotion_result.get("musical_params")
        audio_path = generate_audio(descriptor, duration=duration, musical_params=musical_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return FileResponse(audio_path, media_type="audio/wav",
                        filename=f"synaesthesia_generated.wav")

@app.get("/compose/{emotion}")
def compose_endpoint(emotion: str, bpm: float = 120.0):
    from engine.melody_composer import compose_arrangement
    from engine.predict_emotion_v2 import va_to_musical_params, EMOTION_DESCRIPTORS
    
    # Route 1: Derive Emotion Profile
    matched = None
    for d in EMOTION_DESCRIPTORS:
        if d[2].replace(" ", "") == emotion.lower().replace(" ", ""):
            matched = d
            break
            
    if matched:
        mp = va_to_musical_params(matched[0], matched[1])
    else:
        mp = va_to_musical_params(0, 0)
        
    # Route 2: Generate Algorithmic Baseline Foundation (Bass/Inner Harmony)
    arrangement = compose_arrangement(emotion.upper(), 60, 20, bpm, musical_params=mp)
    
    # Route 3: NEURAL COMPOSER OVERRIDE (The Masterpiece Sequence)
    # If the user has trained and generated 'melody_lstm.pt', we override the melody with pure AI.
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'melody_lstm.pt')
    if os.path.exists(model_path):
        import torch
        try:
            from engine.melody_lstm import EmotionConditionedLSTM, generate_conditional_melody
            model = EmotionConditionedLSTM(vocab_size=128, emotion_dim=4)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            emotion_vector = torch.tensor([[mp["syncopation"], mp["dissonance"], mp["arpeggiation"], mp["groove"]]], dtype=torch.float32)
            start_seq = torch.tensor([[60, 62, 64, 65]]) # Base C-major scale prompt, but network will deviate 
            
            neural_melody = generate_conditional_melody(model, start_seq, emotion_vector, length=16, temperature=1.1)
            arrangement["melody"] = neural_melody
        except Exception as e:
            print("Neural Network Override Failed - Falling back to math logic:", e)

    return {
        "melody": arrangement["melody"],
        "bass": arrangement["bass"],
        "inner": arrangement["inner"],
        "musical_params": mp
    }

@app.post("/feedback/music")
def collect_music_feedback(payload: MusicFeedback):
    from engine.emotion_tracker import log_music_feedback
    log_music_feedback(
        session_id=payload.session_id,
        mp=payload.mp,
        start_v=payload.start_v,
        start_a=payload.start_a,
        end_v=payload.end_v,
        end_a=payload.end_a
    )
    return {"status": "success", "message": "Feedback loop closed."}

@app.get("/therapist/narrative/{session_id}")
def therapist_narrative(session_id: str):
    from engine.therapist import generate_session_narrative
    narrative = generate_session_narrative(session_id)
    return {"narrative": narrative}


@app.post("/export-midi")
def export_midi(req: MidiRequest):
    EMOTION_BPM = {"HAPPY": 128, "SAD": 72, "ANGRY": 145, "CALM": 88}
    bpm = EMOTION_BPM.get(req.emotion, req.bpm)
    spb = 60.0 / bpm
    note_dur = spb * 0.45

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    PROGRAMS = {"melody": 0, "bass": 32, "inner": 4}
    OCTAVE_SHIFT = {"melody": 0, "bass": 0, "inner": 0}

    voices = {"melody": req.notes}
    if hasattr(req, "bass") and req.bass:
        voices["bass"] = req.bass
    if hasattr(req, "inner") and req.inner:
        voices["inner"] = req.inner

    for voice_name, pitches in voices.items():
        instrument = pretty_midi.Instrument(program=PROGRAMS.get(voice_name, 0))
        for i, pitch in enumerate(pitches):
            jitter = random.uniform(-0.015, 0.015)
            sine_vel = math.sin(i * math.pi / 4)
            base_vel = 85 if voice_name == "melody" else (65 if voice_name == "inner" else 75)
            velocity = int(max(40, min(127, base_vel + sine_vel * 12 + random.randint(-6, 6))))
            start = max(0.0, i * spb * 0.5 + jitter)
            dur = note_dur * (1.8 if voice_name == "bass" else 1.0)
            note = pretty_midi.Note(velocity=velocity, pitch=int(pitch),
                                    start=start, end=start + dur)
            instrument.notes.append(note)
        midi.instruments.append(instrument)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        midi.write(tmp.name)
        tmp_path = tmp.name

    return FileResponse(tmp_path, media_type="audio/midi",
                        filename=f"synaesthesia_{req.emotion.lower()}_melody.mid")
