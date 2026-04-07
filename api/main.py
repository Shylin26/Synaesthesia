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
from engine.pipeline import run_pipeline
from engine.emotion_tracker import get_emotional_arc, new_session_id
from engine.performance_analyzer import analyze_performance
from engine.transition_engine import get_transition_path
from api.ws_stream import stream_emotion
class MidiRequest(BaseModel):
    notes: List[int]
    emotion: str
    bpm: int = 120
    bass: List[int] = []
    inner: List[int] = []


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
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


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
