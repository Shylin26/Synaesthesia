import os
import sys
import shutil
import tempfile
import pretty_midi
from pydantic import BaseModel
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.pipeline import run_pipeline
from engine.emotion_tracker import get_emotional_arc, new_session_id
class MidiRequest(BaseModel):
    notes: List[int]
    emotion : str
    bpm : int=120


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


@app.get("/health")
def health():
    return {"status": "ok", "service": "SYNAESTHESIA"}

@app.post("/export-midi")
def export_midi(req: MidiRequest):
    bpm = req.bpm if req.bpm != 120 else {"HAPPY": 128, "SAD": 72, "ANGRY": 145, "CALM": 88}.get(req.emotion, 120)
    seconds_per_beat = 60.0 / bpm
    note_duration = seconds_per_beat * 0.5
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)
    for i, pitch in enumerate(req.notes):
        start = i * seconds_per_beat * 0.5
        note = pretty_midi.Note(velocity=90, pitch=int(pitch),
                                start=start, end=start + note_duration)
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        midi.write(tmp.name)
        tmp_path = tmp.name
    return FileResponse(
        tmp_path,
        media_type="audio/midi",
        filename=f"synaesthesia_{req.emotion.lower()}_melody.mid"
    )
