import sqlite3
import os
import uuid
from datetime import datetime
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fingerprints.db')
VA_MAP = {
    "HAPPY":     (0.7,  0.6),
    "SAD":       (-0.6, -0.5),
    "ANGRY":     (-0.4, 0.75),
    "CALM":      (0.55, -0.6),
    "UNCERTAIN": (0.0,  0.0),
}
def setup_tracker():
    conn=sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emotion_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            emotion TEXT,
            confidence REAL,
            secondary_emotion TEXT,
            valence REAL,
            arousal REAL,
            bpm REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_emotion(session_id:str,pipeline_result:dict):
    emotion=pipeline_result['emotion']
    valence,arousal=VA_MAP.get(emotion,(0.0,0.0))
    secondary=pipeline_result.get("secondary")
    secondary_label=secondary["emotion"] if secondary else None
    conn=sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO emotion_sessions
        (session_id, emotion, confidence, secondary_emotion, valence, arousal, bpm, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        emotion,
        pipeline_result["confidence"],
        secondary_label,
        valence,
        arousal,
        pipeline_result.get("bpm", 0.0),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def get_session(session_id:str)->list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT emotion, confidence, secondary_emotion, valence, arousal, bpm, timestamp
        FROM emotion_sessions WHERE session_id = ? ORDER BY timestamp ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {"emotion":r[0],"confidence":r[1],"secondary":r[2],"valence":r[3],"arousal":r[4],"bpm":r[5],"timestamp":r[6]}
        for r in rows
    ]
def get_emotional_arc(session_id: str) -> dict:
    readings = get_session(session_id)
    if not readings:
        return {"readings": [], "dominant_emotion": None, "avg_valence": 0, "avg_arousal": 0}
    avg_valence = sum(r["valence"] for r in readings) / len(readings)
    avg_arousal = sum(r["arousal"] for r in readings) / len(readings)
    emotion_counts = {}
    for r in readings:
        emotion_counts[r["emotion"]] = emotion_counts.get(r["emotion"], 0) + 1
    dominant = max(emotion_counts, key=emotion_counts.get)
    return {
        "readings": readings,
        "dominant_emotion": dominant,
        "avg_valence": round(avg_valence, 3),
        "avg_arousal": round(avg_arousal, 3),
        "total_readings": len(readings)
    }


def new_session_id()->str:
    return str(uuid.uuid4())[:8]

if __name__=="__main__":
    setup_tracker()
    print("Emotion tracker DB ready.")



    