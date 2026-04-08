import os
import sys
import tempfile
import json
from fastapi import WebSocket,WebSocketDisconnect
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.predict_emotion_v2 import predict_emotion_v2
async def stream_emotion(websocket:WebSocket):
    await websocket.accept()
    try:
        while True:
            data=await websocket.receive_bytes()
            with tempfile.NamedTemporaryFile(delete=False,suffix='.webm') as tmp:
                tmp.write(data)
                tmp_path=tmp.name
            try:
                result = predict_emotion_v2(tmp_path)
                await websocket.send_text(json.dumps({
                    "emotion":    result["emotion"],
                    "confidence": result["confidence"],
                    "blend":      result.get("descriptor", result["emotion"]),
                    "scores":     result.get("emotion_scores", {}),
                    "musical_params": result.get("musical_params", {})
                }))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
            finally:
                os.remove(tmp_path)
    
    except WebSocketDisconnect:
        pass



