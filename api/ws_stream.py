import os
import sys
import tempfile
import json
from fastapi import WebSocket,WebSocketDisconnect
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.predict_emotion import predict_emotion
async def stream_emotion(websocket:WebSocket):
    await websocket.accept()
    try:
        while True:
            data=await websocket.receive_bytes()
            with tempfile.NamedTemporaryFile(delete=False,suffix='.webm') as tmp:
                tmp.write(data)
                tmp_path=tmp.name
            try:
                result = predict_emotion(tmp_path)
                await websocket.send_text(json.dumps({
                    "emotion":    result["emotion"],
                    "confidence": result["confidence"],
                    "blend":      result.get("blend", result["emotion"]),
                    "scores":     result["scores"]
                }))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
            finally:
                os.remove(tmp_path)
    
    except WebSocketDisconnect:
        pass



