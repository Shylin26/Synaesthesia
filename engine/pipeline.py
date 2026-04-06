import os,sys,torch,numpy as np,librosa
import scipy.ndimage as ndimage
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.predict_emotion import predict_emotion,load_model as load_emotion_model
from engine.melody_transformer import MelodyTransformer,generate_melody
from engine.fingerprinter import generate_hashes
from engine.db import recognize_audio
from engine.emotion_tracker import setup_tracker, log_emotion, new_session_id
from engine.chord_generator import generate_chords_from_pipeline

setup_tracker()

MELODY_MODEL_PATH="/Users/parishachauhan/SYNAESTHESIA/models/melody_transformer.pt"
def load_melody_model():
    model=MelodyTransformer()
    model.load_state_dict(torch.load(MELODY_MODEL_PATH,map_location='cpu'))
    model.eval()
    return model
EMOTION_PROMPTS = {
    "HAPPY":     torch.tensor([[60, 64, 67, 72]]),
    "SAD":       torch.tensor([[57, 60, 64, 67]]),
    "ANGRY":     torch.tensor([[60, 61, 66, 67]]),
    "CALM":      torch.tensor([[60, 62, 64, 67]]),
    "UNCERTAIN": torch.tensor([[60, 62, 64, 67]]),  # default to calm
}

EMOTION_IDS = {"HAPPY": 0, "SAD": 1, "ANGRY": 2, "CALM": 3, "UNCERTAIN": 3}
def run_pipeline(audio_path:str, melody_length:int=20, temperature:float=0.8, session_id:str=None)->dict:
    y,sr=librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.squeeze(tempo)), 1)
    auto_temperature = min(1.5, max(0.5, bpm / 120.0))
    D=librosa.stft(y)
    S_db=librosa.amplitude_to_db(np.abs(D),ref=np.max)
    neighbourhood=ndimage.maximum_filter(S_db,size=10)
    local_maxima=(S_db==neighbourhood)
    loud_peaks=(S_db>-50)
    peak_freqs,peak_times=np.where(local_maxima&loud_peaks)
    hashes=generate_hashes(peak_freqs,peak_times)
    song_match=recognize_audio(hashes)
    emotion_result=predict_emotion(audio_path)
    emotion_label=emotion_result["emotion"]
    confidence=emotion_result["confidence"]
    scores=emotion_result['scores']
    melody_model=load_melody_model()
    emotion_id = EMOTION_IDS.get(emotion_label, 3)
    start_sequence=EMOTION_PROMPTS[emotion_label]
    melody_notes = generate_melody(melody_model, start_sequence, emotion_id,
                                   length=melody_length, temperature=auto_temperature)
    result = {
        "song_match": song_match,
        "emotion": emotion_label,
        "confidence": confidence,
        "secondary": emotion_result.get("secondary"),
        "blend": emotion_result.get("blend", emotion_label),
        "emotion_scores": scores,
        "bpm": bpm,
        "melody": melody_notes,
        "melody_length": len(melody_notes),
        "session_id": session_id,
        "chords": generate_chords_from_pipeline(emotion_result)
    }
    if session_id:
        log_emotion(session_id, result)
    return result
   
if __name__ == "__main__":
    import librosa
    test_file = librosa.ex('trumpet')
    result = run_pipeline(test_file)
    print(f"\n=== SYNAESTHESIA RESULT ===")
    print(f"Song Match  : {result['song_match']}")
    print(f"Emotion     : {result['emotion']} ({result['confidence']}% confidence)")
    print(f"All Scores  : {result['emotion_scores']}")
    print(f"Generated   : {result['melody']}")
    print(f"===========================\n")
                        


