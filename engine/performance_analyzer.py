import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from engine.predict_emotion import predict_emotion
from engine.feature_extractor import extract_features_from_file
TARGET_PROFILES = {
    "HAPPY":  {"zcr_high": True,  "centroid_high": True,  "tempo_fast": True,  "rms_high": True},
    "SAD":    {"zcr_high": False, "centroid_high": False, "tempo_fast": False, "rms_high": False},
    "ANGRY":  {"zcr_high": True,  "centroid_high": True,  "tempo_fast": True,  "rms_high": True},
    "CALM":   {"zcr_high": False, "centroid_high": False, "tempo_fast": False, "rms_high": False},
}

FEATURE_LABELS = {
    "zcr_high":      ("Zero Crossing Rate", "too noisy/harsh", "too smooth/flat"),
    "centroid_high": ("Spectral Brightness", "too bright/thin", "too dark/dull"),
    "tempo_fast":    ("Tempo", "too fast/rushed", "too slow/dragging"),
    "rms_high":      ("Energy/Dynamics", "too loud/aggressive", "too quiet/weak"),
}

def analyze_performance(audio_path:str,target_emotion:str)->dict:
    target_emotion=target_emotion.upper()
    if target_emotion not in TARGET_PROFILES:
        return {"error": f"Unknown target emotion: {target_emotion}"}
    
    result=predict_emotion(audio_path)
    detected_emotion=result["emotion"]
    confidence=result["confidence"]
    scores=result["scores"]
    features=extract_features_from_file(audio_path)
    zcr_mean=features[122]
    rms_mean=features[124]
    centroid=features[126]

    import librosa ,numpy as np
    y,sr=librosa.load(audio_path,duration=3.0)
    tempo,_=librosa.beat.beat_track(y=y,sr=sr)
    bpm=float(np.squeeze(tempo))
    actual = {
        "zcr_high":      zcr_mean > 0.05,
        "centroid_high": centroid > 2000.0,
        "tempo_fast":    bpm > 100,
        "rms_high":      rms_mean > 0.05,
    }
    target=TARGET_PROFILES[target_emotion]
    feedback=[]
    score=0

    for key,target_val in target.items():
        label,too_high_msg,too_low_msg=FEATURE_LABELS[key]
        if actual[key]==target_val:
            score+=1
        else:
            msg=too_high_msg if actual[key] else too_low_msg
            feedback.append(f"{label} is {msg} for {target_emotion}")
    
    match_score=round((score/len(target))*100)
    if not feedback:
        feedback.append(f"Performance matches {target_emotion} well.")
    return {
        "target_emotion": target_emotion,
        "detected_emotion": detected_emotion,
        "detected_confidence": confidence,
        "match_score": match_score,
        "feedback": feedback,
        "emotion_scores": scores,
        "bpm": round(bpm, 1),
    }
if __name__ == "__main__":
    import librosa
    test = librosa.ex("trumpet")
    result = analyze_performance(test, "HAPPY")
    print(f"Target:   {result['target_emotion']}")
    print(f"Detected: {result['detected_emotion']} ({result['detected_confidence']}%)")
    print(f"Match:    {result['match_score']}%")
    for f in result["feedback"]:
        print(f"  - {f}")



