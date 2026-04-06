import math
VA_COORDS = {
    "HAPPY":  (0.7,  0.6),
    "SAD":    (-0.6, -0.5),
    "ANGRY":  (-0.4, 0.75),
    "CALM":   (0.55, -0.6),
}
TRANSITION_GRAPH = {
    "ANGRY": ["ANGRY", "HAPPY", "CALM"],
    "SAD":   ["SAD",   "CALM",  "HAPPY"],
    "HAPPY": ["HAPPY", "CALM"],
    "CALM":  ["CALM"],
}
TRANSITION_DESCRIPTIONS = {
    ("ANGRY", "HAPPY"): "Channel the energy — shift from tension to brightness",
    ("ANGRY", "CALM"):  "Release the tension — let the energy dissolve into stillness",
    ("SAD",   "CALM"):  "Sit with the feeling — let stillness hold the sadness",
    ("SAD",   "HAPPY"): "Gentle lift — from introspection toward warmth",
    ("HAPPY", "CALM"):  "Wind down — from bright energy to peaceful rest",
    ("CALM",  "CALM"):  "You're already in a good place — stay here",
}
def get_transition_path(current_emotion:str,target_emotion:str="CALM")->dict:
    current_emotion=current_emotion.upper()
    target_emotion=target_emotion.upper()
    if current_emotion not in VA_COORDS:
        current_emotion="CALM"
    if target_emotion not in VA_COORDS:
        target_emotion="CALM"
    
    path=TRANSITION_GRAPH.get(current_emotion,["CALM"])
    if target_emotion not in path:
        path=[current_emotion,target_emotion]
    steps=[]
    for i,emotion in enumerate(path):
        v,a=VA_COORDS[emotion]
        key=(current_emotion,emotion) if i>0 else(emotion,emotion)
        desc=TRANSITION_DESCRIPTIONS.get(key,f"Moving toward {emotion}")
        steps.append({
            "step": i + 1,
            "emotion": emotion,
            "valence": v,
            "arousal": a,
            "description": desc,
            "suggested_tempo": int(60 + (a + 1) * 50),
        })
    c1,c2=VA_COORDS[current_emotion]
    t1,t2=VA_COORDS[target_emotion]
    distance=round(math.sqrt((t1-c1)**2 + (t2-c2)**2), 3)
    return {
        "current_emotion": current_emotion,
        "target_emotion": target_emotion,
        "steps": steps,
        "emotional_distance": distance,
        "num_steps": len(steps),
    }

if __name__=="__main__":
    result = get_transition_path("ANGRY", "CALM")
    print(f"Path: {result['current_emotion']} → {result['target_emotion']}")
    print(f"Distance: {result['emotional_distance']}")
    for step in result["steps"]:
        print(f"  Step {step['step']}: {step['emotion']} — {step['description']}")
