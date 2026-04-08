import os
import requests
import json
from engine.emotion_tracker import get_emotional_arc, get_longitudinal_context

def generate_session_narrative(session_id: str) -> str:
    """ Generates the brilliant second-person journal entry using Groq LLM API """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "I need a GROQ_API_KEY exported in your terminal to write your emotional story!"
    
    arc = get_emotional_arc(session_id)
    history = get_longitudinal_context()
    
    prompt = f"""
    Act as a highly empathetic, brilliant psychoanalyst and journaling AI for the Synaesthesia engine.
    The user just completed a generative music therapy session where their brainwaves/voice were tracked.
    
    [CURRENT SESSION DATA]
    - Dominant Emotion: {arc.get('dominant_emotion')}
    - Average Valence: {arc.get('avg_valence')} (-1.0 to 1.0 scale)
    - Average Arousal: {arc.get('avg_arousal')} (-1.0 to 1.0 scale)
    - Length of Session: {arc.get('total_readings')} intervals.
    
    [LONGITUDINAL PATTERNS]
    - This user has generated {history['total_historical_datapoints']} emotional data points entirely.
    - {len(history['successful_music_profiles'])} times generative music has successfully improved their valence/mood in the past.

    [TASK]
    Write a beautifully crafted, one-paragraph "emotional story" of what happened in the session.
    Write it in the second person ("You started tense..."). 
    Do not be clinical. Sound like a deeply observant, philosophical journal entry exploring how the music shifted them.
    Do not output any markdown, titles, or intro text—just the raw paragraph.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Use llama3-8b for blazing fast conversational depth
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are Synaesthesia, a brilliant empathetic system that reads human patterns."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.75,
        "max_tokens": 300
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Could not generate narrative... Therapist offline: {str(e)}"
