# SYNAESTHESIA

> *Most music apps tell you what song is playing. This one tells you what it feels — then composes something new from that feeling.*

<br>

An end-to-end music intelligence system built from scratch. Upload any audio. SYNAESTHESIA fingerprints it, decodes its emotional coordinates on the Russell Circumplex Model using a custom Transformer trained on 1,744 real music clips, and composes an original multi-voice arrangement in response — melody, bass, and inner harmony, all driven by continuous valence and arousal values.

No pre-built emotion APIs. No Spotify SDK. No shortcuts.

---

## What it actually does

```
Audio input
    │
    ├── Shazam-style fingerprinting
    │   Constellation map → SHA-1 hashing → SQLite time-alignment matching
    │
    ├── Emotion regression  
    │   176-dim feature vector → custom TransformerEncoder → valence ∈ [1,9], arousal ∈ [1,9]
    │   Trained from scratch on DEAM dataset — 1,744 annotated music clips
    │   MAE ~0.7 on held-out test set · 16 nuanced emotional descriptors
    │
    ├── Emotion-conditioned composition
    │   V/A coordinates → mode (major/dorian/minor/phrygian/harmonic minor)
    │   → tempo → note density → reverb → 3-voice MIDI arrangement
    │   Motif-based melodic development · humanized timing ±15ms jitter
    │
    └── Iso-Principle therapy engine
        Linear interpolation across 12 steps from current → target emotional state
        Music parameters morph continuously · real-time VA canvas visualization
```

---

## Five modes

**Analyse** — Upload or record audio. Get emotion coordinates, a generated chord progression, and an original multi-voice melody composed in response.

**Therapy** — The Iso-Principle from clinical music therapy, implemented. Music that mirrors your current emotional state, then guides you toward your target over 12 interpolated steps. Visualized as moving dots on a real-time valence/arousal canvas.

**Education** — Upload a performance against a target emotion. Get feature-level feedback: where your audio diverges from the intended emotional signature, and why.

**Memory** — Your emotional arc across sessions. Valence/arousal timeline, dominant emotion, trend analysis, longitudinal session history stored in SQLite.

**Live** — Real-time WebSocket streaming. Emotion detected every 2 seconds from microphone. Continuous VA canvas updates as you play or speak.

---

## Why this is hard

Emotion in music is not a classification problem. Happy/Sad/Angry/Calm loses the nuance that makes music emotionally powerful. The psychological literature has known this since Russell (1980) — emotion is a continuous 2D space, not discrete buckets.

SYNAESTHESIA treats it that way. The model outputs continuous valence and arousal coordinates and maps them to 16 nuanced descriptors — *triumphant*, *melancholic*, *tense*, *serene* — that actually mean something.

The Iso-Principle is a real clinical technique used by music therapists: start where the patient is emotionally, not where you want them to be. Moving someone from anxious to calm directly doesn't work. You match the state first, then shift gradually. Implementing this as a generative system — where the music parameters interpolate continuously across 12 steps — is not something you find in a tutorial.

---

## Technical depth

**Emotion detection**
- Custom TransformerEncoder: `d_model=64`, 2 layers, LayerNorm, trained from scratch
- 176-dimensional feature vector: MFCC (40 coefficients) + Chroma + Spectral Contrast + Tonnetz + Spectral Flux + Mel Spectrogram statistics
- Huber loss with cosine annealing scheduler
- Continuous V/A regression on DEAM — not classification, not fine-tuned, trained from scratch

**Music generation**
- V/A → mode selection: `major → dorian → minor → phrygian → harmonic minor`
- Motif-based melodic development — 4-note phrase established and varied across the piece
- 3-voice arrangement: melody + bass + inner harmony
- Humanized MIDI export: ±15ms timing jitter + sine-wave velocity envelope
- Emotion-appropriate instrument programs per track

**Fingerprinting**
- Spectrogram → constellation map via local maxima extraction
- Fan-out hashing: each peak paired with 15 future peaks → SHA-1 hash
- Time-alignment matching: O(1) hash lookup + histogram voting
- Batch indexer for entire music libraries

**Iso-Principle engine**
- Linear interpolation between current and target V/A across 12 steps
- All musical parameters (mode, tempo, reverb, note density) morph at each step
- Real-time visualization: red dot (current) → white dot (music position) → green dot (target)

---

## Stack

```
ML          PyTorch 2.8 · scikit-learn · NumPy
Audio       librosa · soundfile · pretty_midi · Web Audio API
Backend     Python 3.9 · FastAPI · SQLite · WebSocket
Frontend    Vanilla JS · Tone.js · Canvas API
Data        DEAM (1,744 clips) · RAVDESS (1,440 clips)
```

Zero third-party emotion APIs. Zero pre-trained music models. Every model trained from scratch.

---

## Run it

```bash
git clone https://github.com/Shylin26/Synaesthesia
cd Synaesthesia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from engine.db import setup_db; from engine.emotion_tracker import setup_tracker; setup_db(); setup_tracker()"
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000`

---

## Research context

SYNAESTHESIA sits at the intersection of Music Information Retrieval, affective computing, and generative AI.

The core contribution is the end-to-end pipeline: audio fingerprinting → continuous emotion regression → emotion-conditioned music generation → Iso-Principle therapeutic interpolation, implemented as a real-time interactive system.

Treating music emotion as a continuous 2D space rather than discrete categories is consistent with the psychological literature (Russell, 1980; Thayer, 1989) and more expressive than classification-based approaches used in prior systems. The generative component closes the loop — not just detecting emotion in music, but using those coordinates to compose music that responds to and gradually shifts emotional state.

---

*Built by Parisha Chauhan — Undergraduate, B.Tech Computer Science*
