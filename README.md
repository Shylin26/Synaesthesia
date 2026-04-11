<div align="center">

# SYNAESTHESIA

**Music doesn't just play. It feels.**

*An end-to-end music intelligence system that decodes the emotional DNA of any audio and composes original music in response — built from scratch using custom Transformers, trained on real music psychology datasets, and deployed as a full-stack web application.*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-synaesthesia.onrender.com-black?style=for-the-badge)](https://synaesthesia.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange?style=flat-square)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green?style=flat-square)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

</div>

---

## The Idea

Most music apps tell you *what* is playing. SYNAESTHESIA tells you *what it feels* — and then creates something new from that feeling.

Upload any audio. In seconds, the system:

- **Identifies it** — Shazam-style audio fingerprinting using constellation maps and SHA-1 hashing
- **Understands it** — a custom Transformer encoder predicts continuous valence and arousal coordinates on the Russell Circumplex Model, mapping to 16 nuanced emotional descriptors (*melancholic, euphoric, tense, serene...*)
- **Responds to it** — composes an original multi-voice arrangement (melody + bass + inner harmony) with mode, tempo, reverb, and note density all derived from the emotional coordinates
- **Guides you** — the Therapy mode implements the **Iso-Principle** from clinical music therapy: music that mirrors your current emotional state, then gradually shifts toward your target state over 12 steps

---

## What Makes This Different

This is not a wrapper around an existing model. Every component was built from scratch:

| Component | What it is |
|-----------|-----------|
| Emotion Regressor | Custom TransformerEncoder trained on DEAM — 1,744 real music clips with continuous valence/arousal annotations |
| Feature Extractor | 176-dimensional audio feature vector: MFCC + Chroma + Spectral Contrast + Tonnetz + Spectral Flux + Mel Spectrogram |
| Melody Composer | Music-theory-driven multi-voice generator with motif development, mode selection from V/A coordinates, and humanized MIDI export |
| Fingerprinting Engine | Shazam-style constellation map → SHA-1 hashing → time-alignment matching in SQLite |
| Iso-Principle Engine | Real-time valence/arousal interpolation with continuous Tone.js music synthesis that morphs between emotional states |

---

## Architecture

```
Audio Input
    │
    ├─── Fingerprinting
    │    └── Spectrogram → Constellation Map → SHA-1 Hashes → SQLite Lookup
    │
    ├─── Emotion Regression  
    │    ├── 176-dim features (MFCC, Chroma, Tonnetz, Mel, Spectral Flux)
    │    ├── TransformerEncoder (d_model=64, LayerNorm, Huber Loss)
    │    ├── Trained on DEAM — 1,744 songs, continuous V/A labels
    │    └── Output: valence ∈ [1,9], arousal ∈ [1,9] → 16 descriptors
    │
    ├─── Music Generation
    │    ├── V/A → mode (major/dorian/minor/phrygian/harmonic minor)
    │    ├── V/A → tempo, note density, reverb, leap probability
    │    ├── Motif-based melody with phrase development
    │    └── 3-voice arrangement: melody + bass + inner harmony
    │
    └─── FastAPI Backend
         ├── 9 REST endpoints + WebSocket
         ├── Spotify recommendations via audio feature matching
         └── Music library with browser-side WAV rendering
```

---

## Five Modes

### Analyse
Upload or record audio. Get emotion detection with 16 nuanced descriptors, a generated multi-voice melody, chord progression with music theory context, Spotify song recommendations matched to the detected V/A coordinates, and a downloadable MIDI file with humanized timing.

### Therapy — Iso-Principle
Select your current emotional state and where you want to be. The system generates a 12-step musical journey using the Iso-Principle from clinical music therapy: music starts by matching your current state, then gradually interpolates toward the target. Watch the transition in real time on a valence/arousal canvas — red dot (you), green dot (target), white dot (where the music is now).

### Education
Upload a musical performance and select the emotion you were trying to express. The system compares the detected emotion against your intent and provides feature-level feedback: *"Spectral brightness is too dark for HAPPY — try playing in a higher register"*.

### Memory
Your emotional arc across a session. Valence and arousal plotted over time, dominant emotion, trend analysis. Every analysis is logged to SQLite with timestamps.

### Live
Real-time emotion detection via WebSocket. Streams 2-second audio chunks from your microphone, detects emotion continuously, updates the display live.

---

## Technical Depth

**Emotion Detection**
- Continuous regression on the Russell Circumplex Model — not discrete classification
- 176-dimensional feature vector including Tonnetz (harmonic relationships between notes) and Mel spectrogram statistics
- Huber loss with cosine annealing learning rate schedule
- MAE ~0.7 on a 1-9 scale on held-out DEAM test set
- 16 emotional descriptors mapped from V/A coordinates using nearest-neighbor in descriptor space

**Music Generation**
- Mode selection from V/A: positive valence → major/lydian, negative → minor/phrygian/harmonic minor
- Motif-based melodic development: 4-note phrase established, then repeated, varied, and developed across the piece
- Humanized MIDI: ±15ms timing jitter + sine-wave velocity envelope simulating human "pulse"
- 3-track MIDI export with emotion-appropriate instrument programs (piano, strings, bass)
- Browser-side WAV rendering using OfflineAudioContext — what you hear is what you save

**Iso-Principle Implementation**
- Linear interpolation between current and target V/A coordinates over 12 steps
- Music parameters (mode, tempo, reverb wet, note density) morph continuously at each step
- Tone.js Transport loop with continuous playback — not one-shot notes
- Session feedback logged to SQLite for longitudinal analysis

**Audio Fingerprinting**
- Spectrogram → local maxima extraction (constellation map)
- Fan-out hashing: each peak paired with 15 future peaks → SHA-1 → 20-char hex hash
- Time-alignment matching: O(1) hash lookup + histogram voting for song identification
- Batch indexer for entire music libraries

---

## Stack

```
ML:        PyTorch 2.8, scikit-learn, numpy, librosa
Backend:   FastAPI, SQLite, pretty_midi, soundfile, requests
Frontend:  Vanilla JS, Tone.js, Web Audio API, Canvas API
Data:      DEAM (1,744 clips), RAVDESS (1,440 clips)
Deploy:    Render.com
```

---

## Running Locally

```bash
git clone https://github.com/Shylin26/Synaesthesia
cd Synaesthesia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000`

**Environment variables (optional):**
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

---

## Research Context

SYNAESTHESIA sits at the intersection of Music Information Retrieval (MIR), affective computing, and generative AI. The core contribution is an end-to-end pipeline connecting:

- Audio fingerprinting (signal processing)
- Continuous emotion regression from music (MIR + deep learning)  
- Emotion-conditioned multi-voice music generation (generative AI)
- Iso-Principle implementation from clinical music therapy (real-time systems)

The valence/arousal regression approach treats music emotion as a continuous 2D space rather than discrete categories — consistent with the psychological literature (Russell, 1980; Thayer, 1989) and more nuanced than classification-based approaches used in prior work.

**Potential research directions:**
- Fine-tuning on genre-specific datasets for style-aware generation
- Longitudinal emotional pattern analysis from session history
- Comparison of MFCC-only vs 176-dim features for emotion regression (ablation study)

---

## Project Structure

```
engine/          Core ML and audio processing
  feature_extractor.py    176-dim audio features
  emotion_regressor.py    Custom Transformer for V/A regression
  predict_emotion_v2.py   Inference with 16 descriptors
  melody_composer.py      Multi-voice music generation
  chord_generator.py      Music theory chord progressions
  fingerprinter.py        Shazam-style audio fingerprinting
  transition_engine.py    Iso-Principle path computation
  emotion_tracker.py      Session logging to SQLite
  spotify_recommender.py  Emotion-matched song recommendations

train/           Training pipelines
  build_deam_dataset.py   Feature extraction from DEAM
  train_deam.py           Emotion regressor training
  build_music_dataset.py  Music theory training data
  train_melody_v2.py      Melody transformer training

api/             FastAPI backend
  main.py                 9 endpoints + WebSocket
  ws_stream.py            Real-time emotion streaming

frontend/        Single-page application
  index.html              Vanilla JS, Tone.js, Canvas API
```

---

<div align="center">

Built by **Parisha Chauhan** · Undergraduate Research Project

*"Music gives a soul to the universe, wings to the mind, flight to the imagination, and life to everything." — Plato*

</div>
