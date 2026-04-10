<div align="center">

# SYNAESTHESIA
### *Music Emotion AI — Hear. Feel. Create.*

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange?style=flat-square)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

**An end-to-end music intelligence system that listens to audio, decodes its emotional DNA using a custom Transformer trained on 1,744 real music clips, and composes original multi-voice music in response — all in real time.**

</div>

---

## What This Actually Does

Most music apps tell you what song is playing. SYNAESTHESIA tells you what it *feels* — and then creates something new from that feeling.

Upload any audio. The system:

1. **Fingerprints it** — Shazam-style constellation map + SHA-1 hashing, identifies the track from a database of indexed songs
2. **Decodes its emotion** — a Transformer encoder trained on the DEAM dataset (1,744 annotated music clips) predicts continuous valence and arousal coordinates on the Russell Circumplex Model, mapping to 16 nuanced emotional descriptors
3. **Composes a response** — a multi-voice arrangement (melody + bass + inner harmony) generated using music theory rules derived from the emotional coordinates, with mode selection (major/dorian/minor/phrygian/harmonic minor), tempo, reverb, and note density all driven by the V/A values
4. **Guides you** — the Therapy mode implements the **Iso-Principle** from music therapy: music that matches your current emotional state, then gradually shifts toward your target state over 12 steps, visualized as moving dots on a real-time valence/arousal canvas

---

## Architecture

```
Audio Input
    │
    ├── Fingerprinting Engine
    │   └── Constellation map → SHA-1 hashing → SQLite time-alignment matching
    │
    ├── Emotion Regressor (custom Transformer)
    │   ├── 176-dim feature vector (MFCC + Chroma + Spectral Contrast +
    │   │   Tonnetz + Spectral Flux + Mel Spectrogram)
    │   ├── TransformerEncoder (d_model=64, 2 layers, LayerNorm)
    │   ├── Trained on DEAM dataset — 1,744 songs, continuous V/A labels
    │   └── Output: valence ∈ [1,9], arousal ∈ [1,9] → 16 descriptors
    │
    ├── Melody Composer
    │   ├── V/A → musical parameters (mode, tempo, density, reverb)
    │   ├── Motif-based generation with phrase development
    │   └── 3-voice arrangement: melody + bass + inner harmony
    │
    └── FastAPI Backend (9 endpoints + WebSocket)
```

---

## Five Modes

| Mode | What it does |
|------|-------------|
| **Analyse** | Upload or record audio → emotion detection + melody generation + chord progression |
| **Therapy** | Iso-Principle journey — music that mirrors your current state and guides you toward a target emotion over 12 steps |
| **Education** | Upload a performance → compare detected emotion against intended emotion → actionable feedback per audio feature |
| **Memory** | Session emotional arc — valence/arousal timeline, dominant emotion, trend analysis |
| **Live** | Real-time WebSocket streaming — emotion detected every 2 seconds from microphone |

---

## Technical Highlights

**Emotion Detection**
- Custom Transformer encoder with LayerNorm, trained from scratch on DEAM
- 176-dimensional feature vector including Tonnetz (harmonic relationships) and Mel spectrogram statistics
- Continuous valence/arousal regression using Huber loss with cosine annealing
- 16 nuanced emotional descriptors mapped from the Russell Circumplex Model
- MAE ~0.7 on a 1-9 scale on held-out DEAM test set

**Music Generation**
- Emotion-conditioned multi-voice composer: melody, bass, inner harmony
- Mode selection from V/A coordinates: major → dorian → minor → phrygian → harmonic minor
- Motif-based melodic development — 4-note phrase established and varied across the piece
- Humanized MIDI export: ±15ms timing jitter + sine-wave velocity envelope
- 3-track MIDI with emotion-appropriate instrument programs

**Iso-Principle Therapy Engine**
- Linear interpolation between current and target V/A coordinates over 12 steps
- Music parameters (mode, tempo, reverb, note density) morph continuously
- Real-time visualization: red dot (current state) → white dot (music position) → green dot (target)
- Session feedback logged to SQLite for longitudinal analysis

**Shazam-Style Fingerprinting**
- Spectrogram → constellation map (local maxima extraction)
- Fan-out hashing: each peak paired with 15 future peaks → SHA-1 hash
- Time-alignment matching: O(1) hash lookup + histogram voting
- Batch indexer for entire music libraries

---

## Stack

```
Backend:   Python 3.9, PyTorch 2.8, librosa, FastAPI, SQLite, pretty_midi, soundfile
Frontend:  Vanilla JS, Tone.js (audio synthesis), Web Audio API, Canvas API
ML:        Custom Transformer (PyTorch), scikit-learn, numpy
Data:      DEAM (1,744 clips), RAVDESS (1,440 clips), synthetic music theory corpus
```

---

## Running Locally

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

## Research Context

SYNAESTHESIA sits at the intersection of Music Information Retrieval (MIR), affective computing, and generative AI. The core contribution is the end-to-end pipeline connecting audio fingerprinting, continuous emotion regression from music, emotion-conditioned music generation, and the Iso-Principle from clinical music therapy — implemented as a real-time interactive system.

The valence/arousal regression approach treats music emotion as a continuous 2D space rather than discrete categories — consistent with the psychological literature (Russell, 1980; Thayer, 1989) and more nuanced than classification-based approaches used in prior work.

---

## What's Next

- [ ] Spotify integration — recommend real songs matching detected V/A coordinates
- [ ] MusicGen integration — actual audio generation
- [ ] Deployment
- [ ] Research paper submission

---

<div align="center">
Built by Parisha Chauhan · Undergraduate Research Project
</div>
