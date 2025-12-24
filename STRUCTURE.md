# PNEUMA: Anatomical Structure

Pneuma is not structured like an application; it is structured like a nervous system. The codebase is divided into three distinct biological functions: **Sensation (Input)**, **Psyche (Processing)**, and **Expression (Output)**.

## Directory Tree

```text
pneuma/
├── corpus/                 # The Memory Bank
│   ├── raw_audio/          # Untrained .wav files (your past jams)
│   ├── processed/          # Spectrograms and feature extractions
│   └── checkpoints/        # Saved personality weights (The "Soul" snapshots)
│
├── cortex/                 # SENSATION (The Ear)
│   ├── __init__.py
│   ├── listener.py         # PyAudio stream handler (Real-time buffer)
│   ├── analyzer.py         # Librosa feature extraction (Timbre, Density, BPM)
│   └── vibe_check.py       # Classification logic (Aggressive vs. Melancholic)
│
├── psyche/                 # PSYCHOLOGY (The Brain)
│   ├── __init__.py
│   ├── personality.py      # State Machine (Cooperativeness, Temperament variables)
│   ├── memory.py           # Short-term LSTM (Recall of last 10s)
│   └── decision.py         # Logic: "Should I harmonize or disrupt?"
│
├── voice/                  # EXPRESSION (The Throat)
│   ├── __init__.py
│   ├── neural_synth.py     # DDSP / RAVE inference engine
│   ├── hallucinator.py     # Generates raw audio from Psyche intent
│   └── streamer.py         # Outputs audio stream (OSC/Virtual Cable)
│
├── config/
│   ├── bio_metrics.yaml    # Sensitivity settings (Input gain, reaction time)
│   └── persona.yaml        # Personality definitions (e.g., "The Heckler", " The Shadow")
│
├── main.py                 # The Spark of Life (Entry point)
├── requirements.txt
└── README.md

```

## Core Data Flow

1. **Auditory Input** -> `cortex.listener` captures the live signal.
2. **Perception** -> `cortex.analyzer` extracts the "Vibe" (not just notes).
3. **Cognition** -> `psyche.decision` compares Input Vibe against Current Mood + Memory.
4. **Intent** -> `psyche` sends a high-level command (e.g., "Swell", "Fracture", "Echo") to `voice`.
5. **Manifestation** -> `voice.neural_synth` renders the audio and pushes it to the speakers.

