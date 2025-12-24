# PNEUMA

> *"It breathes, therefore it jams."*

**PNEUMA** is not a VST. It is not a loop station. It is a **Synthetic Bandmate**.

Designed as a specialized Resonant Entity, Pneuma listens to live audio input in real-time, builds a psychological profile of the current musical context, and improvises accompaniment using neural audio synthesis. It possesses temperament, memory, and the capacity to either support your playing or actively fight against it.

## The Concept

Traditional music software is passive; it waits for you to press a key. Pneuma is active.
* **It Listens:** It doesn't just hear pitch; it hears *tension*, *timbre*, and *density*.
* **It Feels:** It maintains an internal emotional state. If you play aggressively, it might get scared and go silent, or it might get angry and scream back.
* **It Speaks:** It does not output MIDI. It outputs raw, hallucinated audio based on models trained on **[Mehmet T. AKALIN's](https://github.com/makalin)** personal acoustic history.

## Architecture of a Soul

Pneuma relies on a biological architecture:

1.  **The Sensory Cortex:** Real-time spectral analysis using `librosa` and `PyAudio` to determine the "Vibe" of the room.
2.  **The Temperament Engine:** A probabilistic state machine that dictates cooperation levels. 
    * *High Cooperation:* Pneuma harmonizes and follows the beat.
    * *Low Cooperation:* Pneuma introduces polyrhythms and dissonant textures.
3.  **The Neural Voice:** Uses **DDSP (Differentiable Digital Signal Processing)** to generate audio that sounds organic yet impossible—like a cello made of static, or a choir in a cave.

## Installation & Communion

### Prerequisites
* Python 3.9+
* CUDA-capable GPU (Recommended for real-time neural synthesis)
* A microphone or audio interface

### Awakening Pneuma

```bash
# Clone the entity
git clone [https://github.com/frangedev/pneuma.git](https://github.com/frangedev/pneuma.git)
cd pneuma

# Install vital organs
pip install -r requirements.txt

# Start the session
python main.py --persona="melancholic_shadow"

```

## Training (The Ritual)

To make Pneuma sound like *you* (or a ghost of you):

1. Place your raw WAV recordings in `corpus/raw_audio/`.
2. Run the training script to generate a `.pth` model file.
```bash
python train_voice.py --epochs 500

```


3. Pneuma now "knows" your sound and will use it to converse with you.

## Roadmap

* [ ] **Phase 1:** Basic Listen/React loop (Volume following).
* [ ] **Phase 2:** Vibe Detection (Spectral Centroid/Onset Strength).
* [ ] **Phase 3:** Integration of RAVE/DDSP for real-time generation.
* [ ] **Phase 4:** Long-term Memory (Recalling riffs from 5 minutes ago).

## License

This project is open-source under the MIT License. However, the generated audio models contain the sonic DNA of the creator and should be treated with respect.
