# cortex/analyzer.py
# Librosa feature extraction (Timbre, Density, BPM)

import librosa
import numpy as np

class Analyzer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def analyze(self, audio_chunk):
        """
        Analyzes an audio chunk to extract musical features.
        Returns a dictionary of features.
        """
        
        # Basic features
        features = {
            'rmse': np.mean(librosa.feature.rms(y=audio_chunk)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio_chunk, sr=self.sample_rate)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=audio_chunk, sr=self.sample_rate)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio_chunk, sr=self.sample_rate)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y=audio_chunk)),
        }
        
        # Advanced features
        # Onset detection for rhythmic density
        onsets = librosa.onset.onset_detect(y=audio_chunk, sr=self.sample_rate)
        features['onset_density'] = len(onsets) / (len(audio_chunk) / self.sample_rate)

        # Tempo (BPM)
        # This is computationally more expensive and may not be accurate on short chunks.
        # A larger buffer might be needed for reliable BPM detection.
        try:
            tempo, _ = librosa.beat.beat_track(y=audio_chunk, sr=self.sample_rate)
            features['tempo'] = tempo if tempo > 0 else 0
        except librosa.util.exceptions.ParameterError:
            features['tempo'] = 0 # Not enough data to compute tempo

        # Chroma features for harmonic content
        chroma = librosa.feature.chroma_stft(y=audio_chunk, sr=self.sample_rate)
        features['chroma'] = np.mean(chroma, axis=1)

        return features

if __name__ == '__main__':
    # Example usage with a dummy audio chunk
    sample_rate = 44100
    duration = 2  # Use a slightly longer chunk for better analysis
    frequency = 440  # A4
    t = np.linspace(0., duration, int(sample_rate * duration))
    # Create a simple melody with onsets
    audio_chunk = np.zeros_like(t)
    for i in range(int(duration * 2)): # Add some notes
        start = int(i * sample_rate / 2)
        end = start + int(sample_rate / 10)
        note_freq = frequency * (1.059463 ** (i % 12)) # Chromatic scale
        audio_chunk[start:end] = 0.5 * np.sin(2. * np.pi * note_freq * t[start:end])


    analyzer = Analyzer(sample_rate)
    features = analyzer.analyze(audio_chunk)
    print("--- Extracted Features ---")
    for key, value in features.items():
        if key == 'chroma':
            print(f"  {key}:")
            for i, v in enumerate(value):
                print(f"    C{i}: {v:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
