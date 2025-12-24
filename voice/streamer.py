# voice/streamer.py
# Outputs audio stream (OSC/Virtual Cable)

import pyaudio
import numpy as np

class Streamer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.sample_rate,
                                  output=True)

    def play(self, audio_buffer):
        """
        Plays a chunk of audio data.
        `audio_buffer` should be a NumPy array of float32 samples.
        """
        # Ensure data is in the correct format
        if not isinstance(audio_buffer, np.ndarray):
            audio_buffer = np.array(audio_buffer, dtype=np.float32)

        self.stream.write(audio_buffer.tobytes())

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    streamer = Streamer()
    print("Playing a 1-second sine wave at 440 Hz.")

    # Generate a sine wave to test
    sample_rate = 44100
    duration = 1
    frequency = 440
    t = np.linspace(0., duration, sample_rate * duration)
    amplitude = 0.5
    sine_wave = amplitude * np.sin(2. * np.pi * frequency * t)
    
    try:
        streamer.play(sine_wave)
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        print("Stopping stream.")
        streamer.stop()
