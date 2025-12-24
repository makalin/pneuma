# cortex/listener.py
# PyAudio stream handler (Real-time buffer)

import pyaudio
import numpy as np

class Listener:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size)

    def listen(self):
        """Returns a chunk of audio data from the stream."""
        data = np.frombuffer(self.stream.read(self.chunk_size), dtype=np.float32)
        return data

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    listener = Listener()
    print("Listening...")
    try:
        while True:
            data = listener.listen()
            print(f"RMS: {np.sqrt(np.mean(data**2)):.4f}")
    except KeyboardInterrupt:
        print("Stopping.")
        listener.stop()
