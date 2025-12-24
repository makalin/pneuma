# voice/hallucinator.py
# Generates raw audio from Psyche intent

from voice.neural_synth import NeuralSynth

class Hallucinator:
    def __init__(self, model_path=None):
        """
        The Hallucinator uses the neural synthesizer to generate audio.
        """
        self.synth = NeuralSynth(model_path)

    def generate(self, intent, context):
        """
        Generates audio based on intent and context.
        `intent` is the command from the DecisionMaker.
        `context` is the musical data from the analyzer.
        """
        print(f"Hallucinating with intent: '{intent}'")
        raw_audio = self.synth.synthesize(intent, context)
        return raw_audio

if __name__ == '__main__':
    # This requires the NeuralSynth to be defined
    hallucinator = Hallucinator(model_path="corpus/checkpoints/my_soul.pth")
    
    # Dummy data
    test_intent = "harmonize"
    test_context = {"rmse": 0.5, "spectral_centroid": 1500} # Simplified context

    # Generate audio
    audio_output = hallucinator.generate(test_intent, test_context)
    
    print(f"\nGenerated audio buffer of shape: {audio_output.shape}")
    # In a real application, you would now play this audio.
