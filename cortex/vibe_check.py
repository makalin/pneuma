# cortex/vibe_check.py
# Classification logic (Aggressive vs. Melancholic)

class VibeCheck:
    def __init__(self):
        # These thresholds are arbitrary and should be tuned
        self.aggresive_thresholds = {
            'rmse': 0.1,
            'spectral_centroid': 2000
        }
        self.melancholic_thresholds = {
            'rmse': 0.05,
            'spectral_centroid': 1000
        }

    def get_vibe(self, features):
        """
        Classifies the vibe based on extracted features.
        Returns a string descriptor (e.g., "aggressive", "melancholic", "neutral").
        """
        if features['rmse'] > self.aggresive_thresholds['rmse'] and \
           features['spectral_centroid'] > self.aggresive_thresholds['spectral_centroid']:
            return "aggressive"
        elif features['rmse'] < self.melancholic_thresholds['rmse'] and \
             features['spectral_centroid'] < self.melancholic_thresholds['spectral_centroid']:
            return "melancholic"
        else:
            return "neutral"

if __name__ == '__main__':
    vibe_check = VibeCheck()
    
    aggressive_features = {'rmse': 0.2, 'spectral_centroid': 2500}
    melancholic_features = {'rmse': 0.01, 'spectral_centroid': 800}
    neutral_features = {'rmse': 0.08, 'spectral_centroid': 1500}

    print(f"Aggressive features -> Vibe: {vibe_check.get_vibe(aggressive_features)}")
    print(f"Melancholic features -> Vibe: {vibe_check.get_vibe(melancholic_features)}")
    print(f"Neutral features -> Vibe: {vibe_check.get_vibe(neutral_features)}")
