# psyche/personality.py
# State Machine (Cooperativeness, Temperament variables)

import random

class Personality:
    def __init__(self, persona_config):
        """
        Initializes the personality with a given configuration.
        `persona_config` is a dictionary defining the personality traits.
        """
        self.cooperativeness = persona_config.get('cooperativeness', 0.5)
        self.temperament = persona_config.get('temperament', 0.5) # e.g., 0.0=calm, 1.0=volatile
        self.mood = "neutral" # Can be 'inspired', 'bored', 'annoyed', etc.

    def update(self, vibe):
        """
        Updates the personality state based on the musical vibe.
        """
        if vibe == "aggressive":
            # High temperament might lead to annoyance
            if random.random() < self.temperament:
                self.mood = "annoyed"
                self.cooperativeness -= 0.1
            else:
                self.mood = "inspired"
                self.cooperativeness += 0.1
        elif vibe == "melancholic":
            self.mood = "thoughtful"
            self.cooperativeness += 0.05
        else: # neutral
            self.mood = "neutral"

        # Clamp cooperativeness between 0 and 1
        self.cooperativeness = max(0, min(1, self.cooperativeness))

    def get_state(self):
        return {
            "cooperativeness": self.cooperativeness,
            "temperament": self.temperament,
            "mood": self.mood
        }

if __name__ == '__main__':
    # Example persona
    heckler_persona = {
        'cooperativeness': 0.2,
        'temperament': 0.9 
    }
    
    personality = Personality(heckler_persona)
    print(f"Initial state: {personality.get_state()}")
    
    print("\nReacting to 'aggressive' vibe...")
    personality.update("aggressive")
    print(f"New state: {personality.get_state()}")
    
    print("\nReacting to 'melancholic' vibe...")
    personality.update("melancholic")
    print(f"New state: {personality.get_state()}")
