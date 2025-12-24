# psyche/decision.py
# Logic: "Should I harmonize or disrupt?"

import random
import torch

class DecisionMaker:
    def __init__(self, activity_threshold=0.1):
        self.activity_threshold = activity_threshold

    def decide(self, personality_state, memory_context):
        """
        Makes a decision based on personality and memory.
        Returns a high-level command for the voice.
        """
        cooperativeness = personality_state.get('cooperativeness', 0.5)
        mood = personality_state.get('mood', 'neutral')

        # Use the memory context to influence the decision
        memory_activity = 0.0
        if memory_context is not None:
            # A simple metric for 'activity' could be the norm of the hidden state
            hidden_state, _ = memory_context
            memory_activity = torch.norm(hidden_state).item()

        # If memory activity is very low, bias towards silence
        if memory_activity < self.activity_threshold:
            if random.random() < 0.5: # 50% chance of staying silent if it's quiet
                return "silence"

        # High cooperativeness -> more likely to harmonize
        if cooperativeness > 0.7:
            return "harmonize"
        # Low cooperativeness -> more likely to disrupt
        elif cooperativeness < 0.3:
            return "disrupt"
        
        # In-between cooperativeness -> depends on mood and a bit of randomness
        if mood == 'inspired':
            return "harmonize" if random.random() < 0.8 else "echo"
        elif mood == 'annoyed':
            return "disrupt" if random.random() < 0.8 else "silence"
        elif mood == 'thoughtful':
            return "echo" if random.random() < 0.8 else "silence"
        
        # Default action
        return "silence"

if __name__ == '__main__':
    decision_maker = DecisionMaker()

    # Example personality states
    cooperative_state = {'cooperativeness': 0.8, 'mood': 'inspired'}
    disruptive_state = {'cooperativeness': 0.2, 'mood': 'annoyed'}
    thoughtful_state = {'cooperativeness': 0.5, 'mood': 'thoughtful'}

    # Dummy memory context (a tuple of tensors like an LSTM hidden state)
    dummy_hidden = torch.randn(1, 1, 64)
    dummy_cell = torch.randn(1, 1, 64)
    active_context = (dummy_hidden, dummy_cell)
    
    inactive_context = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))

    print("--- Decisions with Active Memory ---")
    print(f"Cooperative state -> Decision: {decision_maker.decide(cooperative_state, active_context)}")
    print(f"Disruptive state  -> Decision: {decision_maker.decide(disruptive_state, active_context)}")
    print(f"Thoughtful state  -> Decision: {decision_maker.decide(thoughtful_state, active_context)}")
    
    print("\n--- Decisions with Inactive Memory ---")
    print(f"Cooperative state -> Decision: {decision_maker.decide(cooperative_state, inactive_context)}")

