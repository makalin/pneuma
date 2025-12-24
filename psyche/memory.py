# psyche/memory.py
# Short-term LSTM (Recall of last 10s)

import torch
import torch.nn as nn
import numpy as np

def flatten_features(features_dict):
    """
    Flattens a dictionary of features (including numpy arrays) into a single tensor.
    """
    feature_vector = []
    # The order of features must be consistent
    for key in sorted(features_dict.keys()):
        value = features_dict[key]
        if isinstance(value, np.ndarray):
            feature_vector.extend(value.flatten())
        else:
            feature_vector.append(value)
    return torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


class ShortTermMemory:
    def __init__(self, input_size, hidden_size, num_layers=1, sequence_length=10):
        """
        A simple LSTM-based short-term memory.
        `input_size` should match the flattened number of features from the analyzer.
        `sequence_length` is the number of past feature sets to consider.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden = self.init_hidden()
        self.memory_sequence = []

    def init_hidden(self, batch_size=1):
        # The hidden state is a tuple of the hidden state and cell state
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def remember(self, features_tensor):
        """
        Adds a new set of features to the memory and updates the LSTM state.
        `features_tensor` should be a tensor of shape (1, 1, input_size).
        """
        self.memory_sequence.append(features_tensor)
        # Keep memory to a fixed size
        if len(self.memory_sequence) > self.sequence_length:
            self.memory_sequence.pop(0)

        # The input to the LSTM should be a sequence
        sequence = torch.cat(self.memory_sequence, dim=1)
        
        # Detach the hidden state to prevent backpropagating through the entire history
        detached_hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        lstm_out, self.hidden = self.lstm(sequence, detached_hidden)
        return lstm_out, self.hidden

    def get_context(self):
        """
        Returns the current hidden state of the LSTM as the context.
        """
        return self.hidden

if __name__ == '__main__':
    # This corresponds to the features from the enhanced analyzer
    # 12 (chroma) + 1 (onset_density) + 1 (rmse) + 1 (spec_band) + 1 (spec_cent) + 1 (spec_roll) + 1 (tempo) + 1 (zcr) = 19
    INPUT_SIZE = 19
    HIDDEN_SIZE = 64

    memory = ShortTermMemory(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)

    # Example features dictionary from the analyzer
    dummy_features_dict = {
        'chroma': np.random.rand(12),
        'onset_density': 0.8,
        'rmse': 0.1,
        'spectral_bandwidth': 2000,
        'spectral_centroid': 1800,
        'spectral_rolloff': 3000,
        'tempo': 120.0,
        'zero_crossing_rate': 0.05
    }

    print(f"--- Testing memory with {INPUT_SIZE} features ---")
    # Simulate feeding features into memory
    for i in range(15):
        # Create a dummy feature tensor
        feature_tensor = flatten_features(dummy_features_dict)
        # Add some noise to make it different each time
        feature_tensor += torch.randn_like(feature_tensor) * 0.1

        print(f"Remembering step {i+1}...")
        lstm_out, hidden_state = memory.remember(feature_tensor)

    print("\nFinal context (hidden state shape):", hidden_state[0].shape)
    print("Final context (cell state shape):", hidden_state[1].shape)
    print("Feature vector shape:", feature_tensor.shape)
