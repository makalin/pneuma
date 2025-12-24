# tests/test_memory.py

import torch
import numpy as np
import pytest
from psyche.memory import flatten_features, ShortTermMemory

@pytest.fixture
def feature_dict():
    """A sample feature dictionary as returned by the Analyzer."""
    return {
        'chroma': np.random.rand(12),
        'onset_density': 0.8,
        'rmse': 0.1,
        'spectral_bandwidth': 2000.0,
        'spectral_centroid': 1800.0,
        'spectral_rolloff': 3000.0,
        'tempo': 120.0,
        'zero_crossing_rate': 0.05
    }

def test_flatten_features(feature_dict):
    """Tests that the feature dictionary is flattened into a tensor of the correct shape."""
    tensor = flatten_features(feature_dict)
    # Shape should be (1, 1, num_features)
    assert tensor.shape == (1, 1, 19)
    assert isinstance(tensor, torch.Tensor)

def test_memory_initialization():
    """Tests the initialization of the ShortTermMemory module."""
    memory = ShortTermMemory(input_size=19, hidden_size=64)
    assert memory.input_size == 19
    assert memory.hidden_size == 64
    assert isinstance(memory.lstm, torch.nn.LSTM)

def test_memory_remember(feature_dict):
    """Tests that the memory module can process a feature tensor."""
    memory = ShortTermMemory(input_size=19, hidden_size=64)
    feature_tensor = flatten_features(feature_dict)
    
    # The `remember` method should not throw an error
    try:
        lstm_out, hidden_state = memory.remember(feature_tensor)
    except Exception as e:
        pytest.fail(f"memory.remember() raised an exception: {e}")

    # Check the output shapes
    assert lstm_out.shape == (1, 1, 64) # (batch, seq_len, hidden_size)
    assert hidden_state[0].shape == (1, 1, 64) # (num_layers, batch, hidden_size)
    assert hidden_state[1].shape == (1, 1, 64) # (num_layers, batch, hidden_size)

def test_memory_sequence_length(feature_dict):
    """Tests that the memory sequence is capped at the correct length."""
    memory = ShortTermMemory(input_size=19, hidden_size=64, sequence_length=5)
    
    for i in range(10):
        feature_tensor = flatten_features(feature_dict)
        memory.remember(feature_tensor)

    assert len(memory.memory_sequence) == 5
