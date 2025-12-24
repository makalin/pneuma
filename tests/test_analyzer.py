# tests/test_analyzer.py

import numpy as np
import pytest
from cortex.analyzer import Analyzer

@pytest.fixture
def analyzer():
    return Analyzer(sample_rate=44100)

@pytest.fixture
def dummy_audio():
    # 2 seconds of a simple sine wave
    sr = 44100
    duration = 2
    frequency = 440
    t = np.linspace(0., duration, int(sr * duration))
    audio = 0.5 * np.sin(2. * np.pi * frequency * t)
    return audio

def test_analyzer_output_keys(analyzer, dummy_audio):
    """Tests that the analyzer returns the expected feature keys."""
    features = analyzer.analyze(dummy_audio)
    expected_keys = [
        'rmse', 'spectral_centroid', 'spectral_bandwidth', 
        'spectral_rolloff', 'zero_crossing_rate', 'onset_density', 
        'tempo', 'chroma'
    ]
    assert sorted(list(features.keys())) == sorted(expected_keys)

def test_analyzer_chroma_shape(analyzer, dummy_audio):
    """Tests that the chroma feature has the correct shape (12 pitch classes)."""
    features = analyzer.analyze(dummy_audio)
    assert features['chroma'].shape == (12,)

def test_analyzer_feature_types(analyzer, dummy_audio):
    """Tests that all features are of a numeric type."""
    features = analyzer.analyze(dummy_audio)
    for key, value in features.items():
        if key == 'chroma':
            assert isinstance(value, np.ndarray)
        else:
            assert isinstance(value, (float, int, np.number))

