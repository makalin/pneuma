# tests/test_decision.py

import torch
import pytest
from psyche.decision import DecisionMaker

@pytest.fixture
def decision_maker():
    return DecisionMaker()

@pytest.fixture
def active_context():
    hidden = torch.ones(1, 1, 64) # High activity
    cell = torch.ones(1, 1, 64)
    return (hidden, cell)

@pytest.fixture
def inactive_context():
    hidden = torch.zeros(1, 1, 64) # No activity
    cell = torch.zeros(1, 1, 64)
    return (hidden, cell)


def test_decision_output(decision_maker, active_context):
    """Tests that the decision is one of the valid commands."""
    valid_decisions = ["harmonize", "disrupt", "echo", "silence"]
    
    personality_state = {'cooperativeness': 0.5, 'mood': 'neutral'}
    
    decision = decision_maker.decide(personality_state, active_context)
    assert decision in valid_decisions

def test_decision_cooperative(decision_maker, active_context):
    """A highly cooperative personality should always harmonize."""
    personality_state = {'cooperativeness': 0.9, 'mood': 'neutral'}
    decision = decision_maker.decide(personality_state, active_context)
    assert decision == "harmonize"

def test_decision_disruptive(decision_maker, active_context):
    """A highly uncooperative personality should always disrupt."""
    personality_state = {'cooperativeness': 0.1, 'mood': 'annoyed'}
    decision = decision_maker.decide(personality_state, active_context)
    assert decision == "disrupt"

def test_decision_inactive_memory(decision_maker, inactive_context):
    """
    With no memory activity, the decision maker should be biased towards silence,
    though not guaranteed to be silent every time. We test for a high probability.
    """
    personality_state = {'cooperativeness': 0.5, 'mood': 'neutral'}
    
    # Run it multiple times to check for a trend
    decisions = [decision_maker.decide(personality_state, inactive_context) for _ in range(20)]
    
    # Check if 'silence' is a frequent outcome
    assert decisions.count('silence') > 5, "With inactive memory, silence should be a more common choice."

