import pytest
import torch
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

@pytest.fixture
def mock_state():
    """Mock state tensor (144x160 grayscale image)"""
    return np.random.randint(0, 255, (144, 160), dtype=np.uint8)

@pytest.fixture
def mock_transition():
    """Mock transition tuple for replay memory"""
    state = np.random.randint(0, 255, (144, 160), dtype=np.uint8)
    next_state = np.random.randint(0, 255, (144, 160), dtype=np.uint8)
    return Transition(
        state=state,
        action=2,
        reward=1.0,
        next_state=next_state,
        done=False
    )

@pytest.fixture
def mock_transitions():
    """Mock list of transitions for batch testing"""
    transitions = []
    for _ in range(100):
        state = np.random.randint(0, 255, (144, 160), dtype=np.uint8)
        next_state = np.random.randint(0, 255, (144, 160), dtype=np.uint8)
        transitions.append(Transition(
            state=state,
            action=int(np.random.randint(0, 4)),
            reward=float(np.random.uniform(-1, 1)),
            next_state=next_state,
            done=bool(np.random.choice([True, False]))
        ))
    return transitions

@pytest.fixture
def force_cpu_device(monkeypatch):
    """Force CPU device for consistent testing"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)