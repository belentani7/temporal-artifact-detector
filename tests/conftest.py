"""Test fixtures for temporal artifact detector."""

import pytest
import numpy as np


@pytest.fixture
def sample_frame():
    """Create a sample video frame."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames():
    """Create a sequence of sample video frames."""
    frames = []
    for i in range(10):
        # Create frames with slight variations
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def flickering_frames():
    """Create frames with intentional brightness flickering."""
    frames = []
    base = np.full((100, 100, 3), 128, dtype=np.uint8)
    
    for i in range(10):
        if i % 2 == 0:
            # Bright frame
            frame = np.clip(base + 80, 0, 255).astype(np.uint8)
        else:
            # Dark frame
            frame = np.clip(base - 80, 0, 255).astype(np.uint8)
        frames.append(frame)
    
    return frames


@pytest.fixture
def stable_frames():
    """Create frames with minimal variation (stable video)."""
    frames = []
    base = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    
    for i in range(10):
        # Add very small random noise
        noise = np.random.randint(-2, 3, (100, 100, 3), dtype=np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    
    return frames


@pytest.fixture
def drift_frames():
    """Create frames with gradual identity drift."""
    frames = []
    
    # Start with one identity
    base_identity = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    
    for i in range(10):
        # Gradually shift the identity
        shift = int(i * 10)
        frame = np.clip(
            base_identity.astype(np.int16) + shift,
            0,
            255,
        ).astype(np.uint8)
        frames.append(frame)
    
    return frames