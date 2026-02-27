"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fixed_seed():
    """Set a fixed random seed before each test."""
    np.random.seed(42)
    yield


@pytest.fixture
def sample_labels():
    """Balanced binary labels."""
    return np.array([0] * 50 + [1] * 50)


@pytest.fixture
def sample_probas():
    """Reasonable predicted probabilities."""
    rng = np.random.RandomState(42)
    y = np.array([0] * 50 + [1] * 50)
    p = np.where(y == 1, rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.4, 100))
    return p
