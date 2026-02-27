"""Unit tests for utility functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from musae_inv.utils.io import save_pickle, load_pickle, save_numpy, load_numpy, save_json, load_json
from musae_inv.utils.seed import set_seed


class TestSeed:
    """Test reproducibility utilities."""

    def test_set_seed_deterministic(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)


class TestPickleIO:
    """Test pickle serialization."""

    def test_roundtrip_dict(self):
        data = {"a": [1, 2, 3], "b": np.array([4, 5, 6])}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.pkl"
            save_pickle(data, p)
            loaded = load_pickle(p)
            assert loaded["a"] == [1, 2, 3]
            np.testing.assert_array_equal(loaded["b"], np.array([4, 5, 6]))


class TestNumpyIO:
    """Test numpy array serialization."""

    def test_roundtrip(self):
        arr = np.random.randn(10, 20)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.npy"
            save_numpy(arr, p)
            loaded = load_numpy(p)
            np.testing.assert_array_almost_equal(arr, loaded)


class TestJsonIO:
    """Test JSON serialization."""

    def test_roundtrip(self):
        data = {"key": "value", "numbers": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.json"
            save_json(data, p)
            loaded = load_json(p)
            assert loaded == data
