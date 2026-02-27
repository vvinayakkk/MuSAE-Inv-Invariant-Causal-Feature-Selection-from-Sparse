"""Unit tests for data preprocessing and feature building."""

import numpy as np
import pytest

from musae_inv.data.preprocessing import build_icfs_features


class TestBuildICFSFeatures:
    """Test ICFS feature matrix construction."""

    @pytest.fixture
    def mock_feature_dict(self):
        rng = np.random.RandomState(42)
        n = 50
        d = 1024
        return {
            "sae_features": {
                6: rng.randn(n, d).astype(np.float32),
                12: rng.randn(n, d).astype(np.float32),
                18: rng.randn(n, d).astype(np.float32),
                25: rng.randn(n, d).astype(np.float32),
            },
            "labels": np.random.randint(0, 2, n),
        }

    @pytest.fixture
    def mock_indices(self):
        return {
            6: np.arange(0, 10),
            12: np.arange(100, 110),
            18: np.arange(200, 210),
            25: np.arange(300, 310),
        }

    def test_output_shape(self, mock_feature_dict, mock_indices):
        X = build_icfs_features(mock_feature_dict, mock_indices, top_k=10)
        assert X.shape == (50, 40)  # 50 samples × 4 layers × 10 features

    def test_correct_slicing(self, mock_feature_dict, mock_indices):
        X = build_icfs_features(mock_feature_dict, mock_indices, top_k=10)
        # First 10 columns should match layer 6, indices 0-9
        np.testing.assert_array_equal(
            X[:, :10],
            mock_feature_dict["sae_features"][6][:, :10],
        )

    def test_no_nans(self, mock_feature_dict, mock_indices):
        X = build_icfs_features(mock_feature_dict, mock_indices, top_k=10)
        assert not np.any(np.isnan(X))
