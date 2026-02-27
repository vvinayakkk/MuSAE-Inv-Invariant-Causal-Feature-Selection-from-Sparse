"""Unit tests for ICFS v2 scoring logic."""

import numpy as np
import pytest

from musae_inv.features.icfs import compute_icfs_v2


class TestICFSv2:
    """Test the ICFS v2 invariant feature selection."""

    @pytest.fixture
    def mock_cf_cache(self):
        """Create synthetic counterfactual cache for testing."""
        rng = np.random.RandomState(42)
        n_qa, n_dial, n_summ = 100, 50, 50
        d = 1024  # SAE width for test

        def make_delta(n, d, informative_idx):
            delta = rng.randn(n, d).astype(np.float32) * 0.01
            # Make some features informative with consistent sign
            for idx in informative_idx:
                delta[:, idx] = np.abs(rng.randn(n)) + 0.5
            return delta

        informative = list(range(0, 20))  # First 20 features are informative

        return {
            "qa": {"delta": {6: make_delta(n_qa, d, informative),
                             12: make_delta(n_qa, d, informative),
                             18: make_delta(n_qa, d, informative),
                             25: make_delta(n_qa, d, informative)}},
            "dialogue": {"delta": {6: make_delta(n_dial, d, informative),
                                    12: make_delta(n_dial, d, informative),
                                    18: make_delta(n_dial, d, informative),
                                    25: make_delta(n_dial, d, informative)}},
            "summarisation": {"delta": {6: make_delta(n_summ, d, informative),
                                         12: make_delta(n_summ, d, informative),
                                         18: make_delta(n_summ, d, informative),
                                         25: make_delta(n_summ, d, informative)}},
        }

    def test_returns_correct_number_of_indices(self, mock_cf_cache):
        target_layers = [6, 12, 18, 25]
        top_k = 10
        result = compute_icfs_v2(mock_cf_cache, target_layers, top_k)
        for layer in target_layers:
            assert len(result.indices[layer]) == top_k

    def test_selected_features_have_nonzero_scores(self, mock_cf_cache):
        target_layers = [6, 12, 18, 25]
        result = compute_icfs_v2(mock_cf_cache, target_layers, top_k=10)
        for layer in target_layers:
            for idx in result.indices[layer]:
                assert result.scores[layer][idx] > 0

    def test_informative_features_ranked_higher(self, mock_cf_cache):
        """Informative features (0-19) should rank higher than random."""
        target_layers = [6, 12, 18, 25]
        result = compute_icfs_v2(mock_cf_cache, target_layers, top_k=20)
        for layer in target_layers:
            informative_selected = sum(1 for idx in result.indices[layer] if idx < 20)
            # At least half of informative features should be selected
            assert informative_selected >= 5, (
                f"Layer {layer}: only {informative_selected}/20 informative features selected"
            )

    def test_different_topk_values(self, mock_cf_cache):
        target_layers = [6, 12, 18, 25]
        for k in [1, 5, 50, 100]:
            result = compute_icfs_v2(mock_cf_cache, target_layers, top_k=k)
            for layer in target_layers:
                assert len(result.indices[layer]) == k

    def test_scores_are_finite(self, mock_cf_cache):
        result = compute_icfs_v2(mock_cf_cache, [6, 12, 18, 25], top_k=10)
        for layer in result.scores:
            assert np.all(np.isfinite(result.scores[layer]))
