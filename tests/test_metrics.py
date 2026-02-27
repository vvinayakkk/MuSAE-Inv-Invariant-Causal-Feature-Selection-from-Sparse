"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from musae_inv.evaluation.metrics import compute_metrics, ResultsTracker
from musae_inv.analysis.statistical import hanley_mcneil_ci, cohens_d


class TestComputeMetrics:
    """Test metric computation functions."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.9, 1.0])
        m = compute_metrics(y_true, y_prob)
        assert m["auroc"] > 0.95
        assert m["auprc"] > 0.95
        assert m["bal_acc"] > 0.95

    def test_random_predictions(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        m = compute_metrics(y_true, y_prob)
        assert 0.3 < m["auroc"] < 0.7
        assert 0.3 < m["bal_acc"] < 0.7

    def test_all_metrics_present(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        m = compute_metrics(y_true, y_prob)
        for key in ["auroc", "auprc", "bal_acc", "f1", "brier"]:
            assert key in m
            assert isinstance(m[key], float)

    def test_brier_score_range(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y_true, y_prob)
        assert 0.0 <= m["brier"] <= 1.0

    def test_brier_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_prob)
        assert m["brier"] < 0.01


class TestResultsTracker:
    """Test the results tracking class."""

    def test_register_and_retrieve(self):
        tracker = ResultsTracker()
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.8, 0.3, 0.7])
        tracker.register("TestMethod", "QA", y, p)
        assert ("TestMethod", "QA") in tracker.results

    def test_multiple_methods(self):
        tracker = ResultsTracker()
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.8, 0.3, 0.7])
        tracker.register("Method1", "QA", y, p)
        tracker.register("Method2", "QA", y, p)
        assert len(tracker.results) == 2


class TestStatisticalFunctions:
    """Test statistical testing utilities."""

    def test_hanley_mcneil_ci(self):
        rng = np.random.RandomState(42)
        y = np.concatenate([np.zeros(500), np.ones(500)])
        p = np.where(y == 1, rng.uniform(0.6, 1.0, 1000), rng.uniform(0.0, 0.4, 1000))
        auc, ci_lo, ci_hi = hanley_mcneil_ci(y, p)
        assert 0.8 < auc < 1.0
        assert ci_lo < auc
        assert ci_hi > auc
        assert ci_hi - ci_lo > 0

    def test_cohens_d_equal_groups(self):
        a = np.random.randn(100) + 1
        b = np.random.randn(100) - 1
        d = cohens_d(a, b)
        assert d > 0.5  # Should be large for well-separated groups

    def test_cohens_d_identical_groups(self):
        a = np.random.randn(100)
        d = cohens_d(a, a)
        assert abs(d) < 0.1
