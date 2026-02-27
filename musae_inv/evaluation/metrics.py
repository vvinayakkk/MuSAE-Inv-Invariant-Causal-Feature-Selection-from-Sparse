"""
Evaluation metrics for hallucination detection.

Computes AUROC, AUPRC, Balanced Accuracy, Macro-F1, and Brier Score
for binary classification (0=truth, 1=hallucination).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics for binary hallucination detection.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Ground truth binary labels (0=truth, 1=hallucination).
    y_score : np.ndarray, shape (n,)
        Predicted hallucination probabilities in [0, 1].

    Returns
    -------
    dict[str, float]
        Dictionary with keys: auroc, auprc, balacc, f1, brier.
    """
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "balacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


class ResultsTracker:
    """Track evaluation results across methods and domains.

    Provides a central registry for all method × domain metric combinations.
    """

    def __init__(self):
        self.results: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    def register(
        self,
        method: str,
        domain: str,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate and register results for a method on a domain.

        Parameters
        ----------
        method : str
            Method name.
        domain : str
            Domain name (QA, Dialogue, Summ, TruthfulQA).
        y_true : np.ndarray
            Ground truth labels.
        y_score : np.ndarray
            Predicted probabilities.

        Returns
        -------
        dict[str, float]
            Computed metrics.
        """
        m = compute_metrics(y_true, y_score)
        self.results[method][domain] = m
        print(
            f"  [{method}] {domain}: AUROC={m['auroc']:.4f} "
            f"BalAcc={m['balacc']:.4f} F1={m['f1']:.4f}"
        )
        return m

    def get_auroc(self, method: str, domain: str) -> float:
        """Get AUROC for a specific method-domain pair."""
        return self.results.get(method, {}).get(domain, {}).get("auroc", float("nan"))


# Module-level convenience
_global_tracker = ResultsTracker()


def register_result(method, domain, y_true, y_score):
    """Register a result using the global tracker."""
    return _global_tracker.register(method, domain, y_true, y_score)
