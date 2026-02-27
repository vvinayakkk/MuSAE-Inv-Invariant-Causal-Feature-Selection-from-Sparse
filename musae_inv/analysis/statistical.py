"""
Statistical testing for model comparison.

Implements:
- Hanley-McNeil AUROC confidence intervals
- DeLong's test for comparing two AUROCs
- Holm-Bonferroni correction for multiple comparisons
- Cohen's d effect size
- Bootstrap 10-restart variance check

Reference:
    Hanley & McNeil (1982). "The meaning and use of the area under a ROC curve."
    DeLong et al. (1988). "Comparing the areas under two or more correlated ROC curves."
    Holm (1979). "A simple sequentially rejective multiple test procedure."
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def hanley_mcneil_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute Hanley-McNeil 95% CI for AUROC.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_score : np.ndarray
        Predicted scores.
    alpha : float
        Significance level (default 0.05 → 95% CI).

    Returns
    -------
    tuple[float, float, float]
        (auroc, ci_lower, ci_upper)
    """
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_true, y_score)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    # Hanley-McNeil variance approximation
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se = math.sqrt(
        (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2))
        / (n_pos * n_neg)
    )

    z = stats.norm.ppf(1 - alpha / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se)


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for DeLong's test."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    result = np.zeros(n)
    i = 0
    while i < n:
        j_start = i
        while i < n - 1 and z[i] == z[i + 1]:
            i += 1
        midrank = (j_start + i) / 2.0
        for k in range(j_start, i + 1):
            result[j[k]] = midrank
        i += 1
    return result


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> Tuple[float, float]:
    """DeLong's test for comparing two correlated AUROCs.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_score_a : np.ndarray
        Predicted scores from method A.
    y_score_b : np.ndarray
        Predicted scores from method B.

    Returns
    -------
    tuple[float, float]
        (z_statistic, p_value) for the two-sided test.
    """
    from sklearn.metrics import roc_auc_score

    auc_a = roc_auc_score(y_true, y_score_a)
    auc_b = roc_auc_score(y_true, y_score_b)

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    # Structural components for method A
    pos_scores_a = y_score_a[pos_mask]
    neg_scores_a = y_score_a[neg_mask]
    pos_scores_b = y_score_b[pos_mask]
    neg_scores_b = y_score_b[neg_mask]

    # Placement values
    v_a_pos = np.zeros(n_pos)
    v_a_neg = np.zeros(n_neg)
    v_b_pos = np.zeros(n_pos)
    v_b_neg = np.zeros(n_neg)

    for i in range(n_pos):
        v_a_pos[i] = np.mean(pos_scores_a[i] > neg_scores_a) + 0.5 * np.mean(
            pos_scores_a[i] == neg_scores_a
        )
        v_b_pos[i] = np.mean(pos_scores_b[i] > neg_scores_b) + 0.5 * np.mean(
            pos_scores_b[i] == neg_scores_b
        )

    for i in range(n_neg):
        v_a_neg[i] = np.mean(neg_scores_a[i] < pos_scores_a) + 0.5 * np.mean(
            neg_scores_a[i] == pos_scores_a
        )
        v_b_neg[i] = np.mean(neg_scores_b[i] < pos_scores_b) + 0.5 * np.mean(
            neg_scores_b[i] == pos_scores_b
        )

    # Covariance
    s10 = np.cov(v_a_pos, v_b_pos)[0, 1] if n_pos > 1 else 0
    s01 = np.cov(v_a_neg, v_b_neg)[0, 1] if n_neg > 1 else 0

    var_a_pos = np.var(v_a_pos, ddof=1) if n_pos > 1 else 0
    var_a_neg = np.var(v_a_neg, ddof=1) if n_neg > 1 else 0
    var_b_pos = np.var(v_b_pos, ddof=1) if n_pos > 1 else 0
    var_b_neg = np.var(v_b_neg, ddof=1) if n_neg > 1 else 0

    var_diff = (
        var_a_pos / n_pos + var_a_neg / n_neg
        + var_b_pos / n_pos + var_b_neg / n_neg
        - 2 * s10 / n_pos - 2 * s01 / n_neg
    )

    if var_diff <= 0:
        return 0.0, 1.0

    z = (auc_a - auc_b) / math.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return z, p


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    list[tuple[float, bool]]
        (adjusted_p, significant) for each test.
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    results = [None] * m

    for rank, idx in enumerate(sorted_idx):
        adjusted_p = min(p_values[idx] * (m - rank), 1.0)
        significant = adjusted_p < alpha
        results[idx] = (adjusted_p, significant)

    return results


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Parameters
    ----------
    group_a, group_b : np.ndarray
        Score arrays for two groups.

    Returns
    -------
    float
        Cohen's d (positive = group_a > group_b).
    """
    n_a, n_b = len(group_a), len(group_b)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std
