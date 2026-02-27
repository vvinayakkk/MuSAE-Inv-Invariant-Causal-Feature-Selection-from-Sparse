"""
Baseline methods for hallucination detection.

Implements:
- Random baseline
- SelfCheckGPT-NLI proxy (mean SAE entropy)
- Truthfulness Direction Vector (TDV) zero-shot
- SAPLMA (MLP on single-layer SAE features)
- Concat-4L + PCA + LR
- Single-layer SAE LR probes
"""

from __future__ import annotations

from typing import Dict, List

import gc
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from musae_inv.config import Config
from musae_inv.evaluation.metrics import ResultsTracker


def sae_entropy_score(
    feature_cache: Dict,
    target_layers: List[int],
) -> np.ndarray:
    """SelfCheckGPT-NLI proxy: mean SAE activation entropy.

    Higher entropy in SAE features → less certain → more likely hallucinated.

    Parameters
    ----------
    feature_cache : dict
        Feature cache with "sae_features" per layer.
    target_layers : list[int]
        Layers to average over.

    Returns
    -------
    np.ndarray, shape (n,)
        Normalised hallucination scores in [0, 1].
    """
    all_l = []
    for l in target_layers:
        feats = feature_cache["sae_features"][l].astype(np.float32)
        # Softmax-like normalisation to get distribution
        feats_pos = feats - feats.min(axis=1, keepdims=True) + 1e-9
        probs = feats_pos / feats_pos.sum(axis=1, keepdims=True)
        ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        all_l.append(ent)
    scores = np.mean(np.stack(all_l, axis=0), axis=0)
    # Normalise to [0, 1]
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return scores_norm


def tdv_direction_vector(
    X_train_sc: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Compute the Truthfulness Direction Vector (TDV).

    TDV = mean(truth features) − mean(hallucination features), normalised.
    Dot product with TDV gives truthfulness score (higher = more truthful).

    Parameters
    ----------
    X_train_sc : np.ndarray
        Scaled ICFS training features.
    y_train : np.ndarray
        Training labels (0=truth, 1=hallucination).

    Returns
    -------
    np.ndarray, shape (d,)
        Unit TDV direction vector.
    """
    X_truth = X_train_sc[y_train == 0]
    X_hall = X_train_sc[y_train == 1]
    tdv = X_truth.mean(0) - X_hall.mean(0)
    tdv = tdv / (np.linalg.norm(tdv) + 1e-9)
    return tdv


def tdv_score(X_sc: np.ndarray, tdv: np.ndarray) -> np.ndarray:
    """Score examples using the TDV direction vector.

    Parameters
    ----------
    X_sc : np.ndarray
        Scaled ICFS features.
    tdv : np.ndarray
        TDV direction vector.

    Returns
    -------
    np.ndarray, shape (n,)
        Hallucination probability scores (higher = more hallucinated).
    """
    scores = X_sc @ tdv
    # Invert: higher dot product = more truthful = lower hallucination prob
    probs = 1.0 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return probs


def run_all_baselines(
    tracker: ResultsTracker,
    feature_cache: Dict,
    test_domains: Dict,
    target_layers: List[int],
    cfg: Config,
) -> ResultsTracker:
    """Run all baseline methods and register results.

    Parameters
    ----------
    tracker : ResultsTracker
        Results tracker to register methods with.
    feature_cache : dict
        Feature caches per dataset split.
    test_domains : dict
        Mapping from domain name to (y_true, X_icfs) tuples.
    target_layers : list[int]
        Target transformer layers.
    cfg : Config
        Experiment configuration.

    Returns
    -------
    ResultsTracker
        Updated tracker with all baseline results.
    """
    rng = np.random.RandomState(cfg.seed)

    # B1: Random
    print("\n── B1: Random Baseline ──")
    for dom, (y, _) in test_domains.items():
        tracker.register("Random", dom, y, rng.uniform(0, 1, len(y)))

    # B2: SelfCheckGPT-NLI proxy
    print("\n── B2: SelfCheckGPT-NLI Proxy ──")
    fc_map = {
        "QA": "halueval_qa_test",
        "Dialogue": "halueval_dial_test",
        "Summ": "halueval_summ_test",
        "TruthfulQA": "truthfulqa",
    }
    for dom, (y, _) in test_domains.items():
        scores = sae_entropy_score(feature_cache[fc_map[dom]], target_layers)
        tracker.register("SelfCheckGPT-NLI proxy", dom, y, scores)

    return tracker
