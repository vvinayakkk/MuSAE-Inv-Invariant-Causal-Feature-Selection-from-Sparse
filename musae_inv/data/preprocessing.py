"""
Preprocessing utilities for building ICFS feature matrices.

Converts raw SAE feature caches into ICFS-selected feature matrices
suitable for probe training and evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from musae_inv.config import Config


def build_icfs_features(
    feature_cache: Dict,
    icfs_indices: Dict[int, np.ndarray],
    top_k: int = 128,
    layers: Optional[List[int]] = None,
) -> np.ndarray:
    """Build an ICFS feature matrix by selecting top-k features per layer.

    Parameters
    ----------
    feature_cache : dict
        Feature cache with key "sae_features" → {layer: np.ndarray}.
    icfs_indices : dict[int, np.ndarray]
        Mapping from layer to sorted feature indices (best first).
    top_k : int
        Number of features to select per layer.
    layers : list[int], optional
        Which layers to include. Defaults to all keys in icfs_indices.

    Returns
    -------
    np.ndarray, shape (n_samples, n_layers * top_k)
        Concatenated ICFS feature matrix.
    """
    if layers is None:
        layers = sorted(icfs_indices.keys())

    parts = []
    for layer in layers:
        idx = icfs_indices[layer][:top_k]
        parts.append(feature_cache["sae_features"][layer][:, idx])

    return np.concatenate(parts, axis=1)


def build_env_icfs_from_counterfactual(
    cf_cache: Dict,
    domain: str,
    label: int,
    icfs_indices: Dict[int, np.ndarray],
    top_k: int = 128,
    layers: Optional[List[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ICFS features from counterfactual cache for a specific domain/label side.

    Parameters
    ----------
    cf_cache : dict
        Counterfactual cache with keys per domain → {"feat_true", "feat_false", "delta"}.
    domain : str
        Domain key (e.g., "qa", "dialogue", "summarisation").
    label : int
        0 for truth side, 1 for hallucination side.
    icfs_indices : dict[int, np.ndarray]
        ICFS feature indices per layer.
    top_k : int
        Features per layer.
    layers : list[int], optional
        Target layers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (features, labels) arrays.
    """
    if layers is None:
        layers = sorted(icfs_indices.keys())

    side = "feat_true" if label == 0 else "feat_false"
    parts = []
    for layer in layers:
        idx = icfs_indices[layer][:top_k]
        parts.append(cf_cache[domain][side][layer][:, idx])

    X = np.concatenate(parts, axis=1)
    y = np.full(parts[0].shape[0], label, dtype=np.float32)
    return X, y
