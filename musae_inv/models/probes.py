"""
MuSAE-Inv probe and baseline logistic regression probes.

The MuSAE-Inv probe is an L1-regularised logistic regression trained on
domain-augmented ICFS features (QA + Dialogue). Invariance is achieved
through ICFS feature selection, not through a training penalty.

Also provides:
- ERM ablation probe (QA-only, L2 regularisation)
- Single-layer SAE probes (per-layer ablation)
- Concat-4L + PCA + LR baseline
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from musae_inv.config import Config


class MuSAEInvProbe:
    """MuSAE-Inv: Domain-augmented L1-regularised logistic regression probe.

    Trains on ICFS-selected features from multiple domains jointly.
    L1 regularisation induces additional sparsity in the probe weight vector.

    Attributes
    ----------
    probe : LogisticRegression
        Fitted sklearn logistic regression model.
    scaler : StandardScaler
        Feature scaler fitted on domain-augmented training data.
    cfg : Config
        Experiment configuration.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler(with_mean=False)
        self.probe = LogisticRegression(
            C=cfg.musae_C,
            penalty="l1",
            solver="saga",
            max_iter=cfg.musae_max_iter,
            random_state=cfg.seed,
            n_jobs=-1,
            class_weight="balanced",
        )

    def fit(
        self,
        X_qa: np.ndarray,
        y_qa: np.ndarray,
        X_dial: np.ndarray,
        y_dial: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MuSAEInvProbe":
        """Train on domain-augmented ICFS features (QA + Dialogue).

        Parameters
        ----------
        X_qa : np.ndarray, shape (n_qa, d_icfs)
            QA training features.
        y_qa : np.ndarray, shape (n_qa,)
            QA training labels (0=truth, 1=hallucination).
        X_dial : np.ndarray, shape (n_dial, d_icfs)
            Dialogue training features.
        y_dial : np.ndarray, shape (n_dial,)
            Dialogue training labels.
        X_val : np.ndarray, optional
            Validation features for logging.
        y_val : np.ndarray, optional
            Validation labels for logging.

        Returns
        -------
        MuSAEInvProbe
            Self, for method chaining.
        """
        # Domain augmentation: concatenate QA + Dialogue
        X_aug = np.vstack([X_qa, X_dial])
        y_aug = np.concatenate([y_qa, y_dial]).astype(int)
        print(f"Domain-augmented training: n={len(X_aug)} (QA={len(X_qa)}, Dial={len(X_dial)})")

        # Fit scaler on combined data
        X_aug_sc = self.scaler.fit_transform(X_aug)

        # Train L1-LR probe
        print(f"Training MuSAE-Inv probe (C={self.cfg.musae_C}, L1, saga)...")
        self.probe.fit(X_aug_sc, y_aug)

        # Report sparsity
        n_zero = (self.probe.coef_[0] == 0).sum()
        n_total = self.probe.coef_.shape[1]
        print(f"L1 sparsity: {n_zero}/{n_total} ({n_zero / n_total * 100:.1f}%) features zeroed")

        # Optional validation AUROC
        if X_val is not None and y_val is not None:
            p_val = self.predict_proba(X_val)
            val_auroc = roc_auc_score(y_val, p_val)
            print(f"Validation AUROC (QA): {val_auroc * 100:.2f}%")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict hallucination probability.

        Parameters
        ----------
        X : np.ndarray, shape (n, d_icfs)
            ICFS feature matrix (raw, unscaled).

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted probability of hallucination.
        """
        X_sc = self.scaler.transform(X)
        return self.probe.predict_proba(X_sc)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels.

        Parameters
        ----------
        X : np.ndarray, shape (n, d_icfs)
            ICFS feature matrix.
        threshold : float
            Decision threshold.

        Returns
        -------
        np.ndarray, shape (n,)
            Binary predictions (0=truth, 1=hallucination).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def n_active_features(self) -> int:
        """Number of non-zero probe coefficients."""
        return int((self.probe.coef_[0] != 0).sum())


def train_musae_inv_probe(
    cfg: Config,
    X_qa_train: np.ndarray,
    y_qa_train: np.ndarray,
    X_dial_train: np.ndarray,
    y_dial_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> MuSAEInvProbe:
    """Convenience function to create and train a MuSAE-Inv probe.

    Parameters
    ----------
    cfg : Config
        Experiment configuration.
    X_qa_train, y_qa_train : np.ndarray
        QA training data.
    X_dial_train, y_dial_train : np.ndarray
        Dialogue training data.
    X_val, y_val : np.ndarray, optional
        Validation data for logging.

    Returns
    -------
    MuSAEInvProbe
        Trained probe.
    """
    probe = MuSAEInvProbe(cfg)
    probe.fit(X_qa_train, y_qa_train, X_dial_train, y_dial_train, X_val, y_val)
    return probe


class ERMAblationProbe:
    """QA-only L2-LR probe (ERM ablation baseline).

    Same ICFS features as MuSAE-Inv, but trained only on QA without
    domain augmentation. Isolates the contribution of multi-domain training.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler(with_mean=False)
        self.probe = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="saga",
            max_iter=500,
            random_state=cfg.seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ERMAblationProbe":
        """Train on QA-only ICFS features."""
        X_sc = self.scaler.fit_transform(X)
        self.probe.fit(X_sc, y.astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler.transform(X)
        return self.probe.predict_proba(X_sc)[:, 1]


class SingleLayerProbe:
    """Per-layer SAE LR probe for ablation studies."""

    def __init__(self, layer: int, cfg: Config):
        self.layer = layer
        self.cfg = cfg
        self.scaler = StandardScaler(with_mean=False)
        self.probe = LogisticRegression(
            C=0.5,
            max_iter=300,
            random_state=cfg.seed,
            solver="saga",
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleLayerProbe":
        X_sc = self.scaler.fit_transform(X)
        self.probe.fit(X_sc, y.astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler.transform(X)
        return self.probe.predict_proba(X_sc)[:, 1]


class ConcatPCAProbe:
    """Concat all 4 SAE layers → PCA → LR baseline."""

    def __init__(self, n_components: int = 256, cfg: Optional[Config] = None):
        seed = cfg.seed if cfg else 42
        self.scaler = StandardScaler(with_mean=False)
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.probe = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConcatPCAProbe":
        X_sc = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_sc)
        self.probe.fit(X_pca, y.astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler.transform(X)
        X_pca = self.pca.transform(X_sc)
        return self.probe.predict_proba(X_pca)[:, 1]
