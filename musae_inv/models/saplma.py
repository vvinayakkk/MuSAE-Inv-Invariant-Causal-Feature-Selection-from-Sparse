"""
SAPLMA probe baseline.

Implements the Self-Assessment of the LLM's Performance through Metacognitive
Awareness (SAPLMA) approach using an MLP classifier on SAE features from a
single layer (default: layer 18).

Reference:
    Azaria & Mitchell (2023). The Internal State of an LLM Knows When
    It's Lying. EMNLP Findings 2023.
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from musae_inv.config import Config


class SAPLMAProbe(nn.Module):
    """MLP probe on single-layer SAE features (SAPLMA baseline).

    Architecture: Linear → BN → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Sigmoid.
    """

    def __init__(self, d_in: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, d_in)
            Input features.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Hallucination probability.
        """
        return self.net(x).squeeze(-1)


def train_saplma_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    layer: int = 18,
) -> tuple[SAPLMAProbe, StandardScaler]:
    """Train the SAPLMA MLP probe.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, d_sae)
        Training SAE features from the target layer.
    y_train : np.ndarray, shape (n_train,)
        Training labels.
    X_val : np.ndarray, shape (n_val, d_sae)
        Validation features.
    y_val : np.ndarray, shape (n_val,)
        Validation labels.
    cfg : Config
        Experiment configuration.
    layer : int
        Which layer's SAE features to use.

    Returns
    -------
    tuple[SAPLMAProbe, StandardScaler]
        Trained probe and fitted scaler.
    """
    device = cfg.device
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_vl_sc = scaler.transform(X_val)

    d_in = X_train.shape[1]
    probe = SAPLMAProbe(d_in=d_in, hidden=cfg.saplma_hidden).to(device)
    optimizer = AdamW(probe.parameters(), lr=cfg.saplma_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.saplma_epochs)

    xt = torch.FloatTensor(X_tr_sc).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    xv = torch.FloatTensor(X_vl_sc).to(device)

    best_auroc = 0.0
    best_state = None
    batch_size = 128

    print(f"Training SAPLMA probe on L{layer} SAE features (d={d_in})...")
    for epoch in range(cfg.saplma_epochs):
        probe.train()
        idx = torch.randperm(len(xt))
        for i in range(0, len(xt), batch_size):
            xb = xt[idx[i : i + batch_size]]
            yb = yt[idx[i : i + batch_size]]
            loss = F.binary_cross_entropy(probe(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        probe.eval()
        with torch.no_grad():
            vp = probe(xv).cpu().numpy()
        va = roc_auc_score(y_val, vp)
        if va > best_auroc:
            best_auroc = va
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    probe.eval()
    print(f"  Best val AUROC: {best_auroc * 100:.2f}%")
    return probe, scaler
