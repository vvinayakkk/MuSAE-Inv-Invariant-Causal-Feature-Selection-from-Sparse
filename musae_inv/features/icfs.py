"""
ICFS v2: Invariant Causal Feature Selection.

Scores each SAE feature by its cross-domain discriminability and selects
the top-K features that are causally informative across both QA and Dialogue.

Criterion:
    score(j) = min(|CE_QA(j)|, |CE_Dial(j)|) × 𝟙[sign(CE_QA(j)) = sign(CE_Dial(j))]

where CE_d(j) = mean(Δ_d(j)) is the signed causal effect of feature j in domain d,
and Δ_d = feat_true − feat_false is the counterfactual delta.

Root cause of v1 failure:
    v1 used CE/(DV+ε) across QA+Dialogue+Summarisation. Since Summarisation
    yields Δ≈0 for all features, every QA/Dial-discriminative feature has
    high DV → gets excluded. Selected features were worse than random.

Fix (v2):
    Drop Summarisation from selection. Use min-CE × sign-consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from musae_inv.config import Config


@dataclass
class ICFSResult:
    """Container for ICFS v2 scoring results.

    Attributes
    ----------
    scores : dict[int, dict]
        Per-layer scoring data: CE values, ICFS scores, sign consistency.
    indices : dict[int, np.ndarray]
        Per-layer sorted feature indices (best first).
    """
    scores: Dict[int, Dict]
    indices: Dict[int, np.ndarray]


def compute_icfs_v2(
    cf_cache: Dict,
    target_layers: List[int],
    top_k: int = 128,
    cfg: Optional[Config] = None,
) -> ICFSResult:
    """Compute ICFS v2 scores and select top-K features per layer.

    Parameters
    ----------
    cf_cache : dict
        Counterfactual cache with keys "qa", "dialogue", "summarisation".
        Each value has "delta" → {layer: ndarray of shape (n_pairs, d_sae)}.
    target_layers : list[int]
        Transformer layers to process.
    top_k : int
        Number of features to select per layer.
    cfg : Config, optional
        Configuration.

    Returns
    -------
    ICFSResult
        ICFS scores and sorted indices per layer.
    """
    icfs_scores = {}
    icfs_indices = {}

    for l in target_layers:
        delta_qa = cf_cache["qa"]["delta"][l]       # [N_qa, D_SAE]
        delta_dial = cf_cache["dialogue"]["delta"][l]  # [N_dial, D_SAE]
        delta_summ = cf_cache["summarisation"]["delta"][l]  # kept for analysis

        # Signed mean per domain (causal effect)
        mean_qa = np.mean(delta_qa, axis=0)
        mean_dial = np.mean(delta_dial, axis=0)
        mean_summ = np.mean(delta_summ, axis=0)

        # Absolute causal effect
        ce_qa = np.abs(mean_qa)
        ce_dial = np.abs(mean_dial)
        ce_summ = np.abs(mean_summ)

        # min-CE: feature must be discriminative in BOTH QA and Dialogue
        min_ce = np.minimum(ce_qa, ce_dial)

        # Sign consistency: prevents domain-flipped features
        sign_consistent = (np.sign(mean_qa) == np.sign(mean_dial)).astype(float)

        # ICFS v2 score
        icfs_score = min_ce * sign_consistent

        # Sort descending
        sorted_idx = np.argsort(icfs_score)[::-1]

        icfs_scores[l] = {
            "ce": min_ce,
            "ce_qa": ce_qa,
            "ce_dial": ce_dial,
            "ce_summ": ce_summ,
            "score": icfs_score,
            "sign_consistent": sign_consistent,
            "mean_delta": mean_qa,
            "mean_delta_dial": mean_dial,
            "per_domain_abs_means": np.array([ce_qa, ce_dial, ce_summ]),
            "per_domain_signed": np.array([mean_qa, mean_dial, mean_summ]),
        }
        icfs_indices[l] = sorted_idx

        # Log statistics for selected features
        top_k_idx = sorted_idx[:top_k]
        n_sign = int(sign_consistent[top_k_idx].sum())
        print(
            f"L{l}: min-CE range [{min_ce[top_k_idx].min():.5f}, {min_ce[top_k_idx].max():.5f}] | "
            f"CE_QA={ce_qa[top_k_idx].mean():.5f} CE_Dial={ce_dial[top_k_idx].mean():.5f} "
            f"CE_Summ={ce_summ[top_k_idx].mean():.5f} | sign_ok={n_sign}/{top_k}"
        )

    return ICFSResult(scores=icfs_scores, indices=icfs_indices)
