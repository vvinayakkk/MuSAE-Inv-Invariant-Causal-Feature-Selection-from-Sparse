"""
Mechanistic analysis of ICFS features.

Analyses feature characteristics, domain-specific causal effects,
layer contributions, direction statistics, and domain overlap.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from musae_inv.config import Config


def compute_layer_icfs_stats(
    icfs_scores: Dict,
    icfs_indices: Dict,
    top_k: int,
    target_layers: List[int],
) -> Dict[int, Dict]:
    """Compute per-layer ICFS feature statistics.

    Parameters
    ----------
    icfs_scores : dict
        ICFS scoring data per layer.
    icfs_indices : dict
        Sorted feature indices per layer.
    top_k : int
        Number of selected features.
    target_layers : list[int]
        Layers to analyse.

    Returns
    -------
    dict[int, dict]
        Statistics per layer.
    """
    stats = {}
    for l in target_layers:
        scores = icfs_scores[l]
        top_k_idx = icfs_indices[l][:top_k]
        rest_idx = icfs_indices[l][top_k:]

        stats[l] = {
            "top_k_ce_mean": float(scores["ce"][top_k_idx].mean()),
            "rest_ce_mean": float(scores["ce"][rest_idx].mean()),
            "top_k_ce_qa_mean": float(scores["ce_qa"][top_k_idx].mean()),
            "top_k_ce_dial_mean": float(scores["ce_dial"][top_k_idx].mean()),
            "top_k_ce_summ_mean": float(scores["ce_summ"][top_k_idx].mean()),
            "top_k_score_mean": float(scores["score"][top_k_idx].mean()),
            "n_sign_consistent": int(scores["sign_consistent"][top_k_idx].sum()),
        }
        print(
            f"  L{l}: min-CE={stats[l]['top_k_ce_mean']:.5f} | "
            f"CE_QA={stats[l]['top_k_ce_qa_mean']:.5f} | "
            f"CE_Dial={stats[l]['top_k_ce_dial_mean']:.5f} | "
            f"CE_Summ={stats[l]['top_k_ce_summ_mean']:.5f}"
        )
    return stats


def compute_direction_stats(
    icfs_scores: Dict,
    icfs_indices: Dict,
    top_k: int,
    target_layers: List[int],
) -> Dict[int, Dict]:
    """Analyse signed direction of top ICFS features (truth vs hallucination).

    Parameters
    ----------
    icfs_scores, icfs_indices : dict
        ICFS data.
    top_k : int
        Number of selected features.
    target_layers : list[int]
        Target layers.

    Returns
    -------
    dict[int, dict]
        Direction stats per layer.
    """
    stats = {}
    for l in target_layers:
        top_k_idx = icfs_indices[l][:top_k]
        delta = icfs_scores[l]["mean_delta"][top_k_idx]
        n_truth = int((delta > 0).sum())
        n_hall = int((delta < 0).sum())
        stats[l] = {
            "n_truth_features": n_truth,
            "n_hall_features": n_hall,
            "delta_top_k": delta,
        }
        print(f"  L{l}: {n_truth} truth-assoc | {n_hall} halluc-assoc features")
    return stats


def compute_domain_overlap(
    icfs_scores: Dict,
    top_k: int,
    target_layers: List[int],
) -> Dict:
    """Compute Jaccard overlap of top-K features across domains.

    Parameters
    ----------
    icfs_scores : dict
        ICFS per-layer scoring data.
    top_k : int
        Feature budget per layer.
    target_layers : list[int]
        Target layers.

    Returns
    -------
    dict
        Overlap statistics keyed by (domain_i, domain_j, layer).
    """
    domain_names = ["QA", "Dialogue", "Summarisation"]
    overlap = {}

    for l in target_layers:
        per_domain = icfs_scores[l]["per_domain_abs_means"]  # [3, D_SAE]
        dom_sets = [
            set(np.argsort(per_domain[di])[::-1][:top_k].tolist())
            for di in range(3)
        ]
        for i in range(3):
            for j in range(i + 1, 3):
                inter = len(dom_sets[i] & dom_sets[j])
                union = len(dom_sets[i] | dom_sets[j])
                key = (domain_names[i], domain_names[j], l)
                overlap[key] = {
                    "intersection": inter,
                    "union": union,
                    "jaccard": inter / (union + 1e-9),
                }
                print(
                    f"  L{l}: {domain_names[i]} ∩ {domain_names[j]} = "
                    f"{inter}/{top_k} ({inter / top_k * 100:.0f}%)"
                )
    return overlap


def compute_mechanistic_analysis(
    icfs_scores: Dict,
    icfs_indices: Dict,
    feature_cache: Dict,
    cfg: Config,
) -> Dict:
    """Run full mechanistic analysis suite.

    Returns
    -------
    dict
        Keys: layer_stats, direction_stats, domain_overlap.
    """
    print("Computing mechanistic analyses...")
    top_k = cfg.icfs_top_k
    layers = cfg.target_layers

    layer_stats = compute_layer_icfs_stats(icfs_scores, icfs_indices, top_k, layers)
    dir_stats = compute_direction_stats(icfs_scores, icfs_indices, top_k, layers)
    overlap = compute_domain_overlap(icfs_scores, top_k, layers)

    return {
        "layer_stats": layer_stats,
        "direction_stats": dir_stats,
        "domain_overlap": overlap,
    }
