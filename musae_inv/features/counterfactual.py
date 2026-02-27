"""
Counterfactual delta extraction.

For each (truthful, hallucinated) text pair, extract SAE features for both
sides and compute the delta: Δ = features_true − features_false.
Positive delta values indicate features more active for truthful responses.

This is the key input to ICFS v2 scoring.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import numpy as np
import torch

from musae_inv.config import Config
from musae_inv.features.extraction import get_hidden_states, hidden_to_sae


def extract_counterfactual_pairs(
    pairs_df,
    model,
    tokenizer,
    saes: Dict,
    target_layers: List[int],
    cfg: Config = None,
    true_col: str = "text_true",
    false_col: str = "text_false",
    desc: str = "",
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Extract SAE features for counterfactual (true, hallucinated) pairs.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with true_col and false_col columns.
    model : AutoModelForCausalLM
        Base language model.
    tokenizer : AutoTokenizer
        Tokenizer.
    saes : dict[int, SAE]
        SAE modules per layer.
    target_layers : list[int]
        Target transformer layers.
    cfg : Config, optional
        Configuration.
    true_col : str
        Column name for truthful text.
    false_col : str
        Column name for hallucinated text.
    desc : str
        Description for logging.

    Returns
    -------
    tuple[dict, dict, dict]
        (feat_true, feat_false, delta) where each is {layer: ndarray}.
        delta[l] = feat_true[l] − feat_false[l].
    """
    max_length = cfg.max_length if cfg else 192
    batch_size = cfg.batch_size if cfg else 8
    device = cfg.device if cfg else "cuda"

    true_texts = pairs_df[true_col].astype(str).tolist()
    false_texts = pairs_df[false_col].astype(str).tolist()

    # Extract TRUE side
    print(f"  Extracting TRUE side [{desc}] n={len(true_texts)}...")
    hs_true = get_hidden_states(
        true_texts, model, tokenizer, target_layers,
        max_length=max_length, device=device, batch_size=batch_size,
    )
    gc.collect()
    torch.cuda.empty_cache()
    feats_true = hidden_to_sae(hs_true, saes, target_layers, device=device)
    del hs_true
    gc.collect()
    torch.cuda.empty_cache()

    # Extract FALSE side
    print(f"  Extracting FALSE side [{desc}] n={len(false_texts)}...")
    hs_false = get_hidden_states(
        false_texts, model, tokenizer, target_layers,
        max_length=max_length, device=device, batch_size=batch_size,
    )
    gc.collect()
    torch.cuda.empty_cache()
    feats_false = hidden_to_sae(hs_false, saes, target_layers, device=device)
    del hs_false
    gc.collect()
    torch.cuda.empty_cache()

    # Compute delta: truth − hallucination
    delta = {l: feats_true[l] - feats_false[l] for l in target_layers}

    return feats_true, feats_false, delta
