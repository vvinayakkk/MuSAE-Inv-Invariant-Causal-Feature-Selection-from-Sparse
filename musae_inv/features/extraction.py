"""
Feature extraction from Gemma-2-2B residual streams via Gemma Scope SAEs.

Pipeline:
    1. Register forward hooks on target transformer layers
    2. Run batched inference to collect max-pooled residual streams
    3. Encode residual streams through the corresponding SAE encoder
    4. Return SAE feature activations per layer

Memory-efficient: processes in batches with incremental GPU offloading.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from musae_inv.config import Config


def get_hidden_states(
    texts: List[str],
    model,
    tokenizer,
    target_layers: List[int],
    max_length: int = 192,
    device: str = "cuda",
    batch_size: int = 8,
) -> Dict[int, np.ndarray]:
    """Extract max-pooled residual stream activations at target layers.

    For each input text, we register hooks on the specified layers,
    run a forward pass, and collect the max-pooled hidden state
    across the sequence dimension.

    Parameters
    ----------
    texts : list[str]
        Input texts to process.
    model : AutoModelForCausalLM
        The base language model.
    tokenizer : AutoTokenizer
        Corresponding tokenizer.
    target_layers : list[int]
        Layer indices to extract from.
    max_length : int
        Maximum token length.
    device : str
        Compute device.
    batch_size : int
        Inference batch size.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from layer index to array of shape (n_texts, d_model).
    """
    collector = {l: [] for l in target_layers}
    hooks = []

    def make_hook(layer_idx):
        def fn(module, inp, out):
            hs = out[0].detach().float()  # [B, T, d_model]
            pooled = hs.max(dim=1).values  # [B, d_model]
            collector[layer_idx].append(pooled.cpu().numpy())
        return fn

    for l in target_layers:
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Hidden states", leave=False):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                _ = model(**enc)
            del enc
    finally:
        for h in hooks:
            h.remove()

    return {l: np.concatenate(collector[l], axis=0) for l in target_layers}


def hidden_to_sae(
    hidden_dict: Dict[int, np.ndarray],
    saes: Dict,
    target_layers: List[int],
    device: str = "cuda",
    batch_size: int = 256,
) -> Dict[int, np.ndarray]:
    """Encode hidden states through SAE encoders.

    Parameters
    ----------
    hidden_dict : dict[int, np.ndarray]
        Hidden states per layer, shape (n, d_model).
    saes : dict[int, SAE]
        SAE modules per layer.
    target_layers : list[int]
        Layers to process.
    device : str
        Compute device.
    batch_size : int
        Encoding batch size (can be larger since SAE is lightweight).

    Returns
    -------
    dict[int, np.ndarray]
        SAE feature activations per layer, shape (n, d_sae).
    """
    sae_feats = {}
    for l in target_layers:
        hs = hidden_dict[l]
        n = hs.shape[0]
        feats = []
        for i in range(0, n, batch_size):
            ht = torch.FloatTensor(hs[i : i + batch_size]).to(device)
            with torch.no_grad():
                fa = saes[l].encode(ht)
            feats.append(fa.cpu().numpy())
            del ht, fa
        sae_feats[l] = np.concatenate(feats, axis=0)
    return sae_feats


def extract_features(
    df,
    model,
    tokenizer,
    saes: Dict,
    target_layers: List[int],
    cfg: Optional[Config] = None,
    text_col: str = "text",
    desc: str = "",
) -> Dict:
    """Full extraction pipeline: texts → hidden states → SAE features.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with text column and optional 'label' column.
    model : AutoModelForCausalLM
        Base language model.
    tokenizer : AutoTokenizer
        Tokenizer.
    saes : dict[int, SAE]
        SAE modules.
    target_layers : list[int]
        Target layers.
    cfg : Config, optional
        Configuration (uses defaults if None).
    text_col : str
        Column name containing text.
    desc : str
        Description for progress bars.

    Returns
    -------
    dict
        {"sae_features": {layer: ndarray}, "labels": ndarray or None}
    """
    max_length = cfg.max_length if cfg else 192
    batch_size = cfg.batch_size if cfg else 8
    device = cfg.device if cfg else "cuda"

    texts = df[text_col].astype(str).tolist()
    labels = df["label"].values.astype(np.float32) if "label" in df.columns else None

    print(f"  Extracting hidden states [{desc}] n={len(texts)}...")
    hidden_dict = get_hidden_states(
        texts, model, tokenizer, target_layers,
        max_length=max_length, device=device, batch_size=batch_size,
    )
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Converting to SAE features...")
    sae_feats = hidden_to_sae(hidden_dict, saes, target_layers, device=device)
    del hidden_dict
    gc.collect()

    return {"sae_features": sae_feats, "labels": labels}
