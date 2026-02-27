"""
Logit Lens trajectory consistency score.

Measures how consistently the model "commits" to the final-answer token
across intermediate layers. Low consistency → model is uncertain
→ likely hallucinating.

This is a zero-shot method requiring no training data.

Reference:
    nostalgebraist (2020). "interpreting GPT: the logit lens."
"""

from __future__ import annotations

from typing import Dict, List

import gc

import numpy as np
import torch
from tqdm.auto import tqdm

from musae_inv.config import Config


def logit_lens_commitment_score(
    texts: List[str],
    model,
    tokenizer,
    target_layers: List[int],
    max_length: int = 128,
    device: str = "cuda",
    batch_size: int = 8,
) -> np.ndarray:
    """Compute the Logit Lens commitment score for each text.

    The commitment score measures the fraction of intermediate layers that
    predict the same top-1 token as the final layer at the last position.

    Parameters
    ----------
    texts : list[str]
        Input texts.
    model : AutoModelForCausalLM
        Base language model.
    tokenizer : AutoTokenizer
        Tokenizer.
    target_layers : list[int]
        Intermediate layers to probe.
    max_length : int
        Maximum token length.
    device : str
        Compute device.
    batch_size : int
        Inference batch size.

    Returns
    -------
    np.ndarray, shape (n,)
        Commitment scores in [0, 1]. Higher = more consistent = more truthful.
    """
    collector = {l: [] for l in target_layers}
    hooks = []

    def make_hook(layer_idx):
        def fn(module, inp, out):
            hs = out[0].detach().to(model.lm_head.weight.dtype)
            collector[layer_idx].append(hs[:, -1, :].cpu())  # last token
        return fn

    for l in target_layers:
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    all_scores = []
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Logit lens", leave=False):
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
                out = model(**enc, output_hidden_states=False)
                final_logits = out.logits[:, -1, :]
                final_top1 = final_logits.argmax(-1)
            del out

            bs_actual = len(batch)
            agreements = np.zeros((bs_actual, len(target_layers)))

            for j, l in enumerate(target_layers):
                hs_l = collector[l][-1].to(device)
                # Project through final layer norm + lm_head
                if hasattr(model.model, "norm"):
                    hs_l = model.model.norm(hs_l)
                logits_l = model.lm_head(hs_l)
                pred_l = logits_l.argmax(-1)
                agreements[:, j] = (
                    pred_l.cpu().numpy() == final_top1.cpu().numpy()
                ).astype(float)

            score = agreements.mean(axis=1)
            all_scores.append(score)

            # Clean up
            for l in target_layers:
                collector[l].clear()
            del enc, final_logits, final_top1
    finally:
        for h in hooks:
            h.remove()

    return np.concatenate(all_scores)
