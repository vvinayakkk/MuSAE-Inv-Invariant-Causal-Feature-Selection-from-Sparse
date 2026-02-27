"""
Model loading utilities for Gemma-2-2B and Gemma Scope SAEs.

Loads the base Gemma-2-2B language model in bfloat16 precision and
the corresponding Gemma Scope sparse autoencoders for the target layers.
Designed for NVIDIA Tesla P100-16GB (Kaggle) with ~6.44 GB VRAM budget.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from musae_inv.config import Config


def load_gemma_model(
    cfg: Config,
    hf_token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Gemma-2-2B in bfloat16 with eager attention.

    Parameters
    ----------
    cfg : Config
        Experiment configuration.
    hf_token : str, optional
        HuggingFace API token. Falls back to Config or environment.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        The base model (frozen, eval mode) and its tokenizer.
    """
    token = hf_token or cfg.resolve_hf_token()

    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✅ HuggingFace login OK")

    print(f"Loading tokenizer for {cfg.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading {cfg.model_id} in bfloat16 (~5.2 GB)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map=cfg.device,
        token=token,
        attn_implementation="eager",
    )
    model.eval()
    model.config.use_cache = False
    print(f"Model loaded in {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM after model: {vram:.2f} GB")

    return model, tokenizer


def load_gemma_scope_saes(
    cfg: Config,
) -> Dict[int, "SAE"]:
    """Load Gemma Scope SAEs for all target layers.

    Each SAE is frozen in eval mode. Typical memory: ~0.3 GB per layer
    (4 layers ≈ 1.2 GB).

    Parameters
    ----------
    cfg : Config
        Experiment configuration.

    Returns
    -------
    dict[int, SAE]
        Mapping from layer index to loaded SAE module.
    """
    from sae_lens import SAE

    saes: Dict[int, SAE] = {}
    print("Loading Gemma Scope SAEs...\n")

    for layer in cfg.target_layers:
        sae_id = f"layer_{layer}/width_{cfg.sae_width}/{cfg.sae_l0_tag}"
        print(f"  Loading SAE layer {layer}: {sae_id}")

        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=cfg.sae_release,
            sae_id=sae_id,
            device=cfg.device,
        )

        sae.eval()
        for p in sae.parameters():
            p.requires_grad_(False)

        saes[layer] = sae

        # Log sparsity metadata
        if sparsity is not None:
            try:
                l0_str = f"{float(sparsity.mean()):.1f}"
            except Exception:
                l0_str = "unknown"
        else:
            l0_str = "N/A"

        print(
            f"    d_in = {sae.cfg.d_in:,} | "
            f"d_sae = {sae.cfg.d_sae:,} | "
            f"L0 ≈ {l0_str}"
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"\nVRAM after SAEs: {vram_used:.2f} / {vram_total:.2f} GB")

    print("\n✅ SAEs loaded and frozen.")
    return saes
