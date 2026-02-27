"""
MuSAE-Inv: Multi-Layer Sparse Autoencoder with Invariant Causal Feature Selection
for Cross-Domain Hallucination Detection.

This package implements the complete MuSAE-Inv pipeline:
    1. Feature extraction from Gemma-2-2B residual streams via Gemma Scope SAEs
    2. Counterfactual delta computation (truth − hallucination)
    3. ICFS v2 scoring: min-CE × sign-consistency feature selection
    4. Domain-augmented L1-regularised logistic regression probe
    5. Zero-shot baselines: TDV direction vector, Logit Lens commitment
    6. Evaluation across HaluEval (QA, Dialogue, Summarisation) + TruthfulQA

Reference:
    Kumar, V. (2025). MuSAE-Inv: Multi-Layer Sparse Autoencoder with Invariant
    Causal Feature Selection for Cross-Domain Hallucination Detection.
"""

__version__ = "1.0.0"
__author__ = "Vinayak Kumar"

from musae_inv.config import Config, get_default_config
