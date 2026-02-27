"""
Configuration management for MuSAE-Inv experiments.

Provides a centralised Config dataclass and YAML I/O for reproducible experiments.
All hyperparameters from the paper are set as defaults.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    """Central configuration for the MuSAE-Inv pipeline.

    Attributes
    ----------
    seed : int
        Global random seed for reproducibility.
    device : str
        Compute device ("cuda" or "cpu").
    model_id : str
        HuggingFace model identifier for the base LLM.
    sae_release : str
        SAE-Lens release name for Gemma Scope SAEs.
    sae_l0_tag : str
        Sparsity tag within the SAE release.
    sae_width : str
        SAE dictionary width identifier (e.g., "16k").
    target_layers : list[int]
        Transformer layers to extract residual streams from.
    d_model : int
        Hidden dimension of the base model.
    d_sae : int
        SAE dictionary dimension.
    icfs_top_k : int
        Number of ICFS features to select per layer.
    musae_C : float
        L1 regularisation strength for the MuSAE-Inv probe.
    musae_max_iter : int
        Maximum solver iterations for the logistic regression probe.
    n_train_qa : int
        Number of QA training examples (per class).
    n_val_qa : int
        Number of QA validation examples (per class).
    n_test_qa : int
        Number of QA test examples (per class).
    n_dial_train : int
        Number of Dialogue training examples (per class).
    n_dial_test : int
        Number of Dialogue test examples (per class).
    n_summ_train : int
        Number of Summarisation training examples (per class).
    n_summ_test : int
        Number of Summarisation test examples (per class).
    max_length : int
        Maximum token length for input sequences.
    batch_size : int
        Batch size for feature extraction.
    saplma_hidden : int
        Hidden dimension for the SAPLMA MLP probe.
    saplma_epochs : int
        Training epochs for the SAPLMA probe.
    saplma_lr : float
        Learning rate for the SAPLMA probe.
    output_dir : str
        Root directory for all outputs.
    force_recompute_icfs : bool
        Force re-computation of ICFS scores (skips cache).
    force_recompute_feats : bool
        Force re-extraction of features (skips cache).
    hf_token : str | None
        HuggingFace API token for gated model access.
    """

    # Reproducibility
    seed: int = 42

    # Compute
    device: str = "cuda"

    # Model
    model_id: str = "google/gemma-2-2b"
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_l0_tag: str = "canonical"
    sae_width: str = "16k"
    target_layers: List[int] = field(default_factory=lambda: [6, 12, 18, 25])
    d_model: int = 2304
    d_sae: int = 16384

    # ICFS v2
    icfs_top_k: int = 128

    # MuSAE-Inv probe
    musae_C: float = 0.3
    musae_max_iter: int = 2000

    # Dataset sizes (per class)
    n_train_qa: int = 7000
    n_val_qa: int = 1000
    n_test_qa: int = 1000
    n_dial_train: int = 3000
    n_dial_test: int = 1000
    n_summ_train: int = 3000
    n_summ_test: int = 1000

    # Extraction
    max_length: int = 192
    batch_size: int = 8
    sae_batch_size: int = 256

    # SAPLMA baseline
    saplma_hidden: int = 256
    saplma_epochs: int = 30
    saplma_lr: float = 1e-3

    # Counterfactual pair limits (RAM-safe)
    n_cf_qa: int = 3000
    n_cf_dial: int = 1500
    n_cf_summ: int = 1500

    # Directories
    output_dir: str = "./outputs"

    # Flags
    force_recompute_icfs: bool = True
    force_recompute_feats: bool = False

    # Auth
    hf_token: Optional[str] = None

    # ─── Derived properties ─────────────────────────────────

    @property
    def icfs_dim(self) -> int:
        """Total dimensionality of ICFS feature vector."""
        return len(self.target_layers) * self.icfs_top_k

    @property
    def results_dir(self) -> Path:
        return Path(self.output_dir) / "results"

    @property
    def plots_dir(self) -> Path:
        return Path(self.output_dir) / "plots"

    @property
    def features_dir(self) -> Path:
        return Path(self.output_dir) / "features"

    # ─── I/O ────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def make_dirs(self) -> None:
        """Create output directory structure."""
        for d in [self.results_dir, self.plots_dir, self.features_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def resolve_hf_token(self) -> Optional[str]:
        """Resolve HuggingFace token from config or environment."""
        if self.hf_token:
            return self.hf_token
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def get_default_config(**overrides) -> Config:
    """Create a default configuration with optional overrides.

    Parameters
    ----------
    **overrides
        Keyword arguments to override default values.

    Returns
    -------
    Config
        Configuration instance.
    """
    return Config(**overrides)
