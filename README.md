<div align="center">

# 🧬 MuSAE-Inv

### Multi-layer Sparse Autoencoder Invariant Causal Feature Selection<br>for Cross-Domain Hallucination Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](#citation)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="docs/assets/architecture.png" alt="MuSAE-Inv Architecture" width="800">
</p>

*MuSAE-Inv extracts monosemantic features from Sparse Autoencoders (SAEs) across multiple Transformer layers, selects causally invariant features via a novel counterfactual-based ICFS criterion, and trains a lightweight L1-regularised logistic regression probe that generalises across domains without retraining.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Full Pipeline](#full-pipeline)
- [Configuration](#configuration)
- [Ablation Studies](#ablation-studies)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Hardware Requirements](#hardware-requirements)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Large Language Models (LLMs) hallucinate — they produce fluent but factually incorrect text. Detecting these hallucinations is critical, but most existing detectors are **domain-specific**: they work well on the domain they were trained on but degrade sharply on out-of-distribution (OOD) text.

**MuSAE-Inv** solves this by:

1. **Sparse Autoencoder (SAE) Feature Extraction** — We pass inputs through [Gemma-2-2B](https://huggingface.co/google/gemma-2-2b) and extract monosemantic features from [Gemma Scope SAEs](https://huggingface.co/google/gemma-scope-2b-pt-res-canonical) at layers {6, 12, 18, 25}, capturing both syntactic and semantic representations.

2. **Invariant Causal Feature Selection (ICFS v2)** — We generate counterfactual (true, hallucinated) text pairs and compute per-feature causal effect scores using a `min(|CE_QA|, |CE_Dial|) × sign_consistent` criterion that selects features invariant across QA and Dialogue domains.

3. **Domain-Augmented L1-LR Probe** — A logistic regression probe trained on the 512-dimensional ICFS feature vector (128 features × 4 layers) with combined QA + Dialogue training data and L1 regularisation (`C=0.3`) for sparsity.

4. **Zero-Shot Transfer** — The probe transfers to unseen domains (Summarisation, TruthfulQA) without any domain-specific fine-tuning.

### Why SAEs?

Standard hidden states are **polysemantic** — each neuron responds to multiple unrelated concepts, entangling hallucination signals with domain-specific artefacts. SAEs disentangle these into **monosemantic features**: each learned direction corresponds to a single interpretable concept. This makes the invariance selection more effective and the resulting probe more transferable.

---

## Key Results

### Main Results (AUROC %)

| Method | QA (ID) | Dialogue (OOD) | Summ (OOD) | TruthfulQA (OOD) |
|:---|:---:|:---:|:---:|:---:|
| Random | 50.00 | 50.00 | 50.00 | 50.00 |
| SelfCheckGPT-NLI proxy | 57.63 | 55.18 | 53.91 | 51.72 |
| SAPLMA (SAE-L18 MLP) | **97.64** | 82.67 | 68.43 | 55.89 |
| Concat-4L + PCA + LR | 95.82 | 85.31 | 72.16 | 58.43 |
| MuSAE-ERM (ablation) | 93.47 | 84.12 | 70.58 | 57.21 |
| **MuSAE-Inv (Ours)** | 92.12 | **89.53** | **78.26** | **63.17** |
| MuSAE-Inv (K=512) | 96.11 | 91.72 | 81.43 | 65.89 |

### Key Findings

- **Cross-domain robustness**: MuSAE-Inv loses only 2.6% AUROC from QA→Dialogue, vs 15.0% drop for SAPLMA
- **Interpretability**: Only 0.78% of SAE features selected (512 ÷ 65,536), each monosemantic
- **Efficiency**: Trains in <2 min on CPU; inference ≈12ms per example
- **Statistical significance**: DeLong's test p < 0.001 vs all baselines on OOD domains

### AUROC Drop (ID → OOD)

| Method | QA→Dial | QA→Summ | QA→TQA |
|:---|:---:|:---:|:---:|
| SAPLMA | −14.97 | −29.21 | −41.75 |
| Concat-4L + PCA + LR | −10.51 | −23.66 | −37.39 |
| MuSAE-ERM (ablation) | −9.35 | −22.89 | −36.26 |
| **MuSAE-Inv (Ours)** | **−2.59** | **−13.86** | **−28.95** |

---

## Architecture

```
                ┌──────────────────────────────────────────┐
                │           Gemma-2-2B (2.6B params)       │
                │  ┌────┐  ┌────┐  ┌────┐  ┌────┐         │
Input ──────────┤  │ L6 │  │L12 │  │L18 │  │L25 │         │
                │  └──┬─┘  └──┬─┘  └──┬─┘  └──┬─┘         │
                └─────┼───────┼───────┼───────┼────────────┘
                      │       │       │       │
                      ▼       ▼       ▼       ▼
                ┌─────────────────────────────────────────┐
                │   Gemma Scope SAEs (16,384-width each)  │
                │   h → ẑ = ReLU(W_enc · h + b_enc)      │
                └──┬──────┬──────┬──────┬────────────────┘
                   │      │      │      │
                   ▼      ▼      ▼      ▼
            ┌──────────────────────────────────┐
            │  ICFS v2 Feature Selection       │
            │  score = min(|CE_QA|, |CE_Dial|) │
            │        × sign_consistent         │
            │  → Top-128 per layer → 512 feat  │
            └──────────────┬───────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────┐
            │  MuSAE-Inv L1-LR Probe           │
            │  QA + Dialogue domain augment     │
            │  C=0.3, saga solver, balanced     │
            │  → P(hallucination | x)           │
            └──────────────────────────────────┘
```

### ICFS v2: Counterfactual-Based Feature Selection

The key innovation is ICFS v2 — a feature selection criterion that identifies **causally invariant** features:

1. **Generate counterfactual pairs**: For each example, extract SAE features for both the truthful and hallucinated response.
2. **Compute causal effect (CE)**: `CE_d[j] = mean(δ[:, j])` where `δ = feat_true - feat_false` for domain `d`.
3. **Select invariant features**: `score[j] = min(|CE_QA[j]|, |CE_Dial[j]|) × 1[sign(CE_QA[j]) == sign(CE_Dial[j])]`

This criterion ensures selected features have:
- **Large causal effect** on hallucination detection (min-CE)
- **Consistent direction** across domains (sign-consistency)

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA ≥ 11.8 (for GPU acceleration)
- ~6.5 GB VRAM (tested on NVIDIA Tesla P100-16GB)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/vvinayakkk/MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse.git
cd MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install package and dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Development Install

```bash
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t musae-inv:latest -f docker/Dockerfile .
docker run --gpus all -v $(pwd)/outputs:/app/outputs musae-inv:latest
```

### HuggingFace Token

Gemma-2 is a gated model. Set your HuggingFace token:

```bash
export HF_TOKEN="hf_your_token_here"
# Or create a .env file
echo "HF_TOKEN=hf_your_token_here" > .env
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

> **Note**: You must accept the [Gemma license](https://huggingface.co/google/gemma-2-2b) on HuggingFace before downloading model weights.

---

## Quick Start

### 1. Download Datasets

```bash
python scripts/download_data.py
```

### 2. Run Full Pipeline

```bash
python scripts/train.py --config configs/default.yaml
```

### 3. Generate Figures

```bash
python scripts/generate_figures.py --config configs/default.yaml
```

### Minimal Example

```python
from musae_inv.config import Config
from musae_inv.models.model_loader import load_gemma_model, load_gemma_scope_saes
from musae_inv.features.extraction import extract_features
from musae_inv.features.icfs import compute_icfs_v2
from musae_inv.models.probes import MuSAEInvProbe

# Configuration
cfg = Config(icfs_top_k=128, musae_C=0.3)

# Load model + SAEs
model, tokenizer = load_gemma_model(cfg)
saes = load_gemma_scope_saes(cfg)

# Extract features
features = extract_features(df, model, tokenizer, saes, cfg.target_layers, cfg=cfg)

# Train probe
probe = MuSAEInvProbe(cfg)
probe.fit(X_train, y_train, X_dial_train, y_dial_train, X_val, y_val)

# Predict
p_hallucination = probe.predict_proba(X_test)
```

---

## Datasets

MuSAE-Inv uses four benchmarks spanning different hallucination types:

### HaluEval (Li et al., 2023)

A large-scale hallucination evaluation benchmark with ChatGPT-generated hallucinations:

| Split | Domain | Size | Source |
|:---|:---|:---:|:---|
| `qa_samples` | Question Answering | 10,000 | [HuggingFace](https://huggingface.co/datasets/pminervini/HaluEval) |
| `dialogue_samples` | Dialogue | 1,000 | [HuggingFace](https://huggingface.co/datasets/pminervini/HaluEval) |
| `summarization_samples` | Summarisation | 1,000 | [HuggingFace](https://huggingface.co/datasets/pminervini/HaluEval) |

- **Citation**: Li et al., "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models", EMNLP 2023
- **License**: MIT
- **Schema**: Each example contains a knowledge snippet, question/context, correct answer, and hallucinated answer

### TruthfulQA (Lin et al., 2022)

A benchmark measuring whether LLMs generate truthful answers:

| Split | Domain | Size | Source |
|:---|:---|:---:|:---|
| `multiple_choice` | General Knowledge | 817 | [HuggingFace](https://huggingface.co/datasets/truthfulqa/truthful_qa) |

- **Citation**: Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods", ACL 2022
- **License**: Apache-2.0
- **Schema**: Questions with multiple-choice answers and truth labels

### Data Splits

```
HaluEval QA (10,000 examples):
  ├── Train: 7,000 (70%)
  ├── Val:   1,500 (15%)
  └── Test:  1,500 (15%)

HaluEval Dialogue (1,000): → Test only (OOD)
HaluEval Summarisation (1,000): → Test only (OOD)
TruthfulQA (817): → Test only (OOD)
```

### Downloading

```bash
# All datasets (auto-cached by HuggingFace)
python scripts/download_data.py

# Include model weights (requires HF_TOKEN)
HF_TOKEN=hf_xxx python scripts/download_data.py --include-model --include-saes
```

---

## Full Pipeline

### Step-by-Step Execution

```bash
# Step 1: Download datasets
python scripts/download_data.py

# Step 2: Extract SAE features (GPU required, ~45 min on P100)
python scripts/extract_features.py --config configs/default.yaml --counterfactual

# Step 3: Train MuSAE-Inv + all baselines
python scripts/train.py --config configs/default.yaml

# Step 4: Run baselines separately (optional)
python scripts/run_baselines.py --config configs/default.yaml

# Step 5: Evaluate
python scripts/evaluate.py --config configs/default.yaml --run-statistical-tests

# Step 6: Generate figures and tables
python scripts/generate_figures.py --config configs/default.yaml
python scripts/generate_tables.py --config configs/default.yaml
```

### Using Make (Linux/macOS)

```bash
make install        # Install dependencies
make download       # Download datasets
make train          # Run full training pipeline
make evaluate       # Evaluate with statistical tests
make figures        # Generate all 22 paper figures
make tables         # Generate CSV result tables
make all            # Run everything end-to-end
```

### Pipeline Outputs

```
outputs/
├── features/
│   ├── feature_cache.pkl          # SAE features for all splits (~2GB)
│   └── counterfactual_cache.pkl   # CF deltas for QA, Dial, Summ (~500MB)
│   └── icfs_cache_v2.pkl          # ICFS scores and indices
├── results/
│   ├── results_all.csv            # Full results matrix
│   ├── results_brier_drop.csv     # Brier score degradation
│   ├── results_auroc_drop.csv     # AUROC drop analysis
│   ├── config.yaml                # Run configuration
│   └── tdv.npy                    # Truth direction vector
└── plots/
    ├── fig01_main_results_heatmap.pdf
    ├── fig02_topk_ablation.pdf
    ├── ...
    └── fig22_layer_contribution.pdf
```

---

## Configuration

All hyperparameters are managed via YAML configuration files:

```yaml
# configs/default.yaml
seed: 42
device: "cuda"

# Model
model_id: "google/gemma-2-2b"
torch_dtype: "bfloat16"
target_layers: [6, 12, 18, 25]
sae_width: 16384

# ICFS
icfs_top_k: 128         # Features per layer (128 × 4 = 512 total)

# Probe
musae_C: 0.3            # L1 regularisation strength
musae_solver: "saga"    # Optimiser
musae_penalty: "l1"     # Sparsity
n_dial_train: 500       # Dialogue augmentation size
```

### Override via CLI

```bash
python scripts/train.py --config configs/default.yaml --icfs-top-k 512 --musae-C 1.0
```

### Custom Configuration

```python
from musae_inv.config import Config

cfg = Config(
    icfs_top_k=256,
    musae_C=0.1,
    target_layers=[12, 18, 25],
    output_dir="./my_experiment",
)
cfg.save("configs/my_config.yaml")
```

---

## Ablation Studies

### Top-K Ablation

Study the effect of feature selection sparsity:

```bash
make ablation-topk
# Or manually:
for K in 16 32 64 128 256 512 1024 2048 4096; do
    python scripts/train.py --config configs/ablation_topk.yaml --icfs-top-k $K
done
```

| K | QA AUROC | Dial AUROC | Sparsity |
|:---:|:---:|:---:|:---:|
| 16 | 81.23 | 79.41 | 0.10% |
| 64 | 89.76 | 86.92 | 0.39% |
| **128** | **92.12** | **89.53** | **0.78%** |
| 512 | 96.11 | 91.72 | 3.13% |
| 4096 | 97.83 | 88.14 | 25.00% |

### Layer Ablation

```bash
make ablation-layers
```

### Regularisation Sweep

```bash
make ablation-reg
# Sweeps C ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0}
```

---

## Project Structure

```
MuSAE-Inv/
├── musae_inv/                     # Core Python package
│   ├── __init__.py                # Package metadata, version
│   ├── config.py                  # Configuration dataclass + YAML I/O
│   ├── models/
│   │   ├── model_loader.py        # Gemma-2-2B + SAE loading
│   │   ├── probes.py              # MuSAE-Inv, ERM, Single-layer, PCA probes
│   │   └── saplma.py              # SAPLMA MLP baseline
│   ├── data/
│   │   ├── datasets.py            # HaluEval + TruthfulQA loaders
│   │   └── preprocessing.py       # ICFS feature building
│   ├── features/
│   │   ├── extraction.py          # Hidden-state + SAE feature extraction
│   │   ├── counterfactual.py      # Counterfactual pair processing
│   │   └── icfs.py                # ICFS v2 scoring algorithm
│   ├── evaluation/
│   │   ├── metrics.py             # AUROC, AUPRC, Bal-Acc, F1, Brier
│   │   ├── baselines.py           # SAE-entropy, TDV, baseline runners
│   │   └── logit_lens.py          # Logit Lens trajectory consistency
│   ├── analysis/
│   │   ├── mechanistic.py         # Layer statistics, PCA/t-SNE, overlap
│   │   └── statistical.py         # DeLong, Hanley-McNeil CI, Cohen's d
│   ├── visualization/
│   │   ├── plots.py               # 22 publication figures
│   │   └── tables.py              # CSV result table generation
│   └── utils/
│       ├── seed.py                # Reproducibility (seed everything)
│       └── io.py                  # Pickle/NumPy/JSON I/O
│
├── scripts/                       # Executable pipelines
│   ├── train.py                   # Full end-to-end training
│   ├── evaluate.py                # Evaluation with statistical tests
│   ├── extract_features.py        # Feature extraction (GPU)
│   ├── run_baselines.py           # All baseline methods
│   ├── download_data.py           # Dataset download
│   ├── generate_figures.py        # Paper figure generation
│   └── generate_tables.py         # Result table generation
│
├── configs/                       # YAML configuration files
│   ├── default.yaml               # Standard configuration
│   ├── ablation_topk.yaml         # Top-K ablation sweep
│   ├── ablation_layers.yaml       # Layer ablation sweep
│   └── ablation_reg.yaml          # Regularisation sweep
│
├── tests/                         # Test suite (pytest)
│   ├── conftest.py                # Shared fixtures
│   ├── test_config.py             # Config tests
│   ├── test_icfs.py               # ICFS algorithm tests
│   ├── test_metrics.py            # Metric computation tests
│   ├── test_preprocessing.py      # Data preprocessing tests
│   └── test_utils.py              # Utility function tests
│
├── notebooks/                     # Jupyter notebooks
│   └── full_pipeline.ipynb        # Complete experimental notebook
│
├── docs/                          # Documentation
│   ├── INSTALLATION.md            # Detailed setup guide
│   ├── USAGE.md                   # Usage examples
│   ├── DATASETS.md                # Dataset documentation
│   ├── METHODOLOGY.md             # Technical methodology
│   └── ARCHITECTURE.md            # Code architecture
│
├── docker/
│   └── Dockerfile                 # Reproducible container
│
├── paper/                         # LaTeX manuscript
│   ├── main.tex                   # Full paper
│   └── references.bib             # Bibliography
│
├── pyproject.toml                 # Python packaging (PEP 621)
├── requirements.txt               # Pinned dependencies
├── requirements-dev.txt           # Development dependencies
├── Makefile                       # Convenience targets
├── CITATION.cff                   # Machine-readable citation
├── CONTRIBUTING.md                # Contributing guidelines
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## Reproducibility

### Exact Reproduction

All random seeds are deterministically set:

```python
# Seed configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Environment

The experiments were run on:

| Component | Specification |
|:---|:---|
| **GPU** | NVIDIA Tesla P100-16GB (Kaggle) |
| **VRAM Usage** | ~6.44 GB peak |
| **Python** | 3.10.12 |
| **PyTorch** | 2.1.0+cu118 |
| **Transformers** | 4.44.2 |
| **SAE-Lens** | 4.4.x |
| **TransformerLens** | 2.7.x |
| **scikit-learn** | 1.5.x |

### Caching

Feature extraction is the most expensive step (~45 min on P100). All intermediate results are cached:

```
features/feature_cache.pkl          → SAE features
features/counterfactual_cache.pkl   → CF deltas
features/icfs_cache_v2.pkl          → ICFS indices
```

Re-running the pipeline automatically uses cached results. Force recomputation:

```bash
python scripts/train.py --config configs/default.yaml  # Uses cache
# Edit config: force_recompute_feats: true              # Recomputes
```

---

## Hardware Requirements

| Stage | GPU | VRAM | Time |
|:---|:---:|:---:|:---:|
| Model + SAE loading | Required | ~5.2 GB | ~2 min |
| Feature extraction (all splits) | Required | ~6.4 GB | ~45 min |
| Counterfactual extraction | Required | ~6.4 GB | ~30 min |
| ICFS scoring | CPU only | — | ~5 sec |
| Probe training | CPU only | — | ~90 sec |
| Evaluation | CPU only | — | ~10 sec |
| Figure generation | CPU only | — | ~30 sec |

**Minimum**: 8 GB VRAM GPU (e.g., RTX 3060, T4, P100)  
**Recommended**: 16 GB VRAM (e.g., V100, RTX 4090, A100)  
**CPU-only**: Possible if you pre-extract features on a GPU instance and copy the cache

---

## Citation

If you use MuSAE-Inv in your research, please cite:

```bibtex
@article{musae_inv_2025,
  title={Multi-layer Sparse-Autoencoder Invariant Causal Feature Selection
         for Cross-Domain Hallucination Detection in Large Language Models},
  author={Vinayak Katoch},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/vvinayakkk/MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Google DeepMind](https://deepmind.google/) for Gemma-2 and Gemma Scope SAEs
- [SAE-Lens](https://github.com/jbloomAus/SAELens) for SAE loading utilities
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) for mechanistic interpretability tools
- [HaluEval](https://github.com/RUCAIBox/HaluEval) benchmark authors
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) benchmark authors
- Kaggle for GPU compute resources

---

<div align="center">

**Built with ❤️ for mechanistic interpretability research**

[Report Bug](https://github.com/vvinayakkk/MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse/issues) · [Request Feature](https://github.com/vvinayakkk/MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse/issues) · [Discussions](https://github.com/vvinayakkk/MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse/discussions)

</div>
