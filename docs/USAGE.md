# Usage Guide

## Quick Start

### 1. Download Data

```bash
python scripts/download_data.py
```

### 2. Run Full Pipeline

```bash
python scripts/train.py --config configs/default.yaml
```

This executes the entire MuSAE-Inv pipeline:
- Loads Gemma-2-2B and Gemma Scope SAEs
- Extracts SAE features for all dataset splits
- Computes counterfactual deltas
- Runs ICFS v2 feature selection
- Trains MuSAE-Inv probe and all baselines
- Evaluates on QA, Dialogue, Summarisation, TruthfulQA
- Generates result tables

## Step-by-Step Execution

For more control, run each stage separately:

### Feature Extraction (GPU Required)

```bash
# Extract features for all splits
python scripts/extract_features.py --config configs/default.yaml --counterfactual

# Extract only specific splits
python scripts/extract_features.py --splits halueval_qa_train,halueval_qa_test
```

### Training

```bash
# Full training (loads cached features)
python scripts/train.py --config configs/default.yaml

# Custom hyperparameters
python scripts/train.py --icfs-top-k 512 --musae-C 0.3
```

### Baselines

```bash
python scripts/run_baselines.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml --run-statistical-tests
```

### Figure Generation

```bash
# All figures
python scripts/generate_figures.py --config configs/default.yaml

# Specific figures
python scripts/generate_figures.py --figures 1,3,5 --format pdf --dpi 300
```

### Tables

```bash
python scripts/generate_tables.py --config configs/default.yaml
```

## Python API

### Configuration

```python
from musae_inv.config import Config

# Default config
cfg = Config()

# Custom config
cfg = Config(icfs_top_k=256, musae_C=0.1, device="cuda:0")

# From YAML
cfg = Config.load("configs/default.yaml")

# Save
cfg.save("my_config.yaml")
```

### Feature Extraction

```python
from musae_inv.models.model_loader import load_gemma_model, load_gemma_scope_saes
from musae_inv.features.extraction import extract_features

model, tokenizer = load_gemma_model(cfg)
saes = load_gemma_scope_saes(cfg)
features = extract_features(df, model, tokenizer, saes, cfg.target_layers, cfg=cfg)
```

### ICFS Feature Selection

```python
from musae_inv.features.icfs import compute_icfs_v2
from musae_inv.data.preprocessing import build_icfs_features

result = compute_icfs_v2(cf_cache, cfg.target_layers, cfg.icfs_top_k)
X = build_icfs_features(features, result.indices, cfg.icfs_top_k)
```

### Probe Training

```python
from musae_inv.models.probes import MuSAEInvProbe

probe = MuSAEInvProbe(cfg)
probe.fit(X_train, y_train, X_dial_train, y_dial_train, X_val, y_val)
probabilities = probe.predict_proba(X_test)
```

### Metrics

```python
from musae_inv.evaluation.metrics import compute_metrics

metrics = compute_metrics(y_true, y_prob)
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"Brier: {metrics['brier']:.4f}")
```

## Ablation Studies

### Top-K Sweep

```bash
for K in 16 32 64 128 256 512 1024 2048 4096; do
    python scripts/train.py --config configs/ablation_topk.yaml --icfs-top-k $K
done
```

### Regularisation Sweep

```bash
for C in 0.001 0.003 0.01 0.03 0.1 0.3 1.0 3.0 10.0; do
    python scripts/train.py --config configs/ablation_reg.yaml --musae-C $C
done
```

## Output Files

After a full run, you'll find:

```
outputs/
├── features/
│   ├── feature_cache.pkl          # All SAE features
│   ├── counterfactual_cache.pkl   # CF deltas
│   └── icfs_cache_v2.pkl          # ICFS selections
├── results/
│   ├── results_all.csv            # Method × Domain × Metric
│   ├── results_brier_drop.csv     # Brier degradation
│   ├── results_auroc_drop.csv     # AUROC drops
│   ├── config.yaml                # Saved configuration
│   └── tdv.npy                    # Truth direction vector
└── plots/
    ├── fig01_*.pdf through fig22_*.pdf
    └── (22 publication figures)
```
