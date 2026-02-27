# Code Architecture

## Package Structure

```
musae_inv/
├── config.py           ← Central configuration
├── models/             ← Model loading + probes
├── data/               ← Dataset I/O + preprocessing
├── features/           ← Feature extraction + ICFS
├── evaluation/         ← Metrics + baselines
├── analysis/           ← Statistical + mechanistic
├── visualization/      ← Plots + tables
└── utils/              ← Seed, I/O helpers
```

## Module Dependency Graph

```
config.py ──────────────────────────────────────────┐
    │                                                │
    ├── models/model_loader.py  (Gemma + SAE load)   │
    │       │                                        │
    ├── data/datasets.py  (HaluEval + TruthfulQA)   │
    │       │                                        │
    ├── features/extraction.py  (SAE features)       │
    │       │                                        │
    ├── features/counterfactual.py  (CF deltas)      │
    │       │                                        │
    ├── features/icfs.py  (ICFS v2 scoring)          │
    │       │                                        │
    ├── data/preprocessing.py  (build ICFS matrices) │
    │       │                                        │
    ├── models/probes.py  (MuSAE-Inv, ERM, etc.)    │
    │       │                                        │
    ├── evaluation/metrics.py  (AUROC, etc.)         │
    │       │                                        │
    ├── evaluation/baselines.py  (SAE-entropy, TDV)  │
    │                                                │
    └── visualization/  (plots + tables) ────────────┘
```

## Key Design Decisions

### 1. Config Dataclass

All hyperparameters live in a single `Config` dataclass with YAML I/O. This ensures:
- **Reproducibility**: Every run saves its config
- **CLI override**: Easy parameter sweeps
- **Type safety**: Dataclass fields are typed

### 2. Feature Caching

Feature extraction is the bottleneck (~45 min). All features are cached as pickle files:
- `feature_cache.pkl` → full SAE features per split
- `counterfactual_cache.pkl` → CF deltas per domain
- `icfs_cache_v2.pkl` → selected feature indices

### 3. Modular Scripts

Each script handles one stage:
- `extract_features.py` → GPU stage
- `train.py` → full pipeline
- `run_baselines.py` → comparison methods
- `evaluate.py` → statistical evaluation

### 4. Hook-Based Feature Extraction

SAE features are extracted via PyTorch forward hooks on specific layers, avoiding the need to modify the base model. Max-pooling over sequence positions gives a fixed-size vector.

## Data Flow

```
Raw Text → Tokenizer → Gemma-2-2B → Hidden States (per layer)
    → SAE Encoder → Sparse Features (16,384-dim per layer)
    → ICFS Selection → 128-dim per layer
    → Concatenate → 512-dim feature vector
    → MuSAE-Inv L1-LR → P(hallucination)
```

## Testing Strategy

Tests are organised by module:
- `test_config.py` → Config creation, YAML roundtrip, derived properties
- `test_icfs.py` → ICFS scoring with synthetic data
- `test_metrics.py` → Metric computation, tracker, statistical tests
- `test_preprocessing.py` → Feature matrix building
- `test_utils.py` → I/O roundtrips, seed determinism
