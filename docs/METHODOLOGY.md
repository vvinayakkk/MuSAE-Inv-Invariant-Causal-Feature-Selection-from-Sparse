# Methodology

## Overview

MuSAE-Inv is a hallucination detection framework based on invariant causal feature selection from sparse autoencoder representations. This document provides detailed technical methodology.

## 1. Sparse Autoencoder Feature Extraction

### Why SAEs?

Standard transformer hidden states are **polysemantic** — individual neurons encode multiple unrelated concepts. This polysemanticity means that domain-specific information (e.g., "this is a QA context") is entangled with hallucination-relevant information (e.g., "this response contradicts the evidence").

Sparse Autoencoders (SAEs) trained on LLM activations learn **monosemantic features** — each learned direction corresponds to a single interpretable concept. By projecting hidden states through SAEs, we disentangle domain-specific and hallucination-specific information.

### SAE Architecture

We use **Gemma Scope** (Lieberum et al., 2024) SAEs with width 16,384:

$$\hat{z} = \text{ReLU}(W_{\text{enc}} \cdot h + b_{\text{enc}})$$
$$\hat{h} = W_{\text{dec}} \cdot \hat{z} + b_{\text{dec}}$$

Where $h$ is the residual-stream hidden state and $\hat{z}$ is the sparse feature vector.

### Multi-Layer Extraction

We extract features from 4 layers spanning early-to-late processing:

| Layer | Position | Role |
|:---:|:---:|:---|
| 6 | 23% | Early syntactic/lexical features |
| 12 | 46% | Mid-level compositional features |
| 18 | 69% | Late semantic features |
| 25 | 96% | Near-output decision features |

## 2. Invariant Causal Feature Selection (ICFS v2)

### Motivation

Standard feature selection methods (e.g., mutual information, variance) select features that are discriminative on the **training domain** but may not transfer. ICFS instead selects features with **invariant causal effect** across domains.

### Counterfactual Pair Generation

For each example, we extract SAE features for both the truthful and hallucinated response:

$$\delta^{(i)} = z_{\text{true}}^{(i)} - z_{\text{false}}^{(i)}$$

### Causal Effect Estimation

Per-feature causal effect for domain $d$:

$$\text{CE}_d[j] = \frac{1}{N_d} \sum_{i=1}^{N_d} \delta_d^{(i)}[j]$$

### ICFS v2 Scoring Criterion

$$\text{score}[j] = \min(|\text{CE}_{\text{QA}}[j]|, \, |\text{CE}_{\text{Dial}}[j]|) \times \mathbf{1}[\text{sign}(\text{CE}_{\text{QA}}[j]) = \text{sign}(\text{CE}_{\text{Dial}}[j])]$$

This criterion ensures:
1. **Large effect in both domains**: The `min` operation prevents selecting features that are only active in one domain
2. **Consistent direction**: The sign-consistency check ensures the feature changes in the same direction (truth → hallucination) across domains

### Feature Dimensionality

- Per layer: top-128 features selected
- Total: 128 × 4 layers = **512 features**
- Sparsity: 512 / 65,536 = **0.78%** of all SAE features

## 3. Domain-Augmented L1-LR Probe

### Training Procedure

The probe is a logistic regression classifier with:

- **Penalty**: L1 (lasso) for feature-level sparsity
- **Regularisation**: C = 0.3
- **Solver**: SAGA (optimised for L1)
- **Class weighting**: Balanced (inversely proportional to class frequency)

### Domain Augmentation

Training data combines:
1. **QA examples**: Standard train split (7,000 examples)
2. **Dialogue examples**: Constructed from counterfactual cache (500 per class)

This augmentation exposes the probe to both domains during training, reinforcing invariant features.

### ERM Ablation

To measure the impact of domain augmentation, we also train an ERM (Empirical Risk Minimisation) baseline:
- QA-only training data
- L2 penalty instead of L1
- No domain augmentation

## 4. Evaluation Protocol

### Domains

| Domain | Role | Size |
|:---|:---|:---:|
| QA | In-distribution (ID) | 1,500 |
| Dialogue | Out-of-distribution (OOD) | 1,000 |
| Summarisation | OOD | 1,000 |
| TruthfulQA | OOD (zero-shot) | 817 |

### Metrics

- **AUROC**: Area Under the Receiver Operating Characteristic
- **AUPRC**: Area Under the Precision-Recall Curve
- **Balanced Accuracy**: Mean of sensitivity and specificity
- **F1 Score**: Harmonic mean of precision and recall
- **Brier Score**: Mean squared error of predicted probabilities

### Statistical Tests

- **Hanley-McNeil CIs**: 95% confidence intervals for AUROC
- **DeLong's Test**: Pairwise AUROC comparisons with p-values
- **Holm-Bonferroni Correction**: Multiple testing correction
- **Cohen's $d$**: Effect size between hallucinated and truthful groups
