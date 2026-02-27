# Installation Guide

## Prerequisites

| Requirement | Minimum | Recommended |
|:---|:---|:---|
| Python | 3.10 | 3.10–3.12 |
| CUDA | 11.8 | 12.1 |
| VRAM | 8 GB | 16 GB |
| RAM | 16 GB | 32 GB |
| Disk | 10 GB | 20 GB |

## Step 1: Python Environment

### Using venv (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

### Using conda

```bash
conda create -n musae-inv python=3.10
conda activate musae-inv
```

## Step 2: Install PyTorch

Install PyTorch matching your CUDA version:

```bash
# CUDA 11.8
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only (feature extraction will not work, but probe training will)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

## Step 3: Install MuSAE-Inv

### From Source (Recommended)

```bash
git clone https://github.com/vvinayakkk/MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse.git
cd MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse
pip install -e .
```

### From Requirements

```bash
pip install -r requirements.txt
```

### Development Install

```bash
pip install -e ".[dev]"
```

## Step 4: HuggingFace Token

Gemma-2 requires accepting Google's license agreement:

1. Go to [huggingface.co/google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)
2. Accept the license agreement
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set the token:

```bash
# Option A: Environment variable
export HF_TOKEN="hf_your_token_here"

# Option B: .env file
echo "HF_TOKEN=hf_your_token_here" > .env

# Option C: HuggingFace CLI
huggingface-cli login
```

## Step 5: Verify Installation

```bash
python -c "
from musae_inv.config import Config
cfg = Config()
print(f'MuSAE-Inv installed successfully!')
print(f'Target layers: {cfg.target_layers}')
print(f'ICFS top-k: {cfg.icfs_top_k}')
print(f'ICFS dim: {cfg.icfs_dim}')
"
```

## Troubleshooting

### CUDA Out of Memory

If you get OOM errors during feature extraction:

```yaml
# configs/default.yaml
batch_size: 8            # Reduce from 16
max_length: 128          # Reduce from 256
torch_dtype: "bfloat16"  # Keep bfloat16 (half the memory of float32)
```

### SAE Loading Errors

If `sae-lens` fails to load SAEs:

```bash
pip install sae-lens==4.4.0 transformer-lens==2.7.0
```

### Windows-Specific Issues

```bash
# Use PowerShell (not cmd)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

## Docker Installation

```bash
docker build -t musae-inv:latest -f docker/Dockerfile .
docker run --gpus all -it musae-inv:latest bash
```
