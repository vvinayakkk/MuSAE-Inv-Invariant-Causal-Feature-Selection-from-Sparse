.PHONY: help install install-dev test lint format clean download train evaluate figures tables all docker

# ── Variables ─────────────────────────────────────────
PYTHON  := python
PIP     := pip
PYTEST  := pytest
CONFIG  := configs/default.yaml

# ── Help ──────────────────────────────────────────────
help: ## Display this help message
	@echo "MuSAE-Inv — Multi-layer SAE Invariant Causal Feature Selection"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────
install: ## Install package and dependencies
	$(PIP) install -e .

install-dev: ## Install with development dependencies
	$(PIP) install -e ".[dev]"

# ── Testing ───────────────────────────────────────────
test: ## Run test suite
	$(PYTEST) tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=musae_inv --cov-report=html --cov-report=term

# ── Code Quality ──────────────────────────────────────
lint: ## Run linters (flake8 + mypy)
	flake8 musae_inv/ scripts/ tests/ --max-line-length=120
	mypy musae_inv/ --ignore-missing-imports

format: ## Auto-format code with black + isort
	black musae_inv/ scripts/ tests/
	isort musae_inv/ scripts/ tests/

# ── Pipeline ──────────────────────────────────────────
download: ## Download all datasets
	$(PYTHON) scripts/download_data.py

train: ## Run full training pipeline
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate: ## Evaluate trained model
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --run-statistical-tests

baselines: ## Run all baselines
	$(PYTHON) scripts/run_baselines.py --config $(CONFIG)

features: ## Extract features only
	$(PYTHON) scripts/extract_features.py --config $(CONFIG) --counterfactual

figures: ## Generate publication figures
	$(PYTHON) scripts/generate_figures.py --config $(CONFIG)

tables: ## Generate result tables
	$(PYTHON) scripts/generate_tables.py --config $(CONFIG)

all: download features train baselines evaluate figures tables ## Run entire pipeline

# ── Ablation Studies ──────────────────────────────────
ablation-topk: ## Run top-K ablation sweep
	@for k in 16 32 64 128 256 512 1024 2048 4096; do \
		echo "=== Top-K = $$k ==="; \
		$(PYTHON) scripts/train.py --config configs/ablation_topk.yaml --icfs-top-k $$k; \
	done

ablation-layers: ## Run layer ablation sweep
	$(PYTHON) scripts/train.py --config configs/ablation_layers.yaml

ablation-reg: ## Run regularisation sweep
	@for c in 0.001 0.003 0.01 0.03 0.1 0.3 1.0 3.0 10.0; do \
		echo "=== C = $$c ==="; \
		$(PYTHON) scripts/train.py --config configs/ablation_reg.yaml --musae-C $$c; \
	done

# ── Docker ────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -t musae-inv:latest -f docker/Dockerfile .

docker-run: ## Run experiment in Docker container
	docker run --gpus all -v $(PWD)/outputs:/app/outputs musae-inv:latest

# ── Cleanup ───────────────────────────────────────────
clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-outputs: ## Remove all output files
	rm -rf outputs/
