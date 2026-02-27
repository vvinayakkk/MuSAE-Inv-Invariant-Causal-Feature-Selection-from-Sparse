# Contributing to MuSAE-Inv

Thank you for your interest in contributing to MuSAE-Inv! This guide will help you get started.

## Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse.git
cd MuSAE-Inv-Invariant-Causal-Feature-Selection-from-Sparse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"
```

## Code Style

- **Formatter**: [black](https://github.com/psf/black) (line length = 120)
- **Import sorter**: [isort](https://pycqa.github.io/isort/) (black-compatible profile)
- **Linter**: [flake8](https://flake8.pycqa.org/) (max line length = 120)
- **Type checker**: [mypy](http://mypy-lang.org/) (optional, best-effort)

```bash
# Format
make format

# Lint
make lint
```

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_icfs.py -v
```

## Pull Request Process

1. **Fork** the repository and create your branch from `main`
2. **Add tests** for any new functionality
3. **Run the full test suite** and ensure all tests pass
4. **Update documentation** if you changed APIs or added features
5. **Submit a PR** with a clear description of changes

### PR Checklist

- [ ] Tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Documentation updated if needed
- [ ] CHANGELOG updated for significant changes

## Reporting Issues

When reporting bugs, please include:

1. Python version and OS
2. GPU model and CUDA version (if applicable)
3. Steps to reproduce
4. Expected vs actual behaviour
5. Full error traceback

## Areas for Contribution

- **New SAE architectures**: Support for different SAE widths or training procedures
- **Additional datasets**: New hallucination benchmarks
- **Feature selection methods**: Alternative to ICFS v2
- **Probing methods**: Beyond logistic regression
- **Visualisations**: New analysis plots
- **Documentation**: Tutorials, examples, translations
