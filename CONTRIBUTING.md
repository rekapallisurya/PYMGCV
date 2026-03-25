# Contributing to pymgcv

Thank you for your interest in contributing to **pymgcv**! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/surya/pymgcv.git
cd pymgcv

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install in editable mode with dev dependencies
pip install -e ".[dev,viz]"

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=pymgcv --cov-report=term-missing

# Run a specific test file
pytest tests/test_integration.py -v

# Skip slow / GPU tests
pytest -m "not slow and not gpu"
```

## Code Style

- **Formatter**: [Black](https://black.readthedocs.io/) (line length 100)
- **Linter**: [Ruff](https://docs.astral.sh/ruff/)
- **Import sorting**: handled by Ruff (isort-compatible)
- **Type annotations**: enforced via [mypy](https://mypy.readthedocs.io/) in strict mode

Pre-commit hooks enforce all of the above automatically on every commit.

## Pull Request Process

1. Fork the repo and create a feature branch from `main`.
2. Write tests for any new functionality.
3. Ensure `pytest` passes and coverage does not decrease.
4. Run `pre-commit run --all-files` before pushing.
5. Open a PR with a clear description of what changed and why.
6. Link any related issues.

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: incompatible API changes
- **MINOR**: new features (backward-compatible)
- **PATCH**: bug fixes (backward-compatible)

## Reporting Issues

- Use the [GitHub issue tracker](https://github.com/surya/pymgcv/issues).
- Include a minimal reproducible example.
- Note your Python version, OS, and pymgcv version (`pymgcv.__version__`).

## Code of Conduct

Be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
