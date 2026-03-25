# Installation

## From PyPI (recommended)

```bash
pip install pymgcv
```

## With Optional Extras

```bash
# Visualization support (matplotlib + plotly)
pip install pymgcv[viz]

# GPU acceleration (JAX)
pip install pymgcv[gpu]

# Everything
pip install pymgcv[all]
```

## From Source (development)

```bash
git clone https://github.com/surya/pymgcv.git
cd pymgcv
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows

pip install -e ".[dev,viz]"
```

## From GitHub (latest)

```bash
pip install git+https://github.com/surya/pymgcv.git
```

## Requirements

| Dependency | Minimum | Purpose |
|-----------|---------|---------|
| Python    | 3.11    | Runtime |
| NumPy     | 1.21    | Core numerics |
| SciPy     | 1.7     | Linear algebra, optimization |
| pandas    | 1.3     | Data handling |

### Optional

| Dependency | Purpose |
|-----------|---------|
| JAX / JAXlib | GPU acceleration |
| matplotlib / plotly | Visualization |
| mkdocs-material | Documentation |

## Verify Installation

```python
import pymgcv
print(pymgcv.__version__)

# Check native engine availability
from pymgcv.linalg import backend_info
print(backend_info())
```
