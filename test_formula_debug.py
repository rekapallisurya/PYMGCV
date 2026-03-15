import numpy as np
from pymgcv.utils.formula_parser import FormulaParser
from pymgcv.utils.model_matrix import ModelMatrix

# Test data
np.random.seed(42)
n = 50
data = {
    'y': np.random.randn(n) + 5,
    'x': np.random.randn(n),
}

# Parse formula
parser = FormulaParser('y ~ x')
print("Response:", parser.response)
print("Parametric terms:", parser.parametric_terms)
print("Parametric names:", parser.parametric_names)
print("Smooth terms:", parser.smooth_terms)

# Build model matrix
mm = ModelMatrix(data, 'y ~ x')
print("\nModelMatrix X shape:", mm.X.shape)
print("Column names:", mm.column_names)
print("Parametric indices:", mm.param_indices)
print("Smooth indices:", mm.smooth_indices)
