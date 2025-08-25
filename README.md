# Enhanced Automatic Shifted Log Transformer (EASLT)

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/EASLT.svg)](https://pypi.org/project/EASLT/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Automatically transform skewed data into more normal distributions using Monte Carlo optimized shifted log transformation.**

## Quick Start

```python
from EASLT import AutomaticShiftedLogTransformer
import numpy as np

# Create some skewed data
data = np.random.exponential(2, (1000, 3))

# Transform it
transformer = AutomaticShiftedLogTransformer(random_state=42)
transformed_data = transformer.fit_transform(data)

# That's it! Your data is now more normal
```

## Key Features

- **Automatic Parameter Tuning** - No manual hyperparameter selection needed
- **Fast Processing** - Numba-accelerated computations
- **Robust** - Handles negative values, zeros, and outliers automatically  
- **Adaptive** - Different strategies for different data complexities
- **Multi-metric** - Uses multiple normality tests for reliable results
- **Scikit-learn Compatible** - Drop-in replacement for StandardScaler
- **Reversible** - Full inverse transformation support

## Installation

```bash
pip install EASLT
```


## Basic Usage

### Simple Transformation
```python
from EASLT import AutomaticShiftedLogTransformer

# Initialize
transformer = AutomaticShiftedLogTransformer()

# Fit and transform
X_transformed = transformer.fit_transform(your_data)

# Inverse transform (if needed)
X_original = transformer.inverse_transform(X_transformed)
```

### With Custom Parameters
```python
transformer = AutomaticShiftedLogTransformer(
    mc_iterations=2000,           # More Monte Carlo iterations
    random_state=42,              # Reproducible results
    min_improvement_skewed=0.05   # Higher improvement threshold
)

X_transformed = transformer.fit_transform(your_data)
```

### Get Transformation Details
```python
# See what transformations were applied
summary = transformer.get_transformation_summary()
print(summary)

# Evaluate transformation quality
quality = transformer.evaluate_transformation_quality(your_data)
print(quality)
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | int or None | None | Random seed for reproducible results |
| `mc_iterations` | int | 1000 | Monte Carlo iterations for weight optimization |
| `beta_range` | array-like | np.arange(-8, 8, 0.01) | Range of transformation parameters to search |
| `epsilon` | float | 1e-12 | Small value for numerical stability |

### Adaptive Thresholds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_improvement_normal` | float | 0.001 | Minimum improvement needed for normal-ish data (in percentage) |
| `min_improvement_skewed` | float | 0.01 | Minimum improvement needed for skewed data (in percentage) |
| `early_stop_threshold` | float | 0.85 | Stop optimization if score reaches this |

### Quality Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_kurtosis` | float | 8.0 | Reject transformations with higher kurtosis |
| `max_skewness` | float | 1.0 | Reject transformations with higher skewness |
| `outlier_threshold_normal` | float | 0.05 | Outlier sensitivity for normal data (in percentage) |
| `outlier_threshold_skewed` | float | 0.02 | Outlier sensitivity for skewed data (in percentage) |
| `max_winsor_limits` | float | 0.08 | Maximum winsorization percentage (in percentage) |

### Monte Carlo Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mc_convergence_tolerance` | float | 1e-4 | Convergence threshold for optimization |

## How It Works

The transformer follows these steps:

1. **Data Assessment** - Classifies data complexity (normal, mild issues, needs transformation)
2. **Outlier Handling** - Uses IQR-based detection and adaptive winsorizing
3. **Weight Optimization** - Monte Carlo search for optimal quality score weights
4. **Parameter Search** - Finds best transformation parameters using quality scoring
5. **Transformation** - Applies Feng's shifted log transformation or standardization only
6. **Validation** - Ensures improvement meets minimum thresholds

### Quality Scoring

Uses multiple normality metrics:
- **Anderson-Darling test** - Primary normality assessment
- **Skewness score** - Measures asymmetry
- **Kurtosis score** - Measures tail behavior  
- **Stability score** - Measures numerical stability

Weights are automatically optimized using Monte Carlo sampling.

## Example: Before vs After

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate skewed data
np.random.seed(42)
original = np.random.exponential(2, 1000)

# Transform it
transformer = AutomaticShiftedLogTransformer(random_state=42)
transformed = transformer.fit_transform(original.reshape(-1, 1)).flatten()

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(original, ax=ax1, color='red', alpha=0.7)
ax1.set_title('Before Transformation\n(Highly Skewed)')
ax1.set_ylabel('Frequency')

sns.histplot(transformed, ax=ax2, color='green', alpha=0.7)
ax2.set_title('After Transformation\n(More Normal)')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## Advanced Usage

### Working with Pandas DataFrames
```python
import pandas as pd

df = pd.DataFrame({
    'skewed_col1': np.random.exponential(2, 1000),
    'skewed_col2': np.random.gamma(2, 2, 1000),
    'normal_col': np.random.normal(0, 1, 1000)
})

# Transform
transformer = AutomaticShiftedLogTransformer()
df_transformed = pd.DataFrame(
    transformer.fit_transform(df),
    columns=df.columns,
    index=df.index
)

# Check what happened to each column
for col, info in transformer.get_transformation_summary().items():
    print(f"{col}: {info['method']} (complexity: {info['complexity']})")
```

### Pipeline Integration
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline([
    ('transform', AutomaticShiftedLogTransformer(random_state=42)),
    ('model', RandomForestRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Error Handling

The transformer includes robust error handling:

- **Insufficient Data** - Falls back to standardization for small datasets
- **Numerical Issues** - Automatic domain validation and correction
- **Optimization Failures** - Default weights when Monte Carlo fails
- **Invalid Transformations** - Rejects unstable parameter combinations

## Testing Your Results

```python
from scipy.stats import skew, kurtosis

# Compare before and after
def compare_normality(original, transformed):
    print("Metric | Original | Transformed")
    print("-" * 35)
    print(f"Skewness | {skew(original):.3f} | {skew(transformed):.3f}")
    print(f"Kurtosis | {kurtosis(original):.3f} | {kurtosis(transformed):.3f}")
    
    # Normality scores (if you want to implement quality scoring)
    transformer = AutomaticShiftedLogTransformer()
    orig_score = transformer._quality_score(original)  # Note: private method
    trans_score = transformer._quality_score(transformed)
    print(f"Normality Score | {orig_score:.3f} | {trans_score:.3f}")

compare_normality(original_data, transformed_data)
```

## When to Use This?

**Good for:**
- Machine learning preprocessing
- Statistical analysis requiring normality
- Highly skewed continuous data
- Data with outliers and negative values

**Not ideal for:**
- Categorical data
- Already normal data (though it won't hurt)
- Time series (use with caution)
- Very small datasets (< 8 samples)

## References

- Feng, Q., Hannig, J., & Marron, J. S. (2016). *A Note on Automatic Data Transformation*
- Tukey, J. W. (1977). *Exploratory Data Analysis*
- Box, G. E. P., & Cox, D. R. (1964). *An analysis of transformations*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log/issues)
- **Email**: akmalhusain2003@gmail.com

## Publication
Still on progress

---












