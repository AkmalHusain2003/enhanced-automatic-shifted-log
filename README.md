Got it ✅ — here’s your **PyPI-friendly short README** wrapped in proper Markdown code so you can directly use it as `README.md`:

````markdown
# Enhanced Automatic Shifted Log Transformer

**Automatically transform skewed data into more normal distributions using AI-optimized shifted log transformation.**

---

## 📌 Overview

This transformer improves data normality by applying an **automatically tuned shifted log transformation**.  
It uses **Monte Carlo optimization** with Dirichlet sampling to find the best shift parameters for each feature.

✅ **Reduces skewness**  
✅ **Stabilizes variance**  
✅ **Scikit-learn compatible**  
✅ **Fast** (Numba-accelerated)  

---

## 🔧 Installation

```bash
pip install enhanced-automatic-shifted-log
````

or from source:

```bash
git clone https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log.git
cd enhanced-automatic-shifted-log
pip install -e .
```

---

## 🚀 Quick Start

```python
from enhanced_aslt import AutomaticShiftedLogTransformer

# Fit & transform
transformer = AutomaticShiftedLogTransformer(mc_iterations=1000, random_state=42)
transformed_data = transformer.fit_transform(your_data)
```

---

## 📊 Example: Before vs After

```python
import matplotlib.pyplot as plt, seaborn as sns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(original_data, ax=ax1, color='red'); ax1.set_title('Before')
sns.histplot(transformed_data, ax=ax2, color='green'); ax2.set_title('After')
plt.show()
```

---

## ⚙ Key Parameters

| Parameter                | Description                                   | Default |
| ------------------------ | --------------------------------------------- | ------- |
| `mc_iterations`          | Monte Carlo iterations                        | 1000    |
| `random_state`           | Random seed                                   | None    |
| `min_improvement_skewed` | Minimum skewness improvement for skewed data  | 0.02    |
| `normality_threshold`    | Shapiro-Wilk threshold to skip transformation | 0.9     |

---

## 📚 References

* Feng, C., et al. (2016) – *Shifted log transformation*
* Tukey, J. W. (1977) – *Exploratory Data Analysis*
* Box, G. E. P., & Cox, D. R. (1964) – *Box-Cox transformations*

---

## 📜 License

MIT License – Free to use, modify, and distribute.
Created by **Muhammad Akmal Husain**.

```

