PCA code Kaggle notebook : https://www.kaggle.com/nitsin/pca-demo-1
Video Link:https://youtu.be/tXXnxjj2wM4
# Day 46 - Dimensionality Reduction: Principal Component Analysis (PCA)

## 📌 Overview

Principal Component Analysis (PCA) is one of the most important Dimensionality Reduction techniques in Machine Learning. It reduces the number of features while preserving as much information as possible.

Instead of directly selecting existing features, PCA creates entirely new features called **Principal Components (PCs)** which are linear combinations of the original features.

The primary goal of PCA is:

- Reduce dimensionality
- Remove redundancy
- Reduce multicollinearity
- Improve computational efficiency
- Preserve maximum variance

---

# 1️⃣ Why Do We Need PCA?

## Curse of Dimensionality

As the number of features increases:

- Data becomes sparse
- Distance calculations become unreliable
- Training becomes slower
- Models become prone to overfitting

### Example

Suppose:

```text
Feature Count = 2
```

Space grows as:

```text
10² = 100
```

If:

```text
Feature Count = 10
```

Space grows as:

```text
10¹⁰
```

The feature space becomes extremely large.

---

## Problems Caused by High Dimensions

### Data Sparsity

Data points become spread far apart.

---

### Distance Distortion

Distance-based algorithms struggle:

- KNN
- K-Means
- SVM

because nearest and farthest points become almost equally distant.

---

### Multicollinearity

Highly correlated features create redundancy.

Example:

```text
Shop Size
Shop Rent
```

Both may contain similar information.

---

# 2️⃣ What is PCA?

PCA transforms correlated features into a new set of uncorrelated features called:

```text
Principal Components
```

Properties:

- Orthogonal (90° apart)
- Uncorrelated
- Ranked by variance

---

## Goal of PCA

Find a new coordinate system where:

### PC1

Captures maximum variance.

### PC2

Captures second highest variance.

### PC3

Captures third highest variance.

and so on.

---

# 3️⃣ Geometric Intuition

Imagine a football-shaped dataset:

```text
*******
 ***********
   **************
      ********
```

If we project data onto:

### Wrong Axis

Data collapses together.

Information is lost.

---

### Correct Axis

Data remains spread out.

Variance is preserved.

---

PCA automatically finds this optimal direction.

---

# 4️⃣ Mathematical Foundation

## Step 1: Standardization

PCA is extremely sensitive to feature scales.

Example:

```text
Income : 0 - 1,000,000
Age    : 0 - 100
```

Income dominates variance.

---

### Formula

:contentReference[oaicite:0]{index=0}

Where:

- μ = Mean
- σ = Standard Deviation

---

## Step 2: Covariance Matrix

Measures relationships between features.

### Covariance

Positive:

```text
X ↑  Y ↑
```

Negative:

```text
X ↑  Y ↓
```

No relationship:

```text
Cov ≈ 0
```

---

### Covariance Matrix

For d features:

```text
d × d
```

Matrix.

Diagonal:

```text
Variances
```

Off-diagonal:

```text
Covariances
```

---

## Step 3: Eigenvalues & Eigenvectors

PCA solves:

:contentReference[oaicite:1]{index=1}

Where:

- λ = Eigenvalue
- v = Eigenvector

---

### Eigenvalues

Represent:

```text
Amount of Variance
```

captured by a component.

---

### Eigenvectors

Represent:

```text
Direction
```

of Principal Components.

---

# 5️⃣ Principal Components

After sorting Eigenvalues:

### PC1

Largest Eigenvalue.

Maximum variance.

---

### PC2

Second largest Eigenvalue.

Perpendicular to PC1.

---

### PC3

Third largest Eigenvalue.

Perpendicular to both.

---

# 6️⃣ Data Projection

After selecting top k components:

Project data into new space.

Formula:

:contentReference[oaicite:2]{index=2}

Where:

- Xscaled = Standardized Data
- Vk = Top k Eigenvectors

Output:

```text
Reduced Dataset
```

---

# 7️⃣ PCA Using Scikit-Learn

## Import

```python
from sklearn.decomposition import PCA
```

---

## Standardize Data

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

## Apply PCA

```python
pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)
```

---

## Create DataFrame

```python
import pandas as pd

df_pca = pd.DataFrame(
    X_pca,
    columns=['PC1','PC2']
)
```

---

# 8️⃣ Explained Variance

Shows how much information each component preserves.

```python
print(
    pca.explained_variance_ratio_
)
```

Example:

```text
[0.72, 0.18]
```

Meaning:

```text
PC1 → 72%
PC2 → 18%
```

Total:

```text
90%
```

information retained.

---

# 9️⃣ Choosing Number of Components

## Scree Plot

Visualize cumulative variance.

```python
pca = PCA()

pca.fit(X_scaled)
```

---

```python
import numpy as np

cum_var = np.cumsum(
    pca.explained_variance_ratio_
)
```

---

```python
plt.plot(cum_var)
```

Choose point where curve flattens.

---

## Variance Threshold Method

Automatically preserve:

```text
95%
```

variance.

```python
pca = PCA(
    n_components=0.95
)
```

---

```python
X_pca = pca.fit_transform(
    X_scaled
)
```

Scikit-Learn automatically selects the required number of components.

---

# 🔟 Complete Workflow

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale Data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(
    n_components=0.95
)

X_pca = pca.fit_transform(
    X_scaled
)

print(
    pca.explained_variance_ratio_
)
```

---

# 1️⃣1️⃣ Advantages of PCA

### Reduces Dimensions

✅ Fewer features

---

### Faster Training

✅ Less computation

---

### Removes Multicollinearity

✅ Components are orthogonal

---

### Prevents Overfitting

✅ Removes redundant information

---

### Better Visualization

✅ High-dimensional data can be plotted in 2D/3D

---

# 1️⃣2️⃣ Limitations of PCA

### Loss of Interpretability

Original features disappear.

Instead of:

```text
Age
Salary
Experience
```

we get:

```text
PC1
PC2
PC3
```

---

### Sensitive to Outliers

Extreme values distort variance.

---

### Information Loss

Choosing very small k may discard useful information.

---

### Linear Technique

Captures only linear relationships.

---

# PCA vs Feature Selection

| PCA | Feature Selection |
|------|------------------|
| Creates New Features | Uses Existing Features |
| Reduces Dimensions | Selects Features |
| Loses Interpretability | Keeps Interpretability |
| Removes Correlation | May Keep Correlation |

---

# 📊 Key Takeaways

- PCA stands for Principal Component Analysis.
- Used for Dimensionality Reduction.
- Creates new orthogonal features called Principal Components.
- PC1 captures maximum variance.
- Standardization is mandatory before PCA.
- Uses Covariance Matrix, Eigenvalues, and Eigenvectors.
- Reduces multicollinearity and overfitting.
- Explained Variance helps select components.
- Scree Plot helps determine optimal dimensions.
- Widely used in Machine Learning, Computer Vision, and Data Science.

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## 📚 Concepts Covered

- Dimensionality Reduction
- PCA
- Curse of Dimensionality
- Covariance Matrix
- Eigenvalues
- Eigenvectors
- Principal Components
- Explained Variance
- Scree Plot
- Standardization
- Feature Compression
- Machine Learning Preprocessing