# Day 30 - Feature Engineering: Mathematical Function Transformers

## 📌 Overview

Many Machine Learning algorithms perform better when numerical features follow a **Normal (Gaussian) Distribution**. However, real-world datasets often contain skewed features that can negatively impact model performance.

Mathematical transformations help reshape feature distributions by reducing skewness and making the data more suitable for statistical and machine learning models.

In this notebook, we explore:

* Log Transformation
* Reciprocal Transformation
* Square Root Transformation
* Square Transformation
* FunctionTransformer
* QQ Plots for Normality Analysis
* ColumnTransformer for Selective Feature Transformation

The goal is to identify and apply the most appropriate transformation for a given feature distribution.

---

# 🚀 Quick Revision Notes

## What is a Mathematical Transformation?

A mathematical transformation applies a mathematical function to numerical data in order to modify its distribution.

Common objectives include:

* Reducing skewness
* Improving normality
* Enhancing model performance
* Stabilizing variance

---

## Why Do We Need Transformations?

Many algorithms assume that features are approximately normally distributed.

Examples:

* Linear Regression
* Logistic Regression
* Linear Discriminant Analysis (LDA)
* Gaussian Naive Bayes

Tree-based algorithms do not require normal distributions.

Examples:

* Decision Trees
* Random Forest
* XGBoost
* LightGBM

---

## Understanding QQ Plots

A Quantile-Quantile (QQ) Plot compares:

```text
Actual Data Quantiles
          vs
Theoretical Normal Quantiles
```

If the points closely follow a straight 45° line:

```text
Data ≈ Normal Distribution
```

If significant deviations occur:

```text
Data is Skewed
```

---

# Common Mathematical Transformations

## 1. Log Transformation

```python
np.log1p(x)
```

Formula:

```text
log(x + 1)
```

Best for:

* Highly Right-Skewed Data

Examples:

* Income
* Salary
* House Prices
* Fare

Advantages:

* Compresses large values
* Reduces long right tails
* Handles zero values safely

---

## 2. Reciprocal Transformation

```python
1 / (x + 1e-5)
```

Best for:

* Extremely Right-Skewed Data

Effect:

* Large values become smaller
* Small values become larger

Use carefully because it can drastically alter data relationships.

---

## 3. Square Root Transformation

```python
np.sqrt(x)
```

Best for:

* Moderately Right-Skewed Data

Compared to log transformation:

* Less aggressive
* Preserves more variation

---

## 4. Square Transformation

```python
np.square(x)
```

Best for:

* Left-Skewed Data

Effect:

* Expands larger values
* Stretches the right side of the distribution

---

## 5. FunctionTransformer

```python
from sklearn.preprocessing import FunctionTransformer
```

Allows mathematical functions to be integrated directly into Scikit-Learn pipelines.

Example:

```python
log_transformer = FunctionTransformer(np.log1p)
```

Benefits:

* Pipeline Compatible
* Reusable
* Production Ready
* Cross-Validation Safe

---

# 🔍 Technical Workflow Analysis

## Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
```

---

## Step 2: Load Dataset

```python
df = pd.read_csv(
    'train.csv',
    usecols=['Age', 'Fare', 'Survived']
)
```

Selected Features:

* Age
* Fare

Target:

* Survived

---

## Step 3: Handle Missing Values

```python
df['Age'].fillna(
    df['Age'].mean(),
    inplace=True
)
```

Missing Age values are replaced using Mean Imputation.

---

## Step 4: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    df[['Age', 'Fare']],
    df['Survived'],
    test_size=0.2,
    random_state=42
)
```

---

## Step 5: Analyze Feature Distribution

### KDE Plot

```python
sns.kdeplot(
    X_train['Fare'],
    fill=True
)
```

Used to visualize the feature distribution.

---

### QQ Plot

```python
stats.probplot(
    X_train['Fare'],
    dist="norm",
    plot=plt
)
```

Used to compare feature quantiles against a theoretical normal distribution.

---

## Step 6: Apply Log Transformation

```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(
    func=np.log1p
)
```

---

## Step 7: Transform Data

```python
X_train_log = log_transformer.fit_transform(
    X_train
)

X_test_log = log_transformer.transform(
    X_test
)
```

---

## Step 8: Train Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(
    X_train_log,
    y_train
)
```

---

## Step 9: Evaluate Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy_score(
    y_test,
    clf.predict(X_test_log)
)
```

Compare accuracy before and after transformation.

---

# Selective Feature Transformation

Not every feature requires transformation.

Example:

* Fare → Highly Right-Skewed
* Age → Already Close to Normal

Applying log transformation to Age may actually reduce performance.

---

## Using ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
```

Apply transformations only where necessary.

```python
selective_transformer = ColumnTransformer(
    transformers=[
        (
            'log_fare',
            FunctionTransformer(np.log1p),
            ['Fare']
        )
    ],
    remainder='passthrough'
)
```

This transforms:

```text
Fare
```

while leaving:

```text
Age
```

unchanged.

---

# 📊 Choosing the Right Transformation

| Distribution Type       | Recommended Transformation |
| ----------------------- | -------------------------- |
| Highly Right-Skewed     | Log Transform              |
| Moderately Right-Skewed | Square Root Transform      |
| Extremely Right-Skewed  | Reciprocal Transform       |
| Left-Skewed             | Square Transform           |
| Nearly Normal           | No Transformation          |

---

# 📊 Before vs After Transformation

### Before

```text
Highly Right-Skewed
```

Characteristics:

* Long Right Tail
* Outliers Influence Mean
* Poor Model Assumptions

---

### After Log Transformation

```text
Closer to Normal Distribution
```

Characteristics:

* Reduced Skewness
* Improved Symmetry
* Better Statistical Properties

---

# ⚠️ Common Mistakes

## ❌ Applying One Transformation to Every Feature

Wrong:

```python
log_transformer.fit_transform(X_train)
```

for all columns.

Different features require different transformations.

---

## ❌ Using np.log() with Zero Values

Wrong:

```python
np.log(x)
```

May produce:

```text
-inf
```

for zero values.

Use:

```python
np.log1p(x)
```

instead.

---

## ❌ Ignoring Distribution Analysis

Always inspect:

* Histogram
* KDE Plot
* QQ Plot

before selecting a transformation.

---

# 🎯 Key Takeaways

### ✅ Transformations Reduce Skewness

They help reshape distributions toward normality.

---

### ✅ Choose Transformations Based on Distribution Shape

Different skew patterns require different transformations.

---

### ✅ QQ Plots Help Verify Normality

Closer alignment to a straight line indicates better normality.

---

### ✅ FunctionTransformer Makes Transformations Pipeline-Friendly

Transforms can be integrated directly into machine learning workflows.

---

### ✅ Transform Only Necessary Features

Use ColumnTransformer to selectively target skewed columns.

---

### ✅ Use np.log1p Instead of np.log

Provides safe handling for zero-valued observations.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Seaborn
* Matplotlib
* SciPy
* Jupyter Notebook

---

# 📚 Learning Outcome

By completing this notebook, you will understand:

* Why feature distributions matter
* How skewness affects machine learning models
* How mathematical transformations work
* How to use FunctionTransformer
* How to interpret QQ plots
* How to selectively transform features using ColumnTransformer
* How to improve feature normality for statistical models

---

## 📂 Dataset Used

```text
train.csv
```

### Features

* Age
* Fare

### Target

* Survived

Video Link : https://youtu.be/cTjj3LE8E90
