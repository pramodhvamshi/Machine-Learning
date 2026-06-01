# Day 25 - Feature Scaling: Normalization (Min-Max Scaling)

## 📌 Overview

Many Machine Learning algorithms perform better when numerical features are scaled to a common range. **Normalization**, also known as **Min-Max Scaling**, rescales data so that all values fall within a fixed range, typically between **0 and 1**.

Unlike Standardization, which centers data around a mean of zero, Normalization compresses values into a bounded interval while preserving their relative relationships.

This notebook demonstrates how to perform Min-Max Scaling using Scikit-Learn's `MinMaxScaler`.

---

# 🚀 Quick Revision Notes

## What is Normalization?

Normalization is a feature scaling technique that transforms numerical values into a fixed range.

Most commonly:

```text
0 ≤ x ≤ 1
```

This makes features comparable regardless of their original units or magnitude.

---

## Mathematical Formula

Each value is transformed using:

x' = \frac{x - x_{min}}{x_{max} - x_{min}}

Where:

* **x** → Original Value
* **xmin** → Minimum Value of the Feature
* **xmax** → Maximum Value of the Feature
* **x′** → Normalized Value

---

## MinMaxScaler

Scikit-Learn provides:

```python
from sklearn.preprocessing import MinMaxScaler
```

to automatically perform normalization.

---

## Fit Operation

```python
scaler.fit(X_train)
```

Learns:

* Minimum Value
* Maximum Value

from the training data.

---

## Transform Operation

```python
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Applies normalization using the learned boundaries.

---

## Preventing Data Leakage

Always follow:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Never:

```python
scaler.fit(X_test)
```

The test set must remain completely unseen during training.

---

# Alternative Scaling Techniques

## MaxAbsScaler

Scales values by dividing by the largest absolute value.

```python
from sklearn.preprocessing import MaxAbsScaler
```

Resulting range:

```text
-1 to 1
```

Useful for:

* Sparse datasets
* Zero-centered data

---

## RobustScaler

Uses median and Interquartile Range (IQR) instead of minimum and maximum values.

Formula:

x' = \frac{x - Median}{IQR}

Where:

```text
IQR = Q3 - Q1
```

Useful when datasets contain significant outliers.

---

# 🔍 Technical Workflow Analysis

## Step 1: Load Dataset

```python
import pandas as pd

df = pd.read_csv(
    'wine_data.csv',
    header=None,
    usecols=[0,1,2]
)
```

Loads the Wine Dataset.

---

## Step 2: Assign Column Names

```python
df.columns = [
    'Class label',
    'Alcohol',
    'Malic acid'
]
```

Improves readability.

---

## Step 3: Feature-Target Separation

```python
X = df.drop(
    'Class label',
    axis=1
)

y = df['Class label']
```

Separates:

* Features (X)
* Target (y)

---

## Step 4: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)
```

Prevents data leakage by isolating the test data.

---

## Step 5: Initialize MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

Creates the normalization object.

---

## Step 6: Learn Feature Boundaries

```python
scaler.fit(X_train)
```

Computes:

* Feature Minimum
* Feature Maximum

for each column.

---

## Step 7: Transform Data

```python
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Compresses values into the range:

```text
0 → 1
```

---

## Step 8: Convert Back to DataFrame

```python
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)
```

Makes the transformed data easier to inspect.

---

## Step 9: Verify Scaling

### Original Data

```python
print(X_train.describe())
```

Example:

| Feature    | Min   | Max   |
| ---------- | ----- | ----- |
| Alcohol    | 11.03 | 14.83 |
| Malic Acid | 0.74  | 5.80  |

---

### Normalized Data

```python
print(X_train_scaled.describe())
```

Expected:

| Feature    | Min | Max |
| ---------- | --- | --- |
| Alcohol    | 0.0 | 1.0 |
| Malic Acid | 0.0 | 1.0 |

---

# 📊 Standardization vs Normalization

| Standardization                    | Normalization            |
| ---------------------------------- | ------------------------ |
| Mean = 0                           | Range = [0,1]            |
| Std = 1                            | Fixed Boundaries         |
| Uses Mean & Std                    | Uses Min & Max           |
| Less affected by scale differences | Compresses all values    |
| Common ML default                  | Popular in Deep Learning |

---

# 🎯 When to Use Normalization

### Recommended For

* Neural Networks
* Deep Learning Models
* Image Processing
* Distance-Based Algorithms
* Gradient-Based Optimization

Images are often normalized because pixel values naturally lie within bounded ranges.

Example:

```text
0 → 255
```

becomes:

```text
0 → 1
```

---

# ⚠️ Sensitivity to Outliers

Min-Max Scaling relies directly on:

```text
Minimum Value
Maximum Value
```

A single extreme outlier can dramatically compress the remaining data.

Example:

```text
Normal Values: 10 - 100

Outlier: 10000
```

After scaling, most values become squeezed near zero.

---

# Solution for Outliers

If outliers are present, consider:

```python
from sklearn.preprocessing import RobustScaler
```

because it uses:

* Median
* Interquartile Range (IQR)

instead of minimum and maximum values.

---

# 🎯 Key Takeaways

### ✅ Normalization Scales Data into Fixed Boundaries

Typically:

```text
0 → 1
```

---

### ✅ Relative Ordering Remains Intact

Normalization changes scale but preserves relationships between observations.

---

### ✅ Always Fit Only on Training Data

Correct:

```python
scaler.fit(X_train)
```

Incorrect:

```python
scaler.fit(X_test)
```

---

### ✅ Ideal for Deep Learning

Many activation functions perform better when inputs are bounded.

---

### ✅ Outliers Can Distort Results

Min-Max Scaling is sensitive to extreme values.

Use:

```python
RobustScaler()
```

when significant outliers exist.

---

# 🛠️ Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib
* Jupyter Notebook

---

# 📚 Learning Outcome

By completing this notebook, you will understand:

* What Normalization is
* How Min-Max Scaling works mathematically
* How to use MinMaxScaler
* How to avoid data leakage
* Differences between Standardization and Normalization
* The effect of outliers on scaling
* Alternative scaling techniques such as MaxAbsScaler and RobustScaler

---

## 📂 Dataset Used

```text
wine_data.csv
```

### Features

* Alcohol
* Malic Acid

### Target

* Class Label

Video Link: https://youtu.be/eBrGyuA2MIg
