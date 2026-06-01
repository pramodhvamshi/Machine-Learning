# Day 32 - Feature Engineering: Binning & Binarization

## 📌 Overview

Feature Engineering plays a crucial role in improving machine learning model performance. In this lesson, we explore two important preprocessing techniques:

- **Discretization (Binning)** – Converts continuous numerical data into categorical intervals.
- **Binarization** – Converts numerical values into binary values (0 or 1) based on a threshold.

These techniques help simplify data, reduce the impact of outliers, and capture non-linear patterns.

---

# 1️⃣ Discretization (Binning)

## What is Discretization?

Discretization, also known as **Binning**, is the process of converting continuous numerical values into a finite number of intervals (bins).

### Benefits

- Handles outliers effectively
- Reduces noise in data
- Helps models capture non-linear relationships
- Improves interpretability

---

## Types of Binning

### Equal Width (Uniform) Binning

Divides the entire range into intervals of equal size.

\[
Width = \frac{Max - Min}{Number\ of\ Bins}
\]

**Pros**
- Simple to understand
- Easy to implement

**Cons**
- Highly sensitive to outliers

---

### Equal Frequency (Quantile) Binning

Creates bins containing approximately the same number of observations.

**Pros**
- Handles skewed distributions well
- More balanced bins

**Cons**
- Bin widths are not equal

---

### K-Means Binning

Uses K-Means clustering to group values into bins based on similarity.

**Pros**
- Data-driven
- Captures natural groupings

**Cons**
- Computationally more expensive

---

### Decision Tree (Supervised) Binning

Uses a Decision Tree to determine optimal split points based on the target variable.

**Pros**
- Uses target information
- Often produces highly informative bins

**Cons**
- More complex than unsupervised methods

---

## KBinsDiscretizer

Scikit-Learn provides the `KBinsDiscretizer` class for automatic binning.

### Important Parameters

### `n_bins`

Number of bins to create.

```python
n_bins=10
```

### `strategy`

Determines the binning method.

```python
strategy='uniform'
strategy='quantile'
strategy='kmeans'
```

### `encode`

Determines output format.

```python
encode='ordinal'
encode='onehot'
```

- `ordinal` → Returns integer labels (0,1,2,...)
- `onehot` → Returns one-hot encoded sparse matrix

---

# 2️⃣ Binarization

## What is Binarization?

Binarization converts numerical values into binary values (0 or 1) using a threshold.

### Formula

\[
Output =
\begin{cases}
1 & \text{if } x > threshold \\
0 & \text{if } x \le threshold
\end{cases}
\]

---

## Why Use Binarization?

Useful when only the presence or absence of a condition matters.

### Examples

- Purchased or Not Purchased
- Active User or Inactive User
- Passed or Failed
- Premium Customer or Regular Customer

---

## Binarizer in Scikit-Learn

```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0.0)
```

---

# 🧪 Implementation Workflow

## Step 1: Import Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
```

---

## Step 2: Load Dataset

```python
df = pd.read_csv('train.csv',
                 usecols=['Age', 'Fare', 'Survived'])

df['Age'].fillna(df['Age'].mean(), inplace=True)
```

---

## Step 3: Define Features and Target

```python
X = df[['Age', 'Fare']]
y = df['Survived']
```

---

## Step 4: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## Step 5: Baseline Model

```python
clf = DecisionTreeClassifier()

print(
    np.mean(
        cross_val_score(
            clf,
            X,
            y,
            cv=10,
            scoring='accuracy'
        )
    )
)
```

This provides a benchmark accuracy before applying any feature transformations.

---

## Step 6: Apply Discretization

```python
kbins = KBinsDiscretizer(
    n_bins=10,
    encode='ordinal',
    strategy='quantile'
)

X_train_binned = kbins.fit_transform(X_train)
X_test_binned = kbins.transform(X_test)
```

---

## Step 7: Inspect Bin Edges

```python
print(kbins.bin_edges_)
```

Displays the boundaries created for each feature.

---

## Step 8: Evaluate Binned Data

```python
clf_binned = DecisionTreeClassifier()

print(
    np.mean(
        cross_val_score(
            clf_binned,
            X_train_binned,
            y_train,
            cv=10
        )
    )
)
```

Compare this score with the baseline model.

---

## Step 9: Apply Binarization

```python
binarizer = Binarizer(threshold=0.0)

X_train_binary = binarizer.fit_transform(
    X_train[['Fare']]
)
```

All values greater than 0 become 1, while others become 0.

---

# 📊 Key Takeaways

### Discretization

✅ Converts continuous data into intervals

✅ Helps reduce the impact of outliers

✅ Can improve model interpretability

✅ Supports Uniform, Quantile, and K-Means strategies

---

### Binarization

✅ Converts numerical features into binary indicators

✅ Useful when only a threshold matters

✅ Simplifies feature representation

✅ Common in classification problems

---

# 🚀 Conclusion

Discretization and Binarization are powerful feature engineering techniques that simplify data and improve model learning. While Discretization groups values into meaningful intervals, Binarization creates simple binary indicators that can help models focus on important thresholds rather than exact numerical values.

Understanding when and how to apply these transformations is an important step toward building more effective machine learning pipelines.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Decision Tree Classifier

---

## 📚 Concepts Covered

- Feature Engineering
- Discretization (Binning)
- Equal Width Binning
- Equal Frequency Binning
- K-Means Binning
- Decision Tree Binning
- KBinsDiscretizer
- Binarization
- Binarizer
- Data Preprocessing
- Machine Learning Pipelines
Video Link: https://youtu.be/kKWsJGKcMvo
