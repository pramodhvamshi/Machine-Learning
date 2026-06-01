# Day 31 - Feature Engineering: Power Transformers (Box-Cox & Yeo-Johnson)

## Introduction

Feature Engineering plays a crucial role in improving Machine Learning model performance. One common problem in real-world datasets is **skewed data distributions**, which can negatively affect algorithms that assume normally distributed features.

In this project, I explored **Power Transformations** using Scikit-Learn's `PowerTransformer` to reduce skewness, stabilize variance, and make features more suitable for machine learning models.

---

## What are Power Transformations?

Power Transformations are mathematical techniques used to transform numerical features so that they more closely follow a normal distribution.

### Benefits

- Reduce skewness
- Stabilize variance
- Improve model performance
- Make data closer to Gaussian distribution
- Improve linear model assumptions

Scikit-Learn provides the `PowerTransformer` class to perform these transformations automatically.

---

## Types of Power Transformations

### 1. Box-Cox Transformation

The Box-Cox transformation is designed for **strictly positive data**.

### Formula

\[
x^{(\lambda)}=
\begin{cases}
\frac{x^\lambda-1}{\lambda}, & \lambda \neq 0 \\
\ln(x), & \lambda = 0
\end{cases}
\]

### Advantages

- Produces strong normalization
- Works well for highly skewed positive data

### Limitations

- Cannot handle zero values
- Cannot handle negative values

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='box-cox')
X_transformed = pt.fit_transform(X)
```

---

### 2. Yeo-Johnson Transformation

Yeo-Johnson is an extension of Box-Cox that supports:

- Positive values
- Zero values
- Negative values

### Advantages

- More flexible
- No manual shifting required
- Suitable for real-world datasets

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)
```

---

## Dataset Used

The notebook uses selected columns from the Titanic dataset.

### Features

| Feature | Description |
|----------|------------|
| Age | Passenger Age |
| Fare | Ticket Fare |
| Survived | Target Variable |

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## Step 1: Load and Prepare Data

```python
df = pd.read_csv(
    'train.csv',
    usecols=['Age', 'Fare', 'Survived']
)

df['Age'].fillna(df['Age'].mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df[['Age', 'Fare']],
    df['Survived'],
    test_size=0.2,
    random_state=42
)
```

### Tasks Performed

- Loaded Titanic dataset
- Selected required columns
- Handled missing values
- Split data into training and testing sets

---

## Step 2: Apply Box-Cox Transformation

Since Box-Cox only accepts positive values, a small constant is added to avoid errors caused by zero values.

```python
X_train_pos = X_train.copy()
X_test_pos = X_test.copy()

X_train_pos['Fare'] += 0.00001
X_test_pos['Fare'] += 0.00001

pt_box = PowerTransformer(method='box-cox')

X_train_box = pt_box.fit_transform(X_train_pos)
X_test_box = pt_box.transform(X_test_pos)

print(pt_box.lambdas_)
```

### Why Add a Small Constant?

Box-Cox transformation cannot process:

```python
0
```

or

```python
negative numbers
```

Adding a very small positive value prevents mathematical errors.

---

## Step 3: Apply Yeo-Johnson Transformation

Unlike Box-Cox, Yeo-Johnson can handle positive, zero, and negative values directly.

```python
pt_yeo = PowerTransformer(method='yeo-johnson')

X_train_yeo = pt_yeo.fit_transform(X_train)
X_test_yeo = pt_yeo.transform(X_test)
```

---

## Step 4: Train Logistic Regression

```python
clf = LogisticRegression()

clf.fit(X_train_yeo, y_train)

predictions = clf.predict(X_test_yeo)

print(
    "Accuracy:",
    accuracy_score(y_test, predictions)
)
```

### Purpose

To evaluate whether transformed features improve the performance of a machine learning model.

---

## Understanding Lambda (λ)

PowerTransformer automatically finds the optimal value of λ (lambda) for each feature.

```python
print(pt_box.lambdas_)
```

### Lambda Controls

- Strength of transformation
- Shape correction
- Degree of skewness reduction

This automatic optimization removes the need for manually selecting transformations such as:

```python
np.log()
np.sqrt()
np.cbrt()
```

---

## Box-Cox vs Yeo-Johnson

| Feature | Box-Cox | Yeo-Johnson |
|----------|----------|-------------|
| Positive Values | ✅ | ✅ |
| Zero Values | ❌ | ✅ |
| Negative Values | ❌ | ✅ |
| Requires Data Shift | ✅ Sometimes | ❌ |
| Flexible | ❌ | ✅ |

---

## Key Learnings

- Power Transformations reduce skewness in numerical features.
- Box-Cox works only with strictly positive values.
- Yeo-Johnson works with positive, zero, and negative values.
- PowerTransformer automatically learns the optimal lambda values.
- Normalized distributions often improve machine learning performance.
- Always verify transformations using distribution plots or QQ plots.

---

## Conclusion

Power Transformations are powerful feature engineering techniques for handling skewed numerical data. By using Box-Cox and Yeo-Johnson transformations, we can create more normally distributed features, stabilize variance, and potentially improve the performance of machine learning models.

This experiment demonstrated how Scikit-Learn's `PowerTransformer` simplifies the process by automatically selecting the optimal transformation parameters for each feature.


Video Link : https://youtu.be/lV_Z4HbNAx0
