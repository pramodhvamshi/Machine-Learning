# Day 36 - Missing Data Handling: Numerical Imputation (Mean/Median & Arbitrary)

## 📌 Overview

Missing numerical values are a common problem in machine learning datasets. Since most algorithms cannot handle missing values directly, these gaps must be filled before model training.

One of the most popular approaches is **Numerical Imputation**, where missing values are replaced using statistical estimates or predefined constants.

In this lesson, we explore:

- Mean Imputation
- Median Imputation
- Arbitrary Value Imputation
- SimpleImputer
- Data Leakage Prevention
- Variance and Correlation Distortion

---

# 1️⃣ What is Numerical Imputation?

Numerical Imputation is the process of replacing missing numerical values with estimated values.

### Example

#### Original Data

| Age |
|------|
| 25 |
| 30 |
| NaN |
| 35 |

---

#### Mean Imputation

Mean:

```text
(25 + 30 + 35) / 3 = 30
```

Result:

| Age |
|------|
| 25 |
| 30 |
| 30 |
| 35 |

---

#### Median Imputation

Median:

```text
30
```

Result:

| Age |
|------|
| 25 |
| 30 |
| 30 |
| 35 |

---

# 2️⃣ Univariate Imputation

## Definition

Univariate Imputation fills missing values using information from only the target column.

It does not consider relationships with other features.

---

### Example

```python
Age
```

Missing Age values are filled using:

- Mean of Age
- Median of Age

without considering:

```python
Fare
Pclass
Sex
Embarked
```

---

## Advantages

✅ Simple

✅ Fast

✅ Easy to implement

---

## Limitations

❌ Ignores relationships between features

❌ Can distort correlations

❌ May reduce variance

---

# 3️⃣ Mean Imputation

## Definition

Replace missing values using the average of all non-missing values.

---

## Formula

\[
Mean = \frac{\sum X}{N}
\]

---

### Example

Values:

```text
20, 25, 30, NaN, 35
```

Mean:

```text
(20+25+30+35)/4 = 27.5
```

Imputed Value:

```text
27.5
```

---

## When to Use Mean Imputation?

Use when data follows a roughly normal distribution.

### Example Distribution

```text
      /\
     /  \
    /    \
```

Symmetric distribution.

---

## Advantages

✅ Easy to calculate

✅ Preserves overall mean

---

## Disadvantages

❌ Sensitive to outliers

❌ Can reduce variance

---

# 4️⃣ Median Imputation

## Definition

Replace missing values using the middle value of the distribution.

---

### Example

Values:

```text
10, 15, 20, 1000
```

Median:

```text
17.5
```

Mean:

```text
261.25
```

The median is much more representative.

---

## When to Use Median Imputation?

Use when:

- Data is skewed
- Outliers exist

---

### Example Distribution

```text
|
|\
| \
|  \
|   \____
```

Right-skewed distribution.

---

## Advantages

✅ Robust against outliers

✅ Preserves distribution better

---

## Disadvantages

❌ Still causes variance reduction

❌ May alter feature relationships

---

# 5️⃣ Mean vs Median Imputation

| Feature Type | Recommended Strategy |
|-------------|---------------------|
| Normal Distribution | Mean |
| Skewed Distribution | Median |
| Outliers Present | Median |
| Symmetric Data | Mean |

---

# 6️⃣ SimpleImputer

Scikit-Learn provides the `SimpleImputer` class for handling missing values.

---

## Import

```python
from sklearn.impute import SimpleImputer
```

---

## Mean Imputation

```python
imputer = SimpleImputer(
    strategy='mean'
)
```

---

## Median Imputation

```python
imputer = SimpleImputer(
    strategy='median'
)
```

---

## Most Frequent

```python
imputer = SimpleImputer(
    strategy='most_frequent'
)
```

---

## Constant Value

```python
imputer = SimpleImputer(
    strategy='constant'
)
```

---

# 7️⃣ Dataset Preparation

## Load Dataset

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic_toy.csv')
```

---

## Check Missing Values

```python
print(df.isnull().mean() * 100)
```

---

## Create Features and Target

```python
X = df.drop(columns=['Survived'])
y = df['Survived']
```

---

## Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)
```

---

# 8️⃣ Applying Mean & Median Together

Suppose:

```text
Age  → Normally Distributed
Fare → Skewed Distribution
```

We use:

- Mean for Age
- Median for Fare

---

## ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
```

---

## Configure Imputation

```python
imputer = ColumnTransformer(
    transformers=[
        (
            'impute_age',
            SimpleImputer(strategy='mean'),
            ['Age']
        ),
        (
            'impute_fare',
            SimpleImputer(strategy='median'),
            ['Fare']
        )
    ],
    remainder='passthrough'
)
```

---

## Fit and Transform

```python
X_train_imputed = imputer.fit_transform(
    X_train
)

X_test_imputed = imputer.transform(
    X_test
)
```

---

## View Calculated Statistics

```python
print(
    imputer.named_transformers_
    ['impute_age']
    .statistics_
)

print(
    imputer.named_transformers_
    ['impute_fare']
    .statistics_
)
```

---

# 9️⃣ Data Leakage Prevention

A very important rule:

### ❌ Wrong Approach

```python
imputer.fit(df)
```

Uses information from the entire dataset.

---

### ✅ Correct Approach

```python
imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
```

The imputer learns values only from training data.

This prevents data leakage.

---

# 🔟 Variance Shrinkage Problem

Mean and Median Imputation repeatedly insert the same value.

This reduces data variability.

---

## Variance Comparison

```python
X_train_df = pd.DataFrame(
    X_train_imputed,
    columns=X_train.columns
)

print(
    X_train['Age'].var()
)

print(
    X_train_df['Age'].var()
)
```

---

### Why It Happens

Original:

```text
20
25
30
35
40
```

Missing values replaced with:

```text
30
30
30
30
```

Many identical values reduce variance.

---

# 1️⃣1️⃣ Correlation Distortion

Since imputation ignores other features, relationships may change.

---

## Compare Correlations

```python
print(
    X_train.corr()
)

print(
    X_train_df.corr()
)
```

---

### Risk

Before Imputation:

```text
Age ↔ Fare = 0.45
```

After Imputation:

```text
Age ↔ Fare = 0.31
```

Feature relationships may weaken.

---

# 1️⃣2️⃣ Arbitrary Value Imputation

Instead of using statistical estimates, we intentionally use an unrealistic value.

---

## Examples

```python
-1
999
9999
```

These values lie outside the normal range.

---

## Why Use It?

The model can explicitly recognize:

```text
This value was originally missing
```

---

## Implementation

```python
arbitrary_imputer = SimpleImputer(
    strategy='constant',
    fill_value=-1
)
```

---

## Apply Imputation

```python
X_train_arbitrary = arbitrary_imputer.fit_transform(
    X_train[['Age']]
)
```

---

### Example

Original:

| Age |
|------|
| 25 |
| NaN |
| 35 |

After Imputation:

| Age |
|------|
| 25 |
| -1 |
| 35 |

---

# 1️⃣3️⃣ When to Use Arbitrary Value Imputation?

### Best For

✅ Decision Trees

✅ Random Forest

✅ XGBoost

✅ LightGBM

✅ CatBoost

---

### Avoid For

❌ Linear Regression

❌ Logistic Regression

❌ KNN

❌ SVM

❌ Distance-Based Models

---

### Reason

Large artificial values distort:

- Distances
- Slopes
- Coefficients

---

# 🧪 Complete Workflow

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

imputer = ColumnTransformer(
    transformers=[
        (
            'impute_age',
            SimpleImputer(strategy='mean'),
            ['Age']
        ),
        (
            'impute_fare',
            SimpleImputer(strategy='median'),
            ['Fare']
        )
    ],
    remainder='passthrough'
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_test_imputed = imputer.transform(
    X_test
)

print(
    imputer.named_transformers_
    ['impute_age']
    .statistics_
)

print(
    imputer.named_transformers_
    ['impute_fare']
    .statistics_
)
```

---

# 📊 Key Takeaways

### Mean Imputation

✅ Best for normally distributed data

❌ Sensitive to outliers

---

### Median Imputation

✅ Best for skewed data

✅ Robust to outliers

---

### SimpleImputer

✅ Automates missing value handling

✅ Supports multiple strategies

---

### Data Leakage Prevention

✅ Fit only on training data

✅ Transform train and test separately

---

### Variance Shrinkage

✅ Always compare variance before and after imputation

---

### Correlation Distortion

✅ Check covariance and correlation matrices

---

### Arbitrary Value Imputation

✅ Excellent for tree-based models

❌ Poor choice for linear and distance-based algorithms

---

# 🚀 Conclusion

Numerical Imputation is a critical preprocessing technique for handling missing numerical data. Mean and Median Imputation provide simple statistical replacements, while Arbitrary Value Imputation helps models identify missingness as a separate signal. Although these methods are easy to implement, they can introduce variance shrinkage and correlation distortion, making proper validation essential.

Choosing the right imputation strategy based on feature distribution and model type can significantly improve machine learning performance and data quality.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- SimpleImputer
- ColumnTransformer

---

## 📚 Concepts Covered

- Missing Data Handling
- Numerical Imputation
- Mean Imputation
- Median Imputation
- Arbitrary Value Imputation
- Univariate Imputation
- SimpleImputer
- ColumnTransformer
- Data Leakage
- Variance Shrinkage
- Correlation Distortion
- Data Preprocessing
- Feature Engineering
Video Link : https://youtu.be/mCL2xLBDw8M
