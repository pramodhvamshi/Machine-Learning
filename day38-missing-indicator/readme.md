# Day 38 - Missing Data Handling: Missing Indicator

## 📌 Overview

Missing Indicator is a feature engineering technique used to capture information about missing values. Instead of only filling missing values, we create an additional binary column that records whether a value was originally missing.

This helps machine learning models learn patterns hidden in missing data.

---

## What is Missing Indicator?

A Missing Indicator creates a new column:

- `1` → Value was missing
- `0` → Value was present

### Example

| Age | Age_Missing |
|------|-------------|
| 25 | 0 |
| NaN | 1 |
| 40 | 0 |
| NaN | 1 |

After imputation:

| Age | Age_Missing |
|------|-------------|
| 25 | 0 |
| 30 | 1 |
| 40 | 0 |
| 30 | 1 |

---

## Why Use Missing Indicators?

Sometimes missing values themselves contain useful information.

Example:

- Missing salary may indicate unemployment.
- Missing medical records may indicate healthier patients.
- Missing age may follow specific patterns.

The indicator allows the model to learn these relationships.

---

## MissingIndicator Class

Scikit-Learn provides:

```python
from sklearn.impute import MissingIndicator
```

### Create Indicator

```python
mi = MissingIndicator()
```

### Fit

```python
mi.fit(X_train)
```

### Transform

```python
X_train_mask = mi.transform(X_train)
X_test_mask = mi.transform(X_test)
```

### Missing Columns

```python
print(mi.features_)
```

---

## Standard Imputation

Before using indicators, missing values must still be filled.

```python
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='median')

X_train = si.fit_transform(X_train)
X_test = si.transform(X_test)
```

---

## Using add_indicator=True

Scikit-Learn provides a shortcut:

```python
si = SimpleImputer(
    strategy='median',
    add_indicator=True
)
```

This automatically:

1. Imputes missing values
2. Creates indicator columns

in a single step.

---

## Example

```python
from sklearn.impute import SimpleImputer

si = SimpleImputer(
    strategy='median',
    add_indicator=True
)

X_train_final = si.fit_transform(X_train)
X_test_final = si.transform(X_test)
```

---

## Advantages

✅ Preserves information hidden in missing values

✅ Improves model performance in many cases

✅ Useful for both linear and tree-based models

✅ Easy to implement

---

## Disadvantages

❌ Adds extra features

❌ Not useful when data is MCAR

❌ May introduce noise if missingness has no meaning

---

## When to Use?

Use Missing Indicators when:

- Missing values are informative
- Data is MNAR (Missing Not At Random)
- Missingness carries predictive power

Avoid when:

- Missing values are completely random
- Dataset contains very few missing values

---

## Complete Workflow

```python
from sklearn.impute import SimpleImputer

si = SimpleImputer(
    strategy='median',
    add_indicator=True
)

X_train_final = si.fit_transform(X_train)
X_test_final = si.transform(X_test)
```

---

## Key Takeaways

- Missing Indicator creates binary tracking columns.
- Helps models learn patterns from missing values.
- Usually combined with Mean or Median Imputation.
- `add_indicator=True` is the easiest implementation.
- Most useful when missingness itself contains information.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn

---

## Concepts Covered

- Missing Data Handling
- Missing Indicator
- SimpleImputer
- MissingIndicator
- Median Imputation
- Feature Engineering
- Data Preprocessing