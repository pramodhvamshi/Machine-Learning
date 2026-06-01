# Day 40 - Missing Data Handling: Iterative Imputer (MICE)

## 📌 Overview

Iterative Imputer (MICE) is an advanced multivariate imputation technique that predicts missing values using other features in the dataset. Unlike Mean, Median, or KNN Imputation, it builds predictive models for each column with missing values and iteratively updates them until convergence.

This often produces more accurate imputations because it captures relationships between features.

---

## What is MICE?

**MICE (Multivariate Imputation by Chained Equations)** fills missing values by treating each feature with missing data as a target variable and predicting it using all other available features.

### Process

1. Fill missing values with an initial estimate (Mean).
2. Select one feature with missing values.
3. Train a model using other features.
4. Predict missing values.
5. Repeat for all columns.
6. Perform multiple iterations until values stabilize.

---

## IterativeImputer

Scikit-Learn provides:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

> The experimental import is mandatory.

---

## Basic Implementation

```python
from sklearn.linear_model import LinearRegression

ii = IterativeImputer(
    estimator=LinearRegression(),
    max_iter=10,
    random_state=2
)

X_train_imputed = ii.fit_transform(X_train)
X_test_imputed = ii.transform(X_test)
```

---

## Important Parameters

### estimator

Model used to predict missing values.

```python
estimator=LinearRegression()
```

Other options:

```python
BayesianRidge()
RandomForestRegressor()
KNeighborsRegressor()
```

---

### max_iter

Number of iterative cycles.

```python
max_iter=10
```

Higher values increase accuracy but require more computation.

---

## Example Workflow

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

ii = IterativeImputer(
    estimator=LinearRegression(),
    max_iter=10,
    random_state=2
)

X_train_imputed = ii.fit_transform(X_train)
X_test_imputed = ii.transform(X_test)
```

---

## Advantages

✅ Uses relationships between features

✅ More accurate than Mean/Median Imputation

✅ Handles multivariate data effectively

✅ Faster inference than KNN Imputer

✅ Suitable for production systems

---

## Disadvantages

❌ Computationally expensive during training

❌ More complex than simple imputation

❌ Experimental feature in Scikit-Learn

❌ May require tuning of estimator and iterations

---

## MICE vs KNN Imputer

| Feature | KNN Imputer | MICE |
|----------|------------|------|
| Approach | Neighbor Based | Model Based |
| Uses Feature Relationships | Yes | Yes |
| Training Cost | Low | High |
| Prediction Speed | Slow | Fast |
| Scalability | Moderate | Better |
| Production Use | Limited | Better |

---

## When to Use?

Use MICE when:

- Features are highly correlated
- Dataset is medium or large
- Better accuracy is required
- Production deployment is planned

Avoid when:

- Dataset is very small
- Fast preprocessing is more important than accuracy

---

## Model Training Example

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_imputed, y_train)

pred = clf.predict(X_test_imputed)
```

---

## Complete Workflow

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

ii = IterativeImputer(
    estimator=LinearRegression(),
    max_iter=10,
    random_state=2
)

X_train_imputed = ii.fit_transform(X_train)
X_test_imputed = ii.transform(X_test)

clf.fit(X_train_imputed, y_train)
```

---

## Key Takeaways

- MICE stands for Multivariate Imputation by Chained Equations.
- Uses predictive models instead of averages.
- Implemented using `IterativeImputer`.
- Requires enabling the experimental module.
- `estimator` controls the prediction model.
- `max_iter` controls the number of update cycles.
- More scalable than KNN Imputer.
- Often produces higher-quality imputations.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- IterativeImputer
- Linear Regression
- Logistic Regression

---

## Concepts Covered

- Missing Data Handling
- MICE
- Iterative Imputer
- Multivariate Imputation
- Chained Equations
- Feature Relationships
- Linear Regression
- Model-Based Imputation
- Data Preprocessing
- Machine Learning Pipelines