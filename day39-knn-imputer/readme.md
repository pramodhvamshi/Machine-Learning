# Day 39 - Missing Data Handling: KNN Imputer

## 📌 Overview

KNN Imputer is a multivariate imputation technique that fills missing values using the values of the most similar records in the dataset. Unlike Mean or Median Imputation, it considers relationships between multiple features before predicting missing values.

This often produces more accurate imputations because it uses neighboring observations instead of a single global statistic.

---

## What is KNN Imputer?

KNN (K-Nearest Neighbors) Imputer replaces missing values by finding the **K most similar rows** and averaging their values.

### Example

| Age | Fare |
|------|------|
| 25 | 100 |
| 27 | 110 |
| NaN | 105 |

If K = 2:

```text
Age = (25 + 27) / 2
     = 26
```

Missing Age becomes:

```text
26
```

---

## Why Use KNN Imputer?

Unlike Mean/Median Imputation:

- Uses multiple features
- Preserves feature relationships
- Produces context-aware imputations
- Often improves model performance

---

## KNNImputer Class

```python
from sklearn.impute import KNNImputer
```

### Create Imputer

```python
knn = KNNImputer(
    n_neighbors=5
)
```

---

## Important Parameters

### n_neighbors

Number of nearest neighbors.

```python
KNNImputer(
    n_neighbors=3
)
```

Common values:

```text
1, 3, 5, 7, 10
```

---

### weights

#### Uniform

All neighbors contribute equally.

```python
weights='uniform'
```

#### Distance

Closer neighbors receive higher importance.

```python
weights='distance'
```

---

## Basic Implementation

```python
knn = KNNImputer(
    n_neighbors=3,
    weights='uniform'
)

X_train_knn = knn.fit_transform(X_train)
X_test_knn = knn.transform(X_test)
```

---

## Model Training

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_knn, y_train)

pred = clf.predict(X_test_knn)
```

---

## Hyperparameter Tuning

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('imputer', KNNImputer()),
    ('model', LogisticRegression())
])

param_grid = {
    'imputer__n_neighbors':[1,3,5,7,10],
    'imputer__weights':['uniform','distance']
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

print(grid.best_params_)
```

---

## Advantages

✅ Uses relationships between features

✅ Better than Mean/Median in many cases

✅ Preserves data structure

✅ No assumptions about distribution

---

## Disadvantages

❌ Computationally expensive

❌ Slow on large datasets

❌ Sensitive to feature scaling

❌ Requires memory to store training data

---

## Feature Scaling Requirement

Since KNN uses distance calculations:

```python
Age  : 0 - 80
Fare : 0 - 500
```

Fare dominates the distance metric.

Always scale features before KNN Imputation:

```python
from sklearn.preprocessing import StandardScaler
```

or

```python
from sklearn.preprocessing import MinMaxScaler
```

---

## When to Use?

Use KNN Imputer when:

- Dataset size is moderate
- Feature relationships are important
- Missing percentage is not extremely high

Avoid when:

- Dataset is very large
- Real-time predictions are required
- Computational resources are limited

---

## Complete Workflow

```python
from sklearn.impute import KNNImputer

knn = KNNImputer(
    n_neighbors=3,
    weights='uniform'
)

X_train_knn = knn.fit_transform(X_train)
X_test_knn = knn.transform(X_test)

clf.fit(X_train_knn, y_train)
```

---

## Key Takeaways

- KNN Imputer is a multivariate imputation technique.
- Uses nearest neighbors to estimate missing values.
- `n_neighbors` controls the number of neighbors.
- `weights='distance'` gives more importance to closer points.
- Feature scaling is extremely important.
- Usually performs better than simple Mean/Median Imputation.
- Can be slow on large datasets.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- KNNImputer
- Logistic Regression

---

## Concepts Covered

- Missing Data Handling
- KNN Imputer
- Multivariate Imputation
- Distance-Based Learning
- Feature Scaling
- Hyperparameter Tuning
- GridSearchCV
- Machine Learning Preprocessing