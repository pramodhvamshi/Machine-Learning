# Day 28 - Pipeline Optimization: ColumnTransformer

## 📌 Overview

Real-world datasets often contain a mix of:

* Numerical Features
* Ordinal Categorical Features
* Nominal Categorical Features

Since each feature type requires a different preprocessing technique, managing transformations manually quickly becomes complicated and error-prone.

**ColumnTransformer** solves this problem by allowing different preprocessing operations to be applied to different columns within a single unified workflow.

This notebook demonstrates how to use Scikit-Learn's `ColumnTransformer` to build cleaner, scalable, and production-ready preprocessing pipelines.

---

# 🚀 Quick Revision Notes

## What is ColumnTransformer?

`ColumnTransformer` is a Scikit-Learn utility that applies different transformations to different columns simultaneously.

Instead of manually splitting and combining datasets, all preprocessing steps can be centralized into one object.

---

## Import Statement

```python
from sklearn.compose import ColumnTransformer
```

Provides access to the ColumnTransformer class.

---

## Core Syntax

```python
ColumnTransformer(
    transformers=[
        (
            'name',
            transformer_object,
            [columns]
        )
    ]
)
```

Each transformer tuple contains:

| Component   | Description                       |
| ----------- | --------------------------------- |
| Name        | Identifier for the transformation |
| Transformer | Encoder, Scaler, etc.             |
| Columns     | Columns to transform              |

---

## transformers Parameter

```python
transformers=[
    (
        'tnf1',
        transformer,
        ['column']
    )
]
```

Defines which transformation is applied to which columns.

---

## remainder Parameter

Controls columns that are not explicitly listed.

### Keep Remaining Columns

```python
remainder='passthrough'
```

Retains untouched columns.

---

### Drop Remaining Columns

```python
remainder='drop'
```

Removes unselected columns.

This is the default behavior.

---

## Why Use ColumnTransformer?

Benefits include:

* Cleaner code
* Less manual preprocessing
* Reduced data leakage risk
* Easier integration with Pipelines
* Compatible with GridSearchCV
* Production-ready workflows

---

# 🔍 The Problem: Manual Preprocessing

Before ColumnTransformer, preprocessing looked like this:

```python
X_train_age = scaler.fit_transform(
    X_train[['age']]
)

X_train_gender = ohe.fit_transform(
    X_train[['gender']]
)

X_train_rem = X_train[
    ['fever', 'has_covid']
].values

X_train_transformed = np.hstack(
    (
        X_train_age,
        X_train_gender,
        X_train_rem
    )
)
```

Problems:

* Too much code
* Easy to make mistakes
* Difficult to maintain
* Hard to integrate with pipelines
* Increased risk of data leakage

---

# The Solution: ColumnTransformer

ColumnTransformer centralizes all preprocessing steps into one object.

---

## Step 1: Import Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler
)

from sklearn.compose import ColumnTransformer
```

---

## Step 2: Load Dataset

```python
df = pd.read_csv('covid_toy.csv')
```

Dataset contains:

* Age
* Gender
* City
* Cough Severity
* Has Covid

---

## Step 3: Feature-Target Separation

```python
X = df.drop(
    'has_covid',
    axis=1
)

y = df['has_covid']
```

Separate features and target.

---

## Step 4: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)
```

Always split before fitting transformers.

---

## Step 5: Create ColumnTransformer

```python
transformer = ColumnTransformer(
    transformers=[
        (
            'tnf1',
            OrdinalEncoder(
                categories=[
                    ['Mild', 'Strong']
                ]
            ),
            ['cough']
        ),

        (
            'tnf2',
            OneHotEncoder(
                sparse_output=False,
                drop='first'
            ),
            ['gender', 'city']
        ),

        (
            'tnf3',
            MinMaxScaler(),
            ['age']
        )
    ],

    remainder='passthrough'
)
```

---

# Understanding Each Transformation

## Transformation 1: Ordinal Encoding

```python
OrdinalEncoder(
    categories=[
        ['Mild', 'Strong']
    ]
)
```

Applied to:

```python
['cough']
```

Encoding:

| Cough Severity | Encoded |
| -------------- | ------- |
| Mild           | 0       |
| Strong         | 1       |

---

## Transformation 2: One-Hot Encoding

```python
OneHotEncoder(
    drop='first'
)
```

Applied to:

```python
['gender', 'city']
```

Converts nominal categories into binary columns while avoiding the dummy variable trap.

---

## Transformation 3: Min-Max Scaling

```python
MinMaxScaler()
```

Applied to:

```python
['age']
```

Scales Age values into:

```text
0 → 1
```

range.

---

## Step 6: Fit and Transform

### Training Data

```python
X_train_transformed = transformer.fit_transform(
    X_train
)
```

Learns preprocessing parameters and transforms data.

---

### Test Data

```python
X_test_transformed = transformer.transform(
    X_test
)
```

Applies the same learned transformations.

---

# 📊 Output Structure

After transformation:

```text
Age
Gender Encodings
City Encodings
Cough Encoding
Remaining Columns
```

are merged into a single NumPy array.

---

# Accessing Feature Names

Because ColumnTransformer returns a NumPy array, original DataFrame column names are lost.

Use:

```python
transformer.get_feature_names_out()
```

to retrieve transformed feature names.

Example Output:

```text
tnf1__cough
tnf2__gender_Male
tnf2__city_Delhi
tnf2__city_Mumbai
tnf3__age
```

This is especially useful for debugging and feature inspection.

---

# 📊 ColumnTransformer vs Manual Preprocessing

| Manual Approach               | ColumnTransformer |
| ----------------------------- | ----------------- |
| Multiple preprocessing blocks | Single object     |
| Manual concatenation          | Automatic merging |
| More code                     | Cleaner code      |
| Error-prone                   | Reliable          |
| Difficult to scale            | Production-ready  |

---

# ⚠️ Common Mistakes

## ❌ Forgetting remainder='passthrough'

Wrong:

```python
ColumnTransformer(
    transformers=[...]
)
```

All unspecified columns are dropped.

---

Correct:

```python
ColumnTransformer(
    transformers=[...],
    remainder='passthrough'
)
```

when you want to keep remaining columns.

---

## ❌ Using Column Indexes in Production

Possible:

```python
[0, 1, 2]
```

Better:

```python
['age', 'gender']
```

Column names are safer and easier to maintain.

---

## ❌ Fitting on Test Data

Wrong:

```python
transformer.fit(X_test)
```

Correct:

```python
transformer.fit(X_train)
```

Always learn preprocessing parameters from training data only.

---

# 🎯 Key Takeaways

### ✅ ColumnTransformer Centralizes Preprocessing

Multiple transformations can be applied in a single workflow.

---

### ✅ Eliminates Manual Data Merging

No need for:

```python
np.hstack()
```

or manual concatenation.

---

### ✅ Works Seamlessly with Pipelines

ColumnTransformer is commonly used inside Scikit-Learn Pipelines.

---

### ✅ Helps Prevent Data Leakage

Ensures consistent fit-transform behavior across train and test datasets.

---

### ✅ Supports Mixed Data Types

Can simultaneously handle:

* Numerical Features
* Ordinal Features
* Nominal Features

---

### ✅ Ideal for Production Systems

ColumnTransformer makes preprocessing scalable, maintainable, and deployment-friendly.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Jupyter Notebook

---

# 📚 Learning Outcome

By completing this notebook, you will understand:

* What ColumnTransformer is
* Why it is needed
* How to apply multiple preprocessing techniques simultaneously
* How to combine Ordinal Encoding, One-Hot Encoding, and Scaling
* How to prevent preprocessing errors
* How to retrieve transformed feature names
* How to build production-ready preprocessing workflows

---

## 📂 Dataset Used

```text
covid_toy.csv
```

### Features

* Age
* Gender
* City
* Cough Severity

### Target

* Has Covid

Video Link : https://youtu.be/5TVj6iEBR4I
