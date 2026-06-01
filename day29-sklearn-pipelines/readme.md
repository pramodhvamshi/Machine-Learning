# Day 29 - Preprocessing Chains: Scikit-Learn Pipelines

## 📌 Overview

In real-world Machine Learning projects, data rarely arrives in a form that can be directly fed into a model.

Typical preprocessing steps include:

* Handling Missing Values
* Encoding Categorical Features
* Scaling Numerical Features
* Feature Engineering
* Model Training

Managing these steps separately can quickly become messy and error-prone.

**Scikit-Learn Pipelines** solve this problem by chaining preprocessing steps and machine learning models into a single workflow that behaves like one unified model.

This notebook demonstrates how to build production-ready machine learning pipelines using Scikit-Learn.

---

# 🚀 Quick Revision Notes

## What is a Pipeline?

A Pipeline is a Scikit-Learn utility that sequentially executes multiple preprocessing transformations followed by a machine learning model.

Instead of:

```text
Raw Data
   ↓
Imputation
   ↓
Encoding
   ↓
Scaling
   ↓
Model
```

being managed separately, everything is wrapped into a single object.

---

## Import Statements

### Pipeline

```python
from sklearn.pipeline import Pipeline
```

Creates pipelines using explicitly named steps.

---

### make_pipeline

```python
from sklearn.pipeline import make_pipeline
```

Creates pipelines automatically without manually naming each step.

---

## Pipeline Structure

A Pipeline consists of:

### Intermediate Steps

Must implement:

```python
.fit()
.transform()
```

Examples:

* SimpleImputer
* OneHotEncoder
* MinMaxScaler
* StandardScaler
* PCA

---

### Final Step

Must implement:

```python
.fit()
.predict()
```

Examples:

* Logistic Regression
* Decision Tree
* Random Forest
* SVM

---

## Why Use Pipelines?

Benefits:

* Cleaner Code
* Reduced Data Leakage
* Reproducible Workflows
* Easier Deployment
* Cross-Validation Friendly
* GridSearchCV Compatible

---

# 🔍 Technical Workflow Analysis

## Step 1: Import Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler
)

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
```

---

## Step 2: Load Dataset

```python
df = pd.read_csv('train.csv')
```

The Titanic dataset is used.

---

## Step 3: Remove Unnecessary Columns

```python
df.drop(
    columns=[
        'PassengerId',
        'Name',
        'Ticket',
        'Cabin'
    ],
    inplace=True
)
```

These features are excluded before modeling.

---

## Step 4: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['Survived']),
    df['Survived'],
    test_size=0.2,
    random_state=42
)
```

Split before preprocessing to avoid data leakage.

---

# Building the Pipeline

## Step 5: Missing Value Imputation

```python
trf1 = ColumnTransformer([
    (
        'impute_age',
        SimpleImputer(),
        [2]
    ),

    (
        'impute_embarked',
        SimpleImputer(
            strategy='most_frequent'
        ),
        [6]
    )
],
remainder='passthrough')
```

### Purpose

Handle missing values:

| Column   | Strategy            |
| -------- | ------------------- |
| Age      | Mean Imputation     |
| Embarked | Most Frequent Value |

---

## Step 6: One-Hot Encoding

```python
trf2 = ColumnTransformer([
    (
        'ohe_sex_embarked',
        OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        ),
        [1, 6]
    )
],
remainder='passthrough')
```

### Purpose

Convert categorical features:

* Sex
* Embarked

into binary variables.

---

## Step 7: Feature Scaling

```python
trf3 = ColumnTransformer([
    (
        'scale_features',
        MinMaxScaler(),
        slice(0, 10)
    )
])
```

### Purpose

Normalize feature values into:

```text
0 → 1
```

range.

---

## Step 8: Machine Learning Model

```python
trf4 = DecisionTreeClassifier()
```

The final estimator in the pipeline.

---

# Creating the Pipeline

```python
pipe = Pipeline([
    ('step1_impute', trf1),

    ('step2_encode', trf2),

    ('step3_scale', trf3),

    ('step4_model', trf4)
])
```

Pipeline Flow:

```text
Raw Data
    ↓
Imputation
    ↓
Encoding
    ↓
Scaling
    ↓
Decision Tree
```

---

# Training the Pipeline

```python
pipe.fit(
    X_train,
    y_train
)
```

What happens internally:

```text
Fit Imputer
      ↓
Transform Data
      ↓
Fit Encoder
      ↓
Transform Data
      ↓
Fit Scaler
      ↓
Transform Data
      ↓
Train Decision Tree
```

All steps execute automatically.

---

# Making Predictions

```python
y_pred = pipe.predict(X_test)
```

Notice:

```python
X_test
```

is passed in its raw form.

The pipeline automatically:

* Imputes Missing Values
* Encodes Categories
* Scales Features
* Generates Predictions

using the exact same transformations learned during training.

---

# Hyperparameter Tuning with GridSearchCV

One major advantage of pipelines is seamless integration with Grid Search.

```python
from sklearn.model_selection import GridSearchCV
```

---

## Define Parameters

```python
params = {
    'step4_model__max_depth':
    [1, 2, 3, 4, 5, None]
}
```

Notice the syntax:

```text
step_name__parameter
```

Double underscore (`__`) is mandatory.

---

## Run Grid Search

```python
grid = GridSearchCV(
    pipe,
    param_grid=params,
    cv=5
)

grid.fit(
    X_train,
    y_train
)
```

Grid Search automatically evaluates:

* Preprocessing
* Model Training
* Validation

as one complete workflow.

---

# Accessing Internal Pipeline Steps

Sometimes debugging is necessary.

Example:

```python
pipe.named_steps
```

Returns all pipeline stages.

---

## Access Specific Step

```python
pipe.named_steps[
    'step2_encode'
]
```

Useful for:

* Inspecting Encoders
* Retrieving Feature Names
* Debugging Transformations

---

# 📊 Pipeline vs Manual Workflow

| Manual Approach               | Pipeline Approach    |
| ----------------------------- | -------------------- |
| Multiple preprocessing blocks | Single workflow      |
| Easy to make mistakes         | Consistent execution |
| Higher leakage risk           | Leakage-resistant    |
| Harder deployment             | Production-ready     |
| Separate model object         | Unified object       |

---

# ⚠️ Common Mistakes

## ❌ Transforming Test Data Separately

Wrong:

```python
encoder.fit(X_test)
```

Correct:

```python
pipe.predict(X_test)
```

The pipeline automatically applies training transformations.

---

## ❌ Forgetting Double Underscores

Wrong:

```python
step4_model_max_depth
```

Correct:

```python
step4_model__max_depth
```

Required for GridSearchCV.

---

## ❌ Mixing Models and Transformers

Only the final step should be an estimator.

All preceding steps must implement:

```python
.transform()
```

---

# 🎯 Key Takeaways

### ✅ Pipelines Chain Preprocessing and Models

Everything becomes one reusable object.

---

### ✅ Prevent Data Leakage

Transformations are learned only from training data.

---

### ✅ Simplify Predictions

Pass raw input directly into:

```python
pipe.predict()
```

without manual preprocessing.

---

### ✅ Enable Powerful Hyperparameter Tuning

GridSearchCV can optimize the entire workflow.

---

### ✅ Improve Deployment Readiness

The same object used during training can be saved and deployed.

---

### ✅ Make Code Cleaner and Easier to Maintain

Complex preprocessing logic becomes modular and reusable.

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

* What Scikit-Learn Pipelines are
* How pipelines execute internally
* How to chain multiple preprocessing steps
* How to integrate ColumnTransformers
* How to train and predict using pipelines
* How to perform hyperparameter tuning with GridSearchCV
* How to build production-ready machine learning workflows

---

## 📂 Dataset Used

```text
train.csv
```

### Features

* Passenger Information
* Demographics
* Ticket Details

### Target

* Survived

Video Link : https://youtu.be/xOccYkgRV4Q
