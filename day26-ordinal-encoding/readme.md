# Day 26 - Categorical Encoding: Ordinal Encoding & Label Encoding

## 📌 Overview

Machine Learning algorithms work with numerical data, but real-world datasets often contain categorical values such as:

```text
Poor, Average, Good
Male, Female
Yes, No
```

These categorical values must be converted into numbers before training a model.

This notebook focuses on two important encoding techniques:

1. **Ordinal Encoding** – Used for ordered categorical features.
2. **Label Encoding** – Primarily used for target labels.

Understanding when and where to use each technique is essential for building accurate machine learning pipelines.

---

# 🚀 Quick Revision Notes

## What is Categorical Encoding?

Categorical Encoding is the process of converting categorical (text-based) data into numerical values that machine learning algorithms can understand.

Example:

| Category | Encoded Value |
| -------- | ------------- |
| Poor     | 0             |
| Average  | 1             |
| Good     | 2             |

---

## Ordinal Encoding

Ordinal Encoding is used when categories have a meaningful order.

Examples:

### Education Level

```text
School < UG < PG
```

### Customer Review

```text
Poor < Average < Good
```

Since these categories have a natural ranking, assigning ordered numbers makes sense.

---

## Label Encoding

Label Encoding converts target labels into integers.

Example:

| Purchased | Encoded |
| --------- | ------- |
| No        | 0       |
| Yes       | 1       |

Label Encoding is generally used for the target variable (**y**).

---

## OrdinalEncoder

Scikit-Learn provides:

```python
from sklearn.preprocessing import OrdinalEncoder
```

to encode ordered categorical features.

---

## LabelEncoder

Scikit-Learn provides:

```python
from sklearn.preprocessing import LabelEncoder
```

to encode target labels.

---

## Categories Parameter

The `categories` parameter allows us to explicitly define category rankings.

Example:

```python
categories=[
    ['Poor', 'Average', 'Good'],
    ['School', 'UG', 'PG']
]
```

This ensures the encoding follows the correct logical order.

---

# 🔍 Technical Workflow Analysis

## Step 1: Import Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    OrdinalEncoder,
    LabelEncoder
)
```

Import required libraries for preprocessing.

---

## Step 2: Load Dataset

```python
df = pd.read_csv('customer.csv')
```

Dataset contains:

### Features

* Review
* Education
* Gender

### Target

* Purchased

---

## Step 3: Feature and Target Separation

```python
X = df[
    ['review',
     'education',
     'gender']
]

y = df['purchased']
```

Separate independent features and target variable.

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

Splitting first prevents data leakage.

---

# Ordinal Encoding

## Step 5: Define Category Hierarchy

```python
oe = OrdinalEncoder(
    categories=[
        ['Poor', 'Average', 'Good'],
        ['School', 'UG', 'PG']
    ]
)
```

The order must represent the true ranking of categories.

---

## Step 6: Fit and Transform

```python
X_train_ordered = oe.fit_transform(
    X_train[['review', 'education']]
)

X_test_ordered = oe.transform(
    X_test[['review', 'education']]
)
```

The encoder learns the category mapping from training data and applies it to both datasets.

---

## Step 7: View Encoded Categories

```python
print(oe.categories_)
```

Output:

```text
['Poor', 'Average', 'Good']
['School', 'UG', 'PG']
```

Encoded Mapping:

| Review  | Encoded |
| ------- | ------- |
| Poor    | 0       |
| Average | 1       |
| Good    | 2       |

| Education | Encoded |
| --------- | ------- |
| School    | 0       |
| UG        | 1       |
| PG        | 2       |

---

# Label Encoding

## Step 8: Initialize LabelEncoder

```python
le = LabelEncoder()
```

Creates the label encoder object.

---

## Step 9: Encode Target Variable

```python
y_train_encoded = le.fit_transform(y_train)

y_test_encoded = le.transform(y_test)
```

Converts categorical target values into integers.

---

## Step 10: View Label Mapping

```python
print(le.classes_)
```

Example Output:

```text
['No', 'Yes']
```

Encoded Mapping:

| Purchased | Encoded |
| --------- | ------- |
| No        | 0       |
| Yes       | 1       |

---

# 📊 Ordinal Encoding vs Label Encoding

| Ordinal Encoding            | Label Encoding               |
| --------------------------- | ---------------------------- |
| Used for Features (X)       | Used for Target (y)          |
| Requires Ordered Categories | No Manual Order Required     |
| User Defines Ranking        | Automatically Assigns Labels |
| Multiple Columns Supported  | Single Target Column         |

---

# ⚠️ Common Mistakes

## ❌ Using Ordinal Encoding on Nominal Data

Wrong Example:

```text
Male
Female
```

There is no ranking between Male and Female.

Encoding:

```text
Male = 0
Female = 1
```

creates a false mathematical relationship.

For nominal variables, use:

```python
OneHotEncoder()
```

instead.

---

## ❌ Forgetting to Specify Category Order

Wrong:

```python
OrdinalEncoder()
```

Scikit-Learn may assign categories alphabetically.

Example:

```text
Average = 0
Good = 1
Poor = 2
```

This destroys the intended ranking.

Always specify:

```python
categories=[
    ['Poor', 'Average', 'Good']
]
```

---

## ❌ Using LabelEncoder on Feature Columns

Avoid:

```python
le.fit_transform(X['gender'])
```

LabelEncoder is intended for:

```python
y
```

not for independent features.

---

# 🎯 Key Takeaways

### ✅ Use Ordinal Encoding for Ordered Categories

Examples:

```text
Poor < Average < Good

School < UG < PG
```

---

### ✅ Explicitly Define Category Rankings

Always pass:

```python
categories=[...]
```

to avoid incorrect alphabetical ordering.

---

### ✅ Use LabelEncoder for Target Labels

Correct:

```python
le.fit_transform(y)
```

---

### ✅ Use One-Hot Encoding for Nominal Features

Examples:

* Gender
* City
* Country
* Department

These categories have no inherent order.

---

### ✅ Split Before Encoding

Recommended Workflow:

```text
Train-Test Split
        ↓
Fit Encoder on Training Data
        ↓
Transform Training Data
        ↓
Transform Test Data
```

This prevents data leakage.

---

# 🛠️ Technologies Used

* Python
* Pandas
* Scikit-Learn
* Jupyter Notebook

---

# 📚 Learning Outcome

By completing this notebook, you will understand:

* What categorical encoding is
* Why encoding is required
* How Ordinal Encoding works
* How Label Encoding works
* Differences between Ordinal and Label Encoding
* Common encoding mistakes
* Best practices for preprocessing categorical data

---

## 📂 Dataset Used

```text
customer.csv
```

### Features

* Review
* Education
* Gender

### Target

* Purchased

Video Link: https://youtu.be/w2GglmYHfmM
