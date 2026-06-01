# Day 27 - Categorical Encoding: One-Hot Encoding

## 📌 Overview

Many real-world datasets contain **Nominal Categorical Features**—categories that have no natural ordering or ranking.

Examples:

```text
Brand: Apple, Samsung, Xiaomi

City: New York, Texas, California

Fuel Type: Petrol, Diesel, CNG
```

Unlike Ordinal Data, these categories cannot be converted directly into numbers because doing so would create false mathematical relationships.

To solve this problem, we use **One-Hot Encoding (OHE)**.

One-Hot Encoding converts each category into a separate binary column, allowing machine learning algorithms to process categorical information correctly.

---

# 🚀 Quick Revision Notes

## What is Nominal Data?

Nominal data contains categories with no meaningful order.

Examples:

* Gender
* City
* Brand
* Department
* Fuel Type

Example:

```text
Petrol
Diesel
CNG
```

There is no ranking among these categories.

---

## What is One-Hot Encoding?

One-Hot Encoding creates a new binary column for every unique category.

Example:

| Fuel   | Petrol | Diesel | CNG |
| ------ | ------ | ------ | --- |
| Petrol | 1      | 0      | 0   |
| Diesel | 0      | 1      | 0   |
| CNG    | 0      | 0      | 1   |

Each row contains exactly one active category.

---

## OneHotEncoder

Scikit-Learn provides:

```python
from sklearn.preprocessing import OneHotEncoder
```

for performing One-Hot Encoding.

---

## Dummy Variable Trap

When all encoded columns are retained:

| Petrol | Diesel | CNG |
| ------ | ------ | --- |
| 1      | 0      | 0   |
| 0      | 1      | 0   |
| 0      | 0      | 1   |

one column can always be derived from the others.

This creates:

```text
Perfect Multicollinearity
```

which negatively affects many linear models.

---

## drop='first'

```python
OneHotEncoder(drop='first')
```

Drops the first category and removes redundant information.

Example:

| Diesel | CNG |
| ------ | --- |
| 0      | 0   |
| 1      | 0   |
| 0      | 1   |

The missing category is automatically implied.

---

## sparse_output=False

```python
OneHotEncoder(
    sparse_output=False
)
```

Returns a normal NumPy array instead of a sparse matrix.

Useful for:

* Learning
* Debugging
* Converting to DataFrames

---

## handle_unknown='ignore'

```python
OneHotEncoder(
    handle_unknown='ignore'
)
```

Prevents errors when unseen categories appear during prediction.

Example:

Training Data:

```text
Petrol
Diesel
```

Test Data:

```text
CNG
```

Without this setting, the transformation fails.

---

# 🔍 Technical Workflow Analysis

## Step 1: Import Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
```

Import required preprocessing tools.

---

## Step 2: Load Dataset

```python
df = pd.read_csv('cars.csv')
```

Dataset contains categorical features such as:

* Brand
* Fuel Type

and target:

* Selling Price

---

## Step 3: Feature-Target Separation

```python
X = df[
    ['brand', 'fuel']
]

y = df['selling_price']
```

Separate independent variables and target variable.

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

Split data before encoding to prevent data leakage.

---

# One-Hot Encoding

## Step 5: Initialize Encoder

```python
ohe = OneHotEncoder(
    drop='first',
    sparse_output=False,
    dtype=int
)
```

Configuration:

* Remove dummy variable trap
* Return dense array
* Store values as integers

---

## Step 6: Fit and Transform

```python
X_train_ohe = ohe.fit_transform(
    X_train[['brand', 'fuel']]
)

X_test_ohe = ohe.transform(
    X_test[['brand', 'fuel']]
)
```

The encoder learns categories from training data and applies the same transformation to test data.

---

## Step 7: Retrieve Feature Names

```python
encoded_cols = ohe.get_feature_names_out(
    ['brand', 'fuel']
)
```

Output Example:

```text
brand_Honda
brand_Toyota
fuel_Petrol
fuel_Diesel
```

These become the new feature columns.

---

## Step 8: Convert Back to DataFrame

```python
X_train_rebuilt = pd.DataFrame(
    X_train_ohe,
    columns=encoded_cols
)
```

Makes the transformed data easier to inspect and analyze.

---

# 📊 Example of One-Hot Encoding

Original Data:

| Brand   |
| ------- |
| Apple   |
| Samsung |
| Xiaomi  |

Encoded Data:

| Samsung | Xiaomi |
| ------- | ------ |
| 0       | 0      |
| 1       | 0      |
| 0       | 1      |

Apple is dropped because:

```python
drop='first'
```

---

# High Cardinality Problem

## What is High Cardinality?

A feature containing a large number of unique categories.

Example:

```text
1000 Cities
5000 Products
10000 Users
```

Applying One-Hot Encoding directly would generate thousands of columns.

This causes:

* Large Memory Usage
* Slower Training
* Curse of Dimensionality

---

# Handling High Cardinality

## Step 1: Count Category Frequencies

```python
counts = X_train['brand'].value_counts()
```

Determine how frequently each category appears.

---

## Step 2: Define Threshold

```python
threshold = 100
```

Categories appearing fewer than 100 times are considered rare.

---

## Step 3: Group Rare Categories

```python
repl = counts[
    counts <= threshold
].index

X_train_capped = X_train[
    'brand'
].replace(
    repl,
    'uncommon'
)
```

Rare categories are replaced with:

```text
uncommon
```

before applying One-Hot Encoding.

---

# 📊 Standard OHE vs High Cardinality OHE

| Standard OHE                  | High Cardinality OHE   |
| ----------------------------- | ---------------------- |
| Encodes every category        | Groups rare categories |
| More columns                  | Fewer columns          |
| Higher memory usage           | Lower memory usage     |
| Risk of dimensional explosion | More efficient         |

---

# ⚠️ Common Mistakes

## ❌ Applying OHE Before Train-Test Split

Wrong:

```python
ohe.fit(df)
```

Correct:

```python
ohe.fit(X_train)
```

Always learn categories from training data only.

---

## ❌ Ignoring Unseen Categories

Wrong:

```python
OneHotEncoder()
```

Better:

```python
OneHotEncoder(
    handle_unknown='ignore'
)
```

Prevents prediction-time failures.

---

## ❌ Blindly Encoding High Cardinality Features

Example:

```text
5000 Unique Cities
```

This can create thousands of unnecessary columns.

Use:

* Frequency Encoding
* Target Encoding
* Category Grouping

instead.

---

# 🎯 Key Takeaways

### ✅ Use One-Hot Encoding for Nominal Features

Examples:

* Brand
* Fuel Type
* Gender
* City

These categories have no meaningful order.

---

### ✅ Prevent Dummy Variable Trap

For linear models:

```python
drop='first'
```

is recommended.

---

### ✅ Tree-Based Models Are an Exception

Algorithms such as:

* Decision Trees
* Random Forests
* XGBoost
* LightGBM

are not affected by multicollinearity.

For these models:

```python
drop=None
```

is often acceptable.

---

### ✅ Handle Unseen Categories

Always consider:

```python
handle_unknown='ignore'
```

for production-ready pipelines.

---

### ✅ Control High Cardinality

Group rare categories before encoding to avoid excessive feature expansion.

---

# 🛠️ Technologies Used

* Python
* Pandas
* Scikit-Learn
* NumPy
* Jupyter Notebook

---

# 📚 Learning Outcome

By completing this notebook, you will understand:

* What Nominal Data is
* Why One-Hot Encoding is needed
* How OneHotEncoder works
* What the Dummy Variable Trap is
* How to avoid multicollinearity
* How to handle unseen categories
* How to manage high-cardinality categorical features
* Best practices for categorical preprocessing pipelines

---

## 📂 Dataset Used

```text
cars.csv
```

### Features

* Brand
* Fuel Type

### Target

* Selling Price

Video Link : https://youtu.be/U5oCv3JKWKA
