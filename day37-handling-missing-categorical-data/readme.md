# Day 37 - Missing Data Handling: Categorical Imputation

## 📌 Overview

Categorical features frequently contain missing values in real-world datasets. Since machine learning algorithms cannot directly process missing categorical data, we must replace those missing values using appropriate imputation techniques.

In this lesson, we explore two widely used categorical imputation methods:

- Frequent Category (Mode) Imputation
- Missing Category Imputation

We also discuss when to use each method, their advantages and disadvantages, and how to implement them using Scikit-Learn's `SimpleImputer`.

---

# 1️⃣ What is Categorical Imputation?

Categorical Imputation is the process of replacing missing categorical values with meaningful substitutes.

### Example Dataset

| GarageQual |
|------------|
| TA |
| Gd |
| NaN |
| TA |

The missing value must be replaced before model training.

---

# 2️⃣ Frequent Category Imputation (Mode Imputation)

## Definition

Frequent Category Imputation replaces missing values with the most frequently occurring category (Mode).

---

### Example

Original Data:

| GarageQual |
|------------|
| TA |
| Gd |
| TA |
| NaN |
| TA |

Frequency:

```text
TA → 3
Gd → 1
```

Mode:

```text
TA
```

After Imputation:

| GarageQual |
|------------|
| TA |
| Gd |
| TA |
| TA |
| TA |

---

## Why Use Mode Imputation?

The most common category is assumed to be the safest replacement.

---

## Advantages

✅ Simple and fast

✅ Easy to implement

✅ Preserves dataset size

✅ Works well for low missing percentages

---

## Disadvantages

❌ Increases dominance of the most frequent category

❌ Can distort category distributions

❌ May introduce bias when missing values are high

---

# 3️⃣ The 10% Rule

A common guideline:

```text
Missing Percentage < 10%
```

Mode Imputation is generally safe.

---

### Example

| Total Rows | Missing Rows |
|------------|-------------|
| 10,000 | 500 |

Missing Percentage:

```text
500 / 10000 × 100
= 5%
```

Since:

```text
5% < 10%
```

Mode Imputation is usually acceptable.

---

# 4️⃣ Missing Category Imputation

## Definition

Instead of replacing missing values with the most common category, we create a completely new category.

Common labels:

```text
Missing
Unknown
Not Available
```

---

### Example

Original Data

| FireplaceQu |
|------------|
| Gd |
| TA |
| NaN |
| Ex |

After Imputation

| FireplaceQu |
|------------|
| Gd |
| TA |
| Missing |
| Ex |

---

## Why Use It?

The missing value itself may contain useful information.

Example:

```text
FireplaceQu = Missing
```

might mean:

```text
No Fireplace Exists
```

This information can be predictive.

---

## Advantages

✅ Preserves original category frequencies

✅ Missingness becomes a separate signal

✅ Works well with tree-based models

---

## Disadvantages

❌ Creates an additional category

❌ May increase feature dimensionality after encoding

---

# 5️⃣ SimpleImputer

Scikit-Learn provides the `SimpleImputer` class for handling missing values.

---

## Import

```python
from sklearn.impute import SimpleImputer
```

---

# 6️⃣ Dataset Preparation

## Load Dataset

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv(
    'garage_and_masvnr.csv',
    usecols=[
        'GarageQual',
        'FireplaceQu',
        'SalePrice'
    ]
)
```

---

## Check Missing Ratios

```python
print(
    df.isnull().mean() * 100
)
```

Example Output:

```text
GarageQual      5.48
FireplaceQu    47.26
```

---

## Create Features and Target

```python
X = df.drop(columns=['SalePrice'])

y = df['SalePrice']
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

# 7️⃣ Applying Mode Imputation

## Check Category Frequencies

```python
print(
    X_train['GarageQual']
    .value_counts(normalize=True)
)
```

Example:

```text
TA    0.89
Gd    0.06
Fa    0.03
Ex    0.02
```

The most common category is:

```text
TA
```

---

## Create Imputer

```python
imputer_mode = SimpleImputer(
    strategy='most_frequent'
)
```

---

## Fit and Transform

```python
X_train_mode = imputer_mode.fit_transform(
    X_train[['GarageQual']]
)

X_test_mode = imputer_mode.transform(
    X_test[['GarageQual']]
)
```

---

## Learned Mode

```python
print(
    imputer_mode.statistics_
)
```

Output:

```text
['TA']
```

---

# 8️⃣ Checking Distribution Changes

After Mode Imputation, verify that category frequencies remain stable.

---

## Original Distribution

```python
orig_ratio = (
    X_train['GarageQual']
    .value_counts(normalize=True)
)
```

---

## Imputed Distribution

```python
imputed_ratio = (
    pd.Series(
        X_train_mode.flatten()
    )
    .value_counts(normalize=True)
)
```

---

## Compare Side by Side

```python
print(
    pd.concat(
        [
            orig_ratio,
            imputed_ratio
        ],
        axis=1,
        keys=[
            'Original',
            'Imputed'
        ]
    )
)
```

---

### Interpretation

If the mode category increases dramatically:

❌ Distribution distortion exists

If frequencies remain similar:

✅ Imputation is safe

---

# 9️⃣ Applying Missing Category Imputation

Suppose:

```text
FireplaceQu
```

has:

```text
47% Missing Values
```

Mode Imputation would heavily distort the distribution.

A better solution is to create a new category.

---

## Create Imputer

```python
imputer_missing = SimpleImputer(
    strategy='constant',
    fill_value='Missing'
)
```

---

## Fit and Transform

```python
X_train_missing = (
    imputer_missing.fit_transform(
        X_train[['FireplaceQu']]
    )
)

X_test_missing = (
    imputer_missing.transform(
        X_test[['FireplaceQu']]
    )
)
```

---

## Check Results

```python
print(
    pd.Series(
        X_train_missing.flatten()
    ).value_counts()
)
```

Example Output:

```text
Missing    690
Gd         250
TA         190
Ex          80
```

The new category is now treated as a legitimate feature value.

---

# 🔟 Data Leakage Prevention

Always fit the imputer using only training data.

---

### ❌ Wrong

```python
imputer.fit(df)
```

Uses information from the entire dataset.

---

### ✅ Correct

```python
imputer.fit(X_train)

X_train = imputer.transform(X_train)

X_test = imputer.transform(X_test)
```

Prevents data leakage.

---

# 1️⃣1️⃣ When to Use Each Method?

| Condition | Recommended Strategy |
|------------|---------------------|
| Missing < 10% | Mode Imputation |
| Missing > 20% | Missing Category |
| Strong Mode Dominance | Mode Imputation |
| Missingness Carries Information | Missing Category |
| Tree-Based Models | Missing Category |

---

# 1️⃣2️⃣ Complete Workflow

```python
from sklearn.impute import SimpleImputer

# Mode Imputation
imputer_mode = SimpleImputer(
    strategy='most_frequent'
)

X_train_mode = (
    imputer_mode.fit_transform(
        X_train[['GarageQual']]
    )
)

X_test_mode = (
    imputer_mode.transform(
        X_test[['GarageQual']]
    )
)

# Missing Category Imputation
imputer_missing = SimpleImputer(
    strategy='constant',
    fill_value='Missing'
)

X_train_missing = (
    imputer_missing.fit_transform(
        X_train[['FireplaceQu']]
    )
)

X_test_missing = (
    imputer_missing.transform(
        X_test[['FireplaceQu']]
    )
)
```

---

# 📊 Key Takeaways

### Frequent Category Imputation

✅ Replaces missing values with the Mode

✅ Best when missing percentage is low

❌ Can increase category dominance

---

### Missing Category Imputation

✅ Creates a new category for missing values

✅ Preserves original distributions

✅ Treats missingness as useful information

---

### SimpleImputer

✅ Supports both strategies

```python
strategy='most_frequent'
strategy='constant'
```

---

### Distribution Validation

✅ Compare category frequencies before and after imputation

✅ Ensure class proportions remain reasonable

---

### Data Leakage Prevention

✅ Fit only on training data

✅ Transform train and test separately

---

# 🚀 Conclusion

Categorical Imputation is an essential preprocessing step for handling missing categorical data. Frequent Category Imputation works well when missing values are minimal, while Missing Category Imputation is often preferred when missingness itself contains useful information. Choosing the correct strategy helps preserve feature distributions, minimize bias, and improve model performance.

Understanding when to use each approach is critical for building reliable and robust machine learning pipelines.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- SimpleImputer

---

## 📚 Concepts Covered

- Missing Data Handling
- Categorical Imputation
- Frequent Category Imputation
- Mode Imputation
- Missing Category Imputation
- SimpleImputer
- Data Leakage
- Distribution Preservation
- Feature Engineering
- Data Preprocessing