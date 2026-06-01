# Day 35 - Missing Data Handling: Complete Case Analysis (CCA)

## 📌 Overview

Missing data is one of the most common challenges in real-world datasets. Before training machine learning models, missing values must be handled appropriately to prevent biased predictions and poor performance.

One of the simplest approaches is **Complete Case Analysis (CCA)**, also known as **Listwise Deletion**, where rows containing missing values are completely removed from the dataset.

CCA is easy to implement but should only be used when the amount of missing data is small and the missing values occur completely at random.

---

# 1️⃣ What is Complete Case Analysis (CCA)?

Complete Case Analysis removes every row that contains one or more missing values.

### Example

#### Original Dataset

| Name | Age | Salary |
|--------|--------|--------|
| John | 25 | 50000 |
| Alice | NaN | 60000 |
| Bob | 30 | 55000 |

#### After CCA

| Name | Age | Salary |
|--------|--------|--------|
| John | 25 | 50000 |
| Bob | 30 | 55000 |

The row containing the missing value is completely removed.

---

# 2️⃣ Why Use CCA?

### Advantages

✅ Very easy to implement

✅ No complex imputation techniques required

✅ Preserves original values

✅ Works well when missing data is minimal

---

### Disadvantages

❌ Reduces dataset size

❌ Loss of valuable information

❌ Can introduce bias if missing values follow a pattern

❌ Not suitable for datasets with large amounts of missing data

---

# 3️⃣ The MCAR Assumption

CCA is only reliable when data is:

## Missing Completely At Random (MCAR)

The probability of a value being missing is completely independent of:

- Other variables
- The missing value itself

---

### Example of MCAR

A survey form is accidentally damaged during storage.

Missing responses occur randomly.

```text
Missingness ← Random Accident
```

CCA can safely be applied.

---

### Example of Non-MCAR

People with higher salaries intentionally skip salary questions.

```text
Missingness ← Salary Level
```

Dropping these rows would create selection bias.

CCA should not be used.

---

# 4️⃣ The 5% Rule of Thumb

A commonly used guideline:

```text
Missing Percentage < 5%
```

CCA is generally considered safe.

---

### Example

| Rows | Missing Rows |
|--------|--------|
| 10,000 | 300 |

Missing Percentage:

```text
300 / 10000 × 100 = 3%
```

Since:

```text
3% < 5%
```

CCA is usually acceptable.

---

# 5️⃣ Checking Missing Value Percentages

Before applying CCA, calculate the percentage of missing values in each column.

---

## Load Dataset

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data_science_job.csv')
```

---

## Missing Percentage Calculation

```python
print(df.isnull().mean() * 100)
```

### Example Output

```text
experience              2.3
enrolled_university     1.5
city                    0.0
salary                  15.7
```

Columns below the 5% threshold become candidates for CCA.

---

# 6️⃣ Applying Complete Case Analysis

## Selecting Columns Under the 5% Threshold

```python
cols_to_drop = [
    col for col in df.columns
    if df[col].isnull().mean() < 0.05
    and df[col].isnull().mean() > 0
]
```

This identifies columns that:

- Have missing values
- Have less than 5% missing data

---

## Drop Rows Containing Missing Values

```python
df_clean = df.dropna(
    subset=cols_to_drop
)
```

Only rows missing values in the selected columns are removed.

---

## Compare Dataset Shapes

```python
print("Original Shape:", df.shape)
print("Cleaned Shape:", df_clean.shape)
```

### Example Output

```text
Original Shape: (19158, 13)
Cleaned Shape: (18502, 13)
```

---

# 7️⃣ Why Use the `subset` Parameter?

Instead of:

```python
df.dropna()
```

use:

```python
df.dropna(subset=cols_to_drop)
```

---

### Problem with Plain `dropna()`

```python
df.dropna()
```

Removes rows with missing values in **any column**.

This can accidentally remove a large portion of the dataset.

---

### Better Approach

```python
df.dropna(subset=cols_to_drop)
```

Only targets specific columns that satisfy the CCA assumptions.

---

# 8️⃣ Distribution Preservation Check

After removing rows, verify that feature distributions remain unchanged.

This ensures CCA has not introduced bias.

---

## Numerical Feature Validation

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111)

df['experience'].plot(
    kind='kde',
    ax=ax,
    color='red',
    label='Original'
)

df_clean['experience'].plot(
    kind='kde',
    ax=ax,
    color='green',
    label='CCA'
)

plt.legend()
plt.show()
```

---

### Interpretation

If both curves overlap closely:

✅ Distribution preserved

If curves shift significantly:

❌ CCA may have introduced bias

---

# 9️⃣ Checking Categorical Variables

For categorical features, compare class frequencies before and after deletion.

---

## Frequency Comparison

```python
temp = pd.concat(
    [
        df['enrolled_university'].value_counts() / len(df),
        df_clean['enrolled_university'].value_counts() / len(df_clean)
    ],
    axis=1
)

temp.columns = ['original', 'cca']

print(temp)
```

---

### Example Output

| Category | Original | CCA |
|-----------|----------|-----|
| no_enrollment | 0.68 | 0.67 |
| Full time course | 0.21 | 0.22 |
| Part time course | 0.11 | 0.11 |

The proportions remain nearly identical.

This indicates that CCA has not distorted the categorical distribution.

---

# 🔟 Complete Workflow

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data_science_job.csv')

# Missing percentage
print(df.isnull().mean() * 100)

# Identify columns under 5% threshold
cols_to_drop = [
    col for col in df.columns
    if df[col].isnull().mean() < 0.05
    and df[col].isnull().mean() > 0
]

# Apply CCA
df_clean = df.dropna(
    subset=cols_to_drop
)

# Shape comparison
print(df.shape)
print(df_clean.shape)

# Numerical distribution check
df['experience'].plot(kind='kde')
df_clean['experience'].plot(kind='kde')

# Categorical distribution check
temp = pd.concat(
    [
        df['enrolled_university'].value_counts() / len(df),
        df_clean['enrolled_university'].value_counts() / len(df_clean)
    ],
    axis=1
)

print(temp)
```

---

# 📊 Key Takeaways

### Complete Case Analysis (CCA)

✅ Removes rows containing missing values.

✅ Also known as Listwise Deletion.

✅ Simple and fast to implement.

---

### MCAR Assumption

✅ Missing values must occur completely at random.

✅ Most important requirement for CCA.

---

### 5% Rule

✅ Safe when missing data is less than 5%.

✅ Minimizes loss of statistical power.

---

### `dropna()`

✅ Primary method used for row removal.

✅ Prefer using the `subset` parameter.

---

### Distribution Validation

✅ Compare KDE plots for numerical features.

✅ Compare frequency ratios for categorical features.

✅ Ensure CCA does not introduce bias.

---

# 🚀 Conclusion

Complete Case Analysis is one of the simplest methods for handling missing data. It works best when the amount of missing information is small and the missingness follows the MCAR assumption. Although easy to implement, CCA should always be accompanied by distribution checks to ensure that removing rows does not distort the dataset.

When applied carefully, CCA provides a fast and effective solution for cleaning datasets while maintaining data quality and model reliability.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📚 Concepts Covered

- Missing Data Handling
- Complete Case Analysis (CCA)
- Listwise Deletion
- MCAR Assumption
- Missing Value Analysis
- `dropna()`
- `subset` Parameter
- Distribution Preservation
- KDE Plots
- Categorical Frequency Validation
- Data Cleaning
- Data Preprocessing
Video Link : https://youtu.be/aUnNWZorGmk
