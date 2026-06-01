# Day 44 - Outlier Detection & Removal: Percentile Method (Winsorization)

## 📌 Overview

The Percentile Method is a flexible outlier detection technique that identifies extreme values using custom percentile boundaries instead of statistical assumptions like Mean, Standard Deviation, or IQR.

Since it relies only on data ranking, it works well for both normal and skewed distributions.

---

## What is Percentile-Based Outlier Detection?

Percentiles divide data into 100 equal parts.

Common thresholds:

- Lower Limit → 1st Percentile (P1)
- Upper Limit → 99th Percentile (P99)

Values outside these limits are considered outliers.

---

## Why Use Percentiles?

Unlike:

- Z-Score → Requires normal distribution
- IQR → Designed mainly for skewed data

Percentiles:

✅ Work on any distribution

✅ Easy to understand

✅ Fully customizable

---

## Step 1: Load Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('weight-height.csv')
```

---

## Step 2: Visualize Distribution

```python
sns.histplot(
    df['height'],
    kde=True
)

plt.show()
```

### Summary Statistics

```python
print(
    df['height'].describe()
)
```

---

## Step 3: Calculate Percentile Limits

### Lower Limit (1%)

```python
lower_limit = df['height'].quantile(0.01)
```

### Upper Limit (99%)

```python
upper_limit = df['height'].quantile(0.99)
```

---

## View Limits

```python
print(lower_limit)

print(upper_limit)
```

---

## Find Outliers

```python
outliers = df[
    (df['height'] > upper_limit) |
    (df['height'] < lower_limit)
]

print(outliers.shape)
```

---

## Method 1: Trimming

Remove rows outside percentile limits.

```python
df_trimmed = df[
    (df['height'] >= lower_limit) &
    (df['height'] <= upper_limit)
]
```

### Check Row Count

```python
print(df.shape)

print(df_trimmed.shape)
```

---

### Advantages

✅ Removes extreme values

✅ Cleaner dataset

---

### Disadvantages

❌ Data loss

❌ Not ideal for small datasets

---

## Method 2: Capping (Winsorization)

Replace outliers with percentile limits.

```python
df_capped = df.copy()

df_capped['height'] = np.where(
    df_capped['height'] > upper_limit,
    upper_limit,
    np.where(
        df_capped['height'] < lower_limit,
        lower_limit,
        df_capped['height']
    )
)
```

---

### Advantages

✅ No row loss

✅ Preserves dataset size

✅ Reduces outlier impact

---

### Disadvantages

❌ Original values are modified

❌ May slightly distort data

---

## Common Percentile Choices

| Lower | Upper | Usage |
|---------|---------|---------|
| 1% | 99% | Standard Cleaning |
| 0.5% | 99.5% | Strict Cleaning |
| 0.1% | 99.9% | Extreme Outlier Detection |

---

## Percentile vs IQR vs Z-Score

| Method | Best For |
|----------|----------|
| Z-Score | Normal Distribution |
| IQR | Skewed Distribution |
| Percentile | Any Distribution |

---

## When to Use?

Use Percentile Method when:

- Distribution shape is unknown
- Data is heavily skewed
- Flexible thresholds are required
- Domain-specific limits are needed

Avoid when:

- Exact statistical assumptions are required

---

## Complete Workflow

```python
lower_limit = df['height'].quantile(0.01)

upper_limit = df['height'].quantile(0.99)

# Trimming
df_trimmed = df[
    (df['height'] >= lower_limit) &
    (df['height'] <= upper_limit)
]

# Capping
df_capped = df.copy()

df_capped['height'] = np.where(
    df_capped['height'] > upper_limit,
    upper_limit,
    np.where(
        df_capped['height'] < lower_limit,
        lower_limit,
        df_capped['height']
    )
)
```

---

## Key Takeaways

- Percentile Method uses rank-based thresholds.
- No distribution assumptions are required.
- Common limits are 1st and 99th percentiles.
- Trimming removes outlier rows.
- Capping replaces extreme values with limits.
- More flexible than Z-Score and IQR methods.
- Always calculate percentile limits using training data only.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Concepts Covered

- Outlier Detection
- Percentile Method
- Winsorization
- Trimming
- Capping
- Quantiles
- Data Cleaning
- Feature Engineering
- Data Preprocessing