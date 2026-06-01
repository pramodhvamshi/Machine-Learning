# Day 42 - Outlier Detection & Removal: Z-Score Method

## 📌 Overview

The Z-Score Method is a statistical technique used to identify and handle outliers by measuring how far a data point is from the mean in terms of standard deviations.

It works best on normally distributed data and is commonly used for detecting extreme values before training machine learning models.

---

## What is a Z-Score?

A Z-Score tells us how many standard deviations a value lies away from the mean.

### Formula


::contentReference[oaicite:0]{index=0}


Where:

- **x** = Data Point
- **μ** = Mean
- **σ** = Standard Deviation

---

## Empirical Rule (68-95-99.7)

For a Normal Distribution:

- 68% data lies within ±1σ
- 95% data lies within ±2σ
- 99.7% data lies within ±3σ

Therefore:

```text
|Z| > 3
```

is usually considered an outlier.

---

## Outlier Boundaries

### Upper Limit

:contentReference[oaicite:1]{index=1}

### Lower Limit

:contentReference[oaicite:2]{index=2}

Any value outside these limits is treated as an outlier.

---

## Step 1: Load Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('placement.csv')
```

---

## Step 2: Check Normal Distribution

```python
sns.histplot(
    df['placement_exam_marks'],
    kde=True
)
plt.show()
```

### Check Skewness

```python
print(
    df['placement_exam_marks']
    .skew()
)
```

If skewness is close to:

```text
0
```

the feature is approximately normal.

---

## Step 3: Calculate Limits

```python
mean_val = df['placement_exam_marks'].mean()

std_val = df['placement_exam_marks'].std()

upper_limit = mean_val + 3 * std_val

lower_limit = mean_val - 3 * std_val
```

---

## Find Outliers

```python
outliers = df[
    (df['placement_exam_marks'] > upper_limit) |
    (df['placement_exam_marks'] < lower_limit)
]

print(outliers)
```

---

## Method 1: Trimming

Remove rows containing outliers.

```python
df_trimmed = df[
    (df['placement_exam_marks'] < upper_limit) &
    (df['placement_exam_marks'] > lower_limit)
]
```

### Check Shape

```python
print(df.shape)

print(df_trimmed.shape)
```

---

### Advantages

✅ Removes extreme values completely

✅ Improves model stability

---

### Disadvantages

❌ Data loss

❌ Not suitable for small datasets

---

## Method 2: Capping (Winsorization)

Replace outliers with boundary values.

```python
df_capped = df.copy()

df_capped['placement_exam_marks'] = np.where(
    df_capped['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        df_capped['placement_exam_marks'] < lower_limit,
        lower_limit,
        df_capped['placement_exam_marks']
    )
)
```

---

### Advantages

✅ No row loss

✅ Preserves dataset size

✅ Safer for small datasets

---

### Disadvantages

❌ Outliers still exist in modified form

❌ Can slightly distort distribution

---

## When to Use Z-Score?

Use when:

- Data follows Normal Distribution
- Skewness is close to zero
- Outliers are extreme and rare

Avoid when:

- Data is highly skewed
- Distribution is not Gaussian

---

## Complete Workflow

```python
mean_val = df['placement_exam_marks'].mean()

std_val = df['placement_exam_marks'].std()

upper_limit = mean_val + 3 * std_val

lower_limit = mean_val - 3 * std_val

# Trimming
df_trimmed = df[
    (df['placement_exam_marks'] < upper_limit) &
    (df['placement_exam_marks'] > lower_limit)
]

# Capping
df_capped = df.copy()

df_capped['placement_exam_marks'] = np.where(
    df_capped['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        df_capped['placement_exam_marks'] < lower_limit,
        lower_limit,
        df_capped['placement_exam_marks']
    )
)
```

---

## Key Takeaways

- Z-Score measures distance from the mean.
- Works best for normally distributed data.
- Outliers are typically values with:

```text
|Z| > 3
```

- Trimming removes outlier rows.
- Capping replaces outliers with boundary values.
- Always verify normality before applying Z-Score.
- Calculate limits using training data only.

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
- Z-Score
- Standard Deviation
- Normal Distribution
- Trimming
- Capping
- Winsorization
- Data Cleaning
- Feature Engineering
- Data Preprocessing
Video Link:https://youtu.be/OnPE-Z8jtqM
