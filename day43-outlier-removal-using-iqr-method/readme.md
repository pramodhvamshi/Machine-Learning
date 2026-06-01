# Day 43 - Outlier Detection & Removal: IQR Method

## 📌 Overview

The IQR (Interquartile Range) Method is a robust outlier detection technique that works well for skewed and non-normal distributions. Unlike the Z-Score method, it uses percentiles instead of mean and standard deviation, making it less sensitive to extreme values.

---

## What is IQR?

IQR measures the spread of the middle 50% of the data.

### Formula

:contentReference[oaicite:0]{index=0}

Where:

- **Q1** = 25th Percentile
- **Q3** = 75th Percentile

---

## Outlier Boundaries (Tukey's Fences)

### Upper Limit

:contentReference[oaicite:1]{index=1}

### Lower Limit

:contentReference[oaicite:2]{index=2}

Values outside these limits are treated as outliers.

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

## Step 2: Visualize Distribution

```python
sns.histplot(
    df['placement_exam_marks'],
    kde=True
)

sns.boxplot(
    x=df['placement_exam_marks']
)
```

### Check Skewness

```python
print(
    df['placement_exam_marks']
    .skew()
)
```

If data is highly skewed, prefer IQR over Z-Score.

---

## Step 3: Calculate Quartiles

```python
q1 = df['placement_exam_marks'].quantile(0.25)

q3 = df['placement_exam_marks'].quantile(0.75)
```

### Calculate IQR

```python
iqr = q3 - q1
```

---

## Step 4: Calculate Limits

```python
upper_limit = q3 + 1.5 * iqr

lower_limit = q1 - 1.5 * iqr
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

✅ Improves model performance

---

### Disadvantages

❌ Data loss

❌ Not suitable for small datasets

---

## Method 2: Capping

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

✅ Handles extreme values effectively

---

### Disadvantages

❌ Slightly changes original values

❌ May affect distribution shape

---

## IQR vs Z-Score

| Feature | IQR | Z-Score |
|----------|-----|----------|
| Uses Mean | ❌ | ✅ |
| Uses Percentiles | ✅ | ❌ |
| Sensitive to Outliers | ❌ | ✅ |
| Best for Skewed Data | ✅ | ❌ |
| Best for Normal Data | ❌ | ✅ |

---

## When to Use?

Use IQR when:

- Data is skewed
- Distribution is not normal
- Extreme values affect mean

Avoid when:

- Data is perfectly normal
- Z-Score assumptions hold

---

## Complete Workflow

```python
q1 = df['placement_exam_marks'].quantile(0.25)

q3 = df['placement_exam_marks'].quantile(0.75)

iqr = q3 - q1

upper_limit = q3 + 1.5 * iqr

lower_limit = q1 - 1.5 * iqr

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

- IQR uses quartiles instead of mean and standard deviation.
- Best suited for skewed distributions.
- Outliers lie outside Tukey's fences.
- Trimming removes outlier rows.
- Capping replaces extreme values with limits.
- More robust than Z-Score for non-normal data.
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
- IQR Method
- Quartiles
- Interquartile Range
- Tukey's Fences
- Trimming
- Capping
- Winsorization
- Data Cleaning
- Data Preprocessing