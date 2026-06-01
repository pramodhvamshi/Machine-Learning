# Day 20 - Univariate Analysis (Exploring Features Individually)

## 📌 Overview

Univariate Analysis is the process of analyzing a single feature at a time to understand its distribution, central tendency, spread, and potential anomalies.

In this notebook, we explore both **categorical** and **numerical** variables using statistical summaries and visualization techniques. These insights help identify class imbalances, skewness, and outliers before moving to feature engineering or model training.

---

## 🚀 Quick Revision Notes

## 1. Categorical Data Analysis

### Frequency Counts

```python
df['col'].value_counts()
```

Counts the occurrences of each category within a feature.

Useful for:

* Detecting class imbalance
* Understanding category distribution

---

### Bar Plot

```python
df['col'].value_counts().plot(kind='bar')
```

Visualizes category frequencies using bars.

Best for comparing category counts.

---

### Pie Chart

```python
df['col'].value_counts().plot(
    kind='pie',
    autopct='%.2f%%'
)
```

Displays category proportions as percentages.

Useful for understanding relative shares.

---

### Count Plot (Seaborn)

```python
sns.countplot(data=df, x='col')
```

Creates a categorical frequency chart directly from a DataFrame.

Advantages:

* Cleaner syntax
* Better aesthetics
* Works seamlessly with Seaborn themes

---

## 2. Numerical Data Analysis

### Histogram

```python
plt.hist(df['col'], bins=20)
```

Groups continuous values into bins.

Used to identify:

* Normal distributions
* Skewed distributions
* Multimodal distributions

---

### Histogram with Density Curve

```python
sns.histplot(df['col'], kde=True)
```

Displays:

* Histogram
* Kernel Density Estimation (KDE)

Helps visualize the underlying probability distribution.

---

### Box Plot

```python
sns.boxplot(x=df['col'])
```

Visualizes the Five-Number Summary:

* Minimum
* First Quartile (Q1)
* Median (Q2)
* Third Quartile (Q3)
* Maximum

Useful for detecting outliers.

---

### Skewness

```python
df['col'].skew()
```

Measures distribution asymmetry.

| Skewness Value | Interpretation            |
| -------------- | ------------------------- |
| 0              | Symmetric Distribution    |
| > 0            | Right-Skewed Distribution |
| < 0            | Left-Skewed Distribution  |

---

## 🔍 Technical Workflow Analysis

### Step 1: Import Libraries and Load Dataset

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
```

Loads the dataset and required visualization libraries.

---

## Step 2: Categorical Feature Analysis

### Frequency Distribution

```python
print(df['Survived'].value_counts())
```

Example Output:

```text
0    549
1    342
```

This helps determine whether the target variable is balanced.

---

### Pie Chart Visualization

```python
df['Pclass'].value_counts().plot(
    kind='pie',
    autopct='%.1f%%'
)

plt.show()
```

Displays the percentage distribution of passenger classes.

---

## Step 3: Numerical Feature Analysis

### Histogram + KDE

```python
sns.histplot(
    df['Age'],
    kde=True
)

plt.show()
```

Used to understand:

* Data concentration
* Distribution shape
* Potential skewness

---

### Measuring Skewness

```python
print(df['Age'].skew())
```

Interpretation:

* Positive → Longer right tail
* Negative → Longer left tail
* Near Zero → Approximately normal distribution

---

## Step 4: Outlier Detection

### Box Plot

```python
sns.boxplot(x=df['Fare'])

plt.show()
```

Box plots help identify observations beyond the IQR boundaries.

---

### Statistical Summary

```python
print(df['Fare'].describe())
```

Example Output:

```text
count
mean
std
min
25%
50%
75%
max
```

These metrics provide a quick overview of data spread and extreme values.

---

## 📊 Understanding the Box Plot

A box plot is built using the Five-Number Summary:

```text
Minimum
Q1 (25%)
Median (50%)
Q3 (75%)
Maximum
```

### Interquartile Range (IQR)

```text
IQR = Q3 - Q1
```

Outlier Thresholds:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values outside these limits are considered potential outliers.

---

## 🎯 Key Takeaways

### ✅ Check Class Balance Early

Always inspect categorical variables using:

```python
df['col'].value_counts()
```

Imbalanced classes may require:

* Stratified Sampling
* Oversampling
* Undersampling

---

### ✅ Histograms Reveal Distribution Shape

Use histograms to identify:

* Normal distributions
* Skewed data
* Multiple peaks

---

### ✅ Skewness Guides Feature Transformation

Highly skewed data often benefits from:

* Log Transformation
* Square Root Transformation
* Box-Cox Transformation

Example:

```python
df['Fare'].skew()
```

---

### ✅ Box Plots Expose Outliers

Outliers can significantly affect:

* Mean
* Standard Deviation
* Linear Models

Review box plots before applying preprocessing techniques.

---

### ✅ Numerical and Categorical Features Require Different Approaches

| Feature Type | Common Visualizations           |
| ------------ | ------------------------------- |
| Categorical  | Bar Plot, Pie Chart, Count Plot |
| Numerical    | Histogram, KDE Plot, Box Plot   |

---

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📚 Learning Outcome

By completing this notebook, you will understand:

* The fundamentals of Univariate Analysis
* How to analyze categorical variables
* How to analyze numerical variables
* How to detect skewness and outliers
* How to interpret histograms and box plots
* How to prepare features for future preprocessing and modeling stages



Video Link : https://www.youtube.com/watch?v=4HyTlbHUKSw
