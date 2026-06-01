# Day 19 - Understanding Your Data (Descriptive Statistics)

## 📌 Overview

Before building any machine learning model, it is essential to understand the dataset thoroughly. This notebook focuses on **Exploratory Data Analysis (EDA)** using descriptive statistics to uncover patterns, identify missing values, detect duplicates, and understand relationships between features.

The goal is to gain insights into the data before moving on to preprocessing and model building.

---

## 🚀 Quick Revision Notes

### 1. Dataset Dimensions

```python
df.shape
```

Returns the number of rows and columns in the DataFrame.

Example Output:

```python
(891, 12)
```

---

### 2. Previewing Data

```python
df.head()
df.tail()
df.sample(5)
```

* `head()` → Displays the first few rows.
* `tail()` → Displays the last few rows.
* `sample()` → Displays random rows to avoid bias from ordered data.

---

### 3. Dataset Information

```python
df.info()
```

Provides:

* Column names
* Data types
* Non-null counts
* Memory usage

Useful for detecting missing values and datatype issues.

---

### 4. Missing Value Analysis

```python
df.isnull().sum()
```

Returns the total number of missing values in each column.

Example:

```python
Age         177
Cabin       687
Embarked      2
```

---

### 5. Descriptive Statistics

```python
df.describe()
```

Generates statistical summaries for numerical columns.

Includes:

* Count
* Mean
* Standard Deviation
* Minimum Value
* Maximum Value
* Quartiles (25%, 50%, 75%)

---

### 6. Duplicate Detection

```python
df.duplicated().sum()
```

Identifies duplicate rows that may affect model performance.

---

### 7. Correlation Analysis

```python
df.corr(numeric_only=True)
```

Computes Pearson Correlation Coefficients between numerical variables.

Useful for:

* Feature selection
* Detecting multicollinearity
* Understanding relationships with the target variable

---

## 🔍 Technical Workflow Analysis

### Step 1: Load the Dataset

```python
import pandas as pd

df = pd.read_csv("train.csv")
```

Loads the dataset into a Pandas DataFrame for analysis.

---

### Step 2: Initial Data Inspection

```python
print(df.shape)

print(df.info())

print(df.isnull().sum())
```

This step helps answer:

* How large is the dataset?
* Which columns contain missing values?
* What are the data types?

---

### Step 3: Statistical Summary

```python
print(df.describe())
```

Provides an overview of:

* Central Tendency (Mean, Median)
* Spread (Standard Deviation)
* Range (Min, Max)
* Distribution (Quartiles)

Example:

```python
count
mean
std
min
25%
50%
75%
max
```

---

### Step 4: Correlation Analysis

```python
correlation_matrix = df.corr(numeric_only=True)

print(
    correlation_matrix["Survived"]
    .sort_values(ascending=False)
)
```

This helps identify which features are most strongly related to the target variable.

Example Insights:

* Positive correlation → Feature increases with target.
* Negative correlation → Feature decreases with target.
* Near zero → Weak or no linear relationship.

---

## 📊 Exploratory Data Analysis Concepts

### Univariate Analysis

Studies a single variable independently.

Examples:

* Age Distribution
* Fare Distribution
* Passenger Class Frequency

Questions Answered:

* What is the distribution?
* Are there outliers?
* Is the data skewed?

---

### Bivariate Analysis

Studies the relationship between two variables.

Examples:

* Age vs Survival
* Fare vs Survival
* Passenger Class vs Survival

Questions Answered:

* Is there any relationship?
* Is the relationship positive or negative?

---

### Multivariate Analysis

Studies interactions among multiple variables simultaneously.

Examples:

* Age + Class + Survival
* Fare + Gender + Survival

Questions Answered:

* How do multiple factors influence outcomes together?

---

## 🎯 Key Takeaways

### ✅ Understand Data Before Modeling

Never train a model without first exploring the dataset.

---

### ✅ Detect Missing Values Early

Use:

```python
df.info()

df.isnull().sum()
```

to determine the appropriate imputation strategy.

---

### ✅ Watch for Outliers

When reviewing:

```python
df.describe()
```

Compare:

```text
75th Percentile → Max Value
```

A large gap often indicates outliers or skewed distributions.

---

### ✅ Remove Duplicate Records

Duplicates can bias model learning and evaluation.

```python
df.duplicated().sum()
```

---

### ✅ Check Feature Relationships

Use:

```python
df.corr()
```

to identify:

* Strong predictors
* Redundant features
* Multicollinearity issues

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Jupyter Notebook

---

## 📚 Learning Outcome

By completing this notebook, you will understand:

* How to inspect a dataset effectively
* How to identify missing values and duplicates
* How to interpret descriptive statistics
* How to analyze feature relationships
* The foundations of Exploratory Data Analysis (EDA)
* How to prepare data for preprocessing and machine learning




Video Link: https://www.youtube.com/watch?v=mJlRTUuVr04
