# Day 24 - Feature Scaling: Standardization (Z-Score Normalization)

## 📌 Overview

Many Machine Learning algorithms are sensitive to the scale of input features. When features have vastly different ranges, variables with larger magnitudes can dominate model learning and distance calculations.

**Standardization (Z-Score Normalization)** is one of the most commonly used feature scaling techniques. It transforms numerical features so that they have:

* Mean = 0
* Standard Deviation = 1

This helps algorithms converge faster and improves model performance.

---

# 🚀 Quick Revision Notes

## What is Feature Scaling?

Feature Scaling is a preprocessing technique used to bring numerical features onto a similar scale.

Example:

| Feature | Range            |
| ------- | ---------------- |
| Age     | 18 - 60          |
| Salary  | 15,000 - 150,000 |

Without scaling, Salary values dominate Age values in distance calculations.

---

## What is Standardization?

Standardization transforms data into a standard normal distribution where:

```text
Mean = 0
Standard Deviation = 1
```

The transformed values are called **Z-Scores**.

---

## Mathematical Formula

The standardized value is calculated using:

Where:

* **x** → Original Value
* **μ** → Mean of the Feature
* **σ** → Standard Deviation of the Feature
* **z** → Standardized Value

---

## StandardScaler

Scikit-Learn provides the `StandardScaler` class for standardization.

```python
from sklearn.preprocessing import StandardScaler
```

---

## Fit Operation

```python
scaler.fit(X_train)
```

Computes:

* Mean (μ)
* Standard Deviation (σ)

using only the training data.

---

## Transform Operation

```python
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Applies the previously computed scaling parameters.

---

## The Cardinal Rule: Avoid Data Leakage

✅ Correct:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

❌ Wrong:

```python
scaler.fit(X_test)
```

Never fit on test data.

Doing so leaks future information into the model and produces misleading evaluation results.

---

## Algorithms That Require Standardization

### Highly Sensitive

* K-Nearest Neighbors (KNN)
* K-Means Clustering
* Logistic Regression
* Linear Regression (Gradient Descent)
* Support Vector Machines (SVM)
* Neural Networks
* Principal Component Analysis (PCA)

---

### Usually Not Required

Tree-based models are scale-invariant:

* Decision Trees
* Random Forest
* XGBoost
* CatBoost
* LightGBM

---

# 🔍 Technical Workflow Analysis

## Step 1: Load Dataset

```python
import pandas as pd

df = pd.read_csv("Social_Network_Ads.csv")
```

Load the dataset into a Pandas DataFrame.

---

## Step 2: Feature and Target Separation

```python
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
```

Separate:

* Independent Features (X)
* Target Variable (y)

---

## Step 3: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)
```

Splitting first prevents data leakage.

---

## Step 4: Initialize StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Creates the scaler object.

---

## Step 5: Fit the Training Data

```python
scaler.fit(X_train)
```

Learns:

* Mean
* Standard Deviation

from training data only.

---

## Step 6: Transform the Data

```python
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Converts features into standardized values.

---

## Step 7: Convert Back to DataFrame (Optional)

```python
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)
```

Improves readability and analysis.

---

## Step 8: Verify Scaling

### Original Statistics

```python
print(X_train.describe())
```

Output will show:

* Original Mean
* Original Standard Deviation

---

### Standardized Statistics

```python
print(X_train_scaled.describe())
```

Expected:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

---

# 📊 Before vs After Standardization

### Original Data

| Age | Salary |
| --- | ------ |
| 20  | 20000  |
| 30  | 50000  |
| 40  | 100000 |

Notice the large difference in scales.

---

### Standardized Data

| Age  | Salary |
| ---- | ------ |
| -1.2 | -0.9   |
| 0.0  | 0.1    |
| 1.3  | 1.4    |

Now both features contribute equally.

---

# 🎯 Key Takeaways

## ✅ Standardization Does Not Change Distribution Shape

Scaling only changes:

* Center
* Spread

It does **not** remove skewness or outliers.

---

## ✅ Always Split Before Scaling

Correct Workflow:

```text
Train-Test Split
        ↓
Fit on Training Data
        ↓
Transform Training Data
        ↓
Transform Test Data
```

---

## ✅ Prevent Data Leakage

Never calculate scaling statistics from test data.

```python
scaler.fit(X_train)
```

is correct.

```python
scaler.fit(X_test)
```

is wrong.

---

## ✅ Standardization Improves Distance-Based Algorithms

Algorithms such as:

* KNN
* K-Means
* PCA
* SVM

perform significantly better after scaling.

---

## ✅ Outliers Affect StandardScaler

Since StandardScaler uses:

* Mean
* Standard Deviation

extreme values can distort scaling.

For datasets with severe outliers, consider:

```python
from sklearn.preprocessing import RobustScaler
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib
* Jupyter Notebook

---

# 📚 Learning Outcome

By completing this notebook, you will understand:

* Why Feature Scaling is important
* How Standardization works mathematically
* How to use StandardScaler
* How to prevent data leakage
* Which algorithms require scaling
* The impact of scaling on model performance
* When to use alternative scaling techniques

---

## 📂 Dataset Used

```text
Social_Network_Ads.csv
```

### Features

* Age
* EstimatedSalary

### Target

* Purchased


Video Link : https://youtu.be/1Yw9sC0PNwY
