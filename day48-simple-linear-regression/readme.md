# Day 48 - Machine Learning Foundations: Simple Linear Regression

## 📌 Overview

Simple Linear Regression (SLR) is the most fundamental supervised machine learning algorithm used for predicting continuous numerical values. It models the relationship between a single independent variable (feature) and a dependent variable (target) using a straight line.

The objective is to find the best-fit line that minimizes prediction errors and accurately captures the relationship between the input and output variables.

---

# 1️⃣ What is Regression?

Regression is a supervised learning technique used when the target variable is continuous.

### Examples

| Input (X) | Output (Y) |
|------------|------------|
| CGPA | Salary Package |
| House Area | House Price |
| Experience | Salary |
| Temperature | Electricity Usage |

Unlike Classification:

```text
Classification → Predict Categories
Regression     → Predict Numbers
```

---

# 2️⃣ What is Simple Linear Regression?

Simple Linear Regression models the relationship between:

```text
1 Independent Variable (X)
1 Dependent Variable (Y)
```

using a straight line.

---

## Mathematical Equation

:contentReference[oaicite:0]{index=0}

Where:

| Symbol | Meaning |
|----------|----------|
| y | Predicted Value |
| x | Input Feature |
| β₀ | Intercept |
| β₁ | Slope/Coefficient |
| ε | Error Term |

---

# 3️⃣ Understanding the Components

## Independent Variable (X)

Input feature used for prediction.

Example:

```text
CGPA
```

---

## Dependent Variable (Y)

Target variable.

Example:

```text
Salary Package
```

---

## Intercept (β₀)

The value of Y when:

```text
X = 0
```

Graphically:

```text
Point where line crosses Y-axis
```

---

## Slope (β₁)

Represents change in Y for every one-unit increase in X.

### Example

If:

```text
β₁ = 2
```

then:

```text
X increases by 1
Y increases by 2
```

---

## Error Term (ε)

Difference between:

```text
Actual Value
```

and

```text
Predicted Value
```

---

# 4️⃣ Geometric Interpretation

Consider a dataset:

```text
CGPA vs Package
```

Plotting points gives:

```text
      *
    *
  *
 *
*
```

Simple Linear Regression draws the best possible straight line through these points.

---

## Regression Line

```text
Actual Points

     *
   *
  *
 *
*

Best Fit Line
-------------
```

The goal:

```text
Minimize Distance
```

between data points and line.

---

# 5️⃣ Prediction Equation

Predicted value:

:contentReference[oaicite:1]{index=1}

Example:

```text
β₀ = -3
β₁ = 1.5
CGPA = 8
```

Prediction:

```text
ŷ = -3 + (1.5 × 8)

ŷ = 9
```

---

# 6️⃣ How Does Linear Regression Find the Best Line?

There can be infinitely many lines.

Example:

```text
Line 1
Line 2
Line 3
...
```

Which one is best?

The answer:

```text
Ordinary Least Squares (OLS)
```

---

# 7️⃣ Ordinary Least Squares (OLS)

OLS finds values of:

```text
β₀
β₁
```

that minimize total prediction error.

---

## Residual

Residual =

```text
Actual - Predicted
```

Formula:

:contentReference[oaicite:2]{index=2}

---

## Sum of Squared Errors (SSE)

OLS minimizes:

:contentReference[oaicite:3]{index=3}

Why square?

- Removes negative signs
- Penalizes large errors heavily
- Creates a smooth optimization function

---

# 8️⃣ Deriving the Best-Fit Line

The OLS solution is obtained by differentiating the SSE function and setting derivatives equal to zero.

---

## Intercept Formula

After solving:

:contentReference[oaicite:4]{index=4}

Where:

```text
x̄ = Mean of X
ȳ = Mean of Y
```

---

## Slope Formula

Final OLS solution:

:contentReference[oaicite:5]{index=5}

---

## Interpretation

Numerator:

```text
Covariance(X,Y)
```

Measures relationship strength.

Denominator:

```text
Variance(X)
```

Measures spread of X.

---

# 9️⃣ From-Scratch Implementation

## Custom Linear Regression Class

```python
import numpy as np

class CampusXLinearRegression:

    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):

        X_mean = np.mean(X_train)
        y_mean = np.mean(y_train)

        num = 0
        den = 0

        for i in range(X_train.shape[0]):

            num += (
                (X_train[i] - X_mean)
                *
                (y_train[i] - y_mean)
            )

            den += (
                (X_train[i] - X_mean)
                ** 2
            )

        self.m = num / den

        self.b = y_mean - (
            self.m * X_mean
        )

    def predict(self, X_test):

        return (
            self.m * X_test
        ) + self.b
```

---

# 🔟 Scikit-Learn Implementation

## Import Libraries

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```

---

## Load Dataset

```python
import pandas as pd

df = pd.read_csv(
    'placement.csv'
)
```

---

## Create Features

```python
X = df[['cgpa']]

y = df['package']
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

## Train Model

```python
lr = LinearRegression()

lr.fit(
    X_train,
    y_train
)
```

---

## Predictions

```python
y_pred = lr.predict(
    X_test
)
```

---

## Model Parameters

### Slope

```python
print(
    lr.coef_[0]
)
```

### Intercept

```python
print(
    lr.intercept_
)
```

---

# 1️⃣1️⃣ Evaluating the Model

## R² Score

Measures goodness of fit.

Formula:

:contentReference[oaicite:6]{index=6}

---

### Interpretation

| R² Value | Meaning |
|-----------|----------|
| 1 | Perfect Fit |
| 0.8 | Very Good |
| 0.5 | Moderate |
| 0 | No Relationship |

---

## Calculate R²

```python
from sklearn.metrics import r2_score

print(
    r2_score(
        y_test,
        y_pred
    )
)
```

---

# 1️⃣2️⃣ Assumptions of Linear Regression

### Linearity

Relationship must be linear.

---

### Independence

Observations should be independent.

---

### Constant Variance

Residual variance should remain constant.

---

### Normal Residuals

Errors should be normally distributed.

---

### No Extreme Outliers

Outliers can significantly distort the regression line.

---

# 1️⃣3️⃣ Advantages

✅ Easy to understand

✅ Fast training

✅ Interpretable results

✅ Closed-form mathematical solution

✅ Good baseline model

---

# 1️⃣4️⃣ Limitations

❌ Assumes linear relationships

❌ Sensitive to outliers

❌ Cannot capture complex patterns

❌ Poor performance on nonlinear datasets

---

# OLS vs Gradient Descent

| OLS | Gradient Descent |
|------|----------------|
| Exact Solution | Iterative Solution |
| No Learning Rate | Requires Learning Rate |
| Fast for Small Data | Better for Large Data |
| Closed Form | Optimization Based |

---

# 📊 Key Takeaways

- Simple Linear Regression predicts continuous values.
- Uses one input feature and one target variable.
- Models data using a straight line.
- OLS finds the optimal slope and intercept.
- Slope determines direction and steepness.
- Intercept determines where the line crosses the Y-axis.
- Residuals measure prediction errors.
- SSE is minimized to find the best-fit line.
- Scikit-Learn provides a highly optimized implementation.
- R² Score measures model quality.

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## 📚 Concepts Covered

- Regression Analysis
- Simple Linear Regression
- OLS (Ordinary Least Squares)
- Slope and Intercept
- Residuals
- SSE
- R² Score
- Linear Models
- Model Evaluation
- Machine Learning Foundations
Video Link : https://youtu.be/UZPfbG0jNec
