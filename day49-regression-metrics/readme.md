# Day 49 - Regression Performance Evaluation Metrics

## 📌 Overview

Regression Metrics are used to evaluate how well a regression model predicts continuous numerical values. They measure prediction errors and the overall goodness-of-fit of the model.

The most commonly used metrics are:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Adjusted R² Score

---

# 1️⃣ Mean Absolute Error (MAE)

MAE calculates the average absolute difference between actual and predicted values.

### Formula

:contentReference[oaicite:0]{index=0}

### Characteristics

✅ Easy to understand

✅ Same unit as target variable

✅ Less sensitive to outliers

### Example

```text
Actual    = [10, 20, 30]
Predicted = [12, 18, 28]

Errors = [2, 2, 2]

MAE = 2
```

---

# 2️⃣ Mean Squared Error (MSE)

MSE calculates the average squared difference between actual and predicted values.

### Formula

:contentReference[oaicite:1]{index=1}

### Characteristics

✅ Penalizes large errors heavily

✅ Useful when outliers matter

❌ Unit becomes squared

### Example

```text
Errors = [2, 2, 2]

MSE = 4
```

---

# 3️⃣ Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

### Formula

:contentReference[oaicite:2]{index=2}

### Characteristics

✅ Same unit as target variable

✅ Sensitive to outliers

✅ Most commonly used regression metric

---

# 4️⃣ R² Score (Coefficient of Determination)

Measures how much variance in the target variable is explained by the model.

### Formula

:contentReference[oaicite:3]{index=3}

### Interpretation

| R² Value | Meaning |
|-----------|----------|
| 1 | Perfect Model |
| 0 | Same as Mean Prediction |
| < 0 | Worse than Mean Model |

### Example

```text
R² = 0.85
```

Means:

```text
85% of variance explained
```

---

# 5️⃣ Adjusted R²

Adjusted R² improves R² by penalizing unnecessary features.

### Formula

:contentReference[oaicite:4]{index=4}

Where:

- n = Number of observations
- p = Number of features

### Why Needed?

Standard R² always increases when new features are added.

Adjusted R² increases only if the new feature genuinely improves the model.

---

# Implementation

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# MAE
mae = mean_absolute_error(
    y_test,
    y_pred
)

# MSE
mse = mean_squared_error(
    y_test,
    y_pred
)

# RMSE
rmse = np.sqrt(mse)

# R²
r2 = r2_score(
    y_test,
    y_pred
)
```

---

# MAE vs RMSE

| Metric | Outlier Sensitivity |
|----------|-------------------|
| MAE | Low |
| RMSE | High |

### Use MAE When

- Dataset contains outliers
- Equal importance to all errors

### Use RMSE When

- Large errors should be penalized
- Outliers are important

---

# R² vs Adjusted R²

| R² | Adjusted R² |
|------|------------|
| Always Increases | Can Increase or Decrease |
| No Penalty | Penalizes Extra Features |
| Single Feature Models | Multiple Feature Models |

---

# Key Takeaways

- MAE = Average absolute error.
- MSE = Average squared error.
- RMSE = Square root of MSE.
- R² measures variance explained by the model.
- Adjusted R² penalizes unnecessary features.
- MAE is robust to outliers.
- RMSE heavily penalizes large mistakes.
- Always use Adjusted R² for multiple regression models.

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn

---

## 📚 Concepts Covered

- Regression Metrics
- MAE
- MSE
- RMSE
- R² Score
- Adjusted R²
- Model Evaluation
- Error Analysis
- Machine Learning Foundations
Video Link: https://youtu.be/Ti7c-Hz7GSM
