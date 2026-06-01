# Day 21 - Bivariate & Multivariate Analysis (Exploring Feature Interactions)

## 📌 Overview

After understanding individual features through Univariate Analysis, the next step is to explore relationships between multiple variables.

This notebook focuses on:

* Numerical vs Numerical Analysis
* Categorical vs Numerical Analysis
* Categorical vs Categorical Analysis
* Multivariate Analysis using multiple dimensions simultaneously

Understanding these relationships helps identify strong predictors, hidden patterns, feature interactions, and potential decision boundaries before model building.

---

# 🚀 Quick Revision Notes

## 1. Categorical vs Numerical Analysis

### Bar Plot

```python
sns.barplot(
    data=df,
    x='cat_col',
    y='num_col',
    hue='group_col'
)
```

Displays the average value of a numerical feature across different categories.

Useful for:

* Comparing category means
* Understanding category influence
* Tracking a third variable through `hue`

---

### Box Plot

```python
sns.boxplot(
    data=df,
    x='cat_col',
    y='num_col'
)
```

Compares numerical distributions across categories.

Shows:

* Median
* Quartiles
* Spread
* Outliers

---

### KDE Distribution Comparison

```python
sns.kdeplot(
    df[df['cat_col'] == 'A']['num_col']
)
```

Compares numerical distributions across different categories.

Useful for identifying class separation.

---

## 2. Categorical vs Categorical Analysis

### Cross Tabulation

```python
pd.crosstab(
    df['cat_col1'],
    df['cat_col2']
)
```

Creates a contingency table showing category frequencies.

Example:

| Pclass | Survived=0 | Survived=1 |
| ------ | ---------- | ---------- |
| 1      | 80         | 136        |
| 2      | 97         | 87         |
| 3      | 372        | 119        |

---

### Heatmap

```python
sns.heatmap(
    pd.crosstab(
        df['cat_col1'],
        df['cat_col2']
    )
)
```

Transforms categorical frequencies into a color-coded matrix.

Useful for quickly spotting patterns and concentrations.

---

## 3. Numerical vs Numerical Analysis

### Scatter Plot

```python
sns.scatterplot(
    data=df,
    x='num_col1',
    y='num_col2'
)
```

Displays relationships between two continuous variables.

Useful for identifying:

* Trends
* Clusters
* Correlations
* Outliers

---

### Scatter Plot with Additional Dimensions

```python
sns.scatterplot(
    data=df,
    x='num_col1',
    y='num_col2',
    hue='cat_col',
    style='cat_col2'
)
```

Adds:

* Color (`hue`)
* Marker style (`style`)

to visualize multiple variables simultaneously.

---

### Line Plot

```python
sns.lineplot(
    data=df,
    x='num_col1',
    y='num_col2'
)
```

Useful for:

* Time-series analysis
* Trend visualization
* Sequential relationships

---

## 4. Multivariate Analysis

### Pair Plot

```python
sns.pairplot(
    data=df,
    hue='target_col'
)
```

Creates:

* Scatter plots for all numerical feature combinations
* Histograms/KDE plots on the diagonal
* Color-coded class separation

One of the most powerful exploratory visualization tools.

---

# 🔍 Technical Workflow Analysis

## Step 1: Numerical-Numerical Relationships

The notebook first explores interactions between numerical variables.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

sns.scatterplot(
    data=df,
    x='Age',
    y='Fare',
    hue='Survived',
    style='Sex'
)

plt.show()
```

### Insights

This visualization helps identify:

* Clusters
* Survival patterns
* Gender-based separation
* Relationships between Age and Fare

---

## Step 2: Categorical Influence on Numerical Features

### Bar Plot

```python
sns.barplot(
    data=df,
    x='Pclass',
    y='Fare',
    hue='Survived'
)

plt.show()
```

Used to compare average fare values across passenger classes.

---

### KDE Distribution Comparison

```python
sns.kdeplot(
    df[df['Survived'] == 0]['Age'],
    label='Died',
    fill=True
)

sns.kdeplot(
    df[df['Survived'] == 1]['Age'],
    label='Survived',
    fill=True
)

plt.show()
```

This helps determine whether Age influences survival outcomes.

---

## Step 3: Categorical-Categorical Analysis

### Cross Tabulation

```python
cross_tab = pd.crosstab(
    df['Pclass'],
    df['Survived']
)

print(cross_tab)
```

Creates a frequency matrix between two categorical features.

---

### Heatmap Visualization

```python
sns.heatmap(
    cross_tab,
    annot=True,
    fmt='d',
    cmap='YlGnBu'
)

plt.show()
```

Highlights category interactions using color intensity.

---

# 📊 Understanding Analysis Types

## Univariate Analysis

Studies:

```text
One Variable
```

Examples:

* Age Distribution
* Fare Distribution
* Passenger Class Counts

---

## Bivariate Analysis

Studies:

```text
Two Variables
```

Examples:

* Age vs Fare
* Age vs Survival
* Pclass vs Survival

Questions Answered:

* Is there a relationship?
* Is the relationship strong or weak?

---

## Multivariate Analysis

Studies:

```text
Three or More Variables
```

Examples:

* Age + Fare + Survival
* Age + Sex + Survival
* Pclass + Fare + Survival

Questions Answered:

* How do variables interact together?
* Are there hidden patterns across multiple dimensions?

---

# 🎯 Key Takeaways

### ✅ Scatter Plots Reveal Relationships

Use scatter plots to identify:

* Trends
* Correlations
* Clusters
* Outliers

```python
sns.scatterplot()
```

---

### ✅ KDE Curves Highlight Class Separation

Well-separated KDE curves often indicate strong predictive features.

```python
sns.kdeplot()
```

---

### ✅ Heatmaps Expose Categorical Patterns

Cross-tabulation heatmaps help identify:

* Category dependencies
* Hidden imbalances
* Survival patterns

```python
sns.heatmap()
```

---

### ✅ Bar Plots Compare Numerical Averages

Useful when evaluating the effect of a categorical variable on a numerical variable.

```python
sns.barplot()
```

---

### ✅ Pair Plots Provide Complete Feature Exploration

One command can generate dozens of feature interaction visualizations.

```python
sns.pairplot()
```

This is especially useful before feature engineering and model selection.

---

# 🛠️ Technologies Used

* Python
* Pandas
* Seaborn
* Matplotlib
* Jupyter Notebook

---

# 📚 Learning Outcome

By completing this notebook, you will understand:

* How to perform Bivariate Analysis
* How to perform Multivariate Analysis
* How to compare categorical and numerical variables
* How to visualize feature relationships
* How to identify clusters and decision boundaries
* How to discover hidden patterns within datasets
* How to prepare features for machine learning models
