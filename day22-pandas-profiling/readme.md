# Day 22 - Automated EDA (Pandas Profiling / ydata-profiling)

## 📌 Overview

Exploratory Data Analysis (EDA) is a critical step in every Machine Learning workflow. However, manually inspecting datasets using methods like `df.info()`, `df.describe()`, and custom visualizations can be time-consuming.

This notebook demonstrates how to perform **Automated Exploratory Data Analysis (EDA)** using the **ydata-profiling** library (formerly known as `pandas-profiling`).

With just a few lines of code, an interactive HTML report containing dataset statistics, visualizations, warnings, correlations, and feature interactions can be generated automatically.

---

# 🚀 Quick Revision Notes

## What is Automated EDA?

Automated EDA refers to generating a complete exploratory analysis report using a single command instead of writing multiple visualization and statistical analysis scripts manually.

Benefits include:

* Faster dataset understanding
* Automatic issue detection
* Correlation analysis
* Missing value identification
* Distribution visualization
* Feature interaction discovery

---

## Importing the Library

```python
from ydata_profiling import ProfileReport
```

Imports the automated profiling engine.

> Note: `pandas_profiling` has been renamed to `ydata-profiling`.

---

## Creating a Profile Report

```python
ProfileReport(
    df,
    title="Dataset Profile Report"
)
```

Creates an interactive profiling report from a Pandas DataFrame.

---

## Exporting the Report

```python
profile.to_file("output.html")
```

Exports the report as a standalone HTML file.

The generated file can be opened directly in a web browser.

---

## Report Warnings Section

The report automatically highlights:

* Missing values
* Duplicate records
* High cardinality columns
* Highly correlated features
* Constant features
* Skewed distributions

This helps identify potential data quality issues instantly.

---

## Variables Section

Provides a detailed analysis of every column including:

* Data Type
* Missing Values
* Distinct Values
* Histograms
* Summary Statistics
* Most Frequent Categories

---

## Correlations & Interactions

Automatically generates:

* Pearson Correlation
* Spearman Correlation
* Kendall Correlation
* Phi-K Correlation

along with interactive visualizations and scatter plots.

---

# 🔍 Technical Workflow Analysis

## Step 1: Import Libraries and Load Dataset

```python
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("train.csv")
```

The dataset is loaded into a Pandas DataFrame and the profiling library is imported.

---

## Step 2: Generate the Profile Report

```python
profile = ProfileReport(
    df,
    title="Titanic Dataset EDA Profile"
)
```

This command performs a complete analysis of the dataset including:

* Numerical Variables
* Categorical Variables
* Missing Values
* Correlations
* Interactions
* Duplicates

---

## Step 3: Export the Report

```python
profile.to_file(
    output_file="titanic_eda_report.html"
)
```

Creates a portable HTML report that can be viewed in any browser.

---

# 📊 Key Sections of the Generated Report

## 1. Overview

Provides a quick summary of:

* Number of Rows
* Number of Columns
* Missing Cells
* Duplicate Rows
* Memory Usage

---

## 2. Variables

For each feature, the report shows:

### Numerical Variables

* Mean
* Median
* Standard Deviation
* Minimum
* Maximum
* Quartiles
* Histogram

### Categorical Variables

* Distinct Categories
* Frequency Counts
* Most Common Values

---

## 3. Missing Values

Displays:

* Missing Value Count
* Missing Value Percentage
* Missing Value Visualizations

Helps determine suitable imputation strategies.

---

## 4. Correlation Analysis

Generates interactive correlation matrices using:

### Pearson Correlation

Measures linear relationships.

### Spearman Correlation

Measures monotonic relationships.

### Kendall Correlation

Measures rank-based associations.

### Phi-K Correlation

Works well with mixed variable types.

---

## 5. Interactions

Automatically creates pairwise feature interaction plots.

Useful for discovering hidden relationships between variables.

---

## 6. Duplicates

Identifies duplicate rows and provides statistics on repeated records.

---

## 7. Warnings

One of the most valuable sections of the report.

Automatically flags:

* Missing Values
* Highly Correlated Features
* Constant Features
* High Cardinality Variables
* Skewed Features

These warnings help prioritize data cleaning efforts.

---

# 🎯 Key Takeaways

### ✅ Quickly Understand Unfamiliar Datasets

Automated profiling provides a complete overview without writing extensive analysis code.

---

### ✅ Focus on the Warnings Section

Always review the warnings tab first.

It immediately highlights:

* Missing Data
* Correlated Variables
* Data Quality Issues

---

### ✅ Great for Initial Dataset Exploration

Instead of manually running:

```python
df.shape
df.info()
df.describe()
df.isnull().sum()
```

a single profiling report generates all of this automatically.

---

### ✅ Correlation Analysis Becomes Easier

The report provides multiple correlation methods and visual heatmaps without additional coding.

---

### ✅ Saves Time During EDA

A report that would normally require dozens of analysis commands can be generated in minutes.

---

### ✅ Automated EDA is a Starting Point

While profiling tools are powerful, they should supplement—not replace—manual exploratory analysis and domain-specific feature engineering.

---

# 🛠️ Technologies Used

* Python
* Pandas
* ydata-profiling
* Jupyter Notebook

---

# 📚 Learning Outcome

By completing this notebook, you will understand:

* What Automated EDA is
* How to use ydata-profiling
* How to generate interactive dataset reports
* How to identify missing values and duplicates
* How to analyze correlations automatically
* How to interpret profiling warnings
* How to accelerate the data exploration phase of Machine Learning projects

---

## 📂 Output

The notebook generates:

```text
output.html
```

An interactive HTML dashboard containing a complete exploratory analysis of the dataset.

Video Link:https://www.youtube.com/watch?v=E69Lg2ZgOxg
