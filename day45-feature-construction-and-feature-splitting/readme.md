# Day 45 - Feature Engineering: Feature Construction & Feature Splitting

## 📌 Overview

Feature Engineering is the process of creating meaningful features from existing data to improve machine learning model performance.

Two important techniques are:

- **Feature Construction** → Creating new features from existing ones.
- **Feature Splitting** → Breaking complex features into simpler, meaningful parts.

These techniques help models learn hidden patterns more effectively.

---

## Feature Construction

Feature Construction creates new features by combining or transforming existing columns.

### Example

Titanic Dataset:

```python
SibSp = Number of siblings/spouses
Parch = Number of parents/children
```

Create Family Size:

```python
Family_Size = SibSp + Parch + 1
```

The extra `+1` represents the passenger.

---

### Creating Family Size

```python
X['Family_Size'] = (
    X['SibSp'] +
    X['Parch'] + 1
)
```

Example:

| SibSp | Parch | Family_Size |
|--------|--------|------------|
| 1 | 2 | 4 |
| 0 | 0 | 1 |
| 2 | 1 | 4 |

---

### Individual Traveler Feature

Create a binary feature:

```python
X['Individual_Traveler'] = np.where(
    X['Family_Size'] == 1,
    1,
    0
)
```

Output:

| Family_Size | Individual_Traveler |
|------------|--------------------|
| 1 | 1 |
| 4 | 0 |
| 3 | 0 |

---

## Why Feature Construction?

✅ Captures hidden relationships

✅ Simplifies raw data

✅ Improves model performance

✅ Reduces feature complexity

---

## Removing Redundant Features

After creating:

```python
Family_Size
```

Original columns may become redundant.

```python
X_optimized = X.drop(
    columns=['SibSp','Parch']
)
```

This reduces multicollinearity and unnecessary information.

---

## Feature Splitting

Feature Splitting breaks a complex feature into smaller meaningful components.

---

### Example

Raw Name:

```text
Braund, Mr. Owen Harris
```

Useful information:

```text
Mr.
```

can be extracted separately.

---

## Extracting Titles

```python
df['Title'] = (
    df['Name']
    .str.split(',')
    .str.get(1)
    .str.split('.')
    .str.get(0)
    .str.strip()
)
```

---

### Output

| Name | Title |
|--------|--------|
| Braund, Mr. Owen Harris | Mr |
| Cumings, Mrs. John | Mrs |
| Heikkinen, Miss. Laina | Miss |

---

## Why Feature Splitting?

✅ Converts text into structured information

✅ Extracts hidden categories

✅ Improves interpretability

✅ Creates model-friendly features

---

## Common String Methods

### Split

```python
.str.split()
```

Splits text using a delimiter.

Example:

```python
"Mr. John".split('.')
```

Output:

```python
['Mr', ' John']
```

---

### Get

```python
.str.get()
```

Retrieves a specific element.

```python
.str.get(0)
```

returns first item.

---

### Strip

```python
.str.strip()
```

Removes extra spaces.

---

## Complete Construction Workflow

```python
X['Family_Size'] = (
    X['SibSp'] +
    X['Parch'] + 1
)

X['Individual_Traveler'] = np.where(
    X['Family_Size'] == 1,
    1,
    0
)

X = X.drop(
    columns=['SibSp','Parch']
)
```

---

## Complete Splitting Workflow

```python
df['Title'] = (
    df['Name']
    .str.split(',')
    .str.get(1)
    .str.split('.')
    .str.get(0)
    .str.strip()
)
```

---

## Advantages

### Feature Construction

✅ Creates informative features

✅ Improves predictive power

✅ Uses domain knowledge

---

### Feature Splitting

✅ Extracts useful information from text

✅ Simplifies complex columns

✅ Makes encoding easier

---

## Best Practices

- Use domain knowledge while creating features.
- Avoid random feature combinations.
- Remove redundant source columns.
- Group rare categories into `"Other"`.
- Verify feature importance after engineering.

---

## Key Takeaways

- Feature Construction creates new features from existing columns.
- Feature Splitting extracts useful components from complex features.
- Family Size is a classic example of feature construction.
- Title extraction is a common feature splitting technique.
- Proper feature engineering often improves model accuracy significantly.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn

---

## Concepts Covered

- Feature Engineering
- Feature Construction
- Feature Splitting
- Domain Knowledge
- String Processing
- Family Size Feature
- Title Extraction
- Pandas String Operations
- Data Preprocessing
- Machine Learning Pipelines