# Day 33 - Feature Engineering: Handling Mixed Variables

## 📌 Overview

Real-world datasets often contain **mixed variables**, where numerical values and categorical text are stored together in the same column. Before applying machine learning algorithms, these mixed features must be separated into meaningful numeric and categorical components.

In this lesson, we explore techniques for identifying, parsing, and transforming mixed variables using **Pandas string operations**, **Regular Expressions (Regex)**, and datatype conversion methods.

---

# 1️⃣ What Are Mixed Variables?

Mixed Variables are features that contain both numbers and text within the same column.

### Examples

| Feature | Example Values |
|----------|---------------|
| Cabin | C123, B45, E67 |
| Ticket | A/5 21171, PC 17599, 347082 |

Such columns cannot be directly processed by machine learning models because they contain multiple types of information combined together.

---

# 2️⃣ Common Types of Mixed Variables

## Type 1: Mixed Records

Some rows contain only numbers, while others contain text and numbers.

### Example

```text
21171
347082
A/5 21171
PC 17599
```

This pattern is commonly seen in the **Ticket** feature.

---

## Type 2: Composite Alphanumeric Strings

Every row contains both letters and digits merged together.

### Example

```text
C123
B45
E67
```

This pattern is commonly seen in the **Cabin** feature.

---

# 3️⃣ Important Functions Used

## `.str.extract()`

A Pandas string method that extracts portions of text using Regular Expressions.

```python
df['column'].str.extract(pattern)
```

Useful for separating text and numeric components from mixed variables.

---

## Regex Pattern: Extract Letters

```python
([A-Za-z]+)
```

### Meaning

- `A-Z` → Uppercase letters
- `a-z` → Lowercase letters
- `+` → One or more characters

### Example

```python
"C123" → "C"
"B45"  → "B"
```

---

## Regex Pattern: Extract Numbers

```python
(\d+)
```

### Meaning

- `\d` → Any digit (0-9)
- `+` → One or more digits

### Example

```python
"C123" → 123
"B45"  → 45
```

---

## `pd.to_numeric()`

Converts extracted values into a true numeric datatype.

```python
pd.to_numeric(column, errors='coerce')
```

### Parameter

```python
errors='coerce'
```

Invalid values are converted into:

```python
NaN
```

instead of generating an error.

---

# 🧪 Implementation Workflow

## Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
```

---

## Step 2: Load Dataset

```python
df = pd.read_csv(
    'titanic.csv',
    usecols=['Ticket', 'Cabin', 'Survived']
)
```

---

## Step 3: Inspect Mixed Features

```python
print(df.head(10))
```

Sample output:

```text
Ticket        Cabin
A/5 21171     C123
347082        B45
PC 17599      E67
```

Both columns contain mixed information that must be separated.

---

# 4️⃣ Handling Composite Strings (Cabin)

## Extract Numeric Component

```python
df['Cabin_num'] = df['Cabin'].str.extract('(\d+)')
```

### Example

```text
C123 → 123
B45  → 45
E67  → 67
```

---

## Extract Categorical Component

```python
df['Cabin_cat'] = df['Cabin'].str.extract('([A-Za-z]+)')
```

### Example

```text
C123 → C
B45  → B
E67  → E
```

---

## View Results

```python
print(
    df[['Cabin',
        'Cabin_num',
        'Cabin_cat']].head()
)
```

### Output

| Cabin | Cabin_num | Cabin_cat |
|--------|-----------|-----------|
| C123 | 123 | C |
| B45 | 45 | B |
| E67 | 67 | E |

---

# 5️⃣ Handling Mixed Records (Ticket)

Unlike Cabin, Ticket values have different formats.

### Examples

```text
21171
347082
A/5 21171
PC 17599
```

Some values are purely numeric, while others contain text and numbers.

---

## Extract Ticket Category

```python
df['Ticket_cat'] = df['Ticket'].apply(
    lambda x:
    x.split()[0]
    if not x.isdigit()
    else 'Numeric'
)
```

### Output

| Ticket | Ticket_cat |
|----------|-----------|
| 21171 | Numeric |
| A/5 21171 | A/5 |
| PC 17599 | PC |

---

## Extract Ticket Number

```python
df['Ticket_num'] = df['Ticket'].apply(
    lambda x:
    x.split()[-1]
    if x.split()[-1].isdigit()
    else np.nan
)
```

### Output

| Ticket | Ticket_num |
|----------|-----------|
| 21171 | 21171 |
| A/5 21171 | 21171 |
| PC 17599 | 17599 |

---

## Convert to Numeric Datatype

```python
df['Ticket_num'] = pd.to_numeric(
    df['Ticket_num'],
    errors='coerce'
)
```

This ensures the extracted values can be used in machine learning models.

---

# 📊 Why Separate Mixed Variables?

Machine learning algorithms expect structured numerical or categorical features.

### Incorrect Approach

```python
Cabin = "C123"
```

A model cannot directly understand:

```text
"C123"
```

---

### Correct Approach

```python
Cabin_cat = "C"
Cabin_num = 123
```

Now:

- Cabin_cat can be encoded using One-Hot Encoding.
- Cabin_num can be scaled and used as a numeric feature.

---

# 📌 Key Takeaways

### Mixed Variables

✅ Contain both text and numbers in a single feature.

✅ Common in real-world datasets.

---

### Regular Expressions

✅ `([A-Za-z]+)` extracts letters.

✅ `(\d+)` extracts numbers.

✅ Best tool for separating composite strings.

---

### `.str.extract()`

✅ Simplifies regex-based extraction.

✅ Creates clean independent features.

---

### `pd.to_numeric()`

✅ Converts extracted strings to numbers.

✅ `errors='coerce'` safely handles invalid values.

---

### Best Practice

✅ Always split mixed variables before encoding or scaling.

✅ Treat categorical and numerical components separately.

✅ Ensure numeric parts are converted back to numeric datatypes.

---

# 🚀 Conclusion

Handling mixed variables is an essential feature engineering skill. Real-world datasets frequently contain columns where categorical labels and numeric values are combined. By separating these components using **Regex**, **Pandas string operations**, and **datatype conversions**, we can transform messy data into structured features suitable for machine learning models.

Proper handling of mixed variables leads to cleaner datasets, better feature representations, and improved model performance.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Regular Expressions (Regex)

---

## 📚 Concepts Covered

- Feature Engineering
- Mixed Variables
- Alphanumeric Data
- Regex Extraction
- `.str.extract()`
- Ticket Feature Processing
- Cabin Feature Processing
- Datatype Conversion
- `pd.to_numeric()`
- Data Preprocessing
Video Link : https://youtu.be/9xiX-I5_LQY
