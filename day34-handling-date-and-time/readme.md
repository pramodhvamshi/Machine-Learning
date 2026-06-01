# Day 34 - Feature Engineering: Handling Date & Time Variables

## 📌 Overview

Date and Time features contain valuable information that can significantly improve machine learning model performance. However, models cannot directly understand raw date strings such as `"2026-06-01"` or time values like `"14:35:20"`.

Feature engineering helps transform these timestamps into meaningful numerical and categorical features such as:

- Year
- Month
- Day
- Day of Week
- Quarter
- Weekend Indicator
- Hour
- Minute
- Second
- Elapsed Time

These extracted features help models identify seasonal, weekly, and time-based patterns hidden within the data.

---

# 1️⃣ Converting Strings to Datetime

## Why Conversion is Necessary

When datasets are loaded using Pandas, date columns are usually stored as strings (`object` datatype).

### Example

```python
import pandas as pd

df = pd.read_csv('orders.csv')

print(df.info())
```

Output:

```text
date    object
time    object
```

Since these columns are treated as plain text, date-specific operations cannot be performed.

---

## Using `pd.to_datetime()`

```python
df['date'] = pd.to_datetime(df['date'])
```

After conversion:

```text
date    datetime64[ns]
```

Now Pandas unlocks the powerful `.dt` accessor.

---

## Handling Invalid Dates

```python
df['date'] = pd.to_datetime(
    df['date'],
    errors='coerce'
)
```

### `errors='coerce'`

Invalid dates are converted into:

```text
NaT
```

(Not a Time)

instead of causing program errors.

---

# 2️⃣ The `.dt` Accessor

The `.dt` accessor provides access to various date and time properties.

```python
df['date'].dt
```

It enables extraction of:

- Year
- Month
- Day
- Weekday
- Quarter
- Hour
- Minute
- Second

and many other useful components.

---

# 3️⃣ Extracting Date Components

## Extract Year

```python
df['date_year'] = df['date'].dt.year
```

Example:

```text
2026-06-01 → 2026
```

---

## Extract Month

```python
df['date_month'] = df['date'].dt.month
```

Example:

```text
2026-06-01 → 6
```

---

## Extract Day

```python
df['date_day'] = df['date'].dt.day
```

Example:

```text
2026-06-01 → 1
```

---

## Complete Example

```python
df['date_year'] = df['date'].dt.year
df['date_month'] = df['date'].dt.month
df['date_day'] = df['date'].dt.day
```

---

# 4️⃣ Extracting Weekday Information

Weekdays often contain important behavioral patterns.

Examples:

- Higher shopping activity on weekends
- Increased traffic on weekdays
- Lower demand during specific days

---

## Day of Week

```python
df['date_dow'] = df['date'].dt.dayofweek
```

Mapping:

| Value | Day |
|---------|---------|
| 0 | Monday |
| 1 | Tuesday |
| 2 | Wednesday |
| 3 | Thursday |
| 4 | Friday |
| 5 | Saturday |
| 6 | Sunday |

---

## Day Name

```python
df['date_dow_name'] = df['date'].dt.day_name()
```

Example:

```text
2026-06-01 → Monday
```

---

## Month Name

```python
df['month_name'] = df['date'].dt.month_name()
```

Example:

```text
2026-06-01 → June
```

---

# 5️⃣ Creating Weekend Indicators

Many business problems benefit from knowing whether an event occurred on a weekend.

---

## Weekend Flag

```python
import numpy as np

df['date_is_weekend'] = np.where(
    df['date_dow'].isin([5, 6]),
    1,
    0
)
```

Output:

| Day | Weekend |
|------|---------|
| Monday | 0 |
| Friday | 0 |
| Saturday | 1 |
| Sunday | 1 |

---

# 6️⃣ Extracting Quarters

Businesses often operate using financial quarters.

---

## Quarter Extraction

```python
df['date_quarter'] = df['date'].dt.quarter
```

Mapping:

| Month | Quarter |
|---------|---------|
| Jan-Mar | Q1 |
| Apr-Jun | Q2 |
| Jul-Sep | Q3 |
| Oct-Dec | Q4 |

---

Example:

```text
2026-06-01 → Quarter 2
```

---

# 7️⃣ Handling Time Variables

Time values can reveal important daily patterns.

Examples:

- Peak shopping hours
- Traffic congestion periods
- Website activity spikes

---

## Convert Time Column

```python
df['time'] = pd.to_datetime(df['time'])
```

---

## Extract Hour

```python
df['hour'] = df['time'].dt.hour
```

Example:

```text
14:35:20 → 14
```

---

## Extract Minutes

```python
df['minute'] = df['time'].dt.minute
```

Example:

```text
14:35:20 → 35
```

---

## Extract Seconds

```python
df['second'] = df['time'].dt.second
```

Example:

```text
14:35:20 → 20
```

---

## Complete Time Extraction

```python
df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute
df['second'] = df['time'].dt.second
```

---

# 8️⃣ Calculating Elapsed Time

Raw dates are often less useful than the duration that has passed since that date.

---

## Calculate Days Passed

```python
today = pd.Timestamp.today()

df['days_elapsed'] = (
    today - df['date']
).dt.days
```

---

### Example

```text
Order Date  : 2026-05-20
Current Date: 2026-06-01

Days Elapsed: 12
```

---

## Why Use Elapsed Time?

Instead of:

```text
2026-05-20
```

use:

```text
12 days
```

This provides a meaningful numerical feature for machine learning algorithms.

---

# 🧪 Complete Workflow

```python
import pandas as pd
import numpy as np

df = pd.read_csv('orders.csv')

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Date features
df['date_year'] = df['date'].dt.year
df['date_month'] = df['date'].dt.month
df['date_day'] = df['date'].dt.day

# Weekday features
df['date_dow'] = df['date'].dt.dayofweek
df['date_dow_name'] = df['date'].dt.day_name()

# Weekend indicator
df['date_is_weekend'] = np.where(
    df['date_dow'].isin([5, 6]),
    1,
    0
)

# Quarter
df['date_quarter'] = df['date'].dt.quarter

# Time features
df['time'] = pd.to_datetime(df['time'])

df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute
df['second'] = df['time'].dt.second

# Elapsed time
today = pd.Timestamp.today()

df['days_elapsed'] = (
    today - df['date']
).dt.days
```

---

# 📊 Key Takeaways

### Datetime Conversion

✅ Use `pd.to_datetime()` to convert string dates into datetime objects.

✅ Use `errors='coerce'` to safely handle invalid values.

---

### Date Features

✅ Extract:

- Year
- Month
- Day
- Quarter

to capture seasonal patterns.

---

### Weekday Features

✅ Use:

```python
.dt.dayofweek
.dt.day_name()
```

to identify weekly trends.

---

### Weekend Indicators

✅ Create binary features to distinguish weekdays from weekends.

---

### Time Features

✅ Extract:

- Hour
- Minute
- Second

to capture daily behavior patterns.

---

### Elapsed Time

✅ Prefer duration-based features over raw dates.

✅ Helps models understand chronological distance.

---

# 🚀 Conclusion

Date and Time variables contain hidden seasonal, weekly, and daily patterns that machine learning models cannot directly interpret. By converting raw timestamps into meaningful numerical and categorical features such as year, month, weekday, quarter, hour, and elapsed time, we transform complex temporal information into model-friendly representations.

Effective Date & Time feature engineering often leads to significant improvements in predictive performance and is one of the most valuable preprocessing techniques in real-world machine learning projects.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy

---

## 📚 Concepts Covered

- Feature Engineering
- Date Variables
- Time Variables
- Datetime Conversion
- `pd.to_datetime()`
- `.dt` Accessor
- Day of Week Extraction
- Month Extraction
- Quarter Extraction
- Weekend Detection
- Time Component Extraction
- Elapsed Time Calculation
- Data Preprocessing
Video Link : https://youtu.be/J73mvgG9fFs
