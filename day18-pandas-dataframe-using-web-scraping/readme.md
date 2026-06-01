# Day 18 - Web Scraping (BeautifulSoup to Pandas DataFrame)

## 📌 Overview

This notebook demonstrates how to scrape company data from a website using **BeautifulSoup** and convert the extracted information into a structured **Pandas DataFrame** for further analysis.

The project focuses on:

* Fetching webpage content using `requests`
* Parsing HTML using BeautifulSoup
* Handling anti-scraping restrictions with custom User-Agent headers
* Extracting structured information from company listing pages
* Building a Pandas DataFrame from the collected data

---

## 🚀 Quick Revision Notes

### 1. Fetching Webpage Content

```python
requests.get(url, headers=headers).text
```

Retrieves the raw HTML content of a webpage.

Using a custom User-Agent helps prevent HTTP 403 (Forbidden) errors from websites that block automated requests.

### 2. Parsing HTML

```python
BeautifulSoup(html_text, "lxml")
```

Creates a BeautifulSoup object using the high-performance `lxml` parser.

### 3. Extracting Multiple Elements

```python
soup.find_all("tag_name", class_="class_name")
```

Returns all matching HTML elements from the page.

### 4. Cleaning Extracted Text

```python
.text.strip()
```

Removes unwanted spaces and newline characters.

### 5. Multi-Page Scraping

```python
for page in range(1, 34):
```

Loops through multiple pages to collect larger datasets.

### 6. Data Collection

```python
name = []
rating = []
reviews = []
```

Lists are used to store extracted values during iteration.

### 7. Creating a DataFrame

```python
pd.DataFrame({
    "name": name,
    "rating": rating,
    "reviews": reviews
})
```

Converts collected lists into a structured tabular format.

---

# 🔍 Technical Workflow Analysis

## Step 1: Handling Server-Side Blocking

Many websites restrict automated requests.

To mimic a real browser, a custom User-Agent header is included.

```python
headers = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36"
}

webpage = requests.get(
    "https://www.ambitionbox.com/list-of-companies?page=1",
    headers=headers
).text

soup = BeautifulSoup(webpage, "lxml")
```

---

## Step 2: Extracting Company Cards

Instead of extracting data globally from the page, the notebook first identifies the parent container for each company.

```python
company_cards = soup.find_all(
    "div",
    class_="company-content-wrapper"
)
```

This keeps all company-related information grouped together.

---

## Step 3: Extracting Company Information

Each company card is processed individually.

```python
for item in company_cards:

    name.append(
        item.find("h2").text.strip()
    )

    rating.append(
        item.find(
            "p",
            class_="rating"
        ).text.strip()
    )

    reviews.append(
        item.find(
            "a",
            class_="review-count"
        ).text.strip()
    )
```

### Extracted Fields

| Field   | Description            |
| ------- | ---------------------- |
| Name    | Company Name           |
| Rating  | Company Rating         |
| Reviews | Number of User Reviews |

---

# 📊 Final DataFrame

After collecting all values:

```python
df = pd.DataFrame({
    "name": name,
    "rating": rating,
    "reviews": reviews
})
```

Sample Output:

| Name      | Rating | Reviews    |
| --------- | ------ | ---------- |
| Company A | 4.2    | 5k Reviews |
| Company B | 4.0    | 3k Reviews |
| Company C | 4.5    | 8k Reviews |

---

# 🎯 Key Takeaways

### ✅ Use User-Agent Headers

Many websites block automated requests. Adding a browser-like User-Agent helps bypass basic restrictions.

### ✅ Extract Parent Containers First

Always identify the wrapper element before extracting child elements. This keeps data aligned and prevents mismatched records.

### ✅ Clean Data Immediately

Use:

```python
.text.strip()
```

to remove unwanted spaces and newline characters.

### ✅ Store Data in Lists

Collect extracted values in lists before creating a DataFrame.

### ✅ Convert to DataFrame for Analysis

Pandas DataFrames make it easier to clean, analyze, visualize, and export scraped data.

---

## 🛠️ Technologies Used

* Python
* Requests
* BeautifulSoup4
* lxml
* Pandas

---

## 📚 Learning Outcome

By completing this notebook, you will understand:

* How web scraping works
* HTML parsing using BeautifulSoup
* Multi-page data extraction
* Handling anti-scraping mechanisms
* Creating structured datasets using Pandas
* Preparing scraped data for machine learning workflows



Video Link: https://www.youtube.com/watch?v=8NOdgjC1988
