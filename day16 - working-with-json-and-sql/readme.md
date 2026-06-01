# Day 16 - Working with JSON and SQL Datasets

## Quick Revision Notes

### 1. Working with JSON (JavaScript Object Notation)

* **`pd.read_json('file_path_or_url')`** → Core Pandas function used to parse semi-structured JSON data into a clean, tabbed 2D DataFrame.
* **Loading via API / Remote URL** → Directly passes a live endpoint URL (e.g., a real-time foreign currency exchange rate API) into `pd.read_json()` to stream server-side responses seamlessly into memory.
* **`nrows` parameter** → Constraints the initial row parsing count; highly useful for previewing structural nesting configurations inside enormous JSON text documents.
* **`chunksize` parameter** → Creates an iterable pipeline for massive JSON documents, allowing processing of structured record batches without triggering machine RAM exhaustion.


### 2. Working with SQL Datasets

* **Relational Database Extraction Workflow** → The end-to-end engineering sequence used to establish a localized database connection, execute raw queries, and convert production relational records directly into a Pandas training matrix.
* **`mysql.connector.connect()`** → Establishes a communication bridge from a Python environment to an active MySQL server using parameters like `host`, `user`, `password`, and targeted `database`.
* **Database Target Initialization** → Passing local loopback values (`host='localhost'`) and structural administrative details (`user='root'`) to bind standard web stack database engines (e.g., XAMPP Control Panel) to Python.
* **`pd.read_sql_query('SQL_QUERY', connection_object)`** → Executes an explicit, optimized database query (e.g., `SELECT * FROM table`) over an open pipeline and instantly structured the resulting query matrix into a standard Pandas DataFrame.
* **In-Database Data Filtering** → Leveraging explicit relational constraints inside your string (such as `WHERE country_code = 'IND'` or `WHERE life_expectancy > 60`) to strip out unneeded attributes *before* loading rows into system RAM.
* **`index_col` parameter** → Overrides auto-generated, sequential line indexation during relational translation by elevating a targeted database primary key (like `city_id`) as the DataFrame index tracker.
* **`chunksize` parameter** → Instructs Pandas to fetch query results iteratively in discrete row intervals, providing a key stabilization method when pulling massive analytical data scales out of database servers.

---

## Key Takeaways

* **Server Agnostic API Parsing:** `pd.read_json()` natively processes cross-language architectural responses from network APIs, removing the need for manual text mapping or decoding tricks.
* **Pre-Filter at Database Level:** Always apply SQL aggregation operators (`WHERE`, `LIMIT`, `JOIN`) directly inside the text string parameter of `pd.read_sql_query()` rather than fetching an entire table to filter in Pandas; this keeps data transfers minimal and prevents your environment from stalling.
* **Bridge Architecture Requirements:** Transferring tables from a live database server requires a physical communication engine (`mysql-connector-python`) to establish the baseline driver connection, paired with an analytical parsing loop (`pd.read_sql_query`) to turn query returns into data structures.

Video link: https://www.youtube.com/watch?v=fFwRC-fapIU

Xampp download link: https://www.apachefriends.org/index.html
World dataset : https://www.kaggle.com/busielmorley/worldcities-pop-lang-rank-sql-create-tbls?select=world.sql
Pandas read_json documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_json.html
Pandas read_sql_query documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html#pandas.read_sql_query
