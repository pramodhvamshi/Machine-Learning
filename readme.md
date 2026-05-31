# Day 15 - Working with CSV Files

## Quick Revision Notes

* **`pd.read_csv('file_path')`** → Core Pandas function used to load locally stored Comma-Separated Values (CSV) files into a DataFrame.
* **Loading from URL** → Fetching data directly from a server or remote link by passing the URL string directly into `pd.read_csv()`.
* **`sep` parameter** → Overrides the default comma delimiter; used as `sep='\t'` to successfully load Tab-Separated Values (TSV) files.
* **`names` parameter** → Accepts a custom list of strings to manually assign column names when the incoming dataset lacks a header row.
* **`index_col` parameter** → Specifies an existing column (by name or position) to be used as the DataFrame index instead of the default auto-generated integers.
* **`header` parameter** → Controls which row is treated as column names; setting `header=0` or `header=1` explicitly shifts the start of the data when row formatting is broken.
* **`usecols` parameter** → Optimizes memory usage by loading only a specific subset of specified columns from the file into the DataFrame.
* **`squeeze=True` parameter** → Forces Pandas to return a single-column dataset as a Series object instead of a 2D DataFrame.
* **`skiprows` parameter** → Skips specific rows during loading; can accept an explicit list of indices or an executable lambda function to filter out specific rows.
* **`nrows` parameter** → Limits the total number of rows imported into memory, acting as a crucial tool to quickly sample massive multi-gigabyte datasets.
* **`encoding` parameter** → Changes the standard UTF-8 decoder when text files throw parsing errors; commonly set to `'latin-1'` for datasets with special characters or unique language formats.
* **`error_bad_lines=False` / `on_bad_lines='skip'` parameter** → Forces the parser to automatically skip corrupted lines (rows with extra delimiters or structural errors) instead of crashing execution.
* **`dtype` parameter** → Explicitly forces columns into desired data types at load time (e.g., `dtype={'column_name': int}`) to reduce wasteful floating-point memory allocations.
* **`parse_dates` parameter** → Instructs Pandas to parse specified text strings directly into standard `datetime64` objects instead of treating them as generic objects/strings.
* **`converters` parameter** → Accepts a dictionary mapping column names to custom Python functions to clean, reformat, or transform data strings immediately during the ingestion process.
* **`na_values` parameter** → Defines a custom list of specific string placeholders (like hyphens, symbols, or custom flags) to be explicitly parsed and treated as standard `NaN` missing values.
* **`chunksize` parameter** → Yields an iterable object that splits an overwhelmingly large dataset into manageable block sizes of a specific row count, preventing out-of-memory errors on limited hardware.

---

## Key Takeaways

* **Memory Preservation First:** Never load multi-gigabyte files entirely at once; utilize `nrows` for initial structural inspection, `usecols` to filter out garbage columns, and `chunksize` to build safe, iterative data loops.
* **Don't Clean Data Later, Do It on Ingestion:** Leverage parameters like `parse_dates`, `dtype`, `na_values`, and `converters` to execute your initial data type corrections and structural text transformations immediately during the parsing phase.
* **Handling Corrupted Text:** Structural failures during reading are almost always caused by incorrect delimiter assumptions (`sep`), mismatched character encodings (`encoding`), or uneven row dimensions that need to be skipped (`error_bad_lines`).



Video link : https://www.youtube.com/watch?v=a_XrmKlaGTs

Books dataset link : http://www2.informatik.uni-freiburg.de/~cziegler/BX/
