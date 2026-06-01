# Day 17 - Fetching Data from API to DataFrame

## Quick Revision Notes

* **API (Application Programming Interface)** → A software intermediary that allows two applications to talk to each other, universally serving as the primary bridge to fetch live, dynamic data from a server.
* **`requests` Library** → The standard Python package used to send HTTP requests to a server and retrieve raw response data.
* **`requests.get('URL')`** → Sends an HTTP GET request to the specified API endpoint URL and returns a server response object.
* **`.json()` Method** → A built-in method of the response object that parses the raw JSON string payload directly into a native Python dictionary or list format.
* **Nested Dictionary Extraction** → The process of digging into multi-layered JSON keys (e.g., `response['results'][0]['name']`) to isolate the exact list of records needed for tabular layout.
* **`pd.DataFrame(data_list)`** → Directly converts a clean, flat list of dictionaries into a standard 2D Pandas DataFrame.
* **Pagination & Loop Controls** → Using a `for` loop to append page numbers dynamically to the API query string (e.g., `?page=i`) to extract thousands of rows across multiple server pages.
* **Accumulator Pattern** → Initializing an empty list (`all_records = []`) to iteratively collect rows returned from successive API calls across multiple pages before converting the final collection into a DataFrame.

---

## Key Takeaways

* **Data Aggregation via Pagination:** Most production APIs limit single-request payloads to save bandwidth. To build a robust training dataset, you must scan the API documentation to locate pagination parameters and script an iterative loop to accumulate rows safely.
* **Flattening Structural Nesting:** Real-world APIs rarely return clean, tabular structures. Data engineering requires parsing the primary response dictionary down to the exact array key that holds the actual core entities before attempting to cast it into a Pandas DataFrame.
* **Handling Connection Overhead:** Unlike reading local files, loading data through APIs incurs network latency; wrapping cross-page loops in robust error-checking logic or standard tracking metrics prevents scripts from quietly failing mid-extraction.


Video Link:

TMDB API : https://developers.themoviedb.org/
RapidAPI : https://rapidapi.com/collection/list-of-free-apis
JSON Viewer: http://jsonviewer.stack.hu/
