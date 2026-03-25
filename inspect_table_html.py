#!/usr/bin/env python3
"""
Inspect the HTML structure of the table to better understand how to parse it
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from astropy.table import Table

# URL of the HTML page containing the table
url = 'https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html'

print("Downloading and inspecting HTML structure...")

# Download the page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all tables
tables = soup.find_all('table')
print(f"Found {len(tables)} table(s) in the HTML")

for i, table in enumerate(tables):
    print(f"\n=== Table {i} ===")
    print(f"Table classes: {table.get('class', [])}")
    print(f"Table id: {table.get('id', 'None')}")

    # Find headers
    headers = table.find_all('th')
    if headers:
        print("Headers found:")
        for j, header in enumerate(headers):
            print(f"  {j}: {header.get_text(strip=True)}")

    # Find rows
    rows = table.find_all('tr')
    print(f"Number of rows: {len(rows)}")

    if len(rows) > 0:
        print("First few rows:")
        for j, row in enumerate(rows[:3]):
            cells = row.find_all(['td', 'th'])
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            print(f"  Row {j}: {cell_texts}")

# Try using pandas to read the HTML tables
print("\n=== Trying pandas.read_html ===")
try:
    pd_tables = pd.read_html(url)
    print(f"Pandas found {len(pd_tables)} table(s)")

    for i, df in enumerate(pd_tables):
        print(f"\nTable {i} shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("First few rows:")
        print(df.head())

        # Try to convert to astropy table
        try:
            astropy_table = Table.from_pandas(df)
            print(f"Successfully converted to astropy Table with shape {astropy_table.shape}")
        except Exception as e:
            print(f"Failed to convert to astropy Table: {e}")

except Exception as e:
    print(f"Pandas read_html failed: {e}")

print("\nInspection complete!")
