#!/usr/bin/env python3
"""
Robust solution for reading the table from
https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html
with multiple fallback methods and better error handling
"""

import pandas as pd
from astropy.table import Table
from astropy.io import ascii
import requests
from bs4 import BeautifulSoup
import numpy as np
import time

def read_aanda_table_robust(url='https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html'):
    """
    Read the ice spectroscopy data table from A&A journal using multiple methods.

    Parameters
    ----------
    url : str
        URL to the A&A journal table (default is the T8 table from aa44032-22)

    Returns
    -------
    astropy.table.Table
        Well-formatted table with the ice spectroscopy data
    """

    methods = [
        ("Pandas read_html", _method_pandas),
        ("Astropy Table.read with html format", _method_astropy_html),
        ("Astropy ascii.read with html format", _method_astropy_ascii),
        ("Manual BeautifulSoup parsing", _method_beautifulsoup),
        ("Astropy with custom headers", _method_astropy_custom)
    ]

    for method_name, method_func in methods:
        print(f"\n=== Trying {method_name} ===")
        try:
            result = method_func(url)
            if result is not None:
                print(f"SUCCESS with {method_name}!")
                return result
        except Exception as e:
            print(f"{method_name} failed: {e}")
            time.sleep(1)  # Brief pause between attempts

    raise RuntimeError("All methods failed to read the table")

def _method_pandas(url):
    """Method 1: Use pandas read_html"""
    # Add headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    pd_tables = pd.read_html(url, header=0, attrs={'class': 'fm'})

    if len(pd_tables) == 0:
        raise ValueError("No tables found")

    return _convert_pandas_to_astropy(pd_tables[0], url)

def _method_astropy_html(url):
    """Method 2: Use astropy Table.read with html format"""
    table = Table.read(url, format='html')
    table.meta['source'] = 'A&A journal'
    table.meta['url'] = url
    return table

def _method_astropy_ascii(url):
    """Method 3: Use astropy ascii.read with html format"""
    table = ascii.read(url, format='html')
    table.meta['source'] = 'A&A journal'
    table.meta['url'] = url
    return table

def _method_beautifulsoup(url):
    """Method 4: Manual parsing with BeautifulSoup"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table with class 'fm'
    table_elem = soup.find('table', class_='fm')
    if not table_elem:
        raise ValueError("Table with class 'fm' not found")

    # Extract headers
    header_row = table_elem.find('tr')
    if not header_row:
        raise ValueError("No header row found")

    headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

    # Extract data rows
    rows = table_elem.find_all('tr')[1:]  # Skip header row
    data = []

    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) > 0:
            row_data = [cell.get_text(strip=True) for cell in cells]
            # Pad row if needed
            while len(row_data) < len(headers):
                row_data.append('')
            data.append(row_data[:len(headers)])

    # Create DataFrame and convert
    df = pd.DataFrame(data, columns=headers)
    return _convert_pandas_to_astropy(df, url)

def _method_astropy_custom(url):
    """Method 5: Astropy with custom parsing parameters"""
    table = Table.read(url, format='html', htmldict={'attrs': {'class': 'fm'}})
    table.meta['source'] = 'A&A journal'
    table.meta['url'] = url
    return table

def _convert_pandas_to_astropy(df, url):
    """Convert pandas DataFrame to astropy Table with proper cleaning"""
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Clean up the data
    # Remove rows that are section headers or completely empty
    mask = df.iloc[:, 0].notna() & (df.iloc[:, 0] != 'Pure ices') & (df.iloc[:, 0] != '')
    df_clean = df[mask].copy()

    # Reset index
    df_clean = df_clean.reset_index(drop=True)

    print(f"Cleaned DataFrame shape: {df_clean.shape}")

    if df_clean.empty:
        raise ValueError("No data rows found after cleaning")

    # Clean column names
    column_mapping = {
        'Nicea (cm–2)': 'N_ice_cm2',
        'Nicea(cm–2)': 'N_ice_cm2',
        'Resolution (cm−1)': 'Resolution_cm1',
        'Resolution (cm-1)': 'Resolution_cm1',
        'Temperature (K)/UV radiation (time)': 'Temperature_UV',
        'Substrate/nsubstrate': 'Substrate_n'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df_clean.columns:
            df_clean = df_clean.rename(columns={old_name: new_name})

    # Convert to astropy Table
    astropy_table = Table()

    for col_name in df_clean.columns:
        series = df_clean[col_name]

        # Handle different data types
        try:
            if pd.api.types.is_numeric_dtype(series):
                astropy_table[col_name] = series.values
            else:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(series, errors='coerce')
                if not numeric_series.isna().all():
                    astropy_table[col_name] = numeric_series.values
                else:
                    # Keep as string
                    astropy_table[col_name] = series.fillna('').astype(str).values
        except:
            astropy_table[col_name] = series.fillna('').astype(str).values

    # Add metadata
    astropy_table.meta['source'] = 'A&A journal'
    astropy_table.meta['url'] = url
    astropy_table.meta['description'] = 'Ice spectroscopy data from A&A 668, A44 (2022)'

    return astropy_table

if __name__ == "__main__":
    # Test the function
    url = 'https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html'

    try:
        table = read_aanda_table_robust(url)

        print(f"\n=== FINAL SUCCESS! ===")
        print(f"Final table shape: {table.shape}")
        print(f"Column names: {table.colnames}")
        print("\nFirst 10 rows:")
        print(table[:10])

        print(f"\nMetadata:")
        for key, value in table.meta.items():
            print(f"  {key}: {value}")

        print(f"\nSample of data types:")
        for col in table.colnames[:4]:  # Show first 4 columns
            print(f"  {col}: {type(table[col][0])}")

    except Exception as e:
        print(f"All methods failed. Final error: {e}")
        import traceback
        traceback.print_exc()
