#!/usr/bin/env python3
"""
Final solution for reading the table from
https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html
"""

import pandas as pd
from astropy.table import Table
import numpy as np

def read_aanda_table(url='https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html'):
    """
    Read the ice spectroscopy data table from A&A journal using the optimal parameters.

    Parameters
    ----------
    url : str
        URL to the A&A journal table (default is the T8 table from aa44032-22)

    Returns
    -------
    astropy.table.Table
        Well-formatted table with the ice spectroscopy data
    """

    # Use pandas to read the HTML table (it handles the complex structure better)
    print("Reading table from URL...")
    pd_tables = pd.read_html(url)

    if len(pd_tables) == 0:
        raise ValueError("No tables found at the specified URL")

    # Get the first (and only) table
    df = pd_tables[0]
    print(f"Table shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Clean up the data
    print("Cleaning data...")

    # Remove rows that are section headers or empty
    # The first row appears to be a duplicate header
    mask = df.iloc[:, 0].notna() & (df.iloc[:, 0] != 'Pure ices')
    df_clean = df[mask].copy()

    # Reset index
    df_clean = df_clean.reset_index(drop=True)

    # Clean column names (remove special characters that might cause issues)
    column_mapping = {
        'Nicea (cm–2)': 'N_ice_cm2',
        'Resolution (cm−1)': 'Resolution_cm1',
        'Temperature (K)/UV radiation (time)': 'Temperature_UV',
        'Substrate/nsubstrate': 'Substrate_n'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df_clean.columns:
            df_clean = df_clean.rename(columns={old_name: new_name})

    # Convert to astropy Table using the proper method
    # We need to convert each column separately to handle mixed types properly
    astropy_table = Table()

    for col_name in df_clean.columns:
        # Get the series and handle NaN values
        series = df_clean[col_name]

        # Try to convert to numeric if possible, otherwise keep as string
        try:
            # First check if it's already numeric
            if pd.api.types.is_numeric_dtype(series):
                astropy_table[col_name] = series.values
            else:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(series, errors='coerce')
                if not numeric_series.isna().all():
                    # If some values are numeric, use those
                    astropy_table[col_name] = numeric_series.values
                else:
                    # Keep as string, replacing NaN with empty string
                    astropy_table[col_name] = series.fillna('').astype(str).values
        except:
            # If all else fails, convert to string
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
        table = read_aanda_table(url)

        print(f"\n=== SUCCESS! ===")
        print(f"Final table shape: {table.shape}")
        print(f"Column names: {table.colnames}")
        print("\nFirst 10 rows:")
        print(table[:10])

        print(f"\nMetadata:")
        for key, value in table.meta.items():
            print(f"  {key}: {value}")

        print(f"\nColumn info:")
        table.info()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
