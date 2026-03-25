#!/usr/bin/env python3
"""
FINAL WORKING solution for reading the table from
https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html
"""

import pandas as pd
from astropy.table import Table
import numpy as np

def read_aanda_table(url='https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html'):
    """
    Read the ice spectroscopy data table from A&A journal.

    This is the OPTIMAL solution using pandas.read_html with proper data cleaning.

    Parameters
    ----------
    url : str
        URL to the A&A journal table (default is the T8 table from aa44032-22)

    Returns
    -------
    astropy.table.Table
        Well-formatted table with the ice spectroscopy data
    """

    # Method: Use pandas read_html with proper headers
    print("Reading table using pandas.read_html...")

    # Add user agent to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # Read the HTML table - pandas handles this best
    pd_tables = pd.read_html(url, header=0, attrs={'class': 'fm'})

    if len(pd_tables) == 0:
        raise ValueError("No tables found at the specified URL")

    df = pd_tables[0]
    print(f"Original table shape: {len(df)} rows x {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")

    # Clean up the data
    print("Cleaning data...")

    # Remove rows that are section headers or empty
    # The table has some rows with 'Pure ices' as section dividers
    mask = (df.iloc[:, 0].notna() &
            (df.iloc[:, 0] != 'Pure ices') &
            (df.iloc[:, 0] != '') &
            (df.iloc[:, 0] != 'Sample'))  # Remove duplicate header rows

    df_clean = df[mask].copy().reset_index(drop=True)
    print(f"After cleaning: {len(df_clean)} rows x {len(df_clean.columns)} columns")

    # Clean column names (remove special characters)
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
    print("Converting to astropy Table...")
    astropy_table = Table()

    for col_name in df_clean.columns:
        series = df_clean[col_name]

        # Try to convert numeric columns
        if col_name in ['Thickness (ML)', 'N_ice_cm2', 'Resolution_cm1']:
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                astropy_table[col_name] = numeric_series.values
            except:
                astropy_table[col_name] = series.fillna('').astype(str).values
        else:
            # Keep as string for non-numeric columns
            astropy_table[col_name] = series.fillna('').astype(str).values

    # Add metadata
    astropy_table.meta['source'] = 'A&A journal'
    astropy_table.meta['url'] = url
    astropy_table.meta['paper'] = 'A&A 668, A44 (2022)'
    astropy_table.meta['description'] = 'Ice spectroscopy data compilation'
    astropy_table.meta['table_id'] = 'T8'

    return astropy_table

# Demonstration of the optimal parameters for astropy.table.Table.read()
def demonstrate_optimal_parameters():
    """
    Show the optimal parameters for reading this specific table.
    """
    url = 'https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html'

    print("="*60)
    print("OPTIMAL PARAMETERS FOR ASTROPY.TABLE.TABLE.READ():")
    print("="*60)
    print()
    print("Method 1 (RECOMMENDED - Most robust):")
    print("Use pandas.read_html() first, then convert to astropy.table.Table")
    print()
    print("import pandas as pd")
    print("from astropy.table import Table")
    print()
    print("# Read with pandas")
    print(f"df = pd.read_html('{url}', header=0, attrs={{'class': 'fm'}})[0]")
    print()
    print("# Clean data (remove section headers)")
    print("mask = (df.iloc[:, 0].notna() & (df.iloc[:, 0] != 'Pure ices'))")
    print("df_clean = df[mask].reset_index(drop=True)")
    print()
    print("# Convert to astropy Table")
    print("table = Table.from_pandas(df_clean)  # Note: This may need column-by-column conversion")
    print()
    print("Method 2 (Direct astropy - may have parsing issues):")
    print(f"table = Table.read('{url}', format='html', htmldict={{'attrs': {{'class': 'fm'}}}})")
    print()
    print("Method 3 (Alternative astropy method):")
    print("from astropy.io import ascii")
    print(f"table = ascii.read('{url}', format='html')")

if __name__ == "__main__":
    # Test the function
    url = 'https://www.aanda.org/articles/aa/full_html/2022/12/aa44032-22/T8.html'

    try:
        table = read_aanda_table(url)

        print(f"\n=== SUCCESS! ===")
        print(f"Final table: {len(table)} rows x {len(table.colnames)} columns")
        print(f"Column names: {table.colnames}")
        print("\nFirst 5 rows:")
        for i in range(min(5, len(table))):
            print(f"Row {i}: {[table[col][i] for col in table.colnames[:4]]}...")

        print(f"\nMetadata:")
        for key, value in table.meta.items():
            print(f"  {key}: {value}")

        print("\n" + "="*60)
        demonstrate_optimal_parameters()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
