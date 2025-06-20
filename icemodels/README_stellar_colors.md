# Stellar Colors in JWST Filters

This module provides functionality to compute stellar colors (magnitudes) in JWST filters for stellar atmosphere models with varying properties.

## Files

- `stellar_colors_in_filters.py`: Main script that generates the stellar colors table
- `tests/test_stellar_colors.py`: Demonstration script showing how to use the table
- `data/stellar_colors_in_filters.ecsv`: Output table with computed magnitudes
- `notebooks/Stellar_Colors_Demo.ipynb`: Jupyter notebook with interactive analysis and plots

## Usage

### Generating the Stellar Colors Table

```bash
python -m icemodels.stellar_colors_in_filters
```

This will:
- Generate stellar atmosphere models for temperatures from 2000K to 50000K
- Compute magnitudes in all JWST NIRCam and MIRI filters
- Save results to `icemodels/data/stellar_colors_in_filters.ecsv`

### Using the Table

```python
from astropy.table import Table
import os

# Load the table
table_path = 'icemodels/data/stellar_colors_in_filters.ecsv'
tbl = Table.read(table_path)

# Access magnitudes for different stellar types
m_stars = tbl[tbl['spectral_type'] == 'M']
b_stars = tbl[tbl['spectral_type'] == 'B']

# Get specific colors
f212n_f444w_color = tbl['F212N'] - tbl['F444W']
```

### Running the Demo

```bash
python -m icemodels.tests.test_stellar_colors
```

This will create plots showing:
- How colors vary with stellar temperature
- Color-color diagrams for different spectral types

### Interactive Jupyter Notebook

For interactive analysis with plots and detailed exploration:

```bash
jupyter notebook notebooks/Stellar_Colors_Demo.ipynb
```

The notebook includes:
- Data loading and exploration
- Color vs temperature analysis
- Color-color diagrams
- Summary statistics by spectral type
- Interactive temperature range exploration

## Table Structure

The output table contains:

- `temperature`: Effective temperature in Kelvin
- `model_type`: Always "stellar_atmosphere"
- `spectral_type`: Approximate spectral type (O, B, A, F, G, K, M)
- One column per JWST filter with magnitudes (e.g., `F212N`, `F444W`, `F1000W`)

## Example Results

The table includes 38 stellar models spanning:
- **M stars**: 2000-2980K (2 models)
- **K stars**: 3959-4939K (2 models)
- **G stars**: 5918K (1 model)
- **F stars**: 6898K (1 model)
- **A stars**: 7878-9837K (3 models)
- **B stars**: 10816-29429K (20 models)
- **O stars**: 30408-38245K (9 models)

## Key Features

- **Parallel processing**: Uses multiprocessing for efficient computation
- **JWST filters**: Covers all major NIRCam and MIRI filters
- **Spectral coverage**: 2.5-5.0 μm wavelength range
- **Temperature range**: From cool M dwarfs to hot O stars
- **Magnitude system**: Uses JWST zero points for accurate photometry

## Comparison with Ice Absorption

This module is similar to `absorbance_in_filters.py` but focuses on:
- **Stellar atmosphere models** instead of ice absorption
- **Temperature variation** instead of column density
- **Pure stellar colors** without any absorption effects
- **Spectral type classification** for easy identification

## Dependencies

- `astropy`: For units and table handling
- `numpy`: For numerical operations
- `tqdm`: For progress bars
- `astroquery.svo_fps`: For JWST filter data
- `icemodels.atmo_model`: For stellar atmosphere models
- `icemodels.fluxes_in_filters`: For filter convolution