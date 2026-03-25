# Magnitude Calculations in plot_stellar_seds_demo.ipynb

## Summary

The notebook now includes comprehensive magnitude calculations for every example, demonstrating complete worked examples of stellar photometry through ice and dust.

## New Helper Functions

### `compute_magnitudes()`
Computes magnitudes for stellar SEDs with optional ice absorption and extinction.

**Parameters:**
- `temperatures`: Single or list of stellar temperatures (K)
- `filters`: List of JWST filter IDs
- `xarr`: Wavelength array (optional)
- `ice_model_table`: Ice optical constants table(s) (optional)
- `ice_column`: Ice column density(ies) (optional)
- `molecular_weight`: Molecular weight(s) of ice (optional)
- `ice_labels`: Labels for ice species (optional)
- `extinction_Av`: Visual extinction in magnitudes (optional)
- `extinction_curve`: dust_extinction model instance (optional)

**Returns:**
Dictionary with magnitudes for each temperature/scenario and filter.

### `print_magnitudes()`
Pretty-prints magnitude results in a formatted table.

## Examples Enhanced

All 7 examples in the notebook now include magnitude calculations:

### Example 1: Multiple Temperatures
- Computes magnitudes for 3000K, 5800K, and 8000K stars
- Shows how stellar magnitudes change with temperature
- Demonstrates that cooler stars are brighter in infrared filters

**Sample Output:**
```
3000K:
  F212N     : -14.093
  F444W     : -14.739
  F1000W    : -15.015
```

### Example 2: With Ice Absorption
- Shows CO2 and CO ice absorption effects
- CO2 affects F410M (4.27 μm feature): -14.114 → -13.821 (Δm = -0.293)
- CO affects F466N (4.67 μm feature): -13.990 → -13.816 (Δm = -0.174)
- Demonstrates filter-specific ice absorption

### Example 3: Custom Wavelength Range
- Magnitudes for different temperatures with high-resolution wavelength grid
- Shows how wavelength sampling affects results

### Example 4: Many Filters
- Magnitudes across 6 NIRCam filters
- Demonstrates complete spectral energy distribution sampling

### Example 5: With Extinction
- Shows effect of 17 mag Av extinction
- Extinction makes stars fainter at all wavelengths
- More extinction at shorter wavelengths (steeper extinction curve)

**Sample Output:**
```
4000K (Av=17.0):
  F212N     : -12.071  (vs -14.009 unextincted)
  F444W     : -13.150  (vs -14.030 unextincted)
```

### Example 6: Extinction + Ice
- Combined effects of extinction and ice absorption
- Shows how ice features appear on top of extinction

### Example 7: Scenario Comparison
- **Most comprehensive example**
- Compares three scenarios side-by-side:
  1. Baseline (4000K, no extinction, no ice)
  2. With extinction (Av=17 mag)
  3. With extinction + CO2 ice

- **Quantifies individual effects:**
  - Extinction effect: F212N gets -1.938 mag fainter, F444W gets -0.881 mag fainter
  - Ice effect on extincted star: F444W gets additional -0.171 mag from CO2
  - Total effect: Combined dimming from both processes

**Sample Output:**
```
Effect of extinction (Δm = m_baseline - m_extinct):
  F212N     : -1.938 mag
  F444W     : -0.881 mag

Effect of ice on extincted star (Δm = m_extinct - m_extinct+ice):
  F212N     : +0.000 mag
  F444W     : -0.171 mag
```

## Educational Value

These examples demonstrate:

1. **How to compute JWST magnitudes** from stellar models using SVO filter profiles
2. **Ice absorption signatures** in specific filters based on ice composition
3. **Extinction effects** showing wavelength-dependent dimming
4. **Combined radiative transfer** through dust and ice
5. **Filter selection** for detecting specific ice features
6. **Magnitude differences** to quantify absorption strength

## Technical Details

### Magnitude Calculation
Follows the standard formula:
```
m = -2.5 * log10(F_ν / F_0)
```
where:
- F_ν is the flux density in the filter
- F_0 is the zero-point flux from SVO FPS

### Filter Integration
Uses `fluxes_in_filters()` to:
1. Interpolate spectrum onto filter transmission grid
2. Multiply by transmission curve
3. Integrate and normalize

### Zero Points
Retrieved automatically from SVO Filter Profile Service via astroquery.

## Use Cases

These worked examples are useful for:
- **Observers**: Planning JWST observations to detect ice features
- **Students**: Learning stellar photometry and radiative transfer
- **Researchers**: Modeling ice absorption in YSO envelopes
- **Tool developers**: Understanding the icemodels workflow

## Files Modified

- `/orange/adamginsburg/repos/icemodels/notebooks/plot_stellar_seds_demo.ipynb`:
  - Added `compute_magnitudes()` and `print_magnitudes()` helper functions
  - Added magnitude calculation cells after each of the 7 examples
  - Added comprehensive comparison in Example 7 with delta-magnitude analysis
