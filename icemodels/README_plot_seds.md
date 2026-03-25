# Stellar SED Plotting

The `plot_stellar_seds` function provides visualization of stellar spectral energy distributions (SEDs) using synphot/stsynphot stellar atmosphere grids, with support for:

- Multiple stellar temperatures overlaid
- JWST filter transmission profiles
- Ice absorption effects
- Zoom-in subplots for each filter

## Function Signature

```python
plot_stellar_seds(temperatures, filters, xarr=None, ice_model_table=None, 
                  ice_column=None, molecular_weight=None, ice_labels=None,
                  figsize=None, color_cycle=None, show_ice_absorbed=True,
                  renormalize_insets=False, extinction_Av=None, 
                  extinction_curve=None)
```

## Parameters

- **temperatures** (float or array-like): Stellar effective temperature(s) in Kelvin. Can be a single value or list of multiple temperatures to overlay.

- **filters** (list of str): List of filter IDs (e.g., `['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']`). Each filter gets its own zoom subplot.

- **xarr** (astropy.units.Quantity, optional): Wavelength array for computing the stellar SED. Default: 0.6 to 28 microns with 25000 points.

- **ice_model_table** (astropy.table.Table or list, optional): Ice optical constants table(s) with 'Wavelength' and 'k' columns. Can be a single table or list of tables for multiple ice species.

- **ice_column** (float/Quantity or list, optional): Ice column density in molecules/cm². Can be a single value or list for multiple ices.

- **molecular_weight** (astropy.units.Quantity or list, optional): Molecular weight of ice composition (e.g., 44*u.Da for CO2). Can be single value or list.

- **ice_labels** (list of str, optional): Labels for ice species in legend. Default: ["Ice 1", "Ice 2", ...].

- **figsize** (tuple, optional): Figure size (width, height) in inches. Auto-calculated if None.

- **color_cycle** (list, optional): Colors for different temperatures. Uses matplotlib default if None.

- **show_ice_absorbed** (bool, optional): Whether to show ice-absorbed SEDs. Default: True.

- **renormalize_insets** (bool, optional): If True, adjust y-axis limits of insets to match data range. Default: False.

- **extinction_Av** (float, optional): Visual extinction A_V in magnitudes. If provided, extinction is applied to stellar SEDs.

- **extinction_curve** (dust_extinction model, optional): Extinction curve from dust_extinction package (e.g., CT06_MWGC()). Defaults to CT06_MWGC() if extinction_Av is provided.

## Returns

- **fig**: matplotlib Figure object
- **axes**: Array of axes [main_ax, zoom_ax1, zoom_ax2, ...]

## Plot Layout

The function creates a figure with:
- **Top row**: One large plot showing full SEDs across all wavelengths with shaded regions indicating filter coverage
- **Bottom row**: N small subplots (one per filter) showing zoomed-in views with transmission profiles superposed

## Examples

### Basic Usage - Multiple Temperatures

```python
from icemodels import plot_stellar_seds

temperatures = [3000, 4000, 5000, 6000]
filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W', 'JWST/MIRI.F1000W']

fig, axes = plot_stellar_seds(temperatures=temperatures, filters=filters)
plt.show()
```

### With Ice Absorption

```python
from icemodels import plot_stellar_seds, read_ocdb_file, optical_constants_cache_dir
import astropy.units as u

# Load CO2 ice optical constants
# Using Gerakines CO2 data at 25K  
ice_file = f'{optical_constants_cache_dir}/55_CO2_(1)_25K_Gerakines.txt'
ice_table = read_ocdb_file(ice_file)

fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W'],
    ice_model_table=ice_table,
    ice_column=1e19*u.cm**-2,  # ice column density
    molecular_weight=44*u.Da
)
plt.show()
```

### Custom Wavelength Range

```python
import numpy as np

# High resolution in the NIR
xarr_custom = np.linspace(1.5*u.um, 5.0*u.um, 50000)

fig, axes = plot_stellar_seds(
    temperatures=[3500, 4500, 5500],
    filters=['JWST/NIRCam.F182M', 'JWST/NIRCam.F212N', 'JWST/NIRCam.F444W'],
    xarr=xarr_custom
)
plt.show()
```

### With Interstellar Extinction

```python
from dust_extinction.averages import CT06_MWGC

fig, axes = plot_stellar_seds(
    temperatures=[3000, 4000, 5000],
    filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W', 'JWST/MIRI.F1000W'],
    extinction_Av=17.0,  # 17 magnitudes of visual extinction
    extinction_curve=CT06_MWGC()  # Chiar & Tielens 2006 extinction curve
)
plt.show()
```

### Combined Extinction and Ice Absorption

```python
from dust_extinction.averages import CT06_MWGC
from icemodels import plot_stellar_seds, read_ocdb_file
import astropy.units as u

# Load ice data
ice_table = read_ocdb_file('path/to/CO2_ice.txt')

fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W'],
    ice_model_table=ice_table,
    ice_column=1e19*u.cm**-2,
    molecular_weight=44*u.Da,
    extinction_Av=17.0,
    extinction_curve=CT06_MWGC()
)
plt.show()
```

### Comparing Multiple Scenarios on the Same Plot

You can reuse existing axes to compare multiple scenarios (e.g., with/without extinction/ice):

```python
from dust_extinction.averages import CT06_MWGC
from icemodels import plot_stellar_seds, read_ocdb_file
import astropy.units as u

# Load ice data
ice_table = read_ocdb_file('path/to/CO2_ice.txt')
filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']

# Scenario 1: Baseline (no extinction, no ice)
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=filters,
    label='Baseline'
)

# Scenario 2: With extinction (reuse same axes)
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=filters,
    extinction_Av=17.0,
    extinction_curve=CT06_MWGC(),
    fig=fig,
    axes=axes,
    label='With extinction'
)

# Scenario 3: Extinction + ice (reuse again)
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=filters,
    ice_model_table=ice_table,
    ice_column=1e18*u.cm**-2,
    molecular_weight=44*u.Da,
    extinction_Av=17.0,
    extinction_curve=CT06_MWGC(),
    fig=fig,
    axes=axes,
    label='Extinction + ice'
)
plt.show()
```

### Many Filters

```python
many_filters = [
    'JWST/NIRCam.F182M',
    'JWST/NIRCam.F212N',
    'JWST/NIRCam.F300M',
    'JWST/NIRCam.F444W',
    'JWST/MIRI.F770W',
    'JWST/MIRI.F1000W'
]

fig, axes = plot_stellar_seds(
    temperatures=[3000, 6000],
    filters=many_filters,
    figsize=(20, 8)  # Custom size for many subplots
)
plt.show()
```

## Filter Coverage Display

The function automatically:
1. Retrieves transmission data from the SVO Filter Profile Service
2. Identifies the 50% transmission range for each filter
3. Shows this range as shaded regions on the main plot
4. Creates zoom plots centered on each filter's wavelength range
5. Overlays the transmission profile on each zoom plot

## Ice Absorption

When ice parameters are provided:
- Bare stellar SEDs are shown as solid lines
- Ice-absorbed SEDs are shown with different linestyles (dashed, dash-dot, dotted)
- Multiple ice species can be shown simultaneously
- Both are normalized to the same peak for easy comparison
- The absorption calculation uses the `absorbed_spectrum` function from icemodels.core

## Interstellar Extinction

When extinction parameters are provided:
- Extinction is applied to stellar SEDs using dust_extinction models
- Default extinction curve is CT06_MWGC (Chiar & Tielens 2006)
- Extinction amount specified via A_V in magnitudes
- Can be combined with ice absorption to show both effects
- Label indicates the A_V value: e.g., "4000 K (Av=17.0)"

## Notes

- SEDs are normalized to peak=1 for easier comparison across different temperatures
- The atmosphere models use Kurucz and Phoenix stellar atmosphere grids via synphot/stsynphot
- Filter transmission profiles are retrieved from the SVO Filter Profile Service
- Ice absorption is computed using the full radiative transfer through the ice layer
- Extinction curves from dust_extinction expect wavelength range roughly 1.2-27 μm
- When using extinction, ensure dust_extinction is installed: `pip install dust_extinction`

## See Also

- `absorbance_in_filters.py`: Related ice absorption calculations
- `stellar_colors_in_filters.py`: Compute stellar magnitudes in filters
- Jupyter notebook: `notebooks/plot_stellar_seds_demo.ipynb`
- Example script: `examples/plot_seds_examples.py`
