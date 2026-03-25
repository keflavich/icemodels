# Reusable Axes Feature for plot_stellar_seds

## Summary

The `plot_stellar_seds` function now supports reusing existing figure and axes, allowing you to overlay multiple scenarios on the same plot. This is useful for comparing:
- Baseline stellar SEDs vs extincted SEDs
- No ice vs with ice absorption
- Different combinations of extinction and ice

## New Parameters

- **fig** (matplotlib.figure.Figure, optional): Existing figure to plot on. If None, a new figure is created.
- **axes** (array of matplotlib.axes.Axes, optional): Existing axes [main_ax, zoom_ax1, zoom_ax2, ...]. Must match the number of filters.
- **label** (str, optional): Label prefix for legend entries. Useful for distinguishing scenarios (e.g., 'Baseline', 'With extinction').

## How It Works

When `fig` and `axes` are provided:
1. The function reuses the existing figure and axes instead of creating new ones
2. Filter transmission curves and axis formatting are only applied to new axes
3. New data is plotted on top of existing data
4. The legend is updated to include all plotted lines
5. Y-axis limits can be adjusted if renormalize_insets is used

## Example: Comparing Three Scenarios

```python
from dust_extinction.averages import CT06_MWGC
from icemodels import plot_stellar_seds, read_ocdb_file
import astropy.units as u
import numpy as np

# Setup
ice_table = read_ocdb_file('path/to/CO2_ice.txt')
filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']
xarr = np.linspace(1.5*u.um, 5.2*u.um, 50000)

# Scenario 1: 4000K baseline (no extinction, no ice)
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=filters,
    xarr=xarr,
    label='Baseline'
)

# Scenario 2: 4000K with extinction, no ice (reuse axes)
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=filters,
    xarr=xarr,
    extinction_Av=17.0,
    extinction_curve=CT06_MWGC(),
    fig=fig,
    axes=axes,
    label='With extinction'
)

# Scenario 3: 4000K with extinction + ice (reuse axes again)
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=filters,
    xarr=xarr,
    ice_model_table=ice_table,
    ice_column=1e18 * u.cm**-2,
    molecular_weight=44*u.Da,
    extinction_Av=17.0,
    extinction_curve=CT06_MWGC(),
    fig=fig,
    axes=axes,
    label='Extinction + ice'
)

plt.show()
```

## What Gets Plotted

The above example will show:
- **Baseline: 4000 K** - Clean stellar SED (top curve)
- **With extinction: 4000 K** - Extincted SED (middle curve)
- **Extinction + ice: 4000 K** - Extincted SED (solid line)
- **4000 K (CO2)** - Ice-absorbed version of extincted SED (dashed line)

All four curves appear on the same plot, making it easy to see the cumulative effects of extinction and ice absorption.

## Technical Details

### Axes Detection
The function determines if axes are new or reused by checking if `axes` parameter is None.

### What's Only Done for New Axes
- Creating GridSpec layout
- Setting axis labels and titles
- Adding filter transmission curves
- Setting x-axis limits
- Adding shaded filter regions on main plot
- Calling tight_layout()

### What's Done Every Time
- Plotting SEDs for specified temperatures
- Calculating ice absorption (if requested)
- Applying extinction (if requested)
- Updating legend with new lines
- Tracking data for renormalize_insets

## Use Cases

1. **Comparing extinction levels**: Plot same temperature with different A_V values
2. **Extinction vs no extinction**: Show baseline and extincted SEDs side-by-side
3. **Ice composition effects**: Show SEDs with different ice species
4. **Full radiative transfer path**: Demonstrate stellar → dust → ice → observer

## See Also

- `notebooks/plot_stellar_seds_demo.ipynb` - Example 7 demonstrates this feature
- `icemodels/README_plot_seds.md` - Full documentation with more examples
