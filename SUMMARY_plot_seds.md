# Summary of Changes: Stellar SED Plotting Function

## Overview
Added a new comprehensive plotting function `plot_stellar_seds` to visualize stellar spectral energy distributions (SEDs) with JWST filter transmission profiles and optional ice absorption.

## Files Created/Modified

### New Files

1. **`icemodels/plot_seds.py`** - Main plotting module
   - `plot_stellar_seds()` function with full documentation
   - Supports multiple temperatures, filter transmission overlays, ice absorption
   - Creates sophisticated multi-panel figures (1 main plot + N filter zoom subplots)

2. **`icemodels/README_plot_seds.md`** - Comprehensive documentation
   - Function reference
   - Parameter descriptions
   - Multiple usage examples
   - Notes on implementation details

3. **`examples/plot_seds_examples.py`** - Standalone examples script
   - Three example cases demonstrating different features
   - Saves output PNG files

4. **`notebooks/plot_stellar_seds_demo.ipynb`** - Interactive Jupyter notebook
   - Four examples with explanatory markdown
   - Ready to run demonstrations

### Modified Files

5. **`icemodels/__init__.py`**
   - Added `plot_seds` module import
   - Added `plot_stellar_seds` function import
   - Updated `__all__` to export new functionality

## Key Features

### 1. Multiple Temperatures
- Plot SEDs for multiple stellar temperatures simultaneously
- Each temperature gets a unique color from the color cycle
- Temperatures can range from cool M dwarfs (~2000K) to hot O stars (~40000K)

### 2. Filter Transmission Profiles
- Automatically retrieves transmission data from SVO Filter Profile Service
- Identifies 50% transmission range for each filter
- Creates dedicated zoom subplot for each filter
- Overlays transmission curve on zoom plots (gray shaded area)

### 3. Ice Absorption Support
- Optional ice-absorbed SEDs shown as dashed lines
- Uses existing `absorbed_spectrum` function from icemodels.core
- Requires ice optical constants table and column density
- Works with any ice composition (CO2, H2O, CO, mixtures, etc.)

### 4. Figure Layout
```
┌─────────────────────────────────────────────────┐
│     Main Plot: Full SED with Filter Coverage   │
│  (shows all wavelengths, all temps, ice)       │
└─────────────────────────────────────────────────┘
┌──────────┬──────────┬──────────┬──────────┐
│ Filter 1 │ Filter 2 │ Filter 3 │ Filter N │
│  zoom    │  zoom    │  zoom    │  zoom    │
└──────────┴──────────┴──────────┴──────────┘
```

### 5. Smart Defaults
- Auto-calculated figure size based on number of filters
- Default wavelength range covers all JWST filters (0.6-28 μm)
- Automatic normalization of SEDs for easy comparison
- Intelligent zoom ranges based on filter transmission

## Usage Examples

### Basic Usage
```python
from icemodels import plot_stellar_seds

fig, axes = plot_stellar_seds(
    temperatures=[3000, 4000, 5000],
    filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']
)
```

### With Ice Absorption
```python
from icemodels import plot_stellar_seds, load_molecule
import astropy.units as u

ice_table = load_molecule('CO2')
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=['JWST/NIRCam.F212N'],
    ice_model_table=ice_table,
    ice_column=1e19,
    molecular_weight=44*u.Da
)
```

## Technical Implementation

- Uses `mysg` stellar atmosphere models (Kurucz & Phoenix grids)
- Integrates with existing icemodels infrastructure
- Leverages `astroquery.svo_fps` for filter data
- Compatible with all JWST NIRCam and MIRI filters
- Handles units properly with astropy.units

## Testing

The implementation includes:
- Example script that can be run standalone
- Interactive Jupyter notebook for exploration
- No syntax errors (verified with get_errors)
- Follows existing code style and conventions

## Integration

Fully integrated with the icemodels package:
- Imported in `__init__.py` 
- Follows naming conventions
- Uses existing helper functions (atmo_model, absorbed_spectrum)
- Compatible with existing data structures

## Future Enhancements (Suggestions)

1. Add support for custom color maps
2. Option to show absolute flux units instead of normalized
3. Support for non-JWST filters
4. Ability to save individual zoom plots separately
5. Interactive plot with ipywidgets for parameter adjustment
