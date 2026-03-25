# Extinction Feature for Stellar SED Plotting

## Overview

The `plot_stellar_seds` function now supports interstellar extinction using the `dust_extinction` package. This allows you to visualize how stellar SEDs appear when viewed through dust in the interstellar medium.

## Quick Start

```python
from dust_extinction.averages import CT06_MWGC
from icemodels import plot_stellar_seds

# Plot stellar SEDs with 17 magnitudes of extinction
fig, axes = plot_stellar_seds(
    temperatures=[3000, 4000, 5000],
    filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W', 'JWST/MIRI.F1000W'],
    extinction_Av=17.0,
    extinction_curve=CT06_MWGC()  # Default: Chiar & Tielens 2006
)
```

## Parameters

- **extinction_Av** (float, optional): Visual extinction A_V in magnitudes. Set this to apply extinction.
- **extinction_curve** (dust_extinction model, optional): Extinction curve model. Defaults to CT06_MWGC() if not provided.

## Supported Extinction Curves

The function works with any extinction model from the `dust_extinction` package:

```python
from dust_extinction.averages import CT06_MWGC, CCM89, F99
from dust_extinction.parameter_averages import F04, G16

# Chiar & Tielens 2006 (default)
extinction_curve=CT06_MWGC()

# Cardelli, Clayton, & Mathis 1989
extinction_curve=CCM89()

# Fitzpatrick 1999
extinction_curve=F99()

# Fitzpatrick 2004
extinction_curve=F04()

# Gordon et al. 2016
extinction_curve=G16()
```

## Combined with Ice Absorption

You can show both extinction and ice absorption simultaneously:

```python
from dust_extinction.averages import CT06_MWGC
from icemodels import plot_stellar_seds, read_ocdb_file
import astropy.units as u

# Load ice data
ice_table = read_ocdb_file('path/to/CO2_ice.txt')

# Show extinction + ice
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W'],
    ice_model_table=ice_table,
    ice_column=1e19*u.cm**-2,
    molecular_weight=44*u.Da,
    extinction_Av=17.0,
    extinction_curve=CT06_MWGC()
)
```

This will show:
- Bare stellar SED (after extinction)
- Ice-absorbed SED (with both extinction and ice)

## How It Works

1. Stellar SED is generated using the mysg model
2. Extinction is applied: `flux_extincted = flux * 10^(-A(λ)/2.5)`
   where `A(λ) = A_V × (A(λ)/A_V)` from the extinction curve
3. The extinction curve expects inverse wavelength in units of 1/μm
4. Valid wavelength range depends on the extinction curve (typically ~1.2-27 μm for CT06_MWGC)
5. Ice absorption is applied after extinction if ice parameters are provided

## Installation

To use extinction features, install the dust_extinction package:

```bash
pip install dust_extinction
```

## Examples

See:
- `examples/plot_seds_examples.py` - Example 3 (extinction) and Example 5 (combined)
- `notebooks/plot_stellar_seds_demo.ipynb` - Example 5 (extinction) and Example 6 (combined)
- `icemodels/README_plot_seds.md` - Full documentation

## References

- Chiar, J. E., & Tielens, A. G. G. M. 2006, ApJ, 637, 774 (CT06)
- dust_extinction documentation: https://dust-extinction.readthedocs.io/
