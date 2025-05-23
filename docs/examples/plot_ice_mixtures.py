"""
Ice Mixture Analysis
===================

This example demonstrates how to analyze ice mixtures with different ratios,
specifically looking at H2O:CO2 mixtures.
"""

import icemodels
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d

# Create a common wavelength grid
wavelength = np.linspace(1, 28, 1000) * u.um

# Get the default spectrum and interpolate it to our wavelength grid
default_spectrum = icemodels.core.phx4000['fnu']
default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
spectrum = f(wavelength)

# Load components
h2o = icemodels.load_molecule('h2o')
co2 = icemodels.load_molecule('co2')

# Define mixture ratios (H2O:CO2)
ratios = [(1, 0.2), (1, 0.5), (1, 1)]
base_column = 1e17 * u.cm**-2

plt.figure(figsize=(10, 6))
for h2o_ratio, co2_ratio in ratios:
    # Calculate individual spectra
    h2o_spec = icemodels.absorbed_spectrum(
        ice_column=base_column * h2o_ratio,
        ice_model_table=h2o,
        molecular_weight=18*u.Da,
        xarr=wavelength,
        spectrum=spectrum
    )
    co2_spec = icemodels.absorbed_spectrum(
        ice_column=base_column * co2_ratio,
        ice_model_table=co2,
        molecular_weight=44*u.Da,
        xarr=wavelength,
        spectrum=spectrum
    )
    # Combined spectrum
    combined = h2o_spec * co2_spec
    plt.plot(wavelength, combined,
            label=f'H2O:CO2 = {h2o_ratio}:{co2_ratio}')

plt.xlabel('Wavelength (μm)')
plt.ylabel('Normalized Flux')
plt.title('H2O:CO2 Ice Mixtures')
plt.legend()
plt.show()