"""
Temperature-dependent Ice Analysis
=================================

This example shows how to analyze ice spectra at different temperatures
using the OCDB database.
"""

import icemodels
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Create a common wavelength grid
wavelength = np.linspace(1, 5, 1000) * u.um

# Get the default spectrum and interpolate it to our wavelength grid
default_spectrum = icemodels.core.phx4000['fnu']
default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
spectrum = f(wavelength)

# Load CO data at different temperatures
temperatures = [10, 20, 30, 40]
spectra = []

# Calculate spectra for each temperature
for temp in temperatures:
    data = icemodels.load_molecule_ocdb('co', temperature=temp)
    spec = icemodels.absorbed_spectrum(
        ice_column=1e17 * u.cm**-2,
        ice_model_table=data,
        molecular_weight=28*u.Da,
        xarr=wavelength,
        spectrum=spectrum
    )
    spectra.append(spec)

# Create the plot
plt.figure(figsize=(10, 6))
for temp, spec in zip(temperatures, spectra):
    plt.plot(wavelength, spec, label=f'{temp}K')

plt.xlabel('Wavelength (μm)')
plt.ylabel('Normalized Flux')
plt.title('CO Ice Spectrum at Different Temperatures')
plt.legend()
plt.show()