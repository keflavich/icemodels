"""
CDE Correction Example
=====================

This example demonstrates how to apply the continuous distribution of ellipsoids (CDE)
correction to ice absorption spectra.
"""

import icemodels
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Create a common wavelength grid
wavelength = np.linspace(1, 28, 1000) * u.um

# Load data
co_data = icemodels.load_molecule('co')

# Get the default spectrum and interpolate it to our wavelength grid
default_spectrum = icemodels.core.phx4000['fnu']
default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
spectrum = f(wavelength)

# Calculate absorption spectra with and without CDE correction
spectrum_no_cde = icemodels.absorbed_spectrum(
    ice_column=1e17 * u.cm**-2,
    ice_model_table=co_data,
    molecular_weight=28*u.Da,
    xarr=wavelength,
    spectrum=spectrum,
)

# Interpolate n and k onto our wavelength grid
f_n = interp1d(co_data['Wavelength'], co_data['n'], bounds_error=False, fill_value=1.0)
f_k = interp1d(co_data['Wavelength'], co_data['k'], bounds_error=False, fill_value=0.0)
n = f_n(wavelength)
k = f_k(wavelength)
m = n + 1j * k

# Calculate the CDE-corrected optical depth
freq = wavelength.to(u.cm**-1, u.spectral())
wl = 1.e4/freq
m2 = m**2.0
im_part = ((m2/(m2-1.0))*np.log(m2)).imag
spectrum_cde = (4.0*np.pi/wl)*im_part

# Plot original vs CDE-corrected absorption
plt.figure(figsize=(10, 6))
plt.plot(wavelength, spectrum_no_cde, label='Without CDE')
plt.plot(wavelength, spectrum_cde, label='With CDE')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Optical Depth')
plt.title('Effect of CDE Correction on CO Ice')
plt.legend()
plt.show()