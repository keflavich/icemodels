"""
Filter Analysis
==============

This example shows how to analyze ice spectra through different JWST/MIRI filters.
"""

import icemodels
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from astroquery.svo_fps import SvoFps

# Create a common wavelength grid
wavelength = np.linspace(1, 28, 1000) * u.um

# Get the default spectrum and interpolate it to our wavelength grid
default_spectrum = icemodels.core.phx4000['fnu']
default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
spectrum_base = f(wavelength)

# Calculate spectrum
co2_data = icemodels.load_molecule('co2')
spectrum = icemodels.absorbed_spectrum(
    ice_column=1e17 * u.cm**-2,
    ice_model_table=co2_data,
    molecular_weight=44*u.Da,
    xarr=wavelength,
    spectrum=spectrum_base
)

# Plot the spectrum
plt.figure(figsize=(10, 6))
plt.plot(wavelength, spectrum, label='CO2 Ice')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Normalized Flux')
plt.title('CO2 Ice Spectrum with JWST/MIRI Filters')
plt.legend()

# Calculate and print filter fluxes
filter_ids = ['JWST/MIRI.F1000W', 'JWST/MIRI.F1280W']  # Full SVO FPS IDs
filter_fluxes = {}

transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}

for filter_id in filter_ids:
    flux = icemodels.fluxes_in_filters(
        xarr=wavelength,
        modeldata=spectrum,
        filterids=[filter_id],
        transdata=transdata,
    )
    filter_fluxes[filter_id] = flux
    print(f"Flux through {filter_id}: {flux}")

plt.show()