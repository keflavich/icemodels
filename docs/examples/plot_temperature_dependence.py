"""
Temperature-dependent Ice Analysis
==================================

This example shows how to analyze ice spectra at different temperatures
using the OCDB database.
"""

import icemodels
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create a common wavelength grid
wavelength = np.linspace(1, 5, 1000) * u.um

# Use a simple local continuum to keep docs builds independent of external CDBS data
spectrum = np.ones_like(wavelength.value) * u.Jy

# Load packaged CO data at different temperatures.
# Using local files keeps docs builds deterministic and avoids network dependence.
co_datasets = [
    ('10 K', 'ocdb_85_CO_(1)_10K_Hudgins.txt'),
    ('12.5 K', 'ocdb_1_CO_(1)_12.5K_Baratta.txt'),
    ('15 K', 'ocdb_267_CO_(1)_15K_Palumbo.txt'),
    ('25 K', 'ocdb_63_CO_(1)_25K_Gerakines.txt'),
    ('30 K', 'ocdb_35_CO_(1)_30K_Ehrenfreund.txt'),
]
spectra = []
data_dir = Path(icemodels.__file__).resolve().parent / 'data'

# Calculate spectra for each temperature
for _, filename in co_datasets:
    data = icemodels.read_ocdb_file(data_dir / filename)
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
for (label, _), spec in zip(co_datasets, spectra):
    plt.plot(wavelength, spec, label=label)

plt.xlabel('Wavelength (μm)')
plt.ylabel('Normalized Flux')
plt.title('CO Ice Spectrum at Different Temperatures')
plt.legend()
plt.show()