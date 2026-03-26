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
from scipy.interpolate import interp1d
from pathlib import Path

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

# Create a common wavelength grid
wavelength = np.linspace(1, 5, 1000) * u.um

# Get the default spectrum and interpolate it to our wavelength grid
reference_model = icemodels.atmo_model(4000)
default_spectrum = reference_model['fnu']
default_wavelength = u.Quantity(reference_model['nu'], u.Hz).to(u.um, u.spectral())
f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
spectrum = f(wavelength)

# Load packaged CO data at different temperatures.
# Using local files keeps docs builds deterministic and avoids network dependence.
co_datasets = [
    ('10 K', 'ocdb_85_CO_(1)_10K_Hudgins.txt'),
    ('15 K', 'ocdb_267_CO_(1)_15K_Palumbo.txt'),
    ('25 K', 'ocdb_63_CO_(1)_25K_Gerakines.txt'),
    ('30 K', 'ocdb_35_CO_(1)_30K_Ehrenfreund.txt'),
]
spectra = []
# Use importlib.resources for robust data file access across installation methods
data_package = files('icemodels').joinpath('data')
data_dir = Path(str(data_package))

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