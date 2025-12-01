"""
DREAM Database: Ice Mixture Optical Constants
==============================================

This example demonstrates how to use the DREAM database to load and visualize
optical constants for astrophysically relevant ice mixtures.

The DREAM database (Database Referencing Extraterrestrial/Astrophysical Matter)
provides high-resolution optical constants for various ice mixtures.
"""

import icemodels
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

# %%
# Loading DREAM Database Metadata
# --------------------------------
# First, let's see what's available in the DREAM database

meta_table = icemodels.get_dream_meta_table()
print("Available DREAM datasets:")
print("=" * 60)
for i, row in enumerate(meta_table):
    print(f"{i+1}. {row['composition']} ({row['ratio']}) - {row['reference']}")

# %%
# Loading Optical Constants
# --------------------------
# Load optical constants for different H2O:CO2 ratios

data_100_14 = icemodels.load_molecule_dream('H2O : CO2', ratio='100 : 14')
data_100_50 = icemodels.load_molecule_dream('H2O : CO2', ratio='100 : 50')

print("\nData loaded:")
print(f"  H2O:CO2 (100:14): {len(data_100_14)} data points")
print(f"  H2O:CO2 (100:50): {len(data_100_50)} data points")

# %%
# Plotting Optical Constants
# ---------------------------
# Visualize the real and imaginary parts of the refractive index

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Focus on the infrared region (2-20 microns)
wl_min, wl_max = 2, 20

# Filter data for plotting
mask1 = (data_100_14['Wavelength'] >= wl_min) & (data_100_14['Wavelength'] <= wl_max)
mask2 = (data_100_50['Wavelength'] >= wl_min) & (data_100_50['Wavelength'] <= wl_max)

# Plot n (real part of refractive index)
ax1.plot(data_100_14['Wavelength'][mask1], data_100_14['n'][mask1], 
         label='H$_2$O:CO$_2$ (100:14)', linewidth=1.5)
ax1.plot(data_100_50['Wavelength'][mask2], data_100_50['n'][mask2], 
         label='H$_2$O:CO$_2$ (100:50)', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Wavelength (μm)', fontsize=12)
ax1.set_ylabel('n (real part)', fontsize=12)
ax1.set_title('Real Part of Refractive Index', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot k (imaginary part / extinction coefficient)
ax2.semilogy(data_100_14['Wavelength'][mask1], data_100_14['k'][mask1], 
             label='H$_2$O:CO$_2$ (100:14)', linewidth=1.5)
ax2.semilogy(data_100_50['Wavelength'][mask2], data_100_50['k'][mask2], 
             label='H$_2$O:CO$_2$ (100:50)', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Wavelength (μm)', fontsize=12)
ax2.set_ylabel('k (extinction coefficient)', fontsize=12)
ax2.set_title('Imaginary Part of Refractive Index', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# %%
# Calculating Absorbed Spectrum
# ------------------------------
# Use DREAM data to calculate ice absorption

# Create a simple blackbody spectrum
wavelength = np.linspace(2.5, 5, 1000) * u.um
temperature = 4000  # K

from astropy.modeling.models import BlackBody
bb = BlackBody(temperature=temperature * u.K)
# BlackBody returns surface brightness, convert to flux density by removing steradian
spectrum = bb(wavelength).to(u.erg / u.s / u.cm**2 / u.Hz / u.sr, 
                              equivalencies=u.spectral_density(wavelength)) * u.sr

# Calculate absorbed spectrum with H2O:CO2 (100:14) ice
ice_column = 1e17 / u.cm**2  # Column density in molecules/cm^2

absorbed = icemodels.absorbed_spectrum(
    ice_column=ice_column,
    ice_model_table=data_100_14,
    spectrum=spectrum,
    xarr=wavelength,
    molecular_weight=data_100_14.meta['molwt']
)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(wavelength, spectrum, label='Original spectrum', linewidth=2)
ax.plot(wavelength, absorbed, label='Absorbed spectrum', linewidth=2, alpha=0.7)
ax.set_xlabel('Wavelength (μm)', fontsize=12)
ax.set_ylabel('Flux (erg/s/cm²/Hz)', fontsize=12)
ax.set_title(f'Absorption by H$_2$O:CO$_2$ (100:14) Ice\n' + 
             f'Column density: {ice_column:.1e}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Comparing Different Mixing Ratios
# ----------------------------------
# Compare absorption features for different CO2 concentrations

fig, ax = plt.subplots(figsize=(10, 6))

# Original spectrum
ax.plot(wavelength, spectrum, label='No ice', linewidth=2, color='black')

# H2O:CO2 (100:14)
absorbed_14 = icemodels.absorbed_spectrum(
    ice_column=ice_column,
    ice_model_table=data_100_14,
    spectrum=spectrum,
    xarr=wavelength,
    molecular_weight=data_100_14.meta['molwt']
)
ax.plot(wavelength, absorbed_14, label='H$_2$O:CO$_2$ (100:14)', linewidth=2, alpha=0.8)

# H2O:CO2 (100:50)
absorbed_50 = icemodels.absorbed_spectrum(
    ice_column=ice_column,
    ice_model_table=data_100_50,
    spectrum=spectrum,
    xarr=wavelength,
    molecular_weight=data_100_50.meta['molwt']
)
ax.plot(wavelength, absorbed_50, label='H$_2$O:CO$_2$ (100:50)', linewidth=2, alpha=0.8)

ax.set_xlabel('Wavelength (μm)', fontsize=12)
ax.set_ylabel('Flux (erg/s/cm²/Hz)', fontsize=12)
ax.set_title(f'Comparison of Ice Absorption\nColumn density: {ice_column:.1e}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Citation Information
# --------------------
# When using DREAM database data, cite the original references

print("\nCitation requirements:")
print("=" * 60)
print("If you use data from the DREAM database, you must:")
print("1. Cite the original data references:")
print(f"   {data_100_14.meta.get('citation', 'See file header')}")
print("2. Include: 'This research has made use of the Database")
print("   Referencing Extraterrestrial/Astrophysical Matter (DREAM) database'")
