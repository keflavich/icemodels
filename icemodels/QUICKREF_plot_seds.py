"""
Quick Reference: plot_stellar_seds

BASIC USAGE:
-----------
from icemodels import plot_stellar_seds

# Single temperature, two filters
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=['JWST/NIRCam.F212N', 'JWST/NIRCam.F444W']
)

# Multiple temperatures
fig, axes = plot_stellar_seds(
    temperatures=[3000, 4000, 5000, 6000],
    filters=['JWST/NIRCam.F212N', 'JWST/MIRI.F1000W']
)

WITH ICE ABSORPTION:
-------------------
from icemodels import plot_stellar_seds, load_molecule_ocdb
import astropy.units as u

ice_table = load_molecule_ocdb('co2', temperature=25)
fig, axes = plot_stellar_seds(
    temperatures=4000,
    filters=['JWST/NIRCam.F212N'],
    ice_model_table=ice_table,
    ice_column=1e19*u.cm**-2,  # ice column density
    molecular_weight=44*u.Da
)

AVAILABLE FILTERS (Examples):
----------------------------
NIRCam:
  'JWST/NIRCam.F115W'
  'JWST/NIRCam.F150W'
  'JWST/NIRCam.F182M'
  'JWST/NIRCam.F187N'
  'JWST/NIRCam.F212N'
  'JWST/NIRCam.F444W'

MIRI:
  'JWST/MIRI.F560W'
  'JWST/MIRI.F770W'
  'JWST/MIRI.F1000W'
  'JWST/MIRI.F1280W'
  'JWST/MIRI.F1800W'

CUSTOMIZATION:
-------------
# Custom wavelength range (high resolution)
import numpy as np
xarr = np.linspace(1.5*u.um, 5.0*u.um, 50000)

fig, axes = plot_stellar_seds(
    temperatures=[3000, 5000],
    filters=['JWST/NIRCam.F212N'],
    xarr=xarr,
    figsize=(12, 8),  # Custom size
    color_cycle=['red', 'blue']  # Custom colors
)

RETURNED VALUES:
---------------
fig    : matplotlib Figure object
axes   : Array of Axes [main_plot, zoom1, zoom2, ...]
         axes[0] = main plot
         axes[1:] = filter zoom plots

TYPICAL TEMPERATURES:
--------------------
M dwarf:    2000-3700 K
K dwarf:    3700-5200 K
G (Sun):    5200-6000 K
F:          6000-7500 K
A:          7500-10000 K
B:          10000-30000 K
O:          >30000 K

ICE MOLECULES:
-------------
Available via read_ocdb_file():
  See optical_constants_cache_dir for available files
  Example files:
    55_CO2_(1)_25K_Gerakines.txt
    240_H2O_(1)_25K_Mastrapa.txt
    63_CO_(1)_25K_Gerakines_low.txt

Run download_all_ocdb() to download all OCDB data files.

For ice mixtures, see absorbance_in_filters.py or load_molecule_dream()
"""
