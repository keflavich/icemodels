import icemodels
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os

# Create _static directory if it doesn't exist
os.makedirs('_static', exist_ok=True)

# 1. Temperature-dependent Ice Analysis
def plot_temperature_dependence():
    temperatures = [10, 20, 30, 40]
    spectra = []
    wavelengths = []

    for temp in temperatures:
        data = icemodels.load_molecule_ocdb('co', temperature=temp)
        spectrum = icemodels.absorbed_spectrum(
            ice_column=1e17 * u.cm**-2,
            ice_model_table=data,
            molecular_weight=28*u.Da
        )
        spectra.append(spectrum)
        wavelengths.append(data['Wavelength'])

    # Use the wavelength grid from the first temperature
    common_wavelength = wavelengths[0]

    plt.figure(figsize=(10, 6))
    for temp, spec in zip(temperatures, spectra):
        plt.plot(common_wavelength, spec, label=f'{temp}K')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('CO Ice Spectrum at Different Temperatures')
    plt.legend()
    plt.savefig('_static/temperature_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Ice Mixture Analysis
def plot_ice_mixtures():
    h2o = icemodels.load_molecule('h2o')
    co2 = icemodels.load_molecule('co2')

    ratios = [(1, 0.2), (1, 0.5), (1, 1)]
    base_column = 1e17 * u.cm**-2

    plt.figure(figsize=(10, 6))
    for h2o_ratio, co2_ratio in ratios:
        h2o_spec = icemodels.absorbed_spectrum(
            ice_column=base_column * h2o_ratio,
            ice_model_table=h2o,
            molecular_weight=18*u.Da
        )
        co2_spec = icemodels.absorbed_spectrum(
            ice_column=base_column * co2_ratio,
            ice_model_table=co2,
            molecular_weight=44*u.Da
        )
        combined = h2o_spec * co2_spec
        plt.plot(h2o['Wavelength'], combined,
                label=f'H2O:CO2 = {h2o_ratio}:{co2_ratio}')

    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('H2O:CO2 Ice Mixtures')
    plt.legend()
    plt.savefig('_static/ice_mixtures.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Filter Analysis
def plot_filter_analysis():
    co2_data = icemodels.load_molecule('co2')
    spectrum = icemodels.absorbed_spectrum(
        ice_column=1e17 * u.cm**-2,
        ice_model_table=co2_data,
        molecular_weight=44*u.Da
    )

    filter_fluxes = icemodels.fluxes_in_filters(
        xarr=co2_data['Wavelength'],
        modeldata=spectrum,
        filterids=['JWST/MIRI.F1000W', 'JWST/MIRI.F1280W'],
        doplot=True
    )
    plt.savefig('_static/filter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. CDE Correction
def plot_cde_correction():
    co_data = icemodels.load_molecule('co')
    m = co_data['n'] + 1j * co_data['k']
    freq = (co_data['Wavelength'].quantity.to(u.Hz, u.spectral()))
    abs_coeff = icemodels.cde_correct(freq, m)

    plt.figure(figsize=(10, 6))
    plt.plot(co_data['Wavelength'], co_data['k'], label='Original')
    plt.plot(co_data['Wavelength'], abs_coeff, label='CDE-corrected')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Absorption Coefficient')
    plt.title('Effect of CDE Correction on CO Ice')
    plt.legend()
    plt.savefig('_static/cde_correction.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating example plots...")
    plot_temperature_dependence()
    print("1. Temperature dependence plot generated")
    plot_ice_mixtures()
    print("2. Ice mixtures plot generated")
    plot_filter_analysis()
    print("3. Filter analysis plot generated")
    plot_cde_correction()
    print("4. CDE correction plot generated")
    print("All plots have been generated in the _static directory")