Examples
========

This section contains additional practical examples of using IceModels for various tasks.

Temperature-dependent Ice Analysis
------------------------------

Analyzing how ice spectra change with temperature:

.. code-block:: python

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np

    # Load CO data at different temperatures
    temperatures = [10, 20, 30, 40]
    spectra = []

    for temp in temperatures:
        data = icemodels.load_molecule_ocdb('co', temperature=temp)
        spectrum = icemodels.absorbed_spectrum(
            ice_column=1e17 * u.cm**-2,
            ice_model_table=data,
            molecular_weight=28*u.Da
        )
        spectra.append(spectrum)

    # Plot temperature dependence
    plt.figure(figsize=(10, 6))
    for temp, spec in zip(temperatures, spectra):
        plt.plot(data['Wavelength'], spec, label=f'{temp}K')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('CO Ice Spectrum at Different Temperatures')
    plt.legend()
    plt.show()

Ice Mixture Analysis
-----------------

Analyzing ice mixtures with different ratios:

.. code-block:: python

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
            molecular_weight=18*u.Da
        )
        co2_spec = icemodels.absorbed_spectrum(
            ice_column=base_column * co2_ratio,
            ice_model_table=co2,
            molecular_weight=44*u.Da
        )
        # Combined spectrum
        combined = h2o_spec * co2_spec
        plt.plot(h2o['Wavelength'], combined,
                label=f'H2O:CO2 = {h2o_ratio}:{co2_ratio}')

    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('H2O:CO2 Ice Mixtures')
    plt.legend()
    plt.show()

Filter Analysis
-------------

Analyzing ice spectra through different filters:

.. code-block:: python

    # Calculate spectrum
    co2_data = icemodels.load_molecule('co2')
    spectrum = icemodels.absorbed_spectrum(
        ice_column=1e17 * u.cm**-2,
        ice_model_table=co2_data,
        molecular_weight=44*u.Da
    )

    # Calculate fluxes through filters
    filter_fluxes = icemodels.fluxes_in_filters(
        xarr=co2_data['Wavelength'],
        modeldata=spectrum,
        filterids=['JWST/MIRI.F1000W', 'JWST/MIRI.F1280W'],
        doplot=True
    )

    print("Filter fluxes:", filter_fluxes)

CDE Correction Example
-------------------

Applying continuous distribution of ellipsoids (CDE) correction:

.. code-block:: python

    # Load data
    co_data = icemodels.load_molecule('co')

    # Calculate complex refractive index
    m = co_data['n'] + 1j * co_data['k']

    # Convert wavelength to frequency
    freq = (co_data['Wavelength'].quantity.to(u.Hz, u.spectral()))

    # Apply CDE correction
    abs_coeff = icemodels.cde_correct(freq, m)

    # Plot original vs CDE-corrected absorption
    plt.figure(figsize=(10, 6))
    plt.plot(co_data['Wavelength'], co_data['k'], label='Original')
    plt.plot(co_data['Wavelength'], abs_coeff, label='CDE-corrected')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Absorption Coefficient')
    plt.title('Effect of CDE Correction on CO Ice')
    plt.legend()
    plt.show()

These examples demonstrate more advanced applications of IceModels. The package can be used for many other types of analysis depending on your specific needs.