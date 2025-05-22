Tutorial
========

This tutorial will walk you through common tasks using IceModels.

Basic Ice Spectrum Analysis
-------------------------

Let's analyze a simple CO2 ice spectrum:

.. code-block:: python

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np

    # Load CO2 ice data
    co2_data = icemodels.load_molecule('co2')

    # Define ice parameters
    ice_column = 1e17 * u.cm**-2  # column density
    mol_weight = 44 * u.Da        # CO2 molecular weight

    # Calculate absorption spectrum
    spectrum = icemodels.absorbed_spectrum(
        ice_column=ice_column,
        ice_model_table=co2_data,
        molecular_weight=mol_weight
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(co2_data['Wavelength'], spectrum)
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('CO2 Ice Absorption Spectrum')
    plt.show()

Working with Multiple Ice Components
---------------------------------

Now let's create a more complex example with multiple ice components:

.. code-block:: python

    # Load data for different ices
    h2o_data = icemodels.load_molecule('h2o')
    co_data = icemodels.load_molecule('co')

    # Define columns for each component
    h2o_column = 5e17 * u.cm**-2
    co_column = 1e17 * u.cm**-2

    # Calculate individual spectra
    h2o_spectrum = icemodels.absorbed_spectrum(
        ice_column=h2o_column,
        ice_model_table=h2o_data,
        molecular_weight=18*u.Da
    )

    co_spectrum = icemodels.absorbed_spectrum(
        ice_column=co_column,
        ice_model_table=co_data,
        molecular_weight=28*u.Da
    )

    # Combined spectrum is the product of individual spectra
    combined_spectrum = h2o_spectrum * co_spectrum

    # Plot all components
    plt.figure(figsize=(12, 8))
    plt.plot(h2o_data['Wavelength'], h2o_spectrum, label='H2O')
    plt.plot(co_data['Wavelength'], co_spectrum, label='CO')
    plt.plot(h2o_data['Wavelength'], combined_spectrum, label='Combined')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('Multi-component Ice Spectrum')
    plt.legend()
    plt.show()

Using Gaussian Components
----------------------

Sometimes it's useful to model ice features using Gaussian components:

.. code-block:: python

    # Define Gaussian parameters for CO2 stretch mode
    center = 4.27 * u.um
    width = 0.1 * u.um
    bandstrength = 1e-16 * u.cm/u.molecule
    column = 1e17 * u.cm**-2

    # Calculate Gaussian spectrum
    gauss_spectrum = icemodels.absorbed_spectrum_Gaussians(
        ice_column=column,
        center=center,
        width=width,
        ice_bandstrength=bandstrength
    )

    # Compare with actual CO2 data
    plt.figure(figsize=(10, 6))
    plt.plot(co2_data['Wavelength'], spectrum, label='Real data')
    plt.plot(co2_data['Wavelength'], gauss_spectrum, label='Gaussian model')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('CO2 Ice: Data vs Gaussian Model')
    plt.legend()
    plt.show()

Working with Different Databases
-----------------------------

IceModels can access data from multiple databases. Here's how to compare data from different sources:

.. code-block:: python

    # Get CO data from different sources
    co_builtin = icemodels.load_molecule('co')
    co_ocdb = icemodels.load_molecule_ocdb('co', temperature=10)
    co_univap = icemodels.load_molecule_univap('co')

    # Plot optical constants from each source
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(co_builtin['Wavelength'], co_builtin['n'], label='Built-in')
    plt.plot(co_ocdb['Wavelength'], co_ocdb['n'], label='OCDB')
    plt.plot(co_univap['Wavelength'], co_univap['n'], label='Univap')
    plt.ylabel('n')
    plt.title('CO Ice Optical Constants')
    plt.legend()

    plt.subplot(212)
    plt.plot(co_builtin['Wavelength'], co_builtin['k'], label='Built-in')
    plt.plot(co_ocdb['Wavelength'], co_ocdb['k'], label='OCDB')
    plt.plot(co_univap['Wavelength'], co_univap['k'], label='Univap')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('k')
    plt.legend()
    plt.show()

These examples demonstrate the main functionality of IceModels. For more advanced usage and specific applications, see the :doc:`examples` section.