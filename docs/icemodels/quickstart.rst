Quickstart Guide
==============

This guide will help you get started with IceModels quickly.

Basic Usage
----------

First, import the necessary modules:

.. code-block:: python

    import icemodels
    import astropy.units as u
    import numpy as np

Loading Ice Data
--------------

Load optical constants for a molecule:

.. code-block:: python

    # Load CO2 ice data
    co2_data = icemodels.load_molecule('co2')

    # Access wavelength, n, and k values
    wavelengths = co2_data['Wavelength']  # in microns
    n_values = co2_data['n']  # refractive index
    k_values = co2_data['k']  # extinction coefficient

Computing Absorption Spectra
-------------------------

Calculate the absorption spectrum for an ice layer:

.. code-block:: python

    # Define ice column density
    ice_column = 1e17 * u.cm**-2

    # Calculate absorption spectrum
    spectrum = icemodels.absorbed_spectrum(
        ice_column=ice_column,
        ice_model_table=co2_data,
        molecular_weight=44*u.Da
    )

Working with Multiple Ice Components
--------------------------------

You can work with multiple ice components using Gaussian models:

.. code-block:: python

    # Define Gaussian parameters
    center = 4.27 * u.um  # CO2 stretch mode
    width = 0.1 * u.um
    bandstrength = 1e-16 * u.cm/u.molecule

    # Calculate spectrum with Gaussian components
    spectrum_gauss = icemodels.absorbed_spectrum_Gaussians(
        ice_column=ice_column,
        center=center,
        width=width,
        ice_bandstrength=bandstrength
    )

These examples demonstrate the basic functionality of IceModels. For more detailed examples and advanced usage, see the :doc:`tutorial` and :doc:`examples` sections.