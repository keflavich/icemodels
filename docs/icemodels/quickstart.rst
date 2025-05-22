Quickstart Guide
===============

This guide will help you get started with IceModels quickly.

Basic Usage
----------

IceModels provides tools for working with ice spectroscopy data and models.
The main functionality includes loading ice data, calculating absorption spectra,
and analyzing spectral features.

Loading Ice Data
--------------

To load ice optical constants:

.. code-block:: python

    import icemodels

    # Load CO2 ice data
    co2_data = icemodels.load_molecule('co2')

    # Load temperature-dependent CO data
    co_data = icemodels.load_molecule_ocdb('co', temperature=10)

Computing Absorption Spectra
---------------------------

Calculate absorption spectra for ice layers:

.. code-block:: python

    import astropy.units as u

    # Define ice parameters
    column = 1e17 * u.cm**-2

    # Calculate spectrum
    spectrum = icemodels.absorbed_spectrum(
        ice_column=column,
        ice_model_table=co2_data,
        molecular_weight=44*u.Da
    )

Simple Gaussian models
--------------------

There is some provision for simple Gaussian models.  These are inaccurate and just hacked together; I don't recommend using them.

.. code-block:: python

    # Define Gaussian parameters
    center = 4.27 * u.um  # CO2 stretch mode
    width = 0.1 * u.um
    bandstrength = 1e-16 * u.cm/u.molecule

    # Calculate spectrum with Gaussian components
    spectrum_gauss = icemodels.absorbed_spectrum_Gaussians(
        ice_column=column,
        center=center,
        width=width,
        ice_bandstrength=bandstrength
    )

These examples demonstrate the basic functionality of IceModels. For more detailed examples and advanced usage, see the :doc:`tutorial` and :doc:`examples` sections.