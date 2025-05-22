API Reference
============

Core Functions
-------------

.. automodule:: icemodels.core
   :members:
   :undoc-members:
   :show-inheritance:

Loading Ice Data
---------------

.. py:function:: load_molecule(molname)

   Load optical constants for a specific molecule from predefined data sources.

   :param str molname: Name of the molecule (e.g., 'co2', 'h2o', 'ch3oh')
   :return: Table containing wavelength, n, and k values
   :rtype: astropy.table.Table

.. py:function:: load_molecule_univap(molname, meta_table=None)

   Load optical constants from the Univap database.

   :param str molname: Name of the molecule
   :param meta_table: Optional metadata table
   :return: Table containing optical constants
   :rtype: astropy.table.Table

.. py:function:: load_molecule_ocdb(molname, temperature=10, use_cached=True)

   Load optical constants from the OCDB database.

   :param str molname: Name of the molecule
   :param float temperature: Temperature in Kelvin
   :param bool use_cached: Whether to use cached data
   :return: Table containing optical constants
   :rtype: astropy.table.Table

Spectral Analysis
----------------

.. py:function:: absorbed_spectrum(ice_column, ice_model_table, spectrum=None, xarr=None, molecular_weight=None, minimum_tau=0, return_tau=False)

   Calculate the absorption spectrum for an ice layer.

   :param ice_column: Column density of the ice
   :type ice_column: astropy.units.Quantity
   :param ice_model_table: Table containing optical constants
   :type ice_model_table: astropy.table.Table
   :param spectrum: Input spectrum to modify
   :param xarr: Wavelength array
   :param molecular_weight: Molecular weight of the ice species
   :param float minimum_tau: Minimum optical depth
   :param bool return_tau: Whether to return optical depth instead of spectrum
   :return: Modified spectrum or optical depth
   :rtype: numpy.ndarray

.. py:function:: absorbed_spectrum_Gaussians(ice_column, center, width, ice_bandstrength, spectrum=None, xarr=None)

   Calculate absorption spectrum using Gaussian components.

   :param ice_column: Column density
   :param center: Central wavelength of the Gaussian
   :param width: Width of the Gaussian
   :param ice_bandstrength: Band strength of the ice feature
   :param spectrum: Input spectrum to modify
   :param xarr: Wavelength array
   :return: Modified spectrum
   :rtype: numpy.ndarray

Utility Functions
---------------

.. py:function:: cde_correct(freq, m)

   Apply continuous distribution of ellipsoids (CDE) correction.

   :param freq: Frequency array
   :param m: Complex refractive index
   :return: CDE-corrected absorption coefficient
   :rtype: numpy.ndarray

.. py:function:: fluxes_in_filters(xarr, modeldata, doplot=False, filterids=None, transdata=None)

   Calculate fluxes through specified filters.

   :param xarr: Wavelength array
   :param modeldata: Spectral data
   :param bool doplot: Whether to plot the results
   :param filterids: List of filter IDs
   :param transdata: Transmission data
   :return: Dictionary of filter fluxes
   :rtype: dict