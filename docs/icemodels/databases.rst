Ice Databases
=============

This section describes the various ice spectroscopy databases supported by IceModels.

Built-in Database
-----------------

The built-in database contains a curated set of ice optical constants from various sources.
These data are included with the package and are immediately available without additional downloads.

Available molecules:
- H2O (water ice)
- CO2 (carbon dioxide ice)
- CO (carbon monoxide ice)
- CH3OH (methanol ice)

OCDB Database
-------------

The Optical Constants Database (OCDB) provides temperature-dependent optical constants for various ices.
Data from this database can be accessed using the `load_molecule_ocdb` function.

Features:
- Temperature-dependent measurements
- Multiple ice phases
- Extensive metadata
- Regular updates

Example usage:
```python
data = icemodels.load_molecule_ocdb('co', temperature=10)
```

Data Format
-----------

All databases provide data in a consistent format using astropy Tables with the following columns:

- Wavelength: Wavelength in microns
- n: Real part of the refractive index
- k: Imaginary part of the refractive index (extinction coefficient)

Additional metadata may be included in the table's metadata dictionary.

Univap Database
---------------

The Univap database (from Universidade do Vale do Paraíba) provides optical constants for various ices and ice mixtures.

Available data includes:

* Pure ices (CO, CO2, H2O)
* Ice mixtures (e.g., H2O:CO)
* Different ice phases (amorphous and crystalline)

To load data from Univap:

.. code-block:: python

    data = icemodels.load_molecule_univap('co')

LIDA Database
-------------

The Leiden Ice Database for Astrochemistry (LIDA) contains spectroscopic data for various astronomical ices.

To download all available LIDA data:

.. code-block:: python

    icemodels.download_all_lida()