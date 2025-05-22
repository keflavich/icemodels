Ice Databases
============

IceModels provides access to several databases of ice optical constants:

Built-in Database
---------------

The package includes a built-in database of commonly used ice species with data from various sources:

* CH3OH (methanol)
* CO2 (carbon dioxide)
* CH4 (methane)
* CO (carbon monoxide)
* H2O (water ice, amorphous and crystalline)
* NH3 (ammonia)

These data are automatically downloaded from trusted sources when first accessed.

OCDB Database
-----------

The Optical Constants Database (OCDB) from NASA contains a comprehensive collection of optical constants for various ices at different temperatures.

To download all available OCDB data:

.. code-block:: python

    import icemodels
    icemodels.download_all_ocdb()

To load a specific molecule from OCDB:

.. code-block:: python

    data = icemodels.load_molecule_ocdb('co', temperature=10)

Univap Database
-------------

The Univap database (from Universidade do Vale do Paraíba) provides optical constants for various ices and ice mixtures.

Available data includes:

* Pure ices (CO, CO2, H2O)
* Ice mixtures (e.g., H2O:CO)
* Different ice phases (amorphous and crystalline)

To load data from Univap:

.. code-block:: python

    data = icemodels.load_molecule_univap('co')

LIDA Database
-----------

The Leiden Ice Database for Astrochemistry (LIDA) contains spectroscopic data for various astronomical ices.

To download all available LIDA data:

.. code-block:: python

    icemodels.download_all_lida()

Data Format
----------

All databases return data in a consistent format using astropy Tables with the following columns:

* Wavelength (in microns)
* n (refractive index)
* k (extinction coefficient)

Additional metadata (temperature, density, source reference) is stored in the table's metadata dictionary.