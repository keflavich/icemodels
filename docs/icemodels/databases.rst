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

DREAM Database
--------------

The DREAM database (Database Referencing Extraterrestrial/Astrophysical Matter) provides
optical constants for various ice mixtures relevant to astrophysical environments.

Database URL: https://hebergement.universite-paris-saclay.fr/edartois/dream_database.html

Features:

* Ice mixture optical constants (n and k)
* High spectral resolution data
* Multiple mixing ratios
* Full citation information with DOIs
* Temperature-dependent measurements

Available data includes:

* H₂O:CO₂ mixtures at various ratios (100:14, 100:50, 100:32)
* H₂O:CO₂:CH₃OH mixture (100:21:11)
* H₂O:CO₂:CO:NH₃ mixture (100:16:8:8)
* H₂O:NH₃:CH₄:CO (irradiated)

Getting Started with DREAM
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To download all available DREAM data:

.. code-block:: python

    import icemodels

    # Download all DREAM database files
    icemodels.download_all_dream()

    # Or get metadata first to see what's available
    meta_table = icemodels.get_dream_meta_table()
    print(meta_table)

Loading DREAM Data
^^^^^^^^^^^^^^^^^^

Load optical constants for a specific composition:

.. code-block:: python

    # Load H2O:CO2 mixture with ratio 100:14
    data = icemodels.load_molecule_dream('H2O : CO2', ratio='100 : 14')

    # Inspect the data
    print(f"Wavelength range: {data['Wavelength'].min()} to {data['Wavelength'].max()} microns")
    print(f"Metadata: {data.meta}")

Using DREAM Data with Absorption Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DREAM data integrates seamlessly with existing icemodels functions:

.. code-block:: python

    import numpy as np
    from astropy import units as u

    # Load DREAM data
    dream_data = icemodels.load_molecule_dream('H2O : CO2', ratio='100 : 14')

    # Define wavelength grid and ice column density
    wavelength = np.linspace(2, 20, 1000) * u.um
    ice_column = 1e17 / u.cm**2

    # Calculate absorbed spectrum
    absorbed = icemodels.absorbed_spectrum(
        ice_column=ice_column,
        ice_model_table=dream_data,
        xarr=wavelength,
        molecular_weight=dream_data.meta['molwt']
    )

Data Format
^^^^^^^^^^^

DREAM database files contain:

* Header with citation information and DOI
* Three-column data: wavelength (μm), n (real part), k (imaginary part)
* Description of data processing methods

Each loaded dataset includes comprehensive metadata:

* ``database``: Source database identifier ('dream')
* ``composition``: Ice composition (e.g., "H2O : CO2")
* ``ratio``: Mixing ratio (e.g., "100 : 14")
* ``reference``: Literature citation
* ``doi``: Digital Object Identifier
* ``citation``: Full citation text
* ``molecule``: Primary molecule name
* ``molwt``: Calculated molecular weight with units
* ``density``: Ice density (default 1 g/cm³)

Citation Requirements
^^^^^^^^^^^^^^^^^^^^^

When using DREAM database data, you must:

1. Cite the original data references provided in each file
2. Include the acknowledgment: "This research has made use of the Database Referencing Extraterrestrial/Astrophysical Matter (DREAM) database"

Primary references:

* Dartois, E., Noble, J. A., Ysard, N., Demyk, K., Chabot, M., 2022,
  "Influence of grain growth on CO₂ ice spectroscopic profiles",
  A&A 666, A153, doi:10.1051/0004-6361/202243929

Example: Plotting DREAM Database Spectra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's a complete example showing how to load DREAM data and calculate ice absorption:

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np

    # Download DREAM data (if not already cached)
    icemodels.download_all_dream()

    # Load H2O:CO2 mixture data
    dream_data = icemodels.load_molecule_dream('H2O : CO2', ratio='100 : 14')

    print(f"Loaded {len(dream_data)} data points")
    print(f"Composition: {dream_data.meta['composition']}")
    print(f"Reference: {dream_data.meta['reference']}")

    # Focus on infrared region
    wavelength = np.linspace(2, 20, 1000) * u.um

    # Create a simple blackbody spectrum
    from astropy.modeling.models import BlackBody
    bb = BlackBody(temperature=4000 * u.K)
    # BlackBody returns surface brightness, convert to flux density
    spectrum = bb(wavelength).to(u.erg / u.s / u.cm**2 / u.Hz / u.sr,
                                  equivalencies=u.spectral_density(wavelength)) * u.sr

    # Calculate absorption
    ice_column = 1e17 / u.cm**2
    absorbed = icemodels.absorbed_spectrum(
        ice_column=ice_column,
        ice_model_table=dream_data,
        spectrum=spectrum,
        xarr=wavelength,
        molecular_weight=dream_data.meta['molwt']
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, spectrum, label='Original', linewidth=2)
    plt.plot(wavelength, absorbed, label='With H$_2$O:CO$_2$ ice', linewidth=2, alpha=0.7)
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Flux (erg/s/cm²/Hz)')
    plt.title(f'Ice Absorption from DREAM Database\nColumn: {ice_column:.1e}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


Additional Examples
^^^^^^^^^^^^^^^^^^^

A complete example demonstrating DREAM database usage is available at:
``examples/dream_database_demo.py``

Run it with:

.. code-block:: bash

    python examples/dream_database_demo.py