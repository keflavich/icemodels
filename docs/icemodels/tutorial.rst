Tutorial
========

This tutorial will walk you through common tasks using IceModels.

Basic Ice Spectrum Analysis
--------------------------

Let's analyze a simple CO2 ice spectrum:

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid from 1-5 microns
    wavelength = np.linspace(1, 5, 1000) * u.um

    # Get the default spectrum and interpolate it to our wavelength grid
    default_spectrum = icemodels.core.phx4000['fnu']
    default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
    f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
    spectrum = f(wavelength)

    # Load CO data at different temperatures
    temperatures = [10, 20, 30, 40]
    spectra = []

    # First, download all OCDB data (if not already downloaded)
    icemodels.download_all_ocdb()

    # Calculate spectra for each temperature
    for temp in temperatures:
        # Find the appropriate file for CO at this temperature
        import glob
        files = glob.glob(f'{icemodels.optical_constants_cache_dir}/*CO*{temp}K*.txt')
        if files:
            data = icemodels.read_ocdb_file(files[0])
            spec = icemodels.absorbed_spectrum(
                ice_column=1e17 * u.cm**-2,
                ice_model_table=data,
                molecular_weight=28*u.Da,
                xarr=wavelength,
                spectrum=spectrum
            )
            spectra.append(spec)

    # Create the plot
    plt.figure(figsize=(10, 6))
    for temp, spec in zip(temperatures, spectra):
        plt.plot(wavelength, spec, label=f'{temp}K')

    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('CO Ice Spectrum at Different Temperatures')
    plt.legend()
    plt.show()

Working with Multiple Ice Components
----------------------------------

Now let's create a more complex example with multiple ice components:

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid
    wavelength = np.linspace(2.5, 5, 2000) * u.um

    # Get the default spectrum and interpolate it to our wavelength grid
    default_spectrum = icemodels.core.phx4000['fnu']
    default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
    f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
    spectrum_base = f(wavelength)

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
        molecular_weight=18*u.Da,
        xarr=wavelength,
        spectrum=spectrum_base
    )

    co_spectrum = icemodels.absorbed_spectrum(
        ice_column=co_column,
        ice_model_table=co_data,
        molecular_weight=28*u.Da,
        xarr=wavelength,
        spectrum=spectrum_base
    )

    # Combined spectrum is the product of individual spectra
    combined_spectrum = h2o_spectrum * co_spectrum

    # Plot all components
    plt.figure(figsize=(12, 8))
    plt.plot(wavelength, h2o_spectrum, label='H2O')
    plt.plot(wavelength, co_spectrum, label='CO')
    plt.plot(wavelength, combined_spectrum, label='Combined')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('Multi-component Ice Spectrum')
    plt.legend()
    plt.show()

Using Gaussian Components
-----------------------

Sometimes it's useful to model ice features using Gaussian components:

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid
    wavelength = np.linspace(1, 5, 1000) * u.um

    # Get the default spectrum and interpolate it to our wavelength grid
    default_spectrum = icemodels.core.phx4000['fnu']
    default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
    f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
    spectrum_base = f(wavelength)

    # Load CO2 data for comparison
    co2_data = icemodels.load_molecule('co2')

    # Calculate real CO2 spectrum
    spectrum = icemodels.absorbed_spectrum(
        ice_column=1e17 * u.cm**-2,
        ice_model_table=co2_data,
        molecular_weight=44*u.Da,
        xarr=wavelength,
        spectrum=spectrum_base
    )

    # Define Gaussian parameters for CO2 stretch mode
    center = 4.27 * u.um
    width = 0.1 * u.um
    bandstrength = 1e-16 * u.cm
    column = 1e17 * u.cm**-2

    # Calculate Gaussian spectrum
    gauss_spectrum = icemodels.absorbed_spectrum_Gaussians(
        ice_column=column,
        center=center,
        width=width,
        ice_bandstrength=bandstrength,
        xarr=wavelength,
        spectrum=spectrum_base
    )

    # Compare with actual CO2 data
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, spectrum, label='Real data')
    plt.plot(wavelength, gauss_spectrum, label='Gaussian model')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('CO2 Ice: Data vs Gaussian Model')
    plt.legend()
    plt.show()

Working with Different Databases
--------------------------------

IceModels can access data from multiple databases. Here's how to compare data from different sources:

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid
    # For CO, this should be 4.5-5 microns
    wavelength = np.linspace(4.5, 5, 1000) * u.um

    # First, download all OCDB data (if not already downloaded)
    icemodels.download_all_ocdb()

    # Get CO data from different sources
    co_builtin = icemodels.load_molecule('co')  # Built-in data from Palumbo

    # Find and load the OCDB data for CO at 10K
    import glob
    ocdb_files = glob.glob(f'{icemodels.optical_constants_cache_dir}/*CO*10K*.txt')
    if ocdb_files:
        co_ocdb = icemodels.read_ocdb_file(ocdb_files[0])
    else:
        raise ValueError("Could not find CO data at 10K in OCDB cache")

    # Create interpolation functions for each dataset
    def interpolate_constants(data):
        if 'Wavelength (m)' in data.colnames:
            wl_col = 'Wavelength (m)'
        elif 'Wavelength (µm)' in data.colnames:
            wl_col = 'Wavelength (µm)'
        else:
            wl_col = 'Wavelength'

        if 'k₁' in data.colnames:
            k_col = 'k₁'
        else:
            k_col = 'k'

        if 'n₁' in data.colnames:
            n_col = 'n₁'
        else:
            n_col = 'n'

        f_n = interp1d(data[wl_col], data[n_col], bounds_error=False, fill_value=1.0)
        f_k = interp1d(data[wl_col], data[k_col], bounds_error=False, fill_value=0.0)
        return f_n(wavelength), f_k(wavelength)

    # Interpolate all datasets to common wavelength grid
    n_builtin, k_builtin = interpolate_constants(co_builtin)
    n_ocdb, k_ocdb = interpolate_constants(co_ocdb)

    # Plot optical constants from each source
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(wavelength, n_builtin, label='Built-in (Palumbo)')
    plt.plot(wavelength, n_ocdb, label='OCDB (10 K)')
    plt.ylabel('n')
    plt.title('CO Ice Optical Constants')
    plt.legend()

    plt.subplot(212)
    plt.semilogy(wavelength, k_builtin, label='Built-in (Palumbo)')
    plt.semilogy(wavelength, k_ocdb, label='OCDB (10 K)')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('k')
    plt.legend()
    plt.show()

These examples demonstrate the main functionality of IceModels. For more advanced usage and specific applications, see the :doc:`examples` section.