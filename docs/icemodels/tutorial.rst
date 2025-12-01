Tutorial
========

This tutorial will walk you through common tasks using IceModels.

Basic Ice Spectrum Analysis
---------------------------

Example creating a Phooenix 4000 K stellar spectrum with CO2 absorption:

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
    # For this example, we'll use the built-in data instead of downloading OCDB
    # to avoid timeouts during documentation builds
    import os
    
    # Use built-in data (single temperature, but demonstrates the concept)
    co_data = icemodels.load_molecule('co')
    
    # Calculate spectrum with the built-in data
    spec = icemodels.absorbed_spectrum(
        ice_column=1e17 * u.cm**-2,
        ice_model_table=co_data,
        molecular_weight=28*u.Da,
        xarr=wavelength,
        spectrum=spectrum
    )
    
    # For demonstration, show the same spectrum labeled as "10K" (the built-in data temperature)
    # In practice, you would download OCDB data for multiple temperatures
    spectra = [spec]
    temperatures = [10]
    
    # Note: To get data at multiple temperatures, use:
    # icemodels.download_all_ocdb()
    # Then search for files with glob: glob.glob(f'{icemodels.optical_constants_cache_dir}/ocdb_*CO*{temp}K*.txt')

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
------------------------------------

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

    # Calculate optical depths (tau) for each component
    # When combining ices, we sum their optical depths, not multiply spectra
    h2o_tau = icemodels.absorbed_spectrum(
        ice_column=h2o_column,
        ice_model_table=h2o_data,
        molecular_weight=18*u.Da,
        xarr=wavelength,
        spectrum=spectrum_base,
        return_tau=True
    )

    co_tau = icemodels.absorbed_spectrum(
        ice_column=co_column,
        ice_model_table=co_data,
        molecular_weight=28*u.Da,
        xarr=wavelength,
        spectrum=spectrum_base,
        return_tau=True
    )

    # Combined optical depth is the sum of individual optical depths
    combined_tau = h2o_tau + co_tau
    
    # Apply the combined optical depth to get the final spectrum
    combined_spectrum = spectrum_base * np.exp(-combined_tau)

    # Calculate individual absorbed spectra for comparison
    h2o_spectrum = spectrum_base * np.exp(-h2o_tau)
    co_spectrum = spectrum_base * np.exp(-co_tau)

    # Plot all components
    plt.figure(figsize=(12, 8))
    plt.plot(wavelength, spectrum_base, label='No ice', linestyle='--', alpha=0.5)
    plt.plot(wavelength, h2o_spectrum, label='H2O only')
    plt.plot(wavelength, co_spectrum, label='CO only')
    plt.plot(wavelength, combined_spectrum, label='H2O + CO (combined)', linewidth=2)
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Flux')
    plt.title('Multi-component Ice Spectrum')
    plt.legend()
    plt.show()

Using Gaussian Components
-------------------------

Sometimes it's useful to model ice features using Gaussian components.  However, this example shows that the Gaussian modeling approach isn't correct right now:

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid
    wavelength = np.linspace(4, 4.5, 1000) * u.um

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
    plt.plot(wavelength, spectrum, label='Laboratory data')
    plt.plot(wavelength, gauss_spectrum, label='Gaussian model')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('CO2 Ice: Data vs Gaussian Model')
    plt.legend()
    plt.show()

Working with Different Databases
--------------------------------

IceModels can access data from multiple databases. Here's how to compare data from different sources and retrieve important metadata like author and reference information:

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d
    import os

    # Create a common wavelength grid
    # For CO, this should be 4.5-5 microns
    wavelength = np.linspace(4.5, 5, 1000) * u.um

    # Get CO data from different sources - start with built-in
    co_builtin = icemodels.load_molecule('co')  # Built-in data from Palumbo

    # Only download external databases if not building docs
    skip_downloads = os.environ.get('ICEMODELS_SKIP_DOWNLOADS') == 'true'
    
    co_ocdb = None
    co_lida = None
    
    if not skip_downloads:
        # Download data from different databases (if not already cached)
        icemodels.download_all_ocdb()
        icemodels.download_all_lida()

        # Find and load the OCDB data for CO at 10K
        import glob
        ocdb_files = glob.glob(f'{icemodels.optical_constants_cache_dir}/ocdb_*CO*10K*.txt')
        if ocdb_files:
            co_ocdb = icemodels.read_ocdb_file(ocdb_files[0])

        # Load LIDA data for CO at 15K
        lida_files = glob.glob(f'{icemodels.optical_constants_cache_dir}/*CO*15K*.txt')
        # Filter for LIDA files (they don't start with 'ocdb_' or 'dream_')
        lida_files = [f for f in lida_files if not any(x in f for x in ['ocdb_', 'dream_'])]
        if lida_files:
            co_lida = icemodels.read_lida_file(lida_files[0])

    # Extract metadata from the tables
    # The metadata is stored in the table's .meta dictionary
    builtin_author = co_builtin.meta.get('author', 'Palumbo')
    builtin_ref = co_builtin.meta.get('reference', 'Default')
    
    # Print metadata for user inspection
    print(f"Built-in data: {builtin_author}, {builtin_ref}")
    
    if co_ocdb is not None:
        ocdb_author = co_ocdb.meta.get('author', 'Unknown')
        ocdb_ref = co_ocdb.meta.get('reference', 'OCDB')
        ocdb_temp = co_ocdb.meta.get('temperature', '10K')
        print(f"OCDB data: {ocdb_author}, {ocdb_ref}, Temperature: {ocdb_temp}")
    
    if co_lida is not None:
        lida_author = co_lida.meta.get('author', 'Unknown')
        lida_ref = co_lida.meta.get('reference', 'LIDA')
        lida_temp = co_lida.meta.get('temperature', '15K')
        print(f"LIDA data: {lida_author}, {lida_ref}, Temperature: {lida_temp}")

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
    
    if co_ocdb is not None:
        n_ocdb, k_ocdb = interpolate_constants(co_ocdb)
    
    if co_lida is not None:
        n_lida, k_lida = interpolate_constants(co_lida)

    # Create labels with author information from metadata
    builtin_label = f'{builtin_author}'

    # Plot optical constants from each source
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(wavelength, n_builtin, label=builtin_label)
    if co_ocdb is not None:
        ocdb_label = f'{ocdb_author} ({ocdb_temp})'
        plt.plot(wavelength, n_ocdb, label=ocdb_label)
    if co_lida is not None:
        lida_label = f'{lida_author} ({lida_temp})'
        plt.plot(wavelength, n_lida, label=lida_label, linestyle='--')
    plt.ylabel('n')
    plt.title('CO Ice Optical Constants from Different Sources')
    plt.legend()

    plt.subplot(212)
    plt.semilogy(wavelength, k_builtin, label=builtin_label)
    if co_ocdb is not None:
        plt.semilogy(wavelength, k_ocdb, label=ocdb_label)
    if co_lida is not None:
        plt.semilogy(wavelength, k_lida, label=lida_label, linestyle='--')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('k')
    plt.legend()
    plt.show()


These examples demonstrate the main functionality of IceModels. For more advanced usage and specific applications, see the :doc:`examples` section. For database-specific examples including the DREAM database, see the :doc:`databases` section.