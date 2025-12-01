Examples
========

This section contains additional practical examples of using IceModels for various tasks.

The code for these examples can be found in the ``examples`` directory of the documentation.

Temperature-dependent Ice Analysis
----------------------------------

Analyzing how ice spectra change with temperature:

.. plot::
    :include-source:

    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value')

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid
    wavelength = np.linspace(4.5, 4.8, 1000) * u.um

    # Get the default spectrum and interpolate it to our wavelength grid
    default_spectrum = icemodels.core.phx4000['fnu']
    default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
    f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
    spectrum = f(wavelength)

    # Load CO data at different temperatures
    import glob
    
    # Download all OCDB data (if not already cached)
    icemodels.download_all_ocdb()
    temperatures = [10, 12, 12.5, 25, 30]
    
    spectra = []

    # Calculate spectra for each temperature
    for temp in temperatures:
        # Find the OCDB file for CO at this temperature
        ocdb_files = glob.glob(f'{icemodels.optical_constants_cache_dir}/ocdb*_CO_*{temp}K*.txt')
        if ocdb_files:
            data = icemodels.read_ocdb_file(ocdb_files[0])
            spec = icemodels.absorbed_spectrum(
                ice_column=1e18 * u.cm**-2,
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

Ice Mixture Analysis
--------------------

Analyzing ice mixtures with different ratios:

.. plot::
    :include-source:

    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value')

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid
    wavelength = np.linspace(2.5, 4.5, 1000) * u.um

    # Get the default spectrum and interpolate it to our wavelength grid
    default_spectrum = icemodels.core.phx4000['fnu']
    default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
    f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
    spectrum = f(wavelength)

    # Load components
    h2o = icemodels.load_molecule('h2o')
    co2 = icemodels.load_molecule('co2')

    # Define mixture ratios (H2O:CO2) - these represent molecular ratios
    # The total column density is conserved across all mixtures
    ratios = [(1, 0.2), (1, 0.5), (1, 1)]
    total_column = 1e18 * u.cm**-2  # Total ice column density

    plt.figure(figsize=(10, 6))
    for h2o_ratio, co2_ratio in ratios:
        # Calculate the weighted mean opacity
        # Each component is calculated with ice_column=1, weighted by its ratio,
        # then divided by total ratio to get mean opacity per molecule
        total_ratio = h2o_ratio + co2_ratio
        
        h2o_tau_per_molecule = icemodels.absorbed_spectrum(
            ice_column=1,  # Calculate per molecule
            ice_model_table=h2o,
            molecular_weight=18*u.Da,
            xarr=wavelength,
            spectrum=spectrum,
            return_tau=True
        )
        co2_tau_per_molecule = icemodels.absorbed_spectrum(
            ice_column=1,  # Calculate per molecule
            ice_model_table=co2,
            molecular_weight=44*u.Da,
            xarr=wavelength,
            spectrum=spectrum,
            return_tau=True
        )
        
        # Weighted mean opacity (conserves total column density)
        mean_tau = (h2o_tau_per_molecule * h2o_ratio + 
                    co2_tau_per_molecule * co2_ratio) / total_ratio
        
        # Apply to the total column density
        combined_tau = mean_tau * total_column
        combined = spectrum * np.exp(-combined_tau)
        plt.plot(wavelength, combined,
                label=f'H2O:CO2 = {h2o_ratio}:{co2_ratio}')

    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('H2O:CO2 Ice Mixtures')
    plt.legend()
    plt.show()

Filter Analysis
---------------

Analyzing ice spectra through different filters:

.. plot::
    :include-source:

    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value')

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d
    from astroquery.svo_fps import SvoFps

    # Create a common wavelength grid
    wavelength = np.linspace(1, 28, 1000) * u.um

    # Get the default spectrum and interpolate it to our wavelength grid
    default_spectrum = icemodels.core.phx4000['fnu']
    default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
    f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
    spectrum_base = f(wavelength)

    # Calculate spectrum
    co2_data = icemodels.load_molecule('co2')
    spectrum = icemodels.absorbed_spectrum(
        ice_column=1e17 * u.cm**-2,
        ice_model_table=co2_data,
        molecular_weight=44*u.Da,
        xarr=wavelength,
        spectrum=spectrum_base
    )

    # Calculate and print filter fluxes
    filter_ids = ['JWST/MIRI.F560W', 'JWST/NIRCam.F444W']
    # Get filter transmission data
    transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}
    filter_fluxes = {}
    filter_centers = {}
    
    for filter_id in filter_ids:
        flux = icemodels.fluxes_in_filters(
            xarr=wavelength,
            modeldata=spectrum,
            filterids=[filter_id],
            transdata=transdata
        )
        filter_fluxes[filter_id] = flux
        # Get the effective wavelength (center) of the filter
        trans = transdata[filter_id]
        filter_centers[filter_id] = np.average(trans['Wavelength'], weights=trans['Transmission'])
        print(f"Flux through {filter_id}: {flux}")

    # Plot the spectrum with filter measurements overlaid
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, spectrum, label='CO2 Ice Spectrum', linewidth=2)
    
    # Plot filter measurements as points
    for filter_id in filter_ids:
        plt.plot(filter_centers[filter_id], filter_fluxes[filter_id], 
                'o', markersize=10, label=f'{filter_id}')
    
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('CO2 Ice Spectrum with JWST Filter Measurements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


Stellar Atmosphere Comparison
-----------------------------

Comparing ice absorption features against different stellar atmosphere temperatures:

.. plot::
    :include-source:

    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value')

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from icemodels.core import atmo_model

    # Create wavelength grid focused on 1-5 microns
    wavelength = np.linspace(1, 5, 1000) * u.um

    # Load ice data
    co_data = icemodels.load_molecule('co')
    h2o_data = icemodels.load_molecule('h2o')
    co2_data = icemodels.load_molecule('co2')

    # Create figure with subplots for each stellar temperature
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    temperatures = [2000, 3000, 4000, 5000]

    for ax, temp in zip(axes, temperatures):
        # Get stellar atmosphere model for this temperature
        mod = atmo_model(temp, xarr=wavelength)
        spectrum = mod['fnu']

        # Calculate absorbed spectra for each ice
        co_spec = icemodels.absorbed_spectrum(
            ice_column=1e18 * u.cm**-2,
            ice_model_table=co_data,
            molecular_weight=28*u.Da,
            xarr=wavelength,
            spectrum=spectrum
        )

        h2o_spec = icemodels.absorbed_spectrum(
            ice_column=1e19 * u.cm**-2,
            ice_model_table=h2o_data,
            molecular_weight=18*u.Da,
            xarr=wavelength,
            spectrum=spectrum
        )

        co2_spec = icemodels.absorbed_spectrum(
            ice_column=1e18 * u.cm**-2,
            ice_model_table=co2_data,
            molecular_weight=44*u.Da,
            xarr=wavelength,
            spectrum=spectrum
        )

        # Plot results
        ax.plot(wavelength, spectrum, 'k--', label='Stellar', alpha=0.5)
        ax.plot(wavelength, co_spec, label='CO')
        ax.plot(wavelength, h2o_spec, label='H2O')
        ax.plot(wavelength, co2_spec, label='CO2')

        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'{temp}K Stellar Atmosphere')
        ax.legend()

    axes[-1].set_xlabel('Wavelength (μm)')
    plt.tight_layout()
    plt.show()
