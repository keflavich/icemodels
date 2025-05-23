Examples
========

This section contains additional practical examples of using IceModels for various tasks.

.. include:: ../auto_examples/index.rst

The code for these examples can be found in the ``examples`` directory of the documentation.

Temperature-dependent Ice Analysis
----------------------------------

Analyzing how ice spectra change with temperature:

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid
    wavelength = np.linspace(1, 28, 1000) * u.um

    # Get the default spectrum and interpolate it to our wavelength grid
    default_spectrum = icemodels.core.phx4000['fnu']
    default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
    f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
    spectrum = f(wavelength)

    # Load CO data at different temperatures
    temperatures = [10, 20, 30, 40]
    spectra = []

    # Calculate spectra for each temperature
    for temp in temperatures:
        data = icemodels.load_molecule_ocdb('co', temperature=temp)
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

Ice Mixture Analysis
--------------------

Analyzing ice mixtures with different ratios:

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid
    wavelength = np.linspace(1, 28, 1000) * u.um

    # Get the default spectrum and interpolate it to our wavelength grid
    default_spectrum = icemodels.core.phx4000['fnu']
    default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
    f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
    spectrum = f(wavelength)

    # Load components
    h2o = icemodels.load_molecule('h2o')
    co2 = icemodels.load_molecule('co2')

    # Define mixture ratios (H2O:CO2)
    ratios = [(1, 0.2), (1, 0.5), (1, 1)]
    base_column = 1e17 * u.cm**-2

    plt.figure(figsize=(10, 6))
    for h2o_ratio, co2_ratio in ratios:
        # Calculate individual spectra
        h2o_spec = icemodels.absorbed_spectrum(
            ice_column=base_column * h2o_ratio,
            ice_model_table=h2o,
            molecular_weight=18*u.Da,
            xarr=wavelength,
            spectrum=spectrum
        )
        co2_spec = icemodels.absorbed_spectrum(
            ice_column=base_column * co2_ratio,
            ice_model_table=co2,
            molecular_weight=44*u.Da,
            xarr=wavelength,
            spectrum=spectrum
        )
        # Combined spectrum
        combined = h2o_spec * co2_spec
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

    # Plot the spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, spectrum, label='CO2 Ice')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Normalized Flux')
    plt.title('CO2 Ice Spectrum with JWST/MIRI Filters')
    plt.legend()

    # Calculate and print filter fluxes
    filter_ids = ['JWST/MIRI.F560W', 'JWST/NIRCam.F444W']
    # Get filter transmission data
    transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}
    filter_fluxes = {}
    for filter_id in filter_ids:
        flux = icemodels.fluxes_in_filters(
            xarr=wavelength,
            modeldata=spectrum,
            filterids=[filter_id],
            transdata=transdata
        )
        filter_fluxes[filter_id] = flux
        print(f"Flux through {filter_id}: {flux}")

    plt.show()

CDE Correction Example
----------------------

The continuous distribution of ellipsoids (CDE) correction accounts for the fact that ice grains in space are not perfect spheres.
This correction modifies the absorption spectrum to account for the distribution of grain shapes, which can significantly affect
the optical properties of the ice.

.. plot::
    :include-source:

    import icemodels
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d

    # Create a common wavelength grid
    wavelength = np.linspace(1, 28, 1000) * u.um

    # Load data
    co_data = icemodels.load_molecule('co')

    # Get the default spectrum and interpolate it to our wavelength grid
    default_spectrum = icemodels.core.phx4000['fnu']
    default_wavelength = u.Quantity(icemodels.core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
    f = interp1d(default_wavelength, default_spectrum, bounds_error=False, fill_value=1.0)
    spectrum = f(wavelength)

    # Calculate absorption spectra with and without CDE correction
    spectrum_no_cde = icemodels.absorbed_spectrum(
        ice_column=1e17 * u.cm**-2,
        ice_model_table=co_data,
        molecular_weight=28*u.Da,
        xarr=wavelength,
        spectrum=spectrum,
        return_tau=True
    )

    # Interpolate n and k onto our wavelength grid
    f_n = interp1d(co_data['Wavelength'], co_data['n'], bounds_error=False, fill_value=1.0)
    f_k = interp1d(co_data['Wavelength'], co_data['k'], bounds_error=False, fill_value=0.0)
    n = f_n(wavelength)
    k = f_k(wavelength)
    m = n + 1j * k

    # Calculate the CDE-corrected optical depth
    freq = wavelength.to(u.cm**-1, u.spectral())
    wl = 1.e4/freq
    m2 = m**2.0
    im_part = ((m2/(m2-1.0))*np.log(m2)).imag
    spectrum_cde = (4.0*np.pi/wl)*im_part

    # Plot original vs CDE-corrected absorption
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, spectrum_no_cde, label='Without CDE')
    plt.plot(wavelength, spectrum_cde, label='With CDE')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Optical Depth')
    plt.title('Effect of CDE Correction on CO Ice')
    plt.legend()
    plt.show()

Stellar Atmosphere Comparison
-----------------------------

Comparing ice absorption features against different stellar atmosphere temperatures:

.. plot::
    :include-source:

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
