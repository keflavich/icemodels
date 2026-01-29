import os
import multiprocessing as mp
from functools import partial

import numpy as np
import astropy.units as u
from astropy.table import Table
from tqdm.contrib.concurrent import process_map

from icemodels import fluxes_in_filters, atmo_model
from astroquery.svo_fps import SvoFps

# Get JWST filter list
jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')

# Default JWST filters (same as in absorbance_in_filters.py)
cmd_x_default = (
    'JWST/NIRCam.F115W',
    'JWST/NIRCam.F150W',
    'JWST/NIRCam.F162M',
    'JWST/NIRCam.F182M',
    'JWST/NIRCam.F187N',
    'JWST/NIRCam.F200W',
    'JWST/NIRCam.F210M',
    'JWST/NIRCam.F212N',
    'JWST/NIRCam.F250M',
    'JWST/NIRCam.F277W',
    'JWST/NIRCam.F300M',
    'JWST/NIRCam.F323N',
    'JWST/NIRCam.F322W2',
    'JWST/NIRCam.F335M',
    'JWST/NIRCam.F356W',
    'JWST/NIRCam.F360M',
    'JWST/NIRCam.F405N',
    'JWST/NIRCam.F410M',
    'JWST/NIRCam.F430M',
    'JWST/NIRCam.F444W',
    'JWST/NIRCam.F460M',
    'JWST/NIRCam.F466N',
    'JWST/NIRCam.F470N',
    'JWST/NIRCam.F480M',
    'JWST/MIRI.F560W',
    'JWST/MIRI.F770W',
    'JWST/MIRI.F1000W',
    'JWST/MIRI.F1065C',
    'JWST/MIRI.F1130W',
    'JWST/MIRI.F1140C',
    'JWST/MIRI.F1280W',
    'JWST/MIRI.F1500W',
    'JWST/MIRI.F1550C',
    'JWST/MIRI.F1800W',
    'JWST/MIRI.F2100W',
    'JWST/MIRI.F2300C',
    'JWST/MIRI.F2550W',
)

# Define stellar parameter ranges
# Temperature range from cool M dwarfs to hot O stars
# temperatures = np.linspace(2000, 50000, 50)  # K
temperatures = np.geomspace(2000, 40000, 50)

# Wavelength grid for spectral calculations
# Extended range to cover all JWST filters: F070W (0.699 μm) to F2550W (25.152 μm)
xarr = np.linspace(0.6*u.um, 28.0*u.um, 25000)


def process_stellar_model(args, cmd_x=None, transdata=None, filter_data=None):
    """
    Process a stellar atmosphere model to compute magnitudes in filters.

    Parameters
    ----------
    args : tuple
        Arguments containing (temperature, xarr, cmd_x, transdata, filter_data, basepath)
    cmd_x : tuple or list of str, optional
        List of filter IDs to use. If not provided, defaults to cmd_x_default.
    transdata : dict, optional
        Dictionary mapping filter IDs to their transmission data.
    filter_data : dict, optional
        Dictionary mapping filter IDs to their zero points.

    Returns
    -------
    mag_row : dict
        Dictionary with computed magnitudes for each filter and stellar parameters.
    """
    temperature, xarr, user_cmd_x, user_transdata, user_filter_data, basepath = args

    # Use provided parameters or defaults
    if cmd_x is None:
        cmd_x = user_cmd_x if user_cmd_x is not None else cmd_x_default
    if transdata is None:
        transdata = user_transdata
    if filter_data is None:
        filter_data = user_filter_data

    try:
        # Generate stellar atmosphere model
        stellar_model = atmo_model(temperature, xarr=xarr)
    except IndexError:
        raise ValueError(f"Temperature {temperature}K is out of range for the model")

    try:

        # Calculate fluxes in filters
        fluxes = fluxes_in_filters(xarr, stellar_model['fnu'].quantity,
                                   filterids=cmd_x, transdata=transdata)

        # Calculate magnitudes
        mags = {}
        for filt in cmd_x:
            if filt in fluxes and filt in filter_data:
                # Convert flux to magnitude using zero point
                mag = -2.5 * np.log10(fluxes[filt] / u.Quantity(filter_data[filt], u.Jy))
                # Store with simplified filter name
                if 'JWST/' in filt:
                    mags[filt.replace('JWST/NIRCam.', '').replace('JWST/MIRI.', '')] = mag
                else:
                    mags[filt] = mag

        # Create result row
        mag_row = {
            'temperature': temperature,
            'model_type': 'stellar_atmosphere',
            'spectral_type': get_spectral_type(temperature),
        }
        # Add magnitudes for each filter
        mag_row.update(mags)

        return mag_row

    except Exception as ex:
        print(f"Error processing temperature {temperature}K: {ex}")
        raise


def get_spectral_type(temperature):
    """
    Convert effective temperature to approximate spectral type.

    Parameters
    ----------
    temperature : float
        Effective temperature in Kelvin

    Returns
    -------
    str
        Approximate spectral type
    """
    if temperature < 3700:
        return 'M'
    elif temperature < 5200:
        return 'K'
    elif temperature < 6000:
        return 'G'
    elif temperature < 7500:
        return 'F'
    elif temperature < 10000:
        return 'A'
    elif temperature < 30000:
        return 'B'
    else:
        return 'O'


if __name__ == '__main__':
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the filter set globally for reuse
    cmd_x = cmd_x_default
    filter_data = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in cmd_x}
    transdata = {fid: SvoFps.get_transmission_data(fid) for fid in cmd_x}

    print(f"Processing {len(temperatures)} stellar models...")
    print(f"Temperature range: {temperatures.min():.0f}K to {temperatures.max():.0f}K")
    print(f"Using {len(cmd_x)} JWST filters")

    # Create list of all stellar models to process
    all_models = []
    for temp in temperatures:
        all_models.append((temp, xarr, cmd_x, transdata, filter_data, basepath))

    # Process all models in parallel
    results = process_map(partial(process_stellar_model, cmd_x=cmd_x, transdata=transdata, filter_data=filter_data),
                          all_models,
                          max_workers=mp.cpu_count(),
                          desc="Processing stellar models",
                          unit="model")

    # Combine results
    mag_rows = []
    for result in results:
        if result:
            mag_rows.append(result)

    # Create output table
    mag_tbl = Table(mag_rows)

    # Ensure output directory exists
    output_dir = os.path.join(basepath, 'icemodels', 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Write table
    output_file = f'{output_dir}/stellar_colors_in_filters.ecsv'
    mag_tbl.write(output_file, overwrite=True)

    print(f"Results written to {output_file}")
    print(f"Table contains {len(mag_tbl)} stellar models")
    print(f"Available columns: {list(mag_tbl.colnames)}")

    # Add some useful indices
    mag_tbl.add_index('temperature')
    mag_tbl.add_index('spectral_type')

    # Print summary statistics
    print("\nSummary by spectral type:")
    for spec_type in np.unique(mag_tbl['spectral_type']):
        mask = mag_tbl['spectral_type'] == spec_type
        temp_range = f"{mag_tbl[mask]['temperature'].min():.0f}-{mag_tbl[mask]['temperature'].max():.0f}K"
        count = np.sum(mask)
        print(f"  {spec_type}: {count} models ({temp_range})")

    # Verify we have magnitudes for key filters
    key_filters = ['F212N', 'F444W', 'F1000W']
    for filt in key_filters:
        if filt in mag_tbl.colnames:
            raise ValueError(f"Missing magnitudes for filter {filt}")
