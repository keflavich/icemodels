import os
import re
import glob
import shlex
import json
import numpy as np
from bs4 import BeautifulSoup
import mysg  # Tom Robitaille's YSO grid tool
from astropy.table import Table
from astropy.io import ascii
from astropy import units as u
from astropy import log
import matplotlib.pyplot as pl
import requests
from astroquery.svo_fps import SvoFps
import astropy
import astropy.io.ascii.core
from tqdm.auto import tqdm
from pylatexenc.latex2text import LatexNodes2Text
from molmass import Formula

cache = {}
optical_constants_cache_dir = os.path.dirname(__file__) + "/data"

molecule_data = {
    'ch3oh': {
        'url': ('https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/'
                'main/nk/ch3oh-a-Gerakines2020.txt'),
        'molwt': (12 + 4 + 16) * u.Da,
    },
    'co2': {
        'url': ('https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/'
                'main/nk/co2-a-Gerakines2020.txt'),
        'molwt': (12 + 2 * 16) * u.Da,
    },
    'ch4': {
        'url': ('https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/'
                'main/nk/ch4-a-Gerakines2020.txt'),
        'molwt': (12 + 4) * u.Da,
    },
    'co': {
        'url': ('https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/'
                'main/nk/co-a-Palumbo2006.txt'),
        'molwt': (12 + 16) * u.Da,
    },
    'h2o': {
        'url': ('https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/'
                'main/nk/h2o-a-Hudgins1993.txt'),
        'molwt': (16 + 2) * u.Da,
    },
    'h2o_b': {
        'url': ('https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/'
                'main/nk/h2o_Rocha.txt'),
        'molwt': (16 + 2) * u.Da,
    },
    'nh3': {
        'url': ('https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/'
                'main/nk/nh3_Roser_2021.txt'),
        'molwt': (14 + 3) * u.Da,
        'density': 0.8 * u.g / u.cm**3,  # Satorre+2013 via Roser+2021
    },
}

astrochem_molecule_data = {
    'co': {'url': 'https://ocdb.smce.nasa.gov/dataset/89/'},
    'co_old': {
        'url': 'http://www.astrochem.org/data/CO/CO',
        'molwt': (12 + 16) * u.Da,
    },
    'co_hudgins': {
        'url': 'http://www.astrochem.org/data/CO/CO.Hudgins',
        'molwt': (12 + 16) * u.Da,
    },
}

univap_molecule_data = {
    'co': {
        'url': ("https://www.dropbox.com/s/dgufhmfwleak4ce/G1.txt?dl=1"),
        'molwt': (12 + 16) * u.Da,
    },
    'h2o:co.1:0.4': {
        'url': "http://www1.univap.br/gaa/nkabs-database/D8a.txt",
        'molwt': (12 + 16) * u.Da,
    },
    'co2': {
        'url': ('https://www.dropbox.com/s/hoo9s01knc7p3su/G2.txt?dl=1'),
        'molwt': (16 * 2 + 12) * u.Da,  # pilling
    },
    'h2o_amorphous': {
        'url': ('https://www.dropbox.com/s/dcpqq20766fdf2i/L1.txt?dl=1'),
        'molwt': (16 + 2) * u.Da,
    },
    'h2o_crystal': {
        'url': 'http://www1.univap.br/gaa/nkabs-database/L2.txt',
        'molwt': (16 + 2) * u.Da,
    },
}


def atmo_model(temperature, xarr=np.linspace(1, 28, 15000) * u.um):
    """
    Use https://github.com/astrofrog/mysg to load Kurucz & Phoenix models and interpolate them
    to a specified temperature.

    Then, interpolate those onto a finely-sampled(ish) wavelength grid that covers the JWST filters.

    (the default spectral grid has essentially no sampling from 10-25 microns)
    """
    if mysg is None:
        raise ImportError("mysg module is not available")
    mod = Table(mysg.atmosphere.interp_atmos(temperature))
    mod['nu'].unit = u.Hz
    mod['fnu'].unit = u.erg / u.s / u.cm**2 / u.Hz
    inds = np.argsort(mod['nu'])
    xarrhz = xarr.to(u.Hz, u.spectral())
    mod = Table({
        'fnu': np.interp(xarrhz, mod['nu'].quantity[inds],
                         mod['fnu'].quantity[inds], left=0, right=0),
        'nu': xarrhz
    }, meta={'temperature': temperature})

    return mod


phx4000 = atmo_model(4000)


def load_molecule(molname):
    """
    Load a molecule based on its name from the dictionary of molecular data files above.
    """
    if molname in cache:
        return cache[molname]
    url = molecule_data[molname]['url']
    consts = Table.read(url, format='ascii', data_start=13)
    if 'col1' in consts.colnames:
        consts['col1'].unit = u.um
        consts.rename_column('col1', 'Wavelength')
        consts.rename_column('col2', 'n')
        consts.rename_column('col3', 'k')
    elif 'WaveNum' in consts.colnames:
        consts['Wavelength'] = consts['WaveNum'].quantity.to(
            u.um, u.spectral())
    if 'density' in molecule_data[molname]:
        consts.meta['density'] = molecule_data[molname]['density']
    else:
        lines = requests.get(url).text.split('\n')
        for line in lines:
            if not line.startswith("#"):
                break
        density = float(line.split()[1]) * u.g / u.cm**3
        consts.meta['density'] = density
    cache[molname] = consts
    return consts


def get_univap_meta_table():
    """Get the metadata table from the Univap database."""
    if 'univap_meta_table' in cache:
        return cache['univap_meta_table']
    meta1 = Table.read('https://www1.univap.br/gaa/nkabs-database/data.htm',
                       format='html', htmldict={'table_id': 1})
    meta2 = Table.read('https://www1.univap.br/gaa/nkabs-database/data.htm',
                       format='html', htmldict={'table_id': 2})
    meta1.rename_column('col1', 'datalabel')
    meta1.rename_column('col2', 'temperature')
    meta1.rename_column('col3', 'sample')
    meta1.rename_column('col4', 'reference')
    meta2.rename_column('col1', 'datalabel')
    meta2.rename_column('col2', 'temperature')
    meta2.rename_column('col3', 'sample')
    meta2.rename_column('col4', 'projectile')
    meta2.rename_column('col5', 'flucence')
    meta2.rename_column('col6', 'reference')
    meta_table = Table.vstack([meta1, meta2])
    cache['univap_meta_table'] = meta_table
    return meta_table


def load_molecule_univap(molname, meta_table=None):
    """
    Load a molecule based on its name from the dictionary of molecular data files above.
    """
    if meta_table is None:
        meta_table = get_univap_meta_table()
    meta_table.add_index('datalabel')

    url = univap_molecule_data[molname]['url']
    molid = url.split('/')[-1].split('.')[0]

    consts = Table.read(url, format='ascii', data_start=3)
    if 'col1' in consts.colnames:
        consts['col1'].unit = u.cm**-1
        consts.rename_column('col1', 'WaveNum')
        consts.rename_column('col2', 'absorbance')
        consts.rename_column('col3', 'k')
        consts.rename_column('col4', 'n')
        consts['Wavelength'] = consts['WaveNum'].quantity.to(
            u.um, u.spectral())
    consts.meta['density'] = 1 * u.g / u.cm**3
    consts.meta['author'] = meta_table.loc[molid]['reference'][0]
    consts.meta['source'] = url
    consts.meta['temperature'] = 10
    consts.meta['molecule'] = molname
    consts.meta['composition'] = meta_table.loc[molid]['sample'][0]
    consts.meta['molwt'] = Formula(meta_table.loc[molid]['sample'][0]).mass

    return consts


# defunct def load_molecule_icedb():
# defunct     response = requests.get('https://icedb.strw.leidenuniv.nl/spectrum/download/754/754_15.0K.txt', verify=False)
# defunct     icedb_co = ascii.read(response.text)
# defunct     pl.plot(icedb_co['col1'], icedb_co['col2'])


def download_all_ocdb(n_ocdb=298, redo=False):
    """
    Retrieve and locally cache all data files from the OCDB.
    """
    S = requests.Session()
    S.get('https://ocdb.smce.nasa.gov/search/ice')

    for ii in tqdm(range(1, n_ocdb + 1)):
        if not redo and len(
            glob.glob(
                f'{optical_constants_cache_dir}/{ii}*')) > 0:
            # note that this can miss important parameters when there are many
            # temperatures
            continue
        resp = S.get(
            f'https://ocdb.smce.nasa.gov/dataset/{ii}/download-data/all')
        for row in resp.text.split("\n"):
            if row.startswith('Composition:'):
                molname = shlex.split(row)[1]
            if row.startswith('Temperature:'):
                temperature = shlex.split(row)[1]
            if row.startswith('Reference:'):
                reference = shlex.split(row)[1].split()[0]

        filename = (f'{optical_constants_cache_dir}/{ii}_{molname}_{temperature}_'
                    f'{reference}.txt')
        filename = filename.replace(
            " ",
            "_").replace(
            "'",
            "").replace(
            '\\',
            '')
        filename = filename.replace('"', '')
        with open(filename, 'w') as fh:
            fh.write(resp.text)


def read_ocdb_file(filename):
    """
    Read an OCDB file downloaded with ``download_all_ocdb()`` and return a table.

    The recommended approach is something like:

    .. code-block:: python

        import icemodels
        icemodels.download_all_ocdb()
        tb = icemodels.read_ocdb_file(
            f'{icemodels.optical_constants_cache_dir}/240_H2O_(1)_25K_Mastrapa.txt'
        )
    """
    for ii in range(5, 15):
        try:
            # new header data appear to be added from time to time
            tb = ascii.read(filename,
                            format='tab', delimiter='\t',
                            header_start=ii, data_start=ii + 1)
            break
        except astropy.io.ascii.core.InconsistentTableError:
            if ii == 14:
                raise ValueError("File appears to be invalid")
            continue

    if 'Wavelength (m)' in tb.colnames:
        tb['Wavelength'] = tb['Wavelength (m)'] * u.um  # micron got truncated
        tb['Wavenumber (cm)'] = tb['Wavenumber'] = (
            tb['Wavelength'].to(u.cm**-1, u.spectral())
        )
    elif 'Wavelength (µm)' in tb.colnames:
        tb['Wavelength'] = tb['Wavelength (µm)'] * u.um
        tb['Wavenumber (cm)'] = tb['Wavenumber'] = (
            tb['Wavelength'].to(u.cm**-1, u.spectral())
        )
    elif 'Wavenumber (cm⁻¹)' in tb.colnames:
        tb['Wavelength'] = (
            tb['Wavenumber (cm⁻¹)'] *
            u.cm**-
            1).to(
            u.um,
            u.spectral())
    elif 'Wavenumber (cm)' in tb.colnames:
        tb['Wavelength'] = (
            tb['Wavenumber (cm)'] *
            u.cm**-
            1).to(
            u.um,
            u.spectral())
    else:
        raise ValueError(f"No wavelength column found in {tb.colnames}")

    if 'k₁' in tb.colnames:
        tb['k'] = tb['k₁']

    if 'k' not in tb.colnames:
        raise ValueError("Table had no opacity column")

    tb.meta['density'] = 1 * u.g / u.cm**3

    with open(filename, 'r') as fh:
        keys = [
            'Reference:',
            'DOI:',
            'Composition:',
            'Temperature:',
            'OCdb page:']
        rows = fh.readlines()
        for row in rows[:20]:
            for key in keys:
                if row.startswith(key):
                    kk = key.lower().strip(":")
                    tb.meta[kk] = (
                        " ".join(
                            row.split(":")[
                                1:])).strip().strip('"')

    if 'reference' in tb.meta:
        tb.meta['author'] = tb.meta['reference'].split()[0]
    if 'composition' in tb.meta:
        tb.meta['molecule'] = tb.meta['composition'].split()[0]

    tb.meta['database'] = 'ocdb'
    tb.meta['index'] = int(tb.meta['ocdb page'].split('/')[-1])

    return tb


def load_molecule_ocdb(molname, temperature=10, use_cached=True):
    """
    Load a molecule from the OCDB by performing a query.

    This is not the recommended method; ``download_all_ocdb()`` and ``read_ocdb_file()`` should be used instead.
    """

    if use_cached:
        cache_list = glob.glob(f'{optical_constants_cache_dir}/*.txt')
        if any([molname in x.lower() for x in cache_list]):
            for filename in cache_list:
                if molname in filename.lower():
                    return read_ocdb_file(filename)

    S = requests.Session()
    resp1 = S.get('https://ocdb.smce.nasa.gov/search/ice')
    resp = S.get('https://ocdb.smce.nasa.gov/ajax/datatable',
                 params={'start': 0, 'length': 220,
                         'search[value]': '',
                         'search[regex]': 'false',
                         'form_data[0][name]': 'formula_type',
                         'form_data[0][value]': 'ice',
                         'form_data[1][name]': 'temperature-min',
                         'form_data[1][value]': '',
                         'form_data[2][name]': 'temperature-max',
                         'form_data[2][value]': '',
                         'form_data[3][name]': 'wave-type-selector',
                         'form_data[3][value]': 'wavenumber',
                         'form_data[4][name]': 'wavenumber-min',
                         'form_data[4][value]': '',
                         'form_data[5][name]': 'wavenumber-max',
                         'form_data[5][value]': '',
                         'form_data[6][name]': 'wavelength-min',
                         'form_data[6][value]': '',
                         'form_data[7][name]': 'wavelength-max',
                         'form_data[7][value]': '',
                         "columns[0][data]": "formula_components",
                         "columns[0][name]": "",
                         "columns[0][searchable]": "true",
                         "columns[0][orderable]": "true",
                         "columns[0][search][value]": "",
                         "columns[0][search][regex]": "false",
                         "columns[1][data]": "formula_ratio",
                         "columns[1][name]": "",
                         "columns[1][searchable]": "true",
                         "columns[1][orderable]": "true",
                         "columns[1][search][value]": "",
                         "columns[1][search][regex]": "false",
                         "columns[2][data]": "dataset_temperature",
                         "columns[2][name]": "",
                         "columns[2][searchable]": "true",
                         "columns[2][orderable]": "true",
                         "columns[2][search][value]": "",
                         "columns[2][search][regex]": "false",
                         "columns[3][data]": "wavenumber_min",
                         "columns[3][name]": "",
                         "columns[3][searchable]": "true",
                         "columns[3][orderable]": "true",
                         "columns[3][search][value]": "",
                         "columns[3][search][regex]": "false",
                         "columns[4][data]": "wavenumber_max",
                         "columns[4][name]": "",
                         "columns[4][searchable]": "true",
                         "columns[4][orderable]": "true",
                         "columns[4][search][value]": "",
                         "columns[4][search][regex]": "false",
                         "columns[5][data]": "chart",
                         "columns[5][name]": "",
                         "columns[5][searchable]": "true",
                         "columns[5][orderable]": "true",
                         "columns[5][search][value]": "",
                         "columns[5][search][regex]": "false",
                         "columns[6][data]": "n",
                         "columns[6][name]": "",
                         "columns[6][searchable]": "true",
                         "columns[6][orderable]": "true",
                         "columns[6][search][value]": "",
                         "columns[6][search][regex]": "false",
                         "columns[7][data]": "k",
                         "columns[7][name]": "",
                         "columns[7][searchable]": "true",
                         "columns[7][orderable]": "true",
                         "columns[7][search][value]": "",
                         "columns[7][search][regex]": "false",
                         "columns[8][data]": "t",
                         "columns[8][name]": "",
                         "columns[8][searchable]": "true",
                         "columns[8][orderable]": "true",
                         "columns[8][search][value]": "",
                         "columns[8][search][regex]": "false",
                         "columns[9][data]": "r",
                         "columns[9][name]": "",
                         "columns[9][searchable]": "true",
                         "columns[9][orderable]": "true",
                         "columns[9][search][value]": "",
                         "columns[9][search][regex]": "false",
                         "columns[10][data]": "a",
                         "columns[10][name]": "",
                         "columns[10][searchable]": "true",
                         "columns[10][orderable]": "true",
                         "columns[10][search][value]": "",
                         "columns[10][search][regex]": "false",
                         "columns[11][data]": "o",
                         "columns[11][name]": "",
                         "columns[11][searchable]": "true",
                         "columns[11][orderable]": "true",
                         "columns[11][search][value]": "",
                         "columns[11][search][regex]": "false",
                         "columns[12][data]": "reference",
                         "columns[12][name]": "",
                         "columns[12][searchable]": "true",
                         "columns[12][orderable]": "true",
                         "columns[12][search][value]": "",
                         "columns[12][search][regex]": "false",
                         "order[0][column]": "0",
                         "order[0][dir]": "asc",
                         'draw': '1',
                         }
                 )
    metadata = resp.json()
    soups = [
        BeautifulSoup(
            x['formula_components'],
            features='html5lib') for x in metadata['data']]
    molecules = {soup.find('a').text.lower() + (f".{md['formula_ratio']}" if md['formula_ratio'] != "1" else ""):
                 soup.find('a').attrs['href'] for soup, md in zip(soups, metadata['data'])}
    molecules.update({soup.find('a').text.lower() +
                      (f".{md['formula_ratio']}" if md['formula_ratio'] != "1" else "") +
                      "." +
                      md['dataset_temperature'].lower(): soup.find('a').attrs['href'] for soup, md in zip(soups, metadata['data'])})
    molecules.update({key.replace(" ", ""): value for key,
                     value in molecules.items()})

    # Hudgins > Ehrenfreund; latter doesn't have k-values
    molecules['co.10 k'] = molecules['co.10k'] = molecules['co'] = 'https://ocdb.smce.nasa.gov/dataset/85'
    # non-Hudgins are overwriting Hudgins, but we want Hudgins
    molecules['co2.10 k'] = molecules['co2.10k'] = molecules['co2'] = 'https://ocdb.smce.nasa.gov/dataset/86'
    molecules['h2o.10 k'] = molecules['h2o.10k'] = molecules['h2o'] = 'https://ocdb.smce.nasa.gov/dataset/107'

    log.debug(
        f"molecule name = {molname.lower()}, ID={molecules[molname.lower()]}")

    dtabresp = S.get(f'{molecules[molname.lower()]}/download-data/all')
    for ii in range(5, 12):
        try:
            # new header data appear to be added from time to time
            tb = ascii.read(
                dtabresp.text.encode(
                    'ascii',
                    'ignore').decode(),
                format='tab',
                delimiter='\t',
                header_start=ii,
                data_start=ii + 1)
            break
        except astropy.io.ascii.core.InconsistentTableError:
            continue

    if 'Wavelength (m)' in tb.colnames:
        tb['Wavelength'] = tb['Wavelength (m)'] * u.um  # micron got truncated
    else:
        if 'Wavenumber (cm⁻¹)' in tb.colnames:
            tb['Wavelength'] = (
                tb['Wavenumber (cm⁻¹)'] *
                u.cm**-
                1).to(
                u.um,
                u.spectral())
            tb['Wavenumber (cm)'] = tb['Wavenumber (cm⁻¹)']
        elif 'Wavenumber (cm)' in tb.colnames:
            tb['Wavelength'] = (
                tb['Wavenumber (cm)'] *
                u.cm**-
                1).to(
                u.um,
                u.spectral())
            tb['Wavenumber (cm⁻¹)'] = tb['Wavenumber (cm)']
    tb.meta['density'] = 1 * u.g / u.cm**3
    # Hudgins 1993, page 719:
    # We haveassumedthatthedensitiesofalltheicesare1gcm-3 and that the ices are uniformly thick across the approximately 4 mm diameter focal point of the spectrometer's infrared beam on the sample.
    # "we estimate the uncertainty is no more than 30%"
    return tb


def cde_correct(freq, m):
    """
    cde_correct(freq, m)
    (copied from https://github.com/RiceMunk/omnifit/blob/master/omnifit/utils/utils.py#L181)

    Generate a CDE-corrected spectrum from a complex refractive index
    spectrum.

    Parameters
    ----------
    freq : `numpy.ndarray`
        The frequency data of the input spectrum, in reciprocal
        wavenumbers (cm^-1).
    m : `numpy.ndarray`
        The complex refractive index spectrum.

    Returns
    -------
    A list containing the following numpy arrays, in given order:
        * The spectrum of the absorption cross section of the simulated grain.
        * The spectrum of the absorption cross section of the simulated grain,
            normalized by the volume distribution of the grain. This parameter
            is the equivalent of optical depth in most cases.
        * The spectrum of the scattering cross section of the simulated grain,
            normalized by the volume distribution of the grain.
        * The spectrum of the total cross section of the simulated grain.
    """
    wl = 1.e4 / freq
    m2 = m**2.0
    im_part = ((m2 / (m2 - 1.0)) * np.log(m2)).imag
    cabs_vol = (4.0 * np.pi / wl) * im_part
    cabs = freq * (2.0 * m.imag / (m.imag - 1)) * np.log10(m.imag)
    cscat_vol = (freq**3.0 / (6.0 * np.pi)) * cabs
    ctot = cabs + cscat_vol
    return cabs, cabs_vol, cscat_vol, ctot


def absorbed_spectrum(
    ice_column,
    ice_model_table,
    spectrum=phx4000['fnu'],
    xarr=u.Quantity(phx4000['nu'], u.Hz).to(u.um, u.spectral()),
    molecular_weight=44 * u.Da,
    minimum_tau=0,
    return_tau=False
):
    """
    Calculate the absorbed spectrum for a given ice column.

    Parameters
    ----------
    ice_column : float
        Column density of the ice in molecules/cm^2
    ice_model_table : astropy.table.Table
        Table containing the ice model data
    spectrum : array-like
        Input spectrum to be absorbed
    xarr : array-like
        Wavelength array in microns
    molecular_weight : astropy.units.Quantity
        Molecular weight of the ice
    minimum_tau : float
        Minimum optical depth to consider
    return_tau : bool
        Whether to return the optical depth

    Returns
    -------
    array-like
        The absorbed spectrum
    """
    if not isscalar(ice_column):
        ice_column = ice_column.to(u.cm**-2)
    elif not hasattr(ice_column, 'unit'):
        ice_column = ice_column * u.cm**-2

    if not isscalar(molecular_weight):
        molecular_weight = molecular_weight.to(u.Da)
    elif not hasattr(molecular_weight, 'unit'):
        molecular_weight = molecular_weight * u.Da

    # Convert to mass column density
    mass_column = (ice_column * molecular_weight).to(u.g / u.cm**2)

    # Get the optical constants
    k = np.interp(xarr, ice_model_table['Wavelength'],
                  ice_model_table['k'], left=0, right=0)

    # Calculate optical depth
    tau = 4 * np.pi * k * mass_column / \
        (ice_model_table.meta['density'] * xarr)
    tau = np.maximum(tau, minimum_tau)

    # Calculate transmission
    transmission = np.exp(-tau)

    if return_tau:
        return spectrum * transmission, tau
    else:
        return spectrum * transmission


def isscalar(x):
    """Check if x is a scalar value."""
    return np.isscalar(x) or (hasattr(x, 'isscalar') and x.isscalar)


def absorbed_spectrum_Gaussians(
    ice_column,
    center,
    width,
    ice_bandstrength,
    spectrum=phx4000['fnu'],
    xarr=u.Quantity(
        phx4000['nu'],
        u.Hz).to(
            u.um,
        u.spectral())):
    """
    Calculate the absorbed spectrum using Gaussian absorption bands.

    Parameters
    ----------
    ice_column : float
        Column density of the ice in molecules/cm^2
    center : float
        Center wavelength of the absorption band in microns
    width : float
        Width of the absorption band in microns
    ice_bandstrength : float
        Band strength of the ice in cm/molecule
    spectrum : array-like
        Input spectrum to be absorbed
    xarr : array-like
        Wavelength array in microns

    Returns
    -------
    array-like
        The absorbed spectrum
    """
    if not isscalar(ice_column):
        ice_column = ice_column.to(u.cm**-2)
    else:
        ice_column = ice_column * u.cm**-2

    if not isscalar(ice_bandstrength):
        ice_bandstrength = ice_bandstrength.to(u.cm)
    else:
        ice_bandstrength = ice_bandstrength * u.cm

    # Calculate optical depth
    tau = ice_column * ice_bandstrength * \
        np.exp(-(xarr - center)**2 / (2 * width**2))

    # Calculate transmission
    transmission = np.exp(-tau)

    return spectrum * transmission


def convsum(xarr, model_data, filter_table, doplot=False):
    """
    Convolve model data with filter transmission curves.

    Parameters
    ----------
    xarr : array-like
        Wavelength array in microns
    model_data : array-like
        Model data to be convolved
    filter_table : astropy.table.Table
        Table containing filter transmission curves
    doplot : bool
        Whether to plot the results

    Returns
    -------
    array-like
        The convolved data
    """
    if doplot:
        pl.figure(figsize=(10, 6))
        pl.plot(xarr, model_data, 'k-', alpha=0.5, label='Model')
        pl.plot(
            xarr,
            filter_table['Transmission'],
            'r-',
            alpha=0.5,
            label='Filter')
        pl.xlabel('Wavelength (µm)')
        pl.ylabel('Transmission')
        pl.legend()
        pl.show()

    return np.sum(
        model_data * filter_table['Transmission']) / np.sum(filter_table['Transmission'])


def fluxes_in_filters(
        xarr,
        modeldata,
        doplot=False,
        filterids=None,
        transdata=None):
    """
    Calculate fluxes in various filters.

    Parameters
    ----------
    xarr : array-like
        Wavelength array in microns
    modeldata : array-like
        Model data to be convolved
    doplot : bool
        Whether to plot the results
    filterids : list
        List of filter IDs to use
    transdata : astropy.table.Table
        Table containing filter transmission curves

    Returns
    -------
    dict
        Dictionary of fluxes in each filter
    """
    if filterids is None:
        filterids = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W',
                     'F1800W', 'F2100W', 'F2550W']
    if transdata is None:
        transdata = SvoFps.get_transmission_data(filterids)

    fluxes = {}
    for filt in filterids:
        filter_table = transdata[filt]
        fluxes[filt] = convsum(xarr, modeldata, filter_table, doplot=doplot)

    return fluxes


def retrieve_gerakines_co(resolution='low'):
    import pandas as pd

    cache_file = f'{optical_constants_cache_dir}/CO_n_k_values_25_K_2023.xlsx'

    if os.path.exists(cache_file):
        dd = pd.read_excel(cache_file)
    else:
        dd = pd.read_excel(
            'https://science.gsfc.nasa.gov/691/cosmicice/constants/co/CO_n_k_values_25_K_2023.xlsx')
        dd.to_excel(cache_file)

    # lores
    if resolution == 'low':
        wavenumber = np.array(dd['Unnamed: 2'][1:], dtype='float')
        kk = np.array(dd['Unnamed: 4'][1:], dtype='float')
    # hires
    elif resolution == 'high':
        wavenumber = np.array(dd['Unnamed: 7'][1:], dtype='float')
        kk = np.array(dd['Unnamed: 9'][1:], 'float')
    else:
        raise ValueError("resolution must be 'low' or 'high'")
    keep = np.isfinite(wavenumber) & np.isfinite(kk)
    tbl = Table(
        {
            'Wavenumber': (
                wavenumber[keep]) *
            u.cm**-
            1,
            'Wavelength': (
                wavenumber[keep] *
                u.cm**-
                1).to(
                    u.um,
                    u.spectral()),
            'k': kk[keep]})
    tbl.meta['density'] = 1.029 * u.g / u.cm**3
    tbl.meta['temperature'] = 25
    tbl.meta['author'] = 'Gerakines2023'
    tbl.meta['composition'] = 'CO'
    tbl.meta['molecule'] = 'CO'
    tbl.meta['molwt'] = 28
    tbl.meta['index'] = 63 if resolution == 'low' else 64  # OCDB index

    return tbl


def download_all_lida(
        n_lida=178,
        redo=False,
        baseurl='https://icedb.strw.leidenuniv.nl'):
    S = requests.Session()

    if redo or not os.path.exists(
            f'{optical_constants_cache_dir}/lida_index.json'):
        index = {}
        for ii in range(1, 10):
            resp = S.get(f'{baseurl}/page/{ii}')
            soup = BeautifulSoup(resp.text, features='html5lib')
            mollinks = soup.findAll('a', class_='name')

            tbl = Table.read(
                f'{baseurl}/page/{ii}',
                format='html',
                header_start=0,
                data_start=1,
                htmldict=dict(
                    raw_html_cols=['Analogue']))
            for row, ml in zip(tbl, mollinks):

                mltext = row['Analogue']
                mltext = " ".join(
                    [f'{int(x[:-1]) / 100}' if x.endswith('%') else x for x in mltext.split()])
                moltext = LatexNodes2Text().latex_to_text(mltext)
                ind = int(ml.attrs['href'].split('/')[-1])
                if 'Pure' in moltext:
                    molname = moltext.replace('Pure ', '')
                    ratio = '1'
                elif 'over' in moltext or 'under' in moltext or 'Salt residue' in moltext:
                    molname = moltext
                    ratio = '1'
                elif ':' in moltext:
                    molname, ratio = [x for x in moltext.split() if ':' in x]
                else:
                    molname = moltext
                    ratio = '1'
                author = row['Author']
                index[ind] = {'full': moltext,
                              'name': molname,
                              'ratio': ratio,
                              'url': f'{baseurl}/data/{ind}',
                              'author': author,
                              'latex_molname': row['Analogue'],
                              # 'doi': doi
                              }
        with open(f'{optical_constants_cache_dir}/lida_index.json', 'w') as fh:
            json.dump(index, fh)
    else:
        with open(f'{optical_constants_cache_dir}/lida_index.json', 'r') as fh:
            index = json.load(fh)

    for ii in tqdm(index):
        url = index[ii]['url']
        molname = index[ii]['name']
        ratio = index[ii]['ratio']
        # author = index[ii]['author']
        # doi = index[ii]['doi']

        resp1 = S.get(url)

        soup = BeautifulSoup(resp1.text, features='html5lib')

        datafiles = soup.findAll('a', text='TXT')

        for df in datafiles:
            temperature = os.path.splitext(df.attrs["href"])[0].split(
                "/")[-1].split("_")[1].strip("K")
            index[ii]['temperature'] = float(temperature)
            index[ii]['index'] = int(
                df.attrs["href"].split("/")[-1].split("_")[0])
            outfn = f'{optical_constants_cache_dir}/{ii}_{molname}_{ratio}_{temperature}K.txt'
            if not os.path.exists(outfn) or redo:
                url = f'{baseurl}/{df.attrs["href"]}'
                resp = S.get(url)
                with open(outfn, 'w') as fh:
                    fh.write("# " + json.dumps(index[ii]) + "\n")

                    fh.write(resp.text)


def read_lida_file(filename):

    meta = {}
    with open(filename, 'r') as fh:
        meta = json.loads(fh.readline().lstrip('# '))
    tb = ascii.read(filename, data_start=1)
    tb.meta.update(meta)
    tb.rename_column('col1', 'Wavenumber')
    tb['Wavenumber'].unit = u.cm**-1
    tb.rename_column('col2', 'k')
    tb['Wavelength'] = (tb['Wavenumber'].quantity).to(u.um, u.spectral())

    tb.meta['density'] = 1 * u.g / u.cm**3
    if 'index' not in tb.meta:
        tb.meta['index'] = int(os.path.basename(filename).split('_')[0])
    tb.meta['molecule'] = os.path.basename(filename).split('_')[1]
    tb.meta['ratio'] = os.path.basename(filename).split('_')[2]
    tb.meta['author'] = tb.meta['author']
    tb.meta['composition'] = tb.meta['molecule'] + ' ' + tb.meta['ratio']
    tb.meta['temperature'] = float(
        filename.split("_")[-1].split(".")[0].strip("K"))
    tb.meta['database'] = 'lida'

    return tb


def composition_to_molweight(compstr):
    """
    Return the mean molecular weight assuming the ratio is a number ratio.
    (that's probably correct; it is most likely a number ratio)

    Composition strings look like

    'CO:CH3OH:CH3OCH3 (20:20:1)',
    'CO:CH3OH:CH3CHO (20:20:1)',
    'CO:CH3CHO (20:1)',
    'CO CH4 (20 1)',
    'CO:CO2 (100:16)',
    'CO:CH3OH:CH3CH2OH (20:20:1)',
    'CO:O2:CO2 (100:54:10)',
    'CO:CO2 (1:10)',
    'CO:O2 (100:50)',
    'CO N2 H2O (5 5 1)',
    'CO:HCOOH 1:1',
    'CO CH4 (20 1)',
    """

    if 'under' in compstr or 'over' in compstr or len(compstr.split(" ")) == 1:
        return Formula(compstr.split()[0]).mass * u.Da

    mols, comps = parse_molscomps(compstr)
    molvals = [Formula(mol).mass for mol in mols]

    if len(comps) == 0:
        raise ValueError(f"No comps found for compstr='{compstr}'")

    return sum([m * c for m, c in zip(molvals, comps)]) / sum(comps) * u.Da


def parse_molscomps(comp):
    """
    Given a composition string, return the list of molecules and their
    compositions as two lists
    """
    if len(comp.split(" ")) == 2:
        mols, comps = comp.split(" ")
        comps = list(map(float, re.split("[: ]", comps.strip("()"))))
        mols = re.split("[: ]", mols)
    elif len(comp.split(" (")) == 1:
        mols = [comp]
        comps = [1]
    else:
        mols, comps = comp.split(" (")
        comps = list(map(float, re.split("[: ]", comps.strip(")"))))
        mols = re.split("[: ]", mols)

    return mols, comps
