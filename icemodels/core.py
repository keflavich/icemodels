import os
import re
import glob
import shlex
import json
import numpy as np
from bs4 import BeautifulSoup
import mysg  # Tom Robitaille's YSO grid tool
from astropy.table import Table
from astropy import table
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
optical_constants_cache_dir = os.path.join(os.path.dirname(__file__), "data")

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

univap_molname_lookup = {
    'alpha-gycine': 'C2H5NO2',
    'alpha-glycine': 'C2H5NO2',
    'beta-glycine': 'C2H5NO2',
    'DL-proline': 'C5H9NO2',
    'DL-valine': 'C5H11NO2',
    'adenine': 'C5H5N5',
    'uracil': 'C4H4N2O2',
    'H2O (amorphous)': 'H2O',
    'H2O (crystalline)': 'H2O',
    'acetone': 'C3H6O',
    'acetonitrile': 'C2H3N',
    'acetonirtile': 'C2H3N',
    'cycle-hexane': 'C6H12',
    'Titanaerosol-N2CH4(19:1)': 'N2CH4',
    'Titan aerosol - N2CH4 (19:1)': 'N2CH4',
    'acetic acid': 'C2H4O2',
    'formic acid': 'CH2O2',
    'ethanol': 'C2H6O',
    'methanol': 'CH4O',
    'formamide': 'CH3NO',
    'pyrimidine': 'C4H4N2',
    'benzene': 'C6H6',
    'toluene': 'C7H8',
    'pyridine': 'C5H5N',
    'aniline': 'C6H7N',
    'phenol': 'C6H6O',
    'nitrobenzene': 'C6H5NO2',
    'benzonitrile': 'C7H5N',
    'anisole': 'C7H8O',
    'H2O:NH3:c-C6H6 (1:0.3:0.7)': 'H2O:NH3:C6H6 (1:0.3:0.7)',
    'H2O:NH3:c-C6H12 (1:0.3:0.7)': 'H2O:NH3:C6H12 (1:0.3:0.7)',
    'Enceladus - H2O:CO2:NH3:CH4 (9:1:1:1)': 'H2O:CO2:NH3:CH4 (9:1:1:1)',
    'Europa -H2O:CO2:NH3:SO2 (10:1:1:1)': 'H2O:CO2:NH3:SO2 (10:1:1:1)',
    'c-C6H6': 'C6H6',
    'c-C6H12': 'C6H12',
    'water': 'H2O',
    'ammonia': 'NH3',
    'carbon monoxide': 'CO',
    'carbon dioxide': 'CO2',
    'methane': 'CH4',
    'formaldehyde': 'CH2O',
    'hydrogen sulfide': 'H2S',
    'sulfur dioxide': 'SO2',
    'nitrogen': 'N2',
    'oxygen': 'O2',
    'hydrogen': 'H2',
    'glycine': 'C2H5NO2',
    'proline': 'C5H9NO2',
    'valine': 'C5H11NO2',
    'alanine': 'C3H7NO2',
    'serine': 'C3H7NO3',
    'threonine': 'C4H9NO3',
    'leucine': 'C6H13NO2',
    'isoleucine': 'C6H13NO2',
    'phenylalanine': 'C9H11NO2',
    'tyrosine': 'C9H11NO3',
    'tryptophan': 'C11H12N2O2',
    'histidine': 'C6H9N3O2',
    'lysine': 'C6H14N2O2',
    'arginine': 'C6H14N4O2',
    'aspartic acid': 'C4H7NO4',
    'glutamic acid': 'C5H9NO4',
    'asparagine': 'C4H8N2O3',
    'glutamine': 'C5H10N2O3',
    'cysteine': 'C3H7NOS',
    'methionine': 'C5H11NOS',
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


def get_univap_meta_table(univap_url='https://www1.univap.br/gaa/nkabs-database/data.htm', use_cached=True):
    """Get the metadata table from the Univap database."""
    if 'univap_meta_table' in cache:
        return cache['univap_meta_table']
    elif use_cached and os.path.exists(os.path.join(optical_constants_cache_dir, 'univap_meta_table.ecsv')):
        return Table.read(os.path.join(optical_constants_cache_dir, 'univap_meta_table.ecsv'))

    meta1 = Table.read(univap_url,
                       format='html', htmldict={'table_id': 1})
    meta2 = Table.read(univap_url,
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

    resp = requests.get(univap_url)
    soup = BeautifulSoup(resp.text, features='html5lib')

    for htmltable, meta in zip(soup.find_all('table'), [meta1, meta2]):
        urls = []
        for row in htmltable.find_all('tr'):
            datalabel = row.find('td').string
            if row.find('a'):
                url = row.find('a')['href']
                urls.append(url)
            else:
                urls.append('')
        meta['url'] = urls

    meta_table = table.vstack([meta1, meta2])

    meta_table = meta_table[~(meta_table['datalabel'] == 'Data Label')]

    cache['univap_meta_table'] = meta_table

    meta_table.write(os.path.join(optical_constants_cache_dir, 'univap_meta_table.ecsv'), overwrite=True)

    return meta_table


def load_molecule_univap(molname, meta_table=None, use_cached=True, overwrite=False):
    """
    Load a molecule based on its name from the dictionary of molecular data files above.
    """
    if meta_table is None:
        meta_table = get_univap_meta_table()
    meta_table.add_index('sample')

    row = meta_table.loc[molname]

    url = row['url']
    molid = row['datalabel']
    molname = row['sample']
    temperature = row['temperature']
    reference = row['reference'].replace("\n", " ")
    # Sanitize reference to be safe for file saving
    for ch in [' ', "'", '"', '\\', '/', ':', '*', '?', '<', '>', '|']:
        reference = reference.replace(ch, '_')

    filename = os.path.join(optical_constants_cache_dir, f'univap_{molid}_{molname}_{temperature}_{reference}.txt')

    return read_univap_file(filename, meta_row=row, use_cached=use_cached, overwrite=overwrite)


def read_univap_file(filename, meta_row=None, url=None, use_cached=True, overwrite=False):
    """
    If overwrite is set, the file will be rewritten with attached metadata.
    """
    if use_cached and os.path.exists(filename):
        consts = Table.read(filename, format='ascii', data_start=3)
        if 'source' in consts.meta:
            return consts
        elif meta_row is None:
            raise ValueError(f"File {filename} has no source metadata, so meta_row must be provided.")

    else:
        #url = univap_molecule_data[molname]['url']
        #molid = url.split('/')[-1].split('.')[0]

        consts = Table.read(url, format='ascii', data_start=3)

    if 'col1' in consts.colnames:
        consts['col1'].unit = u.cm**-1
        consts.rename_column('col1', 'WaveNum')
        consts.rename_column('col2', 'absorbance')
        consts.rename_column('col3', 'k')
        consts.rename_column('col4', 'n')
        consts['Wavelength'] = consts['WaveNum'].quantity.to(
            u.um, u.spectral())
    elif consts.colnames[0] == 'Description:':
        col1, col2, col3, col4 = consts.colnames
        consts[col1].unit = u.cm**-1
        consts.rename_column(col1, 'WaveNum')
        consts.rename_column(col2, 'absorbance')
        consts.rename_column(col3, 'k')
        consts.rename_column(col4, 'n')
        consts['Wavelength'] = consts['WaveNum'].quantity.to(
            u.um, u.spectral())


    consts['Wavelength'].unit = u.um
    consts['WaveNum'].unit = u.cm**-1

    url = meta_row['url']
    molid = meta_row['datalabel']
    molname = meta_row['sample']
    if molname in univap_molname_lookup:
        molname = univap_molname_lookup[molname]
    # Should not be needed
    composition = molname.replace('\n', ' ')
    molname = composition.split()[0]
    temperature = meta_row['temperature']
    reference = meta_row['reference']

    if molname in univap_molname_lookup:
        molname = univap_molname_lookup[molname]
    elif ':' in molname:
        spl = molname.split(':')
        spl = (univap_molname_lookup[x] if x in univap_molname_lookup else x for x in spl)
        molname = ':'.join(spl)

    consts.meta['density'] = 1 * u.g / u.cm**3
    consts.meta['author'] = reference
    consts.meta['source'] = url
    consts.meta['temperature'] = int(temperature)
    consts.meta['molecule'] = molname
    consts.meta['composition'] = composition
    try:
        consts.meta['molwt'] = composition_to_molweight(composition)
    except Exception as ex:
        print(f"Error calculating molwt for {molname}: {ex}")
        raise

    consts.meta['database'] = 'univap'
    consts.meta['filename'] = filename
    if not os.path.exists(filename) or overwrite:
        consts.write(filename, format='ascii', overwrite=overwrite)

    return consts


def download_all_univap(meta_table=None, redo=False, redo_meta=True):
    """
    Download all data files from the Univap database.
    """
    if meta_table is None:
        meta_table = get_univap_meta_table()

    for row in tqdm(meta_table):
        url = row['url']
        if url == '':
            continue
        if 'http' not in url:
            if url != '--':
                print(f"Skipping {url} because it is not a valid URL")
            continue

        molid = row['datalabel']
        molname = row['sample']
        temperature = row['temperature']
        reference = str(row['reference']).replace("\n", " ")
        # Sanitize reference to be safe for file saving
        for ch in [' ', "'", '"', '\\', '/', ':', '*', '?', '<', '>', '|']:
            reference = reference.replace(ch, '_')

        filename = os.path.join(optical_constants_cache_dir, f'univap_{molid}_{molname}_{temperature}_{reference}.txt')
        filename = filename.replace(" ", "_").replace("'", "").replace('\\', '')
        filename = filename.replace('"', '')
        if not redo and os.path.exists(filename):
            pass
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            resp = requests.get(url)
            if resp.status_code != 200:
                print(f"Error downloading {url}: {resp.status_code}")

            with open(filename, 'w') as fh:
                fh.write(resp.text)

        if redo_meta or redo:
            read_univap_file(filename, meta_row=row, use_cached=True, overwrite=True)

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
                os.path.join(optical_constants_cache_dir, f'ocdb_{ii}*'))) > 0:
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

        filename = os.path.join(optical_constants_cache_dir, f'ocdb_{ii}_{molname}_{temperature}_{reference}.txt')
        filename = filename.replace(" ", "_").replace("'", "").replace('\\', '')
        filename = filename.replace('"', '')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
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
        raise ValueError(f"No wavelength column found in {tb.colnames} for file {filename}")

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
        cache_list = glob.glob(os.path.join(optical_constants_cache_dir, '*.txt'))
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
    result_list : list
        A list containing the following numpy arrays, in given order:

        * The spectrum of the absorption cross section of the simulated grain.
        * The spectrum of the absorption cross section of the simulated grain, normalized by the volume distribution of the grain. This parameter is the equivalent of optical depth in most cases.
        * The spectrum of the scattering cross section of the simulated grain, normalized by the volume distribution of the grain.
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


phx4000 = atmo_model(4000)


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
    Use an opacity grid to obtain a model absorbed spectrum

    (see also https://github.com/RiceMunk/omnifit/blob/master/omnifit/utils/utils.py#L181)

    Parameters
    ----------
    ice_column : float
        Column density of the ice in molecules/cm^2
    ice_model_table : `astropy.table.Table`
        A table with Wavelength and 'k' constant columns and 'density' in the metadata
        (in units of g/cm^3)
    molecular_weight : `astropy.units.Quantity`
        The molecule mass (gram equivalent)
    minimum_tau : float
        The minimum tau to allow.  Default is 0.  This prevents negative optical
        depths, which create artificial emission.
    return_tau : bool
        If True, return the tau rather than the absorbed spectrum
    """
    xarr_icm = xarr.to(u.cm**-1, u.spectral())
    # not used dx_icm = np.abs(xarr_icm[1]-xarr_icm[0])
    inds = np.argsort(ice_model_table['Wavelength'].quantity)
    kay = np.interp(xarr.to(u.um),
                    ice_model_table['Wavelength'].quantity[inds],
                    ice_model_table['k'][inds],
                    left=0,
                    right=0,
                    )
    # Lambert absorption coefficient: k * 4pi/lambda
    # eqn 4 & 9 of Gerakines 2020
    # eqn 3 in Hudgins 1993
    alpha = kay * xarr_icm * 4 * np.pi
    # this next step just comes from me.
    # Eqn 8 in Hudgins is A = 1/N integral tau dnu
    # A is the integrated absorbance (units cm^-1)
    # N is the column density in cm^-2, so 1/N = cm^2
    # dnu is in cm^-1, and tau is unitless.
    # We can kinda rearrange to N * A / dnu = tau, but.
    # I'm basically assuming A(nu) = alpha / dnu, so tau = N * alpha
    # (I go through this math in DerivationNotes.ipynb)
    rho_n = (ice_model_table.meta['density'] / molecular_weight)
    tau = (alpha * ice_column / rho_n).decompose()
    if return_tau:
        return tau
    if minimum_tau is not None:
        tau[tau < minimum_tau] = minimum_tau

    absorbed_spectrum = ((np.exp(-tau)) * spectrum)
    return absorbed_spectrum


def tau_to_kay(tau, xarr, ice_column, rho_n):
    xarr_icm = xarr.to(u.cm**-1, u.spectral())

    alpha = tau / ice_column * rho_n
    kay = alpha / (xarr_icm * 4 * np.pi)

    return kay.decompose()


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
    spectrum : `numpy.ndarray`
        Input spectrum to be absorbed
    xarr : `numpy.ndarray`
        Wavelength array in microns

    Returns
    -------
    absorbed_spectrum : `numpy.ndarray`
        The absorbed spectrum
    """
    tau = np.zeros(xarr.size)

    cens, wids, strengths = center, width, ice_bandstrength

    if not isscalar(center):
        for center, width, ice_bandstrength in zip(cens, wids, strengths):
            wid_icm = (width / center) * center.to(u.cm**-1, u.spectral())
            line_profile = 1/((2*np.pi)**0.5 * wid_icm) * np.exp(-(xarr-center)**2/(2*width**2))

            assert line_profile.unit.is_equivalent(u.cm)
            assert ice_bandstrength.unit.is_equivalent(u.cm)
            assert ice_column.unit.is_equivalent(u.cm**-2)

            tau = tau + (ice_column * ice_bandstrength * line_profile).decompose()
            assert tau.unit is u.dimensionless_unscaled
    else:
        wid_icm = (width / center) * center.to(u.cm**-1, u.spectral())
        line_profile = 1/((2*np.pi)**0.5 * wid_icm) * np.exp(-(xarr-center)**2/(2*width**2))

        # normalize the line profile: tau is peak tau, not integral tau.
        # nope line_profile = line_profile / line_profile.max()
        tau = tau + (ice_column * ice_bandstrength * line_profile).decompose()
        assert tau.unit is u.dimensionless_unscaled

    absorbed_spectrum = ((np.exp(-tau)) * spectrum)
    return absorbed_spectrum


def convsum(xarr, model_data, filter_table, finite_only=True, doplot=False):
    """
    Convolve model data with filter transmission curves.

    Parameters
    ----------
    xarr : `numpy.ndarray`
        Wavelength array in microns
    model_data : `numpy.ndarray`
        Model data to be convolved
    filter_table : `astropy.table.Table`
        Table containing filter transmission curves
    doplot : bool
        Whether to plot the results
    finite_only : bool
        Whether to include only finite flux values in the filter function
        normalization.  If this is false, the whole bandwidth is included in the
        denominator, even if invalid pixels are excluded from the numerator.

    Returns
    -------
    convolved_data : `numpy.ndarray`
        The convolved data.  Will be nan if there are no valid pixels.
    """
    filtwav = u.Quantity(filter_table['Wavelength'], u.AA).to(u.um)

    inds = np.argsort(xarr.to(u.um))

    interpd = np.interp(filtwav,
                        xarr.to(u.um)[inds],
                        model_data[inds],
                        left=np.nan,
                        right=np.nan,
                        )

    valid = np.isfinite(interpd)

    if valid.sum() == 0:
        if hasattr(model_data, 'unit'):
            return model_data.unit * np.nan
        else:
            return np.nan

    # print(interpd, model_data, filter_table['Transmission'])
    # print(interpd.max(), model_data.max(), filter_table['Transmission'].max())
    result = (interpd * filter_table['Transmission'].value)[valid]
    if doplot:
        L, = pl.plot(filtwav, filter_table['Transmission'])
        pl.plot(filtwav, result, color=L.get_color())
        pl.plot(filtwav, interpd, color=L.get_color())

    # looking for average flux over the filter
    if finite_only:
        result = result.sum() / filter_table['Transmission'][valid].sum()
    else:
        result = result.sum() / filter_table['Transmission'].sum()

    assert np.isfinite(result)

    # dnu = np.abs(xarr[1].to(u.Hz, u.spectral()) - xarr[0].to(u.Hz, u.spectral()))
    if hasattr(model_data, 'unit'):
        return u.Quantity(result, model_data.unit)
    else:
        return result


def fluxes_in_filters(
        xarr,
        modeldata,
        doplot=False,
        filterids=None,
        transdata=None,
        telescope='JWST'):
    """
    Calculate fluxes in various filters.

    Parameters
    ----------
    xarr : `numpy.ndarray`
        Wavelength array in microns
    modeldata : `numpy.ndarray`
        Model data to be convolved
    doplot : bool
        Whether to plot the results
    filterids : list
        List of filter IDs to use
    transdata : `astropy.table.Table`
        Table containing filter transmission curves

    Returns
    -------
    fluxes : dict
        Dictionary of fluxes in each filter
    """

    if filterids is None:
        filterids = [x
                     for instrument in ('NIRCam', 'MIRI')
                     for x in SvoFps.get_filter_list(telescope, instrument=instrument)['filterID']]

    if transdata is None:
        transdata = SvoFps.get_transmission_data(filterids)

    fluxes = {fid: convsum(xarr, modeldata, transdata[fid], doplot=doplot)
              for fid in list(filterids)}

    return fluxes


def retrieve_gerakines_co(resolution='low'):
    import pandas as pd

    cache_file = os.path.join(optical_constants_cache_dir, 'CO_n_k_values_25_K_2023.xlsx')

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
        n_lida=179,
        redo=False,
        baseurl='https://icedb.strw.leidenuniv.nl',
        download_optcon=True,
        ):
    S = requests.Session()

    if redo or not os.path.exists(
            os.path.join(optical_constants_cache_dir, 'lida_index.json')):
        index = {}
        for ii in range(1, 10):
            resp = S.get(f'{baseurl}/page/{ii}')
            soup = BeautifulSoup(resp.text, features='html5lib')
            mollinks = soup.find_all('a', class_='name')

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
        os.makedirs(os.path.dirname(os.path.join(optical_constants_cache_dir, 'lida_index.json')), exist_ok=True)
        with open(os.path.join(optical_constants_cache_dir, 'lida_index.json'), 'w') as fh:
            json.dump(index, fh)
    else:
        with open(os.path.join(optical_constants_cache_dir, 'lida_index.json'), 'r') as fh:
            index = json.load(fh)

    # a monolayer as defined by van Broekhuizen 2006 pg 725 table 1
    # use just 1/cm^-2, not mol/cm^2
    monolayer = 1e15 / u.cm**2

    for ii in tqdm(index):
        url = index[ii]['url']
        molname = index[ii]['name']
        ratio = index[ii]['ratio']
        # author = index[ii]['author']
        # doi = index[ii]['doi']

        resp1 = S.get(url)
        resp1.raise_for_status()

        soup = BeautifulSoup(resp1.text, features='html5lib')

        # units are usually ML
        # ML = monolayer?
        # 1 L =10^15 mol/cm^2 = 1 monolayer, from van Broekhuizen 2006 pg 725 table 1
        ice_thickness = soup.find('strong', string='Ice thickness: ').next_sibling.text.strip()
        #if ice_thickness.endswith('ML'):
        #    ice_thickness = float(ice_thickness.replace('ML', '')) * monolayer

        ice_column_density = soup.find('strong', string='Ice column density: ').next_sibling.text.strip()
        # unit is cm^-2, but -2 is in <sup>
        if ice_column_density.endswith('cm'):
            ice_column_density = ice_column_density.replace('cm', 'cm-2')

        datafiles = soup.find_all('a', string='TXT')

        for df in datafiles:
            temperature = os.path.splitext(df.attrs["href"])[0].split(
                "/")[-1].split("_")[1].strip("K")
            index[ii]['temperature'] = float(temperature)
            index[ii]['ice_thickness'] = ice_thickness
            index[ii]['ice_column_density'] = ice_column_density
            index[ii]['index'] = int(
                df.attrs["href"].split("/")[-1].split("_")[0])
            outfn = os.path.join(optical_constants_cache_dir, f'lida_{ii}_{molname}_{ratio}_{temperature}K.txt')
            os.makedirs(os.path.dirname(outfn), exist_ok=True)
            if not os.path.exists(outfn) or redo:
                url = f'{baseurl}/{df.attrs["href"]}'
                resp = S.get(url)
                with open(outfn, 'w') as fh:
                    fh.write("# " + json.dumps(index[ii]) + "\n")

                    fh.write(resp.text)

    if download_optcon:
        download_lida_optcon(redo=redo, baseurl=baseurl)


def read_lida_file(filename, ice_thickness=None):
    """ Read a LIDA file with absorbance and do the appropriate conversion from absorbance to k """

    if 'lida_optcon' in filename:
        return read_lida_optcon_file(filename)

    meta = {}
    with open(filename, 'r') as fh:
        meta = json.loads(fh.readline().lstrip('# '))
    tb = ascii.read(filename, data_start=1)
    tb.meta.update(meta)
    tb.rename_column('col1', 'Wavenumber')
    tb['Wavenumber'].unit = u.cm**-1
    tb['Wavelength'] = (tb['Wavenumber'].quantity).to(u.um, u.spectral())

    # column 2 is absorbance, not k
    tb.rename_column('col2', 'absorbance')

    if os.path.basename(filename).split('_')[0] == 'lida':
        pp = 1
    else:
        pp = 0

    tb.meta['density'] = 1 * u.g / u.cm**3
    if 'index' not in tb.meta:
        tb.meta['index'] = int(os.path.basename(filename).split('_')[0+pp])
    tb.meta['molecule'] = os.path.basename(filename).split('_')[1+pp]
    tb.meta['ratio'] = os.path.basename(filename).split('_')[2+pp]
    tb.meta['author'] = tb.meta['author']
    tb.meta['composition'] = tb.meta['molecule'] + ' ' + tb.meta['ratio']
    tb.meta['temperature'] = float(
        filename.split("_")[-1].split(".")[0].strip("K"))
    tb.meta['database'] = 'lida'

    # k = absorbance * wavelength / 4 pi d
    # d is the ice thickness, but it's in units of cm, not area, so we have to convert using density
    # this holds IF absorbance is defined as ln(I_0/I), not N^-1 ln(I_0/I), the latter from Hudgins 1993
    # Rocha & Pilling 2014 define absorbance this way; I would call this value optical depth (tau)
    # Rocha+ 2022 LIDA paper defines the terms more clearly
    density = (tb.meta['density'])
    molwt = composition_to_molweight(tb.meta['composition'])
    monolayer = 1e15 / u.cm**2
    if ice_thickness is None:
        if 'None' in tb.meta['ice_thickness']:
            raise ValueError(f"Ice thickness is None for {filename}")
        ice_thickness = float(tb.meta['ice_thickness'].replace('ML', '')) * monolayer
    else:
        ice_thickness = ice_thickness * monolayer
    ice_depth = (ice_thickness / density * molwt).to(u.um)
    tb.meta['ice_layer_depth'] = ice_depth.to(u.um)
    # wrong kay = (tb['absorbance'] * tb['Wavelength'].quantity / (4 * np.pi * ice_depth)).decompose()
    # Equation 10 in Rocha+ 2022 LIDA paper, ignoring second term
    # they use 2.3 exactly, ln(10) = 2.3026
    kay = (tb['absorbance'] * 2.3 / (4 * np.pi * ice_depth * tb['Wavenumber'].quantity)).decompose()
    assert kay.unit.is_equivalent(u.dimensionless_unscaled)


    # use https://icedb.strw.leidenuniv.nl/Kramers_Kronig to derive k
    # inspired by, but not using at all, https://github.com/leiden-laboratory-for-astrophysics/refractive-index
    # This all turns out to be wrong by ~20 orders of magnitude
    #alpha = 1/thickness * (2.3 * tb['absorbance'] + 2 * np.log(1/10**tb['absorbance']))
    #imag = alpha / (12.5 * tb['Wavenumber'].quantity.to(u.cm**-1).value)
    #tb['k'] = imag
    #alpha = 1/ice_thickness.to(u.cm**-2).value * (2.3 * tb['absorbance'] + 2 * np.log(1/10**tb['absorbance']))
    #kay = imag = alpha / (12.5 * tb['Wavenumber'].quantity.to(u.cm**-1).value)

    tb.add_column(kay, name='k', )
    tb.meta['k_comment'] = 'The complex refractive index is estimated from the provided ice depth data using k = A * ln(10) / (4 pi wavenumber d), where A is absorbance, lambda is wavelength, and d is the ice depth.  We assume the ice has a density of 1 g/cm^3 and a molar mass of the composition.'

    return tb


def parse_isotope(mol, excess_mass=0):
    if '^' in mol:
        isoreg = re.compile(r'\^([0-9]+)([A-Z][a-z]?)')
        match = isoreg.search(mol)
        mass = int(match.groups()[0])
        atom_name = match.groups()[1]
        nominal_mass = Formula(atom_name).nominal_mass
        excess_mass = mass - nominal_mass
        nonisotope_mol = isoreg.sub(atom_name, mol)
        if '^' in nonisotope_mol:
            return parse_isotope(nonisotope_mol, excess_mass)
        return nonisotope_mol, excess_mass
    else:
        return mol, 0


def get_molmass(mol):
    mol, excess_mass = parse_isotope(mol)
    return Formula(mol).nominal_mass + excess_mass


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
        return get_molmass(compstr.split()[0]) * u.Da

    mols, comps = parse_molscomps(compstr)

    molvals = [get_molmass(mol) for mol in mols]

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


def molscomps(comp):
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


def download_lida_optcon(
        redo=False,
        baseurl='https://icedb.strw.leidenuniv.nl'):
    """
    Download optical constants data from the Leiden ice database.

    This function downloads n and k constants directly from the optical constants
    section of the database, which provides data for 6 compounds (IDs 2-6).
    Unlike download_all_lida, this doesn't need to compute ice thickness or column
    density since the files directly report n and k constants.

    Parameters
    ----------
    redo : bool
        If True, redownload files even if they already exist
    baseurl : str
        Base URL for the Leiden ice database
    """
    S = requests.Session()

    # The optical constants page lists 6 compounds with IDs 2-6
    # Based on the search results, these are:
    # 2: H2O, 3: CO2, 4: CO, 5: CO:CO2 (100:70), 6: H2O:CO2 (100:14)
    # optcon_ids = [2, 3, 4, 5, 6]

    if redo or not os.path.exists(
            os.path.join(optical_constants_cache_dir, 'lida_optcon_index.json')):
        index = {}

        # Get the main optical constants page to extract metadata
        optcon_resp = S.get(f'{baseurl}/refrac_index')
        if optcon_resp.status_code != 200:
            print(f"Error accessing optical constants page: {optcon_resp.status_code}")
            return

        soup = BeautifulSoup(optcon_resp.text, features='html5lib')

        # Parse the table to get compound information
        table_rows = soup.find_all('tr')[1:]  # Skip header row

        for row in table_rows:
            cells = row.find_all('td')
            if len(cells) >= 3:
                # Extract compound name from first cell
                compound_cell = cells[0]
                compound_link = compound_cell.find('a')
                if compound_link:
                    compound_url = compound_link.get('href')
                    compound_id = int(compound_url.split('/')[-1])
                    compound_name = compound_cell.get_text(strip=True)

                    # Clean up compound name (remove LaTeX formatting)
                    compound_name = LatexNodes2Text().latex_to_text(compound_name)

                    # Extract author from third cell
                    author_cell = cells[2]
                    author = author_cell.get_text(strip=True)

                    index[compound_id] = {
                        'name': compound_name,
                        'author': author,
                        'url': f'{baseurl}/data_opt_const/{compound_id}',
                        'compound_id': compound_id
                    }

        os.makedirs(os.path.dirname(os.path.join(optical_constants_cache_dir, 'lida_optcon_index.json')), exist_ok=True)
        with open(os.path.join(optical_constants_cache_dir, 'lida_optcon_index.json'), 'w') as fh:
            json.dump(index, fh)
    else:
        with open(os.path.join(optical_constants_cache_dir, 'lida_optcon_index.json'), 'r') as fh:
            index = json.load(fh)

    # Download data for each compound
    for compound_id in tqdm(index):
        compound_data = index[compound_id]
        compound_name = compound_data['name']
        author = compound_data['author']

        # Get the compound's data page to find available spectra
        compound_resp = S.get(compound_data['url'])
        compound_resp.raise_for_status()

        compound_soup = BeautifulSoup(compound_resp.text, features='html5lib')

        # Look for spectrum download links
        # These should be in the format: spectrum_nval/download/{id}_{temperature}.txt
        spectrum_links = compound_soup.find_all('a', href=lambda x: x and 'spectrum_nval/download' in x)

        for link in spectrum_links:
            href = link.get('href')
            if href:
                # Extract temperature and spectrum type from the URL
                filename = href.split('/')[-1]
                parts = filename.replace('.txt', '').split('_')
                if len(parts) >= 2:
                    spectrum_id = parts[0]
                    temperature = parts[1]

                    # Determine if this is n or k from the link context
                    # The parent element or nearby text should indicate n or k
                    parent_text = link.parent.get_text() if link.parent else ""
                    if 'Warm-up N' in parent_text:
                        spectrum_type = 'n'
                    elif 'Warm-up K' in parent_text:
                        spectrum_type = 'k'
                    else:
                        raise ValueError(f"Unknown spectrum type for {compound_id} {compound_name} {temperature} {parent_text}")

                    outfn = os.path.join(optical_constants_cache_dir,
                                       f'lida_optcon_{compound_id}_{compound_name}_{temperature}_{spectrum_type}.txt')
                    outfn = outfn.replace(' ', '_').replace(':', '_')

                    os.makedirs(os.path.dirname(outfn), exist_ok=True)

                    if not os.path.exists(outfn) or redo:
                        spectrum_url = f'{baseurl}/{href}'
                        spectrum_resp = S.get(spectrum_url)
                        spectrum_resp.raise_for_status()
                        # Add metadata to the file
                        metadata = {
                            'compound_id': compound_id,
                            'compound_name': compound_name,
                            'author': author,
                            'temperature': temperature,
                            'spectrum_type': spectrum_type,
                            'spectrum_id': spectrum_id,
                            'url': spectrum_url,
                            'database': 'lida_optcon',
                            'index': compound_id,
                        }

                        with open(outfn, 'w') as fh:
                            fh.write("# " + json.dumps(metadata) + "\n")
                            fh.write(spectrum_resp.text)


def read_lida_optcon_file(filename):
    """
    Read a LIDA optical constants file.

    These files contain direct n and k values, unlike the regular LIDA files
    which contain absorbance that needs to be converted to k.

    Parameters
    ----------
    filename : str
        Path to the LIDA optical constants file

    Returns
    -------
    astropy.table.Table
        Table with wavelength and optical constants data
    """
    meta = {}
    with open(filename, 'r') as fh:
        first_line = fh.readline()
        if first_line.startswith('# '):
            meta = json.loads(first_line.lstrip('# '))

    # Read the data starting from line 1 (after metadata)
    tb = ascii.read(filename, data_start=1)
    tb.meta.update(meta)

    # Standard column naming for optical constants files
    if len(tb.colnames) >= 2:
        tb.rename_column('col1', 'Wavenumber')
        tb['Wavenumber'].unit = u.cm**-1
        tb['Wavelength'] = (tb['Wavenumber'].quantity).to(u.um, u.spectral())

        # The second column contains the optical constant (n or k)
        spectrum_type = tb.meta.get('spectrum_type', 'unknown')
        if spectrum_type == 'n':
            tb.rename_column('col2', 'n')
        elif spectrum_type == 'k':
            tb.rename_column('col2', 'k')
        else:
            tb.rename_column('col2', 'optical_constant')

    # Set standard metadata
    tb.meta['density'] = 1 * u.g / u.cm**3  # Default density
    tb.meta['database'] = 'lida_optcon'

    # Set composition based on compound name
    compound_name = tb.meta.get('compound_name', '')
    if 'H2O' in compound_name and 'CO2' in compound_name:
        tb.meta['composition'] = 'H2O:CO2'
        tb.meta['molecule'] = 'H2O:CO2'
    elif 'CO' in compound_name and 'CO2' in compound_name:
        tb.meta['composition'] = 'CO:CO2'
        tb.meta['molecule'] = 'CO:CO2'
    elif 'H2O' in compound_name:
        tb.meta['composition'] = 'H2O'
        tb.meta['molecule'] = 'H2O'
    elif 'CO2' in compound_name:
        tb.meta['composition'] = 'CO2'
        tb.meta['molecule'] = 'CO2'
    elif 'CO' in compound_name:
        tb.meta['composition'] = 'CO'
        tb.meta['molecule'] = 'CO'
    else:
        tb.meta['composition'] = compound_name
        tb.meta['molecule'] = compound_name

    return tb
