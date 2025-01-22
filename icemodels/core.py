import os
import glob
import shlex
import numpy as np
from bs4 import BeautifulSoup
import mysg # Tom Robitaille's YSO grid tool
from astropy.table import Table
from astropy.io import ascii
from astropy import units as u
from astropy import log
import pylab as pl
import requests
from astroquery.svo_fps import SvoFps
import astropy
import astropy.io.ascii.core
from tqdm.auto import tqdm

optical_constants_cache_dir = os.path.dirname(__file__) + "/data"

# "https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/main/nk/co2-a-Gerakines2020.txt",
molecule_data = {'ch3oh':
                 {'url': 'https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/main/nk/ch3oh-a-Gerakines2020.txt',
                  'molwt': (12+4+16)*u.Da, },
                 'co2':
                 {'url': 'https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/main/nk/co2-a-Gerakines2020.txt',
                  'molwt': (12+2*16)*u.Da, },
                 'ch4':
                 {'url': 'https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/main/nk/ch4-a-Gerakines2020.txt',
                  'molwt': (12+4)*u.Da, },
                 'co':
                 {'url': 'https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/main/nk/co-a-Palumbo2006.txt',
                  'molwt': (12+16)*u.Da, },
                 'h2o':
                 {'url': 'https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/main/nk/h2o-a-Hudgins1993.txt',
                  'molwt': (16+2)*u.Da, },
                 'h2o_b':
                 {'url': 'https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/main/nk/h2o_Rocha.txt',
                  'molwt': (16+2)*u.Da, },
                 'nh3':
                 {'url': 'https://raw.githubusercontent.com/willastro/ifw_miri_gto_pstars/main/nk/nh3_Roser_2021.txt',
                  'molwt': (14+3)*u.Da,
                  'density': 0.8*u.g/u.cm**3,  # Satorre+2013 via Roser+2021
                 },
                }

astrochem_molecule_data = {
                 'co': {'url': 'https://ocdb.smce.nasa.gov/dataset/89/'},
                 'co_old':
                 {'url': 'http://www.astrochem.org/data/CO/CO',
                  'molwt': (12+16)*u.Da, },
                 'co_hudgins':
                 {'url': 'http://www.astrochem.org/data/CO/CO.Hudgins',
                  'molwt': (12+16)*u.Da, },
}

univap_molecule_data = {
    # https://www1.univap.br/gaa/nkabs-database/data.htm
    'co': {'url': "https://www.dropbox.com/s/dgufhmfwleak4ce/G1.txt?dl=1", #http://www1.univap.br/gaa/nkabs-database/G1.txt",
           'molwt': (12+16)*u.Da, },
    'co2': {'url': 'https://www.dropbox.com/s/hoo9s01knc7p3su/G2.txt?dl=1', #'https://www1.univap.br/gaa/nkabs-database/G2.txt',
            'molwt': (16*2+12)*u.Da, },
    'h2o_amorphous': {'url': 'https://www.dropbox.com/s/dcpqq20766fdf2i/L1.txt?dl=1', # http://www1.univap.br/gaa/nkabs-database/L1.txt',
                      'molwt': (16+2)*u.Da, },
    'h2o_crystal': {'url': 'http://www1.univap.br/gaa/nkabs-database/L2.txt',
                    'molwt': (16+2)*u.Da},
    'h2o:co.1:0.4': {'url': "http://www1.univap.br/gaa/nkabs-database/D8a.txt",
                     'molwt': (12+16)*u.Da, },
}

def atmo_model(temperature, xarr=np.linspace(1, 28, 15000)*u.um):
    """
    use https://github.com/astrofrog/mysg to load Kurucz & Phoenix models and interpolate them
    to a specified temperature

    then, interpolate those onto a finely-sampled(ish) wavelength grid that covers the JWST filters

    (the default spectral grid has essentially no sampling from 10-25 microns)
    """
    mod = Table(mysg.atmosphere.interp_atmos(temperature))
    mod['nu'].unit = u.Hz
    mod['fnu'].unit = u.erg/u.s/u.cm**2/u.Hz
    inds = np.argsort(mod['nu'])
    xarrhz = xarr.to(u.Hz, u.spectral())
    mod = Table({'fnu': np.interp(xarrhz, mod['nu'].quantity[inds], mod['fnu'].quantity[inds]),
                 'nu': xarrhz},
                meta={'temperature': temperature})

    return mod

phx4000 = atmo_model(4000)


cache = {}
def load_molecule(molname):
    """
    Load a molecule based on its name from the dictionary of molecular data files above
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
        consts['Wavelength'] = consts['WaveNum'].quantity.to(u.um, u.spectral())
    if 'density' in molecule_data[molname]:
        consts.meta['density'] = molecule_data[molname]['density']
    else:
        lines = requests.get(url).text.split('\n')
        for line in lines:
            if not line.startswith("#"):
                break
        density = float(line.split()[1])*u.g/u.cm**3
        consts.meta['density'] = density
    cache[molname] = consts
    return consts


def load_molecule_univap(molname):
    """
    Load a molecule based on its name from the dictionary of molecular data files above
    """
    url = univap_molecule_data[molname]['url']
    consts = Table.read(url, format='ascii', data_start=3)
    if 'col1' in consts.colnames:
        consts['col1'].unit = u.cm**-1
        consts.rename_column('col1', 'WaveNum')
        consts.rename_column('col2', 'absorbance')
        consts.rename_column('col3', 'k')
        consts.rename_column('col4', 'n')
        consts['Wavelength'] = consts['WaveNum'].quantity.to(u.um, u.spectral())
    consts.meta['density'] = 1*u.g/u.cm**3
    return consts


def load_molecule_icedb():
    response = requests.get('https://icedb.strw.leidenuniv.nl/spectrum/download/754/754_15.0K.txt', verify=False)
    icedb_co = ascii.read(response.text)
    pl.plot(icedb_co['col1'], icedb_co['col2'])


def download_all_ocdb(n_ocdb=298):
    S = requests.Session()
    resp1 = S.get('https://ocdb.smce.nasa.gov/search/ice')

    for ii in tqdm(range(1, n_ocdb+1)):
        resp = S.get(f'https://ocdb.smce.nasa.gov/dataset/{ii}/download-data/all')
        for row in resp.text.split("\n"):
            if row.startswith('Composition:'):
                molname = shlex.split(row)[1]
            if row.startswith('Temperature:'):
                temperature = shlex.split(row)[1]
            if row.startswith('Reference:'):
                reference = shlex.split(row)[1].split()[0]
        filename = f'{optical_constants_cache_dir}/{ii}_{molname}_{temperature}_{reference}.txt'
        filename = filename.replace(" ", "_").replace("'", "").replace('\\','').replace('"','')
        with open(filename, 'w') as fh:
            fh.write(resp.text)


def read_ocdb_file(filename):
    for ii in range(5, 15):
        try:
            # new header data appear to be added from time to time
            tb = ascii.read(filename,
                            format='tab', delimiter='\t', header_start=ii, data_start=ii+1)
            break
        except astropy.io.ascii.core.InconsistentTableError:
            if ii == 14:
                raise ValueError("File appears to be invalid")
            continue

    if 'Wavelength (m)' in tb.colnames:
        tb['Wavelength'] = tb['Wavelength (m)'] * u.um # micron got truncated
        tb['Wavenumber (cm)'] = tb['Wavenumber'] = (tb['Wavelength'].to(u.cm**-1, u.spectral()))
    elif 'Wavelength (µm)' in tb.colnames:
        tb['Wavelength'] = tb['Wavelength (µm)'] * u.um
        tb['Wavenumber (cm)'] = tb['Wavenumber'] = (tb['Wavelength'].to(u.cm**-1, u.spectral()))
    elif 'Wavenumber (cm⁻¹)' in tb.colnames:
        tb['Wavelength'] = (tb['Wavenumber (cm⁻¹)'] * u.cm**-1).to(u.um, u.spectral())
    elif 'Wavenumber (cm)' in tb.colnames:
        tb['Wavelength'] = (tb['Wavenumber (cm)'] * u.cm**-1).to(u.um, u.spectral())
    else:
        raise ValueError(f"No wavelength column found in {tb.colnames}")

    if 'k₁' in tb.colnames:
        tb['k'] = tb['k₁']

    tb.meta['density'] = 1*u.g/u.cm**3

    with open(filename, 'r') as fh:
        keys = ['Reference:', 'DOI:', 'Composition:', 'Temperature:', 'OCdb page:']
        rows = fh.readlines()
        for row in rows[:20]:
            for key in keys:
                if row.startswith(key):
                    kk = key.lower().strip(":")
                    tb.meta[kk] = (" ".join(row.split(":")[1:])).strip().strip('"')

    if 'reference' in tb.meta:
        tb.meta['author'] = tb.meta['reference'].split()[0]
    if 'composition' in tb.meta:
        tb.meta['molecule'] = tb.meta['composition'].split()[0]

    return tb


def load_molecule_ocdb(molname, temperature=10, use_cached=True):

    if use_cached:
        cache_list = glob.glob(f'{optical_constants_cache_dir}/*.txt')
        if any([molname in x.lower() for x in cache_list]):
            for filename in cache_list:
                if molname in filename.lower():
                    return read_ocdb_file(filename)

    S = requests.Session()
    resp1 = S.get('https://ocdb.smce.nasa.gov/search/ice')
    resp = S.get('https://ocdb.smce.nasa.gov/ajax/datatable',
                params={'start':0, 'length': 220,
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
    soups = [BeautifulSoup(x['formula_components'], features='html5lib') for x in metadata['data']]
    molecules = {soup.find('a').text.lower() + (f".{md['formula_ratio']}" if md['formula_ratio'] != "1" else ""):
        soup.find('a').attrs['href'] for soup, md in zip(soups, metadata['data'])}
    molecules.update({soup.find('a').text.lower() + (f".{md['formula_ratio']}" if md['formula_ratio'] != "1" else "") + "." + md['dataset_temperature'].lower():
        soup.find('a').attrs['href'] for soup, md in zip(soups, metadata['data'])})
    molecules.update({key.replace(" ",""): value for key, value in molecules.items()})

    # Hudgins > Ehrenfreund; latter doesn't have k-values
    molecules['co.10 k'] = molecules['co.10k'] = molecules['co'] = 'https://ocdb.smce.nasa.gov/dataset/85'
    # non-Hudgins are overwriting Hudgins, but we want Hudgins
    molecules['co2.10 k'] = molecules['co2.10k'] = molecules['co2'] = 'https://ocdb.smce.nasa.gov/dataset/86'
    molecules['h2o.10 k'] = molecules['h2o.10k'] = molecules['h2o'] = 'https://ocdb.smce.nasa.gov/dataset/107'

    log.debug(f"molecule name = {molname.lower()}, ID={molecules[molname.lower()]}")

    dtabresp = S.get(f'{molecules[molname.lower()]}/download-data/all')
    for ii in range(5, 12):
        try:
            # new header data appear to be added from time to time
            tb = ascii.read(dtabresp.text.encode('ascii', 'ignore').decode(),
                            format='tab', delimiter='\t', header_start=ii, data_start=ii+1)
            break
        except astropy.io.ascii.core.InconsistentTableError:
            continue

    if 'Wavelength (m)' in tb.colnames:
        tb['Wavelength'] = tb['Wavelength (m)'] * u.um # micron got truncated
    else:
        if 'Wavenumber (cm⁻¹)' in tb.colnames:
            tb['Wavelength'] = (tb['Wavenumber (cm⁻¹)'] * u.cm**-1).to(u.um, u.spectral())
            tb['Wavenumber (cm)'] = tb['Wavenumber (cm⁻¹)']
        elif 'Wavenumber (cm)' in tb.colnames:
            tb['Wavelength'] = (tb['Wavenumber (cm)'] * u.cm**-1).to(u.um, u.spectral())
            tb['Wavenumber (cm⁻¹)'] = tb['Wavenumber (cm)']
    tb.meta['density'] = 1*u.g/u.cm**3
    # Hudgins 1993, page 719:
    # We haveassumedthatthedensitiesofalltheicesare1gcm-3 and that the ices are uniformly thick across the approximately 4 mm diameter focal point of the spectrometer’s infrared beam on the sample.
    # "we estimate the uncertainty is no more than 30%"
    return tb


def absorbed_spectrum(ice_column,
                      ice_model_table,
                      spectrum=phx4000['fnu'],
                      xarr=u.Quantity(phx4000['nu'], u.Hz).to(u.um, u.spectral()),
                      molecular_weight=44*u.Da,
                      return_tau=False):
    """
    Use an opacity grid to obtain a model absorbed spectrum

    Parameters
    ----------
    ice_column : column density, cm^-2
    ice_model_table : table
        A table with Wavelength and 'k' constant columns and 'density' in the metadata
        (in units of g/cm^3)
    molecular_weight : u.g equivalent
        The molecule mass
    """
    xarr_icm = xarr.to(u.cm**-1, u.spectral())
    # not used dx_icm = np.abs(xarr_icm[1]-xarr_icm[0])
    inds = np.argsort(ice_model_table['Wavelength'].quantity)
    kay = np.interp(xarr.to(u.um),
                    ice_model_table['Wavelength'].quantity[inds],
                    ice_model_table['k'][inds])
    # Lambert absorption coefficient: k * 4pi/lambda
    alpha = kay * xarr_icm * 4 * np.pi
    tau = (alpha * ice_column / (ice_model_table.meta['density'] / molecular_weight)).decompose()
    if return_tau:
        return tau

    absorbed_spectrum = ((np.exp(-tau)) * spectrum)
    return absorbed_spectrum


def isscalar(x):
    try:
        len(x)
        return False
    except Exception:
        return True



def absorbed_spectrum_Gaussians(ice_column, center, width, ice_bandstrength,
                                spectrum=phx4000['fnu'],
                                xarr=u.Quantity(phx4000['nu'], u.Hz).to(u.um, u.spectral())):
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

def convsum(xarr, model_data, filter_table, doplot=False):
    filtwav = u.Quantity(filter_table['Wavelength'], u.AA).to(u.um)

    inds = np.argsort(xarr.to(u.um))

    interpd = np.interp(filtwav,
                        xarr.to(u.um)[inds],
                        model_data[inds])
    # print(interpd, model_data, filter_table['Transmission'])
    # print(interpd.max(), model_data.max(), filter_table['Transmission'].max())
    result = (interpd * filter_table['Transmission'].value)
    if doplot:
        L, = pl.plot(filtwav, filter_table['Transmission'])
        pl.plot(filtwav, result, color=L.get_color())
        pl.plot(filtwav, interpd, color=L.get_color())
    # looking for average flux over the filter
    result = result.sum() / filter_table['Transmission'].sum()
    # dnu = np.abs(xarr[1].to(u.Hz, u.spectral()) - xarr[0].to(u.Hz, u.spectral()))
    return result# * dnu

def fluxes_in_filters(xarr, modeldata, doplot=False):
    telescope = 'JWST'

    if doplot:
        pl.loglog(xarr.to(u.um), modeldata)
        pl.xlabel("Wavelengh [$\\mu$m]")
        pl.ylabel("Flux [Jy]")

    fluxes = {}
    for instrument in ('NIRCam', 'MIRI'):
        filterlist = SvoFps.get_filter_list(telescope, instrument=instrument)
        filterids = filterlist['filterID']
        fluxes_ = {fid: convsum(xarr, modeldata, SvoFps.get_transmission_data(fid), doplot=doplot)
                   for fid in list(filterids)}
        fluxes.update(fluxes_)
    return fluxes
