import numpy as np
import mysg # Tom Robitaille's YSO grid tool
from astropy.table import Table
from astropy import units as u
import pylab as pl
import requests
from astroquery.svo_fps import SvoFps


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
    tau = (kay * xarr_icm * 4 * np.pi * ice_column / (ice_model_table.meta['density'] / molecular_weight)).decompose()
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


# apolar in CO2, apolar pure CO, polar methanol ice
co_ice_wls_icm = ([2143.7, 2139.9, 2136.5] * u.cm**-1)
co_ice_centers = co_ice_wls = co_ice_wls_icm.to(u.um, u.spectral())
co_ice_widths = (([3.0, 3.5, 10.6] * u.cm**-1)/co_ice_wls_icm * co_ice_wls).to(u.um, u.spectral())
co_ice_bandstrength = 1.1e-17 * u.cm # cm per molecule; Jian+1975 via Boogert+2022
co_ice_bandstrengths = [co_ice_bandstrength/3]*3
co_ice_wls, co_ice_widths

ocn_center = 4.62*u.um
ocn_width = 100*u.cm**-1 / (ocn_center.to(u.cm**-1, u.spectral())) * ocn_center
ocs_center = 4.90*u.um
ocn_bandstrength = 1.3e-16*u.cm # per molecule; via Boogert+2022 from van Broekhuizen+2004
ocs_bandstrength = 1.5e-16*u.cm # per molecule; via Boogert+2022 from Palumbo+1997

water_ice_centers_icm = [1666, 3333]*u.cm**-1
water_ice_widths_icm = [160, 390]*u.cm**-1
water_ice_centers = water_ice_centers_icm.to(u.um, u.spectral())
water_ice_widths = (water_ice_widths_icm / water_ice_centers_icm) * water_ice_centers
water_ice_bandstrengths = [1.2e-17, 2e-16]*u.cm
# first is Gerakines+1995 via Boogert+2007
# second Hagen+1981 vis Boogert+2022 - but this is for the 3um band, not the 6um?
nh4_ice_centers = []
nh4_ice_widths = []
nh4_ice_bandstrengths = []

# Hudgins 1993, with centers & widths from Boogert 2015
# these don't always agree well
# some also come from Allamandola 1992 (2832 from their 2825 in tbl 2, 2597 from footnote of tbl 3)
methanol_ice_centers_icm = [2881.8, 2832, 2597, 2538, 1459, 1128, 1026]*u.cm**-1
methanol_ice_widths_icm = [80, 30, 40, 40, 85, 15, 30]*u.cm**-1
methanol_ice_centers = methanol_ice_centers_icm.to(u.um, u.spectral())
methanol_ice_widths = (methanol_ice_widths_icm / methanol_ice_centers_icm) * methanol_ice_centers
methanol_ice_bandstrengths = [5.6e-18, 4e-18, 2.6e-18, 2.8e-18, 1.2e-17, 1.8e-18, 1.8e-17]*u.cm
methanol_ice_centers_icm.to(u.um, u.spectral())

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
        filterlist = SvoFps.get_filter_list(telescope, instrument)
        filterids = filterlist['filterID']
        fluxes_ = {fid: convsum(xarr, modeldata, SvoFps.get_transmission_data(fid), doplot=doplot)
                   for fid in list(filterids)}
        fluxes.update(fluxes_)
    return fluxes
