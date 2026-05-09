"""
4500K stellar SED + CT06 dust + H2O:CO:CO2 (10:1:1) mix ice
at A_V = 0, 10, 50, 100.

CO abundance X(CO) = N(CO)/N(H2) = 2.5e-4 (the icemodels paper default,
abundance is wrt molecular hydrogen, NOT wrt total H).
N(H) = 2.21e21 * A_V cm^-2;  N(H2) = N(H) / 2.
Mix has CO fraction 1/12, so total mix column N_mix = 12 * N(CO).
Mean molecular weight from composition_to_molweight.
Computes [F405N]-[F466N] color of the ice+dust spectrum, shows in legend.
"""
from pathlib import Path

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.table import Table
from astroquery.svo_fps import SvoFps
from dust_extinction.averages import CT06_MWGC

from icemodels import (plot_stellar_seds, atmo_model, absorbed_spectrum,
                       fluxes_in_filters, optical_constants_cache_dir)
from icemodels.core import composition_to_molweight


MIX_COMPOSITION = 'H2O:CO:CO2 (10:1:1)'
MIX_PATH = (Path(optical_constants_cache_dir) / 'mymixes'
            / f'{MIX_COMPOSITION.replace(" ", "_")}.ecsv')
mix_table = Table.read(MIX_PATH)
# Make sure metadata round-trips for absorbed_spectrum + plot_stellar_seds.
mix_table.meta.setdefault('composition', MIX_COMPOSITION)
mix_molwt = u.Quantity(composition_to_molweight(MIX_COMPOSITION), u.Da)

xarr = np.linspace(1.5 * u.um, 5.2 * u.um, 50000)

filternames = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F405N',
               'JWST/NIRCam.F410M', 'JWST/NIRCam.F466N']
color_filters = ['JWST/NIRCam.F405N', 'JWST/NIRCam.F466N']

jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')
zp = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in color_filters}
trans = {fid: SvoFps.get_transmission_data(fid) for fid in color_filters}

NH_PER_AV = 2.21e21      # N(H) per A_V, cm^-2 mag^-1
NH2_TO_NH = 2.0          # N(H) = 2 * N(H2)
X_CO = 2.5e-4            # X(CO) = N(CO) / N(H2)
CO_FRAC_IN_MIX = 1.0 / 12.0  # H2O:CO:CO2 = 10:1:1 -> 12 parts total
TEMP = 4500
LOGG = 2.5

output_dir = Path('plots')
output_dir.mkdir(parents=True, exist_ok=True)

ext_curve = CT06_MWGC()
stellar = atmo_model(TEMP, xarr=xarr, logg=LOGG)
fnu_base = stellar['fnu'].quantity


def f405_f466_color(fnu_arr):
    flx = fluxes_in_filters(xarr, fnu_arr, filterids=color_filters,
                            transdata=trans)
    m405 = -2.5 * np.log10(flx['JWST/NIRCam.F405N']
                           / u.Quantity(zp['JWST/NIRCam.F405N'], u.Jy))
    m466 = -2.5 * np.log10(flx['JWST/NIRCam.F466N']
                           / u.Quantity(zp['JWST/NIRCam.F466N'], u.Jy))
    return float((m405 - m466).value)


def apply_extinction(fnu, Av):
    if Av <= 0:
        return fnu
    x = 1.0 / xarr.to(u.um).value
    xr = ext_curve.x_range
    valid = (x >= xr[0]) & (x <= xr[1])
    ext_alav = np.zeros_like(x)
    ext_alav[valid] = ext_curve(x[valid] / u.um)
    return fnu * 10 ** (-0.4 * ext_alav * Av)


for Av in [0, 10, 50, 100]:
    N_H2 = NH_PER_AV * Av / NH2_TO_NH             # cm^-2 of H2
    N_CO = X_CO * N_H2                            # cm^-2 of CO molecules
    N_mix = N_CO / CO_FRAC_IN_MIX                 # total mix column

    fnu_ext = apply_extinction(fnu_base, Av)
    if N_mix > 0:
        fnu_iced = absorbed_spectrum(ice_column=N_mix * u.cm**-2,
                                     ice_model_table=mix_table,
                                     spectrum=fnu_ext, xarr=xarr,
                                     molecular_weight=mix_molwt)
    else:
        fnu_iced = fnu_ext

    color_val = f405_f466_color(fnu_iced)

    if Av > 0:
        # Single fully-absorbed curve (extinction + ice), normalized to its
        # own max within the plotted range so the shape stays visible at
        # arbitrary A_V.
        ice_legend = (f'CT06 A$_V$={Av}, '
                      f'H$_2$O:CO:CO$_2$ (10:1:1), '
                      f'N(CO)={N_CO:.2e} cm$^{{-2}}$, '
                      f'[F405N]-[F466N]={color_val:+.2f}')
        fig, axes = plot_stellar_seds(
            temperatures=TEMP, logg=LOGG, filters=filternames, xarr=xarr,
            ice_model_table=[mix_table],
            ice_column=[N_mix * u.cm**-2],
            ice_labels=[ice_legend],
            molecular_weight=[mix_molwt],
            extinction_Av=float(Av), extinction_curve=ext_curve,
            show_ice_absorbed=True,
            normalize_per_curve=True,
            show_baseline_when_ice=False,
            color_cycle=['red'])
    else:
        # Av=0: baseline atmosphere (no ext, no ice).
        fig, axes = plot_stellar_seds(
            temperatures=TEMP, logg=LOGG, filters=filternames, xarr=xarr,
            color_cycle=['blue'])
        ax_main = axes[0]
        ax_main.plot([], [], ' ',
                     label=(f'A$_V$=0, no ice, '
                            f'[F405N]-[F466N]={color_val:+.2f}'))
        ax_main.legend(loc='best', fontsize=9)

    out = output_dir / f'{TEMP}KStellarSEDexample_Av{Av:03d}_H2OCOCO2.png'
    fig = axes[0].figure
    fig.canvas.draw()
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Av={Av:3d}: N(CO)={N_CO:.3e} cm^-2, '
          f'N_mix={N_mix:.3e} cm^-2, '
          f'[F405N]-[F466N]={color_val:+.3f} -> {out}')
    plt.close(fig)
