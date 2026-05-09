"""
Driver: produce CCDs, color-vs-column (CMD-like), twoPanel, and diagnostic
SED-overlay spectra for the mixes3 set (= mixes2 + 5% OCN- vs H2O).

Run after icemodels.build_mixes3_with_ocn has appended the 500 mixes3 rows
to combined_ice_absorption_tables.ecsv.

Outputs land under icemodels/figures/.
"""
import os
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as pl

import numpy as np
import astropy.units as u
from astropy.table import Table
from astroquery.svo_fps import SvoFps
from dust_extinction.averages import CT06_MWGC

from icemodels.colorcolordiagrams import (
    example_plots, plot_ccd_icemodels, plot_color_vs_column, propcycle,
    mixes3,
)
from icemodels import (
    plot_stellar_seds, atmo_model, absorbed_spectrum, fluxes_in_filters,
    optical_constants_cache_dir,
)
from icemodels.core import composition_to_molweight


basepath = '/blue/adamginsburg/adamginsburg/repos/icemodels'
savefig = os.path.join(basepath, 'icemodels', 'figures', 'mixes3')
os.makedirs(savefig, exist_ok=True)

dmag_tbl = Table.read(os.path.join(basepath, 'icemodels', 'data',
                                   'combined_ice_absorption_tables.ecsv'))
for k in ('mol_id', 'composition', 'temperature', 'database', 'author'):
    dmag_tbl.add_index(k)


def _ccd_and_color_vs_column():
    cfgs = [p for p in example_plots if p.get('icemix_name') == 'mixes3']
    print(f'mixes3 plot configs: {len(cfgs)}')
    with mpl.rc_context({'axes.prop_cycle': propcycle}):
        for cfg in cfgs:
            pl.figure()
            plot_ccd_icemodels(
                color1=cfg['color1'], color2=cfg['color2'], dmag_tbl=dmag_tbl,
                molcomps=cfg['molcomps'], axlims=cfg['axlims'],
                abundance_wrt_h2=cfg['abundance_wrt_h2'],
                max_column=cfg['max_column'], icemol=cfg['icemol'],
                label_author=cfg.get('label_author', False),
                label_temperature=cfg.get('label_temperature', False),
                av_start=cfg.get('av_start', 0))
            pl.legend(loc='upper left', bbox_to_anchor=(1, 1, 0, 0))
            pl.title(cfg['title'])
            out = os.path.join(savefig, cfg['filename'])
            pl.savefig(out, bbox_inches='tight', dpi=150); pl.close()
            print('  CCD ->', out)

            colors = [cfg['color1'], cfg['color2']]
            for extra in cfg.get('extra_dmag_colors', []):
                colors.append(list(extra))
            seen = set()
            colors = [c for c in colors
                      if not (tuple(c) in seen or seen.add(tuple(c)))]
            for color in colors:
                pl.figure()
                try:
                    plot_color_vs_column(
                        color, dmag_tbl=dmag_tbl,
                        molcomps=cfg['molcomps'],
                        abundance_wrt_h2=cfg['abundance_wrt_h2'],
                        max_column=cfg['max_column'], icemol=cfg['icemol'],
                        label_author=cfg.get('label_author', False),
                        label_temperature=cfg.get('label_temperature', False),
                        av_start=cfg.get('av_start', 0))
                    pl.legend(loc='best', fontsize=8)
                    out2 = os.path.join(
                        savefig,
                        f"colorVScolumn_{color[0]}-{color[1]}_mixes3.png")
                    pl.savefig(out2, bbox_inches='tight', dpi=150)
                    print('  C-vs-col ->', out2)
                except Exception as ex:
                    print(f'  color-vs-col {color}: {ex}')
                pl.close()


def _diagnostic_seds():
    """Per mix in mixes3, plot the absorbed SED at 4500K + Av=30 with
    N_mix scaled so N(CO) = 2.5e-4 * N(H2). Overlay filter markers, save."""
    xarr = np.linspace(1.5 * u.um, 5.2 * u.um, 50000)
    filters = ['JWST/NIRCam.F212N', 'JWST/NIRCam.F405N',
               'JWST/NIRCam.F410M', 'JWST/NIRCam.F466N']
    color_filters = ['JWST/NIRCam.F405N', 'JWST/NIRCam.F466N']
    jf = SvoFps.get_filter_list('JWST'); jf.add_index('filterID')
    zp = {f: float(jf.loc[f]['ZeroPoint']) for f in color_filters}
    trans = {f: SvoFps.get_transmission_data(f) for f in color_filters}
    ext_curve = CT06_MWGC()

    Av = 30.0
    NH_PER_AV, NH2_TO_NH, X_CO = 2.21e21, 2.0, 2.5e-4
    N_H2 = NH_PER_AV * Av / NH2_TO_NH
    N_CO = X_CO * N_H2

    mymix_dir = os.path.join(optical_constants_cache_dir, 'mymixes')

    for comp, T in mixes3:
        mix_path = os.path.join(mymix_dir, f"{comp.replace(' ', '_')}.ecsv")
        if not os.path.exists(mix_path):
            print(f'  missing mix table {mix_path}'); continue
        mix_table = Table.read(mix_path)
        mix_table.meta['composition'] = comp
        mix_molwt = u.Quantity(composition_to_molweight(comp), u.Da)

        mols = comp.split(' ')[0].split(':')
        parts = [float(p) for p in comp.split(' ')[1].strip('()').split(':')]
        co_frac = parts[mols.index('CO')] / sum(parts)
        N_mix = N_CO / co_frac

        stellar = atmo_model(4500, xarr=xarr, logg=2.5)
        fnu0 = stellar['fnu'].quantity
        # extinction
        x = 1.0 / xarr.to(u.um).value
        xr = ext_curve.x_range
        valid = (x >= xr[0]) & (x <= xr[1])
        ext_alav = np.zeros_like(x)
        ext_alav[valid] = ext_curve(x[valid] / u.um)
        fnu_ext = fnu0 * 10 ** (-0.4 * ext_alav * Av)
        fnu_iced = absorbed_spectrum(
            ice_column=N_mix * u.cm**-2, ice_model_table=mix_table,
            spectrum=fnu_ext, xarr=xarr, molecular_weight=mix_molwt)

        flx = fluxes_in_filters(xarr, fnu_iced, filterids=color_filters,
                                transdata=trans)
        m = {f: -2.5*np.log10(flx[f]/u.Quantity(zp[f], u.Jy))
             for f in color_filters}
        cval = float((m[color_filters[0]] - m[color_filters[1]]).value)

        legend = (f'CT06 A$_V$={Av:.0f}, {comp}, '
                  f'N(CO)={N_CO:.2e}, '
                  f'[F405N]-[F466N]={cval:+.2f}')
        fig, axes = plot_stellar_seds(
            temperatures=4500, logg=2.5, filters=filters, xarr=xarr,
            ice_model_table=[mix_table], ice_column=[N_mix * u.cm**-2],
            ice_labels=[legend], molecular_weight=[mix_molwt],
            extinction_Av=Av, extinction_curve=ext_curve,
            show_ice_absorbed=True, normalize_per_curve=True,
            show_baseline_when_ice=False, color_cycle=['red'])
        fig = axes[0].figure; fig.canvas.draw()
        safe = comp.replace(' ', '_').replace(':', '-')
        out = os.path.join(savefig, f'SED_overlay_{safe}.png')
        fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
        pl.close(fig)
        print(f'  SED -> {out}  [F405N]-[F466N]={cval:+.3f}')


if __name__ == '__main__':
    _ccd_and_color_vs_column()
    _diagnostic_seds()
    print('all mixes3 plots written to', savefig)
