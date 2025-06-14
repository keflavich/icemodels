from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
# from molmass import Formula
from icemodels.core import composition_to_molweight
from dust_extinction.averages import CT06_MWGC  # , G21_MWAvg
from tqdm.auto import tqdm
import os

pl.rcParams['axes.prop_cycle'] = pl.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'], ) * pl.cycler(linestyle=['-', '--', ':', '-.'])

x = np.linspace(1.24*u.um, 5*u.um, 1000)
pp_ct06 = np.polyfit(x, CT06_MWGC()(x), 7)


def ext(x, model=CT06_MWGC()):
    if x > 1/model.x_range[1]*u.um and x < 1/model.x_range[0]*u.um:
        return model(x)
    else:
        return np.polyval(pp_ct06, x.value)


def plot_ccd_icemodels(color1, color2, dmag_tbl, molcomps=None, molids=None, axlims=[-1, 4, -2.5, 1],
                       nh2_to_av=2.21e21, abundance=2e-5, av_start=20, max_column=2e20, icemol='CO',
                       icemol2=None, icemol2_col=None, icemol2_abund=None, ext=ext, temperature_id=0,
                       label_author=False, label_temperature=False,
                       pure_ice_no_dust=False):
    """
    Plot only the model tracks for given color combinations and ice compositions.
    """
    def wavelength_of_filter(filtername):
        return u.Quantity(int(filtername[1:-1])/100, u.um).to(u.um, u.spectral())

    E_V_color1 = (ext(wavelength_of_filter(color1[0])) - ext(wavelength_of_filter(color1[1])))
    E_V_color2 = (ext(wavelength_of_filter(color2[0])) - ext(wavelength_of_filter(color2[1])))

    if molcomps is not None:
        if isinstance(molcomps[0][1], tuple):
            molids = [np.unique(dmag_tbl.loc['author', author].loc['composition', mc].loc['temperature', str(tem)]['mol_id']) for (author, (mc, tem)) in molcomps]
            molcomps = [xx[1] for xx in molcomps]
        else:
            molids = [np.unique(dmag_tbl.loc['composition', mc].loc['temperature', str(tem)]['mol_id']) for mc, tem in molcomps]
    else:
        molcomps = np.unique(dmag_tbl.loc[molids]['composition'])

    assert len(molcomps) == len(molids)
    assert len(molcomps) > 0

    dcol = 2
    for mol_id, (molcomp, temperature) in tqdm(zip(molids, molcomps)):
        if isinstance(mol_id, tuple):
            mol_id, database = mol_id
            tb = dmag_tbl.loc[mol_id].loc['database', database].loc['composition', molcomp]
        else:
            tb = dmag_tbl.loc[mol_id].loc['composition', molcomp]
        comp = np.unique(tb['composition'])[0]
        temp = np.unique(tb['temperature'])[temperature_id]
        tb = tb.loc['temperature', temp]

        sel = tb['column'] < max_column
        if sel.sum() == 0:
            print(f"No data for {comp} at {temp} K")
            continue
        try:
            molwt = u.Quantity(composition_to_molweight(comp), u.Da)
            from icemodels.core import molscomps
            mols, comps = molscomps(comp)
        except Exception as ex:
            print(f'Error converting composition {comp} to molwt: {ex}')
            continue
        if icemol in mols:
            mol_frac = comps[mols.index(icemol)] / sum(comps)
        else:
            print(f"icemol {icemol} not in {mols} for {comp}.  tb.meta={tb.meta}")
            continue

        col = tb['column'][sel] * mol_frac
        h2col = col / abundance
        a_color1 = h2col / nh2_to_av * E_V_color1 + av_start * E_V_color1
        a_color2 = h2col / nh2_to_av * E_V_color2 + av_start * E_V_color2

        c1 = (tb[color1[0]][sel] if color1[0] in tb.colnames else 0) - (tb[color1[1]][sel] if color1[1] in tb.colnames else 0) + a_color1 * (not pure_ice_no_dust)
        c2 = (tb[color2[0]][sel] if color2[0] in tb.colnames else 0) - (tb[color2[1]][sel] if color2[1] in tb.colnames else 0) + a_color2 * (not pure_ice_no_dust)

        if icemol2 is not None and icemol2 in mols and icemol2_col is not None:
            mol_frac2 = comps[mols.index(icemol2)] / sum(comps)
            ind_icemol2 = np.argmin(np.abs(tb['column'][sel] * mol_frac2 - icemol2_col))
            L, = pl.plot(c1, c2, label=f'{comp} (X$_{{{icemol2}}}$ = {icemol2_col / h2col[ind_icemol2]:0.1e})', )
        else:
            label = comp
            if label_author:
                label = label + f' ({tb.meta["author"]})'
            elif label_temperature:
                label = label + f' ({tb.meta["temperature"]} K)'
            L, = pl.plot(c1, c2, label=label, )

    pl.axis(axlims)
    return a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb


# Constants for abundances and percent ice
carbon_abundance = 10**(8.7-12)
oxygen_abundance = 10**(9.3-12)
percent_ice = 25  # can be changed per plot if needed

# Example plot configurations
example_plots = [
    # Simple CO/H2O mixes
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F410M', 'F466N'],
        'axlims': (0, 3, -1.5, 1.0),
        'molcomps': [
            ('H2O:CO (0.5:1)', 25.0),
            ('H2O:CO (1:1)', 25.0),
            ('H2O:CO (3:1)', 25.0),
            ('H2O:CO (5:1)', 25.0),
            ('H2O:CO (7:1)', 25.0),
            ('H2O:CO (10:1)', 25.0),
            ('H2O:CO (15:1)', 25.0),
            ('H2O:CO (20:1)', 25.0),
        ],
        'icemol': 'CO',
        'abundance': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F410M-F466N_nodata.png',
    },
    # CO/H2O/CO2/CH3OH/CH3CH2OH mixes
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F466N', 'F480M'],
        'axlims': (-0.2, 10, -1, 2.5),
        'molcomps': [
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)', 25.0),
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:0.1:0.1)', 25.0),
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)', 25.0),
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:1:0.1:0.1:1)', 25.0),
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)', 25.0),
        ],
        'icemol': 'CO',
        'abundance': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F466N-F480M_mixes_nodata.png',
    },
    # OCN mixes
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F410M', 'F466N'],
        'axlims': (0, 3, -1.5, 1.0),
        'molcomps': [
            ('CO:OCN (1:1)', 25.0),
            ('H2O:CO:OCN (1:1:1)', 25.0),
            ('H2O:CO:OCN (1:1:0.02)', 25.0),
            ('H2O:CO:OCN (2:1:0.1)', 25.0),
            ('H2O:CO:OCN (2:1:0.5)', 25.0),
        ],
        'icemol': 'CO',
        'abundance': (percent_ice/100.)*carbon_abundance,
        'max_column': 5e19,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 5e19 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F410M-F466N_OCNmixes_nodata.png',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F410M'],
        'axlims': (0, 3, -0.5, 0.5),
        'molcomps': [
            ('Hudgins', ('CO2 (1)', '70K')),
            ('Gerakines', ('CO2 (1)', '70K')),
            ('Hudgins', ('CO2 (1)', '10K')),
            ('Ehrenfreund', ('CO2 (1)', '10K')),
            ('Hudgins', ('CO2 (1)', '30K')),
            ('Hudgins', ('CO2 (1)', '50K')),
            ('Ehrenfreund', ('CO2 (1)', '50K')),
            ('Gerakines', ('CO2 (1)', '8K')),
        ],
        'icemol': 'CO2',
        'abundance': (percent_ice/100.)*carbon_abundance,
        'max_column': 5e19,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 5e19 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F410M_CO2only_nodata.png',
    },
    # Add more plot configs as needed...
]

if __name__ == "__main__":
    """
    The "main" example is intended to be run in the Brick 2221 project's directory.
    """

    savefig_path = '/orange/adamginsburg/jwst/brick/figures/'

    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dmag_tbl = dmag_all = Table.read(os.path.join(basepath, 'icemodels', 'data', 'combined_ice_absorption_tables.ecsv'))
    dmag_all.add_index('mol_id')
    dmag_all.add_index('composition')
    dmag_all.add_index('temperature')
    dmag_all.add_index('database')
    dmag_tbl.add_index('author')

    for plot_cfg in example_plots:
        if dmag_tbl is None:
            raise ValueError("dmag_tbl not loaded. Please load your model table in the __main__ block.")
        pl.figure()
        plot_ccd_icemodels(
            color1=plot_cfg['color1'],
            color2=plot_cfg['color2'],
            dmag_tbl=dmag_tbl,
            molcomps=plot_cfg['molcomps'],
            axlims=plot_cfg['axlims'],
            abundance=plot_cfg['abundance'],
            max_column=plot_cfg['max_column'],
            icemol=plot_cfg['icemol'],
            label_author=plot_cfg.get('label_author', False),
            label_temperature=plot_cfg.get('label_temperature', False),
        )
        pl.legend(loc='upper left', bbox_to_anchor=(1, 1, 0, 0))
        pl.title(plot_cfg['title'])
        pl.savefig(os.path.join(savefig_path, plot_cfg['filename']), bbox_inches='tight', dpi=150)
        pl.close()
