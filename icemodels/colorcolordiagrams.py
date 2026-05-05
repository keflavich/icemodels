from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
import matplotlib as mpl
# from molmass import Formula
# from icemodels.core import composition_to_molweight
from dust_extinction.averages import CT06_MWGC  # , G21_MWAvg
import os
from icemodels.core import molscomps

# pl.rcParams['axes.prop_cycle']
propcycle = pl.cycler(
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
) * pl.cycler(linestyle=['-', '--', ':', '-.'])

x = np.linspace(1.24*u.um, 5*u.um, 1000)
# this allows extrapolation from CT06, which empirically looks OK - but should be used with caution!
pp_ct06 = np.polyfit(x, CT06_MWGC()(x), 7)


def _format_composition_label(composition):
    """LaTeX-format an ice composition string for legend display.

    Examples
    --------
        'CO2 (1)'                     -> 'CO$_2$'
        'H2O (1)'                     -> 'H$_2$O'
        'H2O:CO:CO2 (10:1:1)'         -> 'H$_2$O:CO:CO$_2$ (10:1:1)'
        'CH3OH:SO2 (1:1)'             -> 'CH$_3$OH:SO$_2$ (1:1)'
        'CO2:CO 10:4'                 -> 'CO$_2$:CO 10:4'
    Strips trailing ``(1)`` or ``1`` (single-component marker), subscripts
    digit runs that follow letters, and leaves ratio parentheses untouched.
    """
    import re as _re
    s = str(composition).strip()
    # Drop a trailing pure-component marker like ' (1)' or ' 1'
    s = _re.sub(r'\s*\(\s*1\s*\)\s*$', '', s)
    s = _re.sub(r'\s+1\s*$', '', s)
    # Subscript digit runs after a letter, but only inside the species
    # tokens (not inside the trailing ratio parentheses).
    def _subscript_outside_ratio(match):
        return _re.sub(r'([A-Za-z])(\d+)', r'\1$_{\2}$', match.group(0))

    paren = _re.search(r'\(.*\)$', s)
    if paren:
        head = s[: paren.start()]
        tail = s[paren.start():]
        head = _re.sub(r'([A-Za-z])(\d+)', r'\1$_{\2}$', head)
        return head + tail
    return _re.sub(r'([A-Za-z])(\d+)', r'\1$_{\2}$', s)


def _format_temperature_label(temperature):
    """Format temperature for legend: append 'K' unit; strip a redundant
    '.0' on integer-valued temperatures; pass through if already formatted."""
    s = str(temperature).strip()
    if s.lower().endswith('k'):
        return s
    try:
        f = float(s)
        if f == int(f):
            return f"{int(f)} K"
        return f"{f:g} K"
    except ValueError:
        return s


def _resolve_single_mol_id(dmag_tbl, author, composition, temperature, verbose=False):
    """
    Look up a unique mol_id for (author, composition, temperature). The
    precomputed combined-ice-absorption table can contain multiple distinct
    mol_ids that share the same (author, composition, temperature) triple
    (different deposit / annealing histories with different wavelength
    coverage; e.g. Mastrapa H2O 40 K appears as both mol_id 241 and 249).
    Selecting by metadata only would return rows from both, and sorting by
    column would interleave them, producing a non-monotonic 'path' in
    color-color space. This helper enforces a single mol_id per request: if
    multiple match, the lowest mol_id is returned and a warning is emitted.
    """
    sub = (dmag_tbl
           .loc['author', author]
           .loc['composition', composition]
           .loc['temperature', float(temperature)])
    ids = np.unique(np.asarray(sub['mol_id']))
    if ids.size == 0:
        raise KeyError(f"No mol_id found for ({author!r}, {composition!r}, {temperature} K)")
    if ids.size > 1:
        import warnings
        warnings.warn(
            f"({author!r}, {composition!r}, {temperature} K) maps to "
            f"mol_ids {list(map(int, ids))}; using {int(ids[0])}. Specify "
            f"mol_id explicitly to disambiguate.",
            stacklevel=2,
        )
    return int(ids[0])


def compute_molecular_column(unextincted_1m2, dmag_tbl, icemol='CO', filter1='F410M', filter2='F466N',
                             maxcol=1e21, verbose=True):
    dmags1 = dmag_tbl[filter1]
    dmags2 = dmag_tbl[filter2]

    assert len(np.unique(dmag_tbl['composition'])) == 1, "dmag_tbl must have only one composition"
    comp = np.unique(dmag_tbl['composition'])[0]
    # molwt = u.Quantity(composition_to_molweight(comp), u.Da)
    mols, comps = molscomps(comp)
    mol_frac = comps[mols.index(icemol)] / sum(comps)

    cols_of_icemol = dmag_tbl['column'] * mol_frac  # molwt * mol_massfrac / (mol_wt_tgtmol)

    dmag_1m2 = np.array(dmags1) - np.array(dmags2)

    if verbose:
        print(f"min(dmag1) = {np.nanmin(dmags1)}, max(dmag1) = {np.nanmax(dmags1)}")
        print(f"min(unextincted_1m2) = {np.nanmin(unextincted_1m2)}, max(unextincted_1m2) = {np.nanmax(unextincted_1m2)}")

    sortorder = np.argsort(dmag_1m2)
    inferred_molecular_column = np.interp(unextincted_1m2,
                                          xp=dmag_1m2[sortorder][cols_of_icemol < maxcol],
                                          fp=cols_of_icemol[sortorder][cols_of_icemol < maxcol])

    return inferred_molecular_column


def compute_dmag_from_column(cols_of_icemol_observed, dmag_tbl, icemol='CO', filter1='F410M', filter2='F466N',
                             maxcol=1e21, verbose=True):

    # nan values in the dmag table mean zero effect on color
    dmags1 = np.nan_to_num(dmag_tbl[filter1])
    dmags2 = np.nan_to_num(dmag_tbl[filter2])

    assert len(np.unique(dmag_tbl['composition'])) == 1, "dmag_tbl must have only one composition"
    comp = np.unique(dmag_tbl['composition'])[0]
    # molwt = u.Quantity(composition_to_molweight(comp), u.Da)
    mols, comps = molscomps(comp)
    mol_frac = comps[mols.index(icemol)] / sum(comps)
    cols_of_icemol_theory = dmag_tbl['column'] * mol_frac

    dmag_1m2 = np.array(dmags1) - np.array(dmags2)

    sortorder = np.argsort(cols_of_icemol_theory)
    dmag_of_icemol = np.interp(cols_of_icemol_observed,
                               xp=cols_of_icemol_theory[sortorder][cols_of_icemol_theory < maxcol],
                               fp=dmag_1m2[sortorder][cols_of_icemol_theory < maxcol],
                               )

    if verbose:
        print(f"min(dmag1) = {np.nanmin(dmags1)}, max(dmag1) = {np.nanmax(dmags1)}")
        print(f"min(cols_of_icemol_theory) = {np.nanmin(cols_of_icemol_theory)}, max(cols_of_icemol_theory) = {np.nanmax(cols_of_icemol_theory)}")
        print(f"min(dmag_of_icemol) = {np.nanmin(dmag_of_icemol)}, max(dmag_of_icemol) = {np.nanmax(dmag_of_icemol)}")

    return dmag_of_icemol


def ext(x, model=CT06_MWGC()):
    if (x > 1/model.x_range[1]*u.um and
            x < 1/model.x_range[0]*u.um):
        return model(x)
    else:
        return np.polyval(pp_ct06, x.value)


@mpl.rc_context({'axes.prop_cycle': propcycle})
def plot_ccd_icemodels(color1, color2, dmag_tbl, molcomps=None, molids=None,
                       axlims=[-1, 4, -2.5, 1], nh_to_av=2.21e21,
                       abundance_wrt_h2=2e-5, av_start=20, max_column=2e20,
                       max_h2_column=None,
                       icemol='CO', icemol2=None, icemol2_col=None,
                       icemol2_abund=None, ext=ext, temperature_id=0,
                       label_author=False, label_temperature=False,
                       column_to_plot_point=None, pure_ice_no_dust=False,
                       verbose=False,
                       **kwargs):
    """
    Plot only the model tracks for given color combinations and ice compositions.

    abundance is with respect to H2.
    """
    def wavelength_of_filter(filtername):
        return u.Quantity(int(filtername[1:-1])/100, u.um).to(
            u.um, u.spectral())

    E_V_color1 = (ext(wavelength_of_filter(color1[0])) -
                  ext(wavelength_of_filter(color1[1])))
    E_V_color2 = (ext(wavelength_of_filter(color2[0])) -
                  ext(wavelength_of_filter(color2[1])))

    if molcomps is not None:
        if isinstance(molcomps[0][1], tuple):
            molids = [_resolve_single_mol_id(dmag_tbl, author, mc, tem, verbose=verbose)
                      for (author, (mc, tem)) in molcomps]
            molcomps = [xx[1] for xx in molcomps]
        else:
            molids = [int(np.unique(dmag_tbl
                                    .loc['composition', mc]
                                    .loc['temperature', float(tem)]['mol_id'])[0])
                      for mc, tem in molcomps]
    else:
        molcomps = np.unique(dmag_tbl.loc['mol_id', molids]['composition'])

    assert len(molcomps) == len(molids)
    assert len(molcomps) > 0

    if max_h2_column is not None:
        if max_column is not None:
            raise ValueError("max_column and max_h2_column cannot both be set")
        max_column = max_h2_column * abundance_wrt_h2

    dcol = 2
    def format_icemol_label(mol):
        import re
        return re.sub(r'(\D)(\d+)', lambda m: f"{m.group(1)}$_{{{m.group(2)}}}$", mol)

    for mol_id, (molcomp, temperature) in (zip(molids, molcomps)):
        if isinstance(mol_id, tuple):
            mol_id, database = mol_id
            tb = dmag_tbl.loc['mol_id', mol_id].loc['database', database].loc['composition', molcomp]
        else:
            tb = dmag_tbl.loc['mol_id', mol_id].loc['composition', molcomp]
        comp = np.unique(tb['composition'])[0]
        temp = np.unique(tb['temperature'])[temperature_id]
        author = np.unique(tb['author'])[0]
        tb = tb.loc['temperature', float(temp)]

        try:
            # molwt = u.Quantity(composition_to_molweight(comp), u.Da)
            from icemodels.core import molscomps
            mols, comps = molscomps(comp)
        except Exception as ex:
            print(f'Error converting composition {comp} to molwt: {ex}')
            raise ex
            continue
        if icemol in mols:
            mol_frac = comps[mols.index(icemol)] / sum(comps)
        else:
            print(f"icemol {icemol} not in {mols} for {comp}.  tb.meta={tb.meta}")
            continue

        icemol_col = np.geomspace(1e17, max_column, 50)
        sel = icemol_col <= max_column
        h2col = icemol_col / abundance_wrt_h2

        dmag_of_icemol_color1 = compute_dmag_from_column(icemol_col, tb, icemol=icemol, maxcol=max_column, filter1=color1[0], filter2=color1[1], verbose=verbose)
        dmag_of_icemol_color2 = compute_dmag_from_column(icemol_col, tb, icemol=icemol, maxcol=max_column, filter1=color2[0], filter2=color2[1], verbose=verbose)

        # a_colors are the extinction colors
        a_color1 = h2col * 2 / nh_to_av * E_V_color1 + av_start * E_V_color1
        a_color2 = h2col * 2 / nh_to_av * E_V_color2 + av_start * E_V_color2

        # nan_to_num used here because nans are returned if there is no overlap with the filter
        # and in that case, the effective color (dmag) is really zero
        c1 = dmag_of_icemol_color1 + a_color1 * (not pure_ice_no_dust)
        c2 = dmag_of_icemol_color2 + a_color2 * (not pure_ice_no_dust)
        # c1 = ((np.nan_to_num(tb[color1[0]][sel]) if color1[0] in tb.colnames else 0) -
        #       (np.nan_to_num(tb[color1[1]][sel]) if color1[1] in tb.colnames else 0) +
        #       a_color1 * (not pure_ice_no_dust))
        # c2 = ((np.nan_to_num(tb[color2[0]][sel]) if color2[0] in tb.colnames else 0) -
        #       (np.nan_to_num(tb[color2[1]][sel]) if color2[1] in tb.colnames else 0) +
        #       a_color2 * (not pure_ice_no_dust))
        assert not np.any(np.isnan(c1))
        assert not np.any(np.isnan(c2))

        if icemol2 is not None and icemol2 in mols and icemol2_col is not None:
            raise NotImplementedError("icemol2 not implemented correctly / I don't know what I was going for")
            mol_frac2 = comps[mols.index(icemol2)] / sum(comps)
            ind_icemol2 = np.argmin(np.abs(tb['column'][sel] * mol_frac2 - icemol2_col))
            L, = pl.plot(c1, c2, label=f'{comp} (X$_{{{format_icemol_label(icemol2)}}}$ = {icemol2_col / h2col[ind_icemol2]:0.1e})', **kwargs)
        else:
            label = _format_composition_label(comp)
            if label_author:
                label = label + f' {author}'
            if label_temperature:
                label = label + f' {_format_temperature_label(temp)}'
            L, = pl.plot(c1, c2, label=label, **kwargs)

        if column_to_plot_point is not None:
            sel2 = np.argmin(np.abs(tb['column'] - column_to_plot_point))
            pl.plot(c1[sel2], c2[sel2], 'o', color='black', markersize=5)

    pl.axis(axlims)
    pl.xlabel(f"{color1[0]} - {color1[1]}")
    pl.ylabel(f"{color2[0]} - {color2[1]}")
    # If axis labels for icemol are used, update to subscripted
    # (This function does not set x/y labels for icemol directly, but legend is handled above)
    return a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb


@mpl.rc_context({'axes.prop_cycle': propcycle})
def plot_color_vs_column(color, dmag_tbl, molcomps=None, molids=None,
                         icemol='CO', abundance_wrt_h2=2e-5, av_start=0,
                         max_column=2e20, max_h2_column=None,
                         nh_to_av=2.21e21, ext=ext, temperature_id=0,
                         xaxis='icemol', include_dust=True,
                         label_author=False, label_temperature=False,
                         ax=None, verbose=False, **kwargs):
    """
    Plot model color (filter1 - filter2) versus column density of H2 for one or
    more ice compositions assuming a specified abundance.

    Parameters
    ----------
    color : tuple of str
        (filter1, filter2). Plotted color is filter1 - filter2.
    dmag_tbl : astropy.table.Table
        Precomputed dmag table (as in `plot_ccd_icemodels`). Must be
        indexed on 'mol_id', 'composition', 'temperature' (and 'database'
        / 'author' if those keys are used in molcomps).
    molcomps, molids : see `plot_ccd_icemodels`.
    icemol : str
        Molecule whose column density is reported on the x-axis (and used
        to scale composition fractions). Must appear in each composition.
    abundance_wrt_h2 : float
        N(icemol) / N(H2). Used to convert icemol column to H2 column
        when `xaxis='h2'` or when `include_dust=True`.
    max_column, max_h2_column : float
        Upper edge of the column grid. Pass at most one.
    xaxis : {'icemol', 'h2'}
        Whether the x-axis is N(icemol) [cm^-2] or N(H2) [cm^-2].
    include_dust : bool
        If True, add foreground dust reddening to the ice color. The
        reddening uses CT06_MWGC (extrapolated polynomially) and an Av
        from `av_start` plus the implied H2 column.
    av_start : float
        Starting (foreground) Av if `include_dust=True`.
    ax : matplotlib axis, optional
        Axis to plot into. Defaults to current axis.
    **kwargs : dict
        Forwarded to `ax.plot`.

    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        ax = pl.gca()

    # Prepare for twinned x-axis
    twin_ax = None

    if max_h2_column is not None:
        if max_column is not None and max_column != 2e20:
            raise ValueError("max_column and max_h2_column cannot both be set")
        max_column = max_h2_column * abundance_wrt_h2

    def wavelength_of_filter(filtername):
        return u.Quantity(int(filtername[1:-1])/100, u.um).to(
            u.um, u.spectral())

    if include_dust:
        E_V_color = (ext(wavelength_of_filter(color[0])) -
                     ext(wavelength_of_filter(color[1])))
    else:
        E_V_color = 0.0

    if molcomps is not None:
        if isinstance(molcomps[0][1], tuple):
            molids = [_resolve_single_mol_id(dmag_tbl, author, mc, tem, verbose=verbose)
                      for (author, (mc, tem)) in molcomps]
            molcomps = [xx[1] for xx in molcomps]
        else:
            molids = [int(np.unique(dmag_tbl
                                    .loc['composition', mc]
                                    .loc['temperature', float(tem)]['mol_id'])[0])
                      for mc, tem in molcomps]
    else:
        molcomps = np.unique(dmag_tbl.loc['mol_id', molids]['composition'])

    assert len(molcomps) == len(molids)
    assert len(molcomps) > 0


    # Store for twinning
    all_icemol_col = None
    all_h2col = None

    for mol_id, (molcomp, temperature) in zip(molids, molcomps):
        if isinstance(mol_id, tuple):
            mol_id, database = mol_id
            tb = (dmag_tbl.loc['mol_id', mol_id]
                          .loc['database', database]
                          .loc['composition', molcomp])
        else:
            tb = dmag_tbl.loc['mol_id', mol_id].loc['composition', molcomp]
        comp = np.unique(tb['composition'])[0]
        temp = np.unique(tb['temperature'])[temperature_id]
        author = np.unique(tb['author'])[0]
        tb = tb.loc['temperature', float(temp)]

        mols, comps = molscomps(comp)
        if icemol not in mols:
            print(f"icemol {icemol} not in {mols} for {comp}")
            continue

        icemol_col = np.geomspace(1e17, max_column, 50)
        h2col = icemol_col / abundance_wrt_h2
        # print("DEBUG:", max_column, max_h2_column, h2col.max(), icemol_col.max(), abundance_wrt_h2)

        # Save for twinned axis
        all_icemol_col = icemol_col
        all_h2col = h2col

        dmag = compute_dmag_from_column(icemol_col, tb, icemol=icemol,
                                        maxcol=max_column,
                                        filter1=color[0], filter2=color[1],
                                        verbose=verbose)

        if include_dust:
            a_color = h2col * 2 / nh_to_av * E_V_color + av_start * E_V_color
            yvals = dmag + a_color
        else:
            yvals = dmag

        xvals = icemol_col if xaxis == 'icemol' else h2col

        label = _format_composition_label(comp)
        if label_author:
            label = label + f' {author}'
        if label_temperature:
            label = label + f' {_format_temperature_label(temp)}'
        ax.plot(xvals, yvals, label=label, **kwargs)


    ax.set_xscale('log')
    def format_icemol_label(mol):
        # Replace trailing digits with subscript in LaTeX
        import re
        return re.sub(r'(\D)(\d+)', lambda m: f"{m.group(1)}$_{{{m.group(2)}}}$", mol)

    if xaxis == 'icemol':
        ax.set_xlabel(f'N({format_icemol_label(icemol)}) [cm$^{{-2}}$]')
        # Add top axis for N(H2)
        twin_ax = ax.twiny()
        twin_ax.set_xscale('log')
        # Set limits to match
        twin_ax.set_xlim(ax.get_xlim())
        # Map icemol_col to h2col for ticks
        if all_icemol_col is not None and all_h2col is not None:
            # Choose a few ticks for icemol_col, map to h2col
            ticks = ax.get_xticks()
            # Remove ticks outside range
            ticks = ticks[(ticks >= all_icemol_col.min()) & (ticks <= all_icemol_col.max())]
            twin_ticks = ticks / abundance_wrt_h2
            twin_ax.set_xticks(ticks)
            twin_ax.set_xticklabels([f"{val:.1e}" for val in twin_ticks])
        twin_ax.set_xlabel('N(H$_2$) [cm$^{-2}$]')
    elif xaxis == 'h2':
        ax.set_xlabel('N(H$_2$) [cm$^{-2}$]')
        # Add top axis for N(icemol)
        twin_ax = ax.twiny()
        twin_ax.set_xscale('log')
        twin_ax.set_xlim(ax.get_xlim())
        if all_h2col is not None and all_icemol_col is not None:
            ticks = ax.get_xticks()
            ticks = ticks[(ticks >= all_h2col.min()) & (ticks <= all_h2col.max())]
            twin_ticks = ticks * abundance_wrt_h2
            twin_ax.set_xticks(ticks)
            twin_ax.set_xticklabels([f"{val:.1e}" for val in twin_ticks])
        twin_ax.set_xlabel(f'N({format_icemol_label(icemol)}) [cm$^{{-2}}$]')
    else:
        raise ValueError(f"xaxis must be 'icemol' or 'h2', got {xaxis!r}")
    ax.set_ylabel(f'{color[0]} - {color[1]}')

    # Add a third axis below for A_V (always linear)
    def h2_to_av(h2):
        return h2 / nh_to_av
    def av_to_h2(av):
        return av * nh_to_av

    # The A_V axis should always correspond to N(H2), which is:
    # - the main axis if xaxis == 'h2'
    # - the twin axis if xaxis == 'icemol'

    if xaxis == 'h2':
        av_secax = ax.secondary_xaxis('bottom', functions=(h2_to_av, av_to_h2))
    else:
        av_secax = twin_ax.secondary_xaxis('bottom', functions=(h2_to_av, av_to_h2))
    av_secax.set_xlabel(r'$A_V$',)
    av_secax.tick_params(axis='x', which='both', pad=5, direction='out')
    av_secax.spines['bottom'].set_position(('outward', 40))

    return ax


# Constants for abundances and percent ice
carbon_abundance = 10**(8.7-12)  # = 1e-3.3 = 5e-4
solar_carbon_abundance = 10**(8.43-12)  # Asplund et al. 2009, Table 1
oxygen_abundance = 10**(9.3-12)
percent_ice = 25  # can be changed per plot if needed


def _fmt_sci(value, digits=1):
    """Format a float as a LaTeX scientific-notation string. Returns
    e.g. ``2.5\\times10^{-4}`` for value=2.5e-4. The result is intended to be
    embedded inside a math-mode block (``$...$``) in a plot title."""
    if value == 0:
        return "0"
    s = f"{value:.{digits}e}"          # e.g. '2.5e-04'
    mantissa, exponent = s.split('e')
    exp_int = int(exponent)
    return f"{mantissa}\\times10^{{{exp_int}}}"

# Ice-mix sets used in the paper. Mirrors the lists in
# brick-jwst-2221/brick2221/analysis/make_ccd_with_icemodels.py so
# colorcolordiagrams.py can render the corresponding dmag-vs-column / CCD
# figures from a single source of truth.
mixes1 = [
    ('H2O:CO (1:1)', 25.0),
    ('H2O:CO (3:1)', 25.0),
    ('H2O:CO (5:1)', 25.0),
    ('H2O:CO (10:1)', 25.0),
    ('H2O:CO (20:1)', 25.0),
    ('H2O:CO:CO2:CH3OH (1:1:0.1:0.1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)', 25.0),
]
mixes2 = [
    ('H2O:CO:CO2 (1:1:1)', 25.0),
    ('H2O:CO:CO2 (2:1:1)', 25.0),
    ('H2O:CO:CO2 (3:1:1)', 25.0),
    ('H2O:CO:CO2 (5:1:1)', 25.0),
    ('H2O:CO:CO2 (10:1:1)', 25.0),
    ('H2O:CO:CO2 (10:1:0.5)', 25.0),
    ('H2O:CO:CO2 (15:1:1)', 25.0),
    ('H2O:CO:CO2 (20:1:1)', 25.0),
    ('H2O:CO:CO2:CH3OH (1:1:1:1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:1:1:1)', 25.0),
]
molcomps_ch3 = [
    ('CO:HCOOH 1:1', 14.0),
    ('CO:CH3OH:CH3CHO (20:20:1)', 15.0),
    ('CO:CH3OH:CH3CH2OH (20:20:1)', 15.0),
    ('CO:CH3OCH3 (20:1)', 15.0),
    ('CO:CH3OH:CH3OCH3 (20:20:1)', 15.0),
    ('H2O:CO:CO2:CH3OH (1:1:0.1:0.1)', 25.0),
    ('H2O:CO:CO2:CH3OH (1:1:0.1:1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:0.1:0.1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:1:0.1:0.1:1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)', 25.0),
]


def _mix_plot_config(name, molcomps, abundance):
    """Build a plot_configs entry that yields three dmag-vs-color outputs
    (F405N-F410M, F405N-F466N, F356W-F444W) per mix set."""
    return {
        'color1': ['F405N', 'F410M'],
        'color2': ['F405N', 'F466N'],
        'extra_dmag_colors': [['F356W', 'F444W']],
        'axlims': (-0.5, 0.5, -1.5, 1.0),
        'molcomps': molcomps,
        'icemol': 'CO',
        'abundance_wrt_h2': abundance,
        'max_column': 5e19,
        'av_start': 0,
        'label_author': False,
        'label_temperature': False,
        'title': (f"{name}, $N(\\mathrm{{CO}})/N(\\mathrm{{H}}_2)"
                  f"={_fmt_sci(abundance)}$"),
        'filename': f'CCD_icemodel_F405N-F410M_F405N-F466N_{name}_nodata.png',
        'icemix_name': name,
    }


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
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F410M-F466N_nodata.png',
        'icemix_name': 'H2O:CO',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F466N'],
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
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F466N_nodata.png',
        'icemix_name': 'H2O:CO',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F466N'],
        'axlims': (0, 4, -1.5, 1.0),
        'molcomps': [
            ('H2O:CO:CO2 (1:1:1)', 25.0),
            ('H2O:CO:CO2 (3:1:1)', 25.0),
            ('H2O:CO:CO2 (5:1:1)', 25.0),
            ('H2O:CO:CO2 (10:1:1)', 25.0),
            ('H2O:CO:CO2 (15:1:1)', 25.0),
            ('H2O:CO:CO2 (20:1:1)', 25.0),
            ('H2O:CO2:CO 72:25:2.7', -999),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 5e19,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 5e19 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F466N_H2OCOCO2_nodata.png',
        'icemix_name': 'H2O:CO:CO2',
    },
    {
        # This one is totally pointless - it's just a vertical line
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.3, 0.3, -2.5, 1.0),
        'molcomps': [
            ('H2O:CO:CO2 (1:1:1)', 25.0),
            ('H2O:CO:CO2 (3:1:1)', 25.0),
            ('H2O:CO:CO2 (5:1:1)', 25.0),
            ('H2O:CO:CO2 (10:1:1)', 25.0),
            ('H2O:CO:CO2 (15:1:1)', 25.0),
            ('H2O:CO:CO2 (20:1:1)', 25.0),
            ('H2O:CO2:CO 72:25:2.7', -999),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'pure_ice_no_dust': True,
        'column_to_plot_point': 1e19,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F466N_H2OCOCO2_pureicenodust_nodata.png',
        'icemix_name': 'H2O:CO:CO2',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.3, 0.3, -2.5, 1.0),
        'molcomps': [
            ('H2O:CO2:CO 72:25:2.7', -999),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': 5.4e-6,
        'max_column': 2e20,
        'pure_ice_no_dust': True,
        'column_to_plot_point': 1e19,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F466N_H2OCOCO2_pureicenodust_nodata_kp5.png',
        'icemix_name': 'H2O:CO:CO2_kp5',
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
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F466N-F480M_mixes_nodata.png',
        'icemix_name': 'H2O:CO:CO2:CH3OH:CH3CH2OH',
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
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 5e19,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 5e19 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F410M-F466N_OCNmixes_nodata.png',
        'icemix_name': 'H2O:CO:OCN',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F410M'],
        'axlims': (-0.1, 2.5, -0.4, 0.15),
        'molcomps': [
            ('Hudgins', ('CO2 (1)', 70)),
            ('Gerakines', ('CO2 (1)', 70)),
            ('Hudgins', ('CO2 (1)', 10)),
            ('Ehrenfreund', ('CO2 (1)', 10)),
            ('Hudgins', ('CO2 (1)', 30)),
            ('Hudgins', ('CO2 (1)', 50)),
            ('Ehrenfreund', ('CO2 (1)', 50)),
            ('Gerakines', ('CO2 (1)', 8)),
            # ('Mastrapa 2024, Gerakines 2020, etc', ('H2O:CO:CO2 (1:1:1)', 25.0)),
            # ('Mastrapa 2024, Gerakines 2020, etc', ('H2O:CO:CO2:CH3OH (1:1:1:1)', 25.0)),
            # ('Mastrapa 2024, Gerakines 2020, etc', ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:1:1:1)', 25.0)),
        ],
        'icemol': 'CO2',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e19,
        'av_start': 0,
        'column_to_plot_point': 1e18,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e19 cm$^{{-2}}$, $N(\\bullet)=1e18 \\mathrm{{cm}}^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F410M_CO2only_nodata.png',
        'icemix_name': 'CO2only',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F410M'],
        'axlims': (-0.1, 2.5, -0.2, 0.15),
        'molcomps': [
            # ('Curtis', ('H2O (1)', '146K')),
            ('Bertie', ('H2O (1)', 100)),
            #('Mastrapa', ('H2O (1)', 100)),
            ('Kitta', ('H2O (1)', 23)),
            #('Mastrapa', ('H2O (1)', 50)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 10)),
            ('Léger', ('H2O (1)', 77)),
            ('Mastrapa', ('H2O (1)', 25)),
            ('Mastrapa', ('H2O (1)', 20)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': (percent_ice/100.)*oxygen_abundance,
        'max_column': 1e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of O in ice, $N_{{max}}$ = 1e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F410M_H2Oonly_nodata.png',
        'icemix_name': 'H2Oonly',
    },
    {
        'color1': ['F356W', 'F444W'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.5, 1.5, -1.5, 1.0),
        'molcomps': [
            # ('Curtis', ('H2O (1)', '146K')),
            ('Bertie', ('H2O (1)', 100)),
            #('Mastrapa', ('H2O (1)', 100)),
            ('Kitta', ('H2O (1)', 23)),
            #('Mastrapa', ('H2O (1)', 50)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 10)),
            ('Léger', ('H2O (1)', 77)),
            ('Mastrapa', ('H2O (1)', 25)),
            ('Mastrapa', ('H2O (1)', 20)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': (percent_ice/100.)*oxygen_abundance,
        'max_column': 1e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of O in ice, $N_{{max}}$ = 1e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F356W-F444W_F405N-F466N_H2Oonly_nodata.png',
        'icemix_name': 'H2Oonly',
    },
    # All pure-H2O variants whose optical-constants tables span [4.0, 4.7] um
    # with non-zero k (i.e. excluding Curtis/Rajaram/Zhang sub-ranges and Mukai
    # which starts at 4.17 um). Used to bracket how strongly H2O alone can bias
    # F405N-F466N color (and therefore CO column inferences).
    #
    # Mastrapa T = 40, 50, 60, 80, 100, 120 K are *excluded*: those temperatures
    # are present twice in the OCDB precomputed table under two different
    # mol_ids (e.g. 241 and 249 both labeled "Mastrapa H2O 40K", from two
    # different deposits / annealing histories with different wavelength
    # coverage). When selected purely by (author, composition, T), the lookup
    # returns both data sets; sorting by column interleaves them, and the
    # resulting "path" in color1-color2 space oscillates between the two
    # measurements rather than being monotonic. Single-mol_id temperatures
    # (15, 20, 25, 30, 70, 90, 110, 130, 140, 150 K) are clean.
    {
        'color1': ['F356W', 'F444W'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.5, 1.5, -1.5, 1.0),
        'molcomps': [
            # amorphous water (cold deposit; Mastrapa T<110, Hudgins T<=100, Kitta, Léger)
            ('Mastrapa', ('H2O (1)', 15)),
            ('Mastrapa', ('H2O (1)', 20)),
            ('Mastrapa', ('H2O (1)', 25)),
            ('Mastrapa', ('H2O (1)', 30)),
            ('Mastrapa', ('H2O (1)', 70)),
            ('Mastrapa', ('H2O (1)', 90)),
            ('Hudgins', ('H2O (1)', 10)),
            ('Hudgins', ('H2O (1)', 40)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 100)),
            ('Kitta', ('H2O (1)', 23)),
            ('Léger', ('H2O (1)', 77)),
            # crystalline water (annealed/warm; Mastrapa T>=110, Hudgins T>=120, Bertie, Clapp)
            ('Mastrapa', ('H2O (1)', 110)),
            ('Mastrapa', ('H2O (1)', 130)),
            ('Mastrapa', ('H2O (1)', 140)),
            ('Mastrapa', ('H2O (1)', 150)),
            ('Hudgins', ('H2O (1)', 120)),
            ('Hudgins', ('H2O (1)', 140)),
            ('Bertie', ('H2O (1)', 100)),
            ('Clapp', ('H2O (1)', 190)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': oxygen_abundance,
        'max_column': 5e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"Pure H$_2$O, $N(\\mathrm{{H_2O}}) = {_fmt_sci(oxygen_abundance)}\\,N(\\mathrm{{H_2}})$",
        'filename': 'CCD_icemodel_F356W-F444W_F405N-F466N_H2Oallvariants_nodata.png',
        'icemix_name': 'H2Oallvariants',
    },
    {
        'color1': ['F356W', 'F444W'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.5, 1.5, -1.5, 1.0),
        'molcomps': [
            # amorphous water (cold deposit; Mastrapa T<110, Hudgins T<=100, Kitta, Léger)
            ('Mastrapa', ('H2O (1)', 15)),
            ('Mastrapa', ('H2O (1)', 20)),
            ('Mastrapa', ('H2O (1)', 25)),
            ('Mastrapa', ('H2O (1)', 30)),
            ('Mastrapa', ('H2O (1)', 70)),
            ('Mastrapa', ('H2O (1)', 90)),
            ('Hudgins', ('H2O (1)', 10)),
            ('Hudgins', ('H2O (1)', 40)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 100)),
            ('Kitta', ('H2O (1)', 23)),
            ('Léger', ('H2O (1)', 77)),
            # crystalline water (annealed/warm; Mastrapa T>=110, Hudgins T>=120, Bertie, Clapp)
            ('Mastrapa', ('H2O (1)', 110)),
            ('Mastrapa', ('H2O (1)', 130)),
            ('Mastrapa', ('H2O (1)', 140)),
            ('Mastrapa', ('H2O (1)', 150)),
            ('Hudgins', ('H2O (1)', 120)),
            ('Hudgins', ('H2O (1)', 140)),
            ('Bertie', ('H2O (1)', 100)),
            ('Clapp', ('H2O (1)', 190)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': oxygen_abundance*0.1,
        'max_column': 5e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"Pure H$_2$O, $N(\\mathrm{{H_2O}}) = {_fmt_sci(oxygen_abundance*0.1)}\\,N(\\mathrm{{H_2}})$",
        'filename': 'CCD_icemodel_F356W-F444W_F405N-F466N_H2Oallvariants_lowabundance_nodata.png',
        'icemix_name': 'H2Oallvariants_lowabundance',
    },
    {
        'color1': ['F356W', 'F444W'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.5, 1.5, -1.5, 1.0),
        'molcomps': [
            # amorphous water (cold deposit; Mastrapa T<110, Hudgins T<=100, Kitta, Léger)
            ('Mastrapa', ('H2O (1)', 15)),
            ('Mastrapa', ('H2O (1)', 20)),
            ('Mastrapa', ('H2O (1)', 25)),
            ('Mastrapa', ('H2O (1)', 30)),
            ('Mastrapa', ('H2O (1)', 70)),
            ('Mastrapa', ('H2O (1)', 90)),
            ('Hudgins', ('H2O (1)', 10)),
            ('Hudgins', ('H2O (1)', 40)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 100)),
            ('Kitta', ('H2O (1)', 23)),
            ('Léger', ('H2O (1)', 77)),
            # crystalline water (annealed/warm; Mastrapa T>=110, Hudgins T>=120, Bertie, Clapp)
            ('Mastrapa', ('H2O (1)', 110)),
            ('Mastrapa', ('H2O (1)', 130)),
            ('Mastrapa', ('H2O (1)', 140)),
            ('Mastrapa', ('H2O (1)', 150)),
            ('Hudgins', ('H2O (1)', 120)),
            ('Hudgins', ('H2O (1)', 140)),
            ('Bertie', ('H2O (1)', 100)),
            ('Clapp', ('H2O (1)', 190)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': oxygen_abundance*0.50,
        'max_column': 5e20,
        'nh_to_av': 2e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"Pure H$_2$O, $N(\\mathrm{{H_2O}}) = {_fmt_sci(oxygen_abundance*0.50)}\\,N(\\mathrm{{H_2}})$, G/D=10",
        'filename': 'CCD_icemodel_F356W-F444W_F405N-F466N_H2Oallvariants_gdr10_50pct_nodata.png',
        'icemix_name': 'H2Oallvariants_gdr10_50pct',
    },
    # All pure-CO variants from the precomputed table. Same axes / format /
    # color cycle as H2Oallvariants. (Gerakines 25 K appears under two
    # mol_ids — the lookup picks the lowest and emits a warning.)
    {
        'color1': ['F356W', 'F444W'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.5, 1.5, -1.5, 1.0),
        'molcomps': [
            ('Baratta', ('CO (1)', 12.5)),
            ('Ehrenfreund', ('CO (1)', 10.0)),
            ('Ehrenfreund', ('CO (1)', 30.0)),
            ('Elsila', ('CO (1)', 12.0)),
            ('Gerakines', ('CO (1)', 25.0)),
            ('Hudgins', ('CO (1)', 10.0)),
            ('Palumbo', ('CO (1)', 15.0)),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': solar_carbon_abundance,
        'max_column': 1e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"Pure CO, $N(\\mathrm{{CO}}) = {_fmt_sci(solar_carbon_abundance)}\\,N(\\mathrm{{H_2}})$",
        'filename': 'CCD_icemodel_F356W-F444W_F405N-F466N_COallvariants_nodata_solarcarbon.png',
        'icemix_name': 'COallvariants_solarcarbon',
    },
    {
        'color1': ['F356W', 'F444W'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.5, 1.5, -1.5, 1.0),
        'molcomps': [
            ('Baratta', ('CO (1)', 12.5)),
            ('Ehrenfreund', ('CO (1)', 10.0)),
            ('Ehrenfreund', ('CO (1)', 30.0)),
            ('Elsila', ('CO (1)', 12.0)),
            ('Gerakines', ('CO (1)', 25.0)),
            ('Hudgins', ('CO (1)', 10.0)),
            ('Palumbo', ('CO (1)', 15.0)),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': carbon_abundance,
        'max_column': 1e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"Pure CO, $N(\\mathrm{{CO}}) = {_fmt_sci(carbon_abundance)}\\,N(\\mathrm{{H_2}})$",
        'filename': 'CCD_icemodel_F356W-F444W_F405N-F466N_COallvariants_nodata.png',
        'icemix_name': 'COallvariants',
    },
    # All pure-CO2 variants from the precomputed table. CO2 affects F410M
    # and F444W (4.27 um fundamental); F466N is mostly CO-clean of CO2.
    {
        'color1': ['F356W', 'F444W'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.5, 1.5, -1.5, 1.0),
        'molcomps': [
            ('Baratta', ('CO2 (1)', 12.5)),
            ('Ehrenfreund', ('CO2 (1)', 10.0)),
            ('Ehrenfreund', ('CO2 (1)', 50.0)),
            ('Gerakines', ('CO2 (1)', 8.0)),
            ('Gerakines', ('CO2 (1)', 70.0)),
            ('Hudgins', ('CO2 (1)', 10.0)),
            ('Hudgins', ('CO2 (1)', 30.0)),
            ('Hudgins', ('CO2 (1)', 50.0)),
            ('Hudgins', ('CO2 (1)', 70.0)),
        ],
        'icemol': 'CO2',
        'abundance_wrt_h2': carbon_abundance,
        'max_column': 5e19,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"Pure CO$_2$, $N(\\mathrm{{CO_2}}) = {_fmt_sci(carbon_abundance)}\\,N(\\mathrm{{H_2}})$",
        'filename': 'CCD_icemodel_F356W-F444W_F405N-F466N_CO2allvariants_nodata.png',
        'icemix_name': 'CO2allvariants',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F410M'],
        'axlims': (-0.1, 2.5, -0.2, 0.15),
        'molcomps': [
            # ('Curtis', ('H2O (1)', '146K')),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (1:0.6:1)", 180.0)),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (1:1:1)", 80.0)),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (9:1:2)", 30.0)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 10)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': (percent_ice/100.)*oxygen_abundance,
        'max_column': 1e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 1e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F410M_H2OandMethanolonly_nodata.png',
        'icemix_name': 'H2O:CH3OH:CO2',
    },
    {
        'color1': ['F200W', 'F356W'],
        'color2': ['F356W', 'F444W'],
        'axlims': (-1, 4, -0.5, 1.5),
        'molcomps': [
            # ('Curtis', ('H2O (1)', '146K')),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (1:0.6:1)", 180.0)),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (1:1:1)", 80.0)),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (9:1:2)", 30.0)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 10)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': (percent_ice/100.)*oxygen_abundance,
        'max_column': 1e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 1e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F200W-F356W_F356W-F444W_H2OandMethanolonly_nodata.png',
        'icemix_name': 'H2O:CH3OH:CO2',
    },
    _mix_plot_config('mixes1', mixes1, 2.5e-4),
    _mix_plot_config('mixes2', mixes2, 2.5e-4),
    _mix_plot_config('molcomps_ch3', molcomps_ch3, 2.5e-4),
    # Add more plot configs as needed...
]

if __name__ == "__main__":
    """
    The "main" example is intended to be run in the Brick 2221 project's directory.
    """

    with mpl.rc_context({'axes.prop_cycle': propcycle}):

        basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import socket
        if 'ufhpc' in socket.gethostname():
            savefig_path = '/orange/adamginsburg/jwst/brick/figures/'
        else:
            savefig_path = os.path.join(basepath, 'icemodels', 'figures')
            os.makedirs(savefig_path, exist_ok=True)

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
                abundance_wrt_h2=plot_cfg['abundance_wrt_h2'],
                max_column=plot_cfg['max_column'],
                icemol=plot_cfg['icemol'],
                label_author=plot_cfg.get('label_author', False),
                label_temperature=plot_cfg.get('label_temperature', False),
                av_start=plot_cfg.get('av_start', 0),
                column_to_plot_point=plot_cfg.get('column_to_plot_point', None),
                pure_ice_no_dust=plot_cfg.get('pure_ice_no_dust', False),
                **plot_cfg.get('kwargs', {})
            )
            pl.legend(loc='upper left', bbox_to_anchor=(1, 1, 0, 0))
            pl.title(plot_cfg['title'])
            pl.savefig(os.path.join(savefig_path, plot_cfg['filename']),
                       bbox_inches='tight', dpi=150)
            pl.close()

            dmag_color_list = [plot_cfg['color1'], plot_cfg['color2']]
            for extra in plot_cfg.get('extra_dmag_colors', []):
                dmag_color_list.append(list(extra))
            # de-duplicate while preserving order
            seen = set()
            dmag_color_list = [c for c in dmag_color_list
                               if not (tuple(c) in seen or seen.add(tuple(c)))]
            for color in dmag_color_list:
                pl.figure()
                ax = plot_color_vs_column(
                    color=color,
                    dmag_tbl=dmag_tbl,
                    molcomps=plot_cfg['molcomps'],
                    abundance_wrt_h2=plot_cfg['abundance_wrt_h2'],
                    max_column=plot_cfg['max_column'],
                    icemol=plot_cfg['icemol'],
                    label_author=plot_cfg.get('label_author', False),
                    label_temperature=plot_cfg.get('label_temperature', False),
                    av_start=plot_cfg.get('av_start', 0),
                    xaxis='h2',
                    include_dust=True,
                    verbose=True,
                )
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1, 0, 0))

                ymin, ymax = ax.get_ylim()
                ax.set_ylim(max(-5, ymin), min(5, ymax))

                pl.title(plot_cfg['title'] + ' (color vs H$_2$ column)')
                pl.savefig(os.path.join(savefig_path, f'dmag_vs_color_{color[0]}-{color[1]}_{plot_cfg["icemix_name"]}.png'),
                           bbox_inches='tight', dpi=150)
                pl.close()
