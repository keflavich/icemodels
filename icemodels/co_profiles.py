"""
CO ice profile decomposition following Pontoppidan et al. (2003) and
Bergner et al. (2024, ApJ, 975, 166).

The 4.67 µm CO ice stretching mode shows profile variations that arise from
CO residing in distinct chemical environments within the ice mantle:

  - **Pure/apolar CO** ("CO_pure"): narrow feature centered at ~2139.9 cm⁻¹
    (4.6729 µm), FWHM ~3.5 cm⁻¹. Arises from CO in a matrix of nonpolar
    species (CO, N₂, O₂).

  - **Polar CO** ("CO_polar"): broad feature centered at ~2136.5 cm⁻¹
    (4.6803 µm), FWHM ~10.6 cm⁻¹. Arises from CO mixed with H₂O, CH₃OH
    and other hydrogen-bonding species.

  - **CO:CO₂ component** ("CO_CO2"): intermediate feature centered at
    ~2143.7 cm⁻¹ (4.6646 µm), FWHM ~3.0 cm⁻¹. Arises from CO mixed
    with CO₂ ice.

This module provides:

  1. Functions to retrieve laboratory optical constants for each environment
     from the databases already supported by icemodels (OCDB, DREAM, LIDA).

  2. A composite CO absorption model that sums optical depths from multiple
     environment-dependent components, each with its own column density.

  3. A lightweight fitting function to decompose an observed CO optical depth
     profile into the three canonical components.

References
----------
- Pontoppidan et al. 2003, A&A, 408, 981
- Öberg et al. 2011, ApJ, 740, 109
- Boogert et al. 2022, ApJ, 941, 32
- Bergner et al. 2024, ApJ, 975, 166
- Tielens et al. 1991, ApJ, 381, 181


This module was originally written by Claude Opus 4.6 on 3/24/2026
"""

import glob
import os
import json
import warnings
import numpy as np
import urllib.request
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import astropy.units as u
from astropy.table import Table

from . import core

__all__ = [
    'CO_ENVIRONMENTS',
    'BERGNER_ZENODO_RECORDS',
    'download_bergner_co_profiles',
    'read_bergner_file',
    'load_co_environment',
    'find_co_mixture_files',
    'co_composite_tau',
    'co_composite_absorbed_spectrum',
    'fit_co_profile',
    'pontoppidan_gaussian_tau',
]

BERGNER_ZENODO_RECORDS = (13948069, 13948083)


def download_bergner_co_profiles(record_ids=BERGNER_ZENODO_RECORDS,
                                 redo=False,
                                 cache_dir=core.optical_constants_cache_dir):
    """
    Download CO profile files from the Bergner et al. Zenodo records.

    Parameters
    ----------
    record_ids : tuple of int
        Zenodo record IDs to download.
    redo : bool
        If True, re-download files even if they already exist.
    cache_dir : str
        Directory where downloaded files are stored.

    Returns
    -------
    downloaded_files : list of str
        Paths to local files that exist after the download step.
    """
    os.makedirs(cache_dir, exist_ok=True)
    downloaded_files = []

    for record_id in record_ids:
        api_url = f'https://zenodo.org/api/records/{int(record_id)}'
        with urllib.request.urlopen(api_url, timeout=60) as fh:
            record = json.load(fh)

        rec_title = record.get('metadata', {}).get('title', '')
        for file_info in record.get('files', []):
            key = file_info['key']
            download_url = file_info['links']['self']

            outfn = os.path.join(cache_dir, f'bergner_{record_id}_{key}')
            outfn = outfn.replace(' ', '_')
            downloaded_files.append(outfn)

            if os.path.exists(outfn) and not redo:
                continue

            with urllib.request.urlopen(download_url, timeout=120) as fh:
                payload = fh.read().decode('utf-8')

            meta = {
                'database': 'bergner',
                'record_id': int(record_id),
                'record_title': rec_title,
                'source_file': key,
                'source_url': download_url,
            }
            with open(outfn, 'w') as fh:
                fh.write('# ' + json.dumps(meta) + '\n')
                fh.write(payload)

    return downloaded_files


def read_bergner_file(filename):
    """
    Read a Bergner Zenodo profile file and return an astropy table.

    The raw files are two-column CSV data: wavenumber (cm^-1) and absorbance.
    """
    with open(filename, 'r') as fh:
        first_line = fh.readline()

    has_meta = first_line.startswith('#')
    meta = json.loads(first_line.lstrip('# ')) if has_meta else {}
    data_start = 1 if has_meta else 0

    tb = Table.read(
        filename,
        format='ascii.csv',
        names=['Wavenumber', 'absorbance'],
        data_start=data_start,
    )

    tb['Wavenumber'].unit = u.cm**-1
    tb['Wavelength'] = tb['Wavenumber'].quantity.to(u.um, u.spectral())
    tb.meta.update(meta)
    tb.meta['database'] = 'bergner'

    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    parts = stem.split('_')
    if parts and parts[0] == 'bergner' and len(parts) > 2:
        descriptor = parts[2]
    else:
        descriptor = parts[0]
    tb.meta['composition'] = descriptor

    temp_tokens = [p for p in parts if p.endswith('K')]
    if temp_tokens:
        tb.meta['temperature'] = float(temp_tokens[-1].rstrip('K'))

    tb.meta['density'] = 1 * u.g / u.cm**3
    tb.meta['reference_column_cm2'] = 1e18

    return tb

# ---------------------------------------------------------------------------
# Canonical Pontoppidan component parameters (Gaussian approximation)
# ---------------------------------------------------------------------------
#
# These are the phenomenological Gaussian parameters from Pontoppidan et al.
# (2003) Table 4, widely adopted by Öberg et al. (2011) and Bergner et al.
# (2024).  They are provided as a fallback when laboratory optical constants
# for the specific mixture are not available.
#
# Keys: center (cm⁻¹), center_um (µm), fwhm (cm⁻¹), label, description.

CO_ENVIRONMENTS = {
    'pure': {
        'center_cm': 2139.9,
        'center_um': 1e4 / 2139.9,       # ≈ 4.6729 µm
        'fwhm_cm': 3.5,
        'label': r'CO$_{\rm pure}$',
        'description': ('Pure / apolar CO ice; narrow component at ~2139.9 cm⁻¹.'
                        ' Dominant in cold, CO-rich outer mantle layers.'),
        'ocdb_search_terms': ['CO_(1)', 'CO (1)'],
        'dream_composition': 'CO',
        'molecular_weight': 28 * u.Da,
    },
    'polar': {
        'center_cm': 2136.5,
        'center_um': 1e4 / 2136.5,       # ≈ 4.6803 µm
        'fwhm_cm': 10.6,
        'label': r'CO$_{\rm polar}$',
        'description': ('CO trapped in polar (H₂O-rich) ice; broad, redshifted '
                        'component at ~2136.5 cm⁻¹.'),
        'ocdb_search_terms': ['H2O_CO', 'H2O:CO'],
        'dream_composition': 'H2O : CO',
        'molecular_weight': 28 * u.Da,    # still CO; matrix is H₂O
    },
    'CO2': {
        'center_cm': 2143.7,
        'center_um': 1e4 / 2143.7,       # ≈ 4.6646 µm
        'fwhm_cm': 3.0,
        'label': r'CO$_{\rm CO_2}$',
        'description': ('CO mixed with CO₂ ice; blueshifted component at '
                        '~2143.7 cm⁻¹.'),
        'ocdb_search_terms': ['CO_CO2', 'CO:CO2'],
        'dream_composition': 'CO : CO2',
        'molecular_weight': 28 * u.Da,
    },
}


# ---------------------------------------------------------------------------
# Database lookup helpers
# ---------------------------------------------------------------------------

def find_co_mixture_files(environment, database='ocdb', temperature=10,
                          cache_dir=core.optical_constants_cache_dir):
    """
    Search the locally-cached database files for CO optical constants
    matching a given ice environment.

    Parameters
    ----------
    environment : str
        One of 'pure', 'polar', 'CO2' (keys of ``CO_ENVIRONMENTS``).
    database : str
        Which database to search: 'ocdb', 'dream', 'lida', or 'bergner'.
    temperature : int or None
        Target temperature in K.  Used to filter filenames for OCDB.
    cache_dir : str or None
        Override the default ``icemodels.optical_constants_cache_dir``.

    Returns
    -------
    list of str
        Matching file paths, sorted alphabetically.
    """


    env_name = environment if isinstance(environment, str) else None
    if isinstance(environment, str):
        environment = CO_ENVIRONMENTS[environment]

    def _parse_temp_k(path):
        base = os.path.splitext(os.path.basename(path))[0]
        for token in base.replace('-', '_').split('_'):
            if token.endswith('K'):
                val = token[:-1]
                if val.replace('.', '', 1).isdigit():
                    return float(val)
        return None

    if database == 'ocdb':
        candidates = []
        for term in environment['ocdb_search_terms']:
            pattern = f'{cache_dir}/*{term}*'
            candidates.extend(glob.glob(pattern))

        # Deduplicate
        candidates = sorted(set(candidates))

        # Keep OCDB-like files only
        candidates = [
            fn for fn in candidates
            if not os.path.basename(fn).startswith(('lida_', 'dream_', 'univap_'))
        ]

        # Prefer explicit ocdb_ files when present
        if any(os.path.basename(fn).startswith('ocdb_') for fn in candidates):
            candidates = [
                fn for fn in candidates
                if os.path.basename(fn).startswith('ocdb_')
            ]

        # Environment-specific composition guards
        if env_name == 'polar':
            candidates = [
                fn for fn in candidates
                if (('H2O:CO' in os.path.basename(fn)) or ('H2O_CO' in os.path.basename(fn))
                    or ('CO:H2O' in os.path.basename(fn)) or ('CO_H2O' in os.path.basename(fn)))
                and ('CO2' not in os.path.basename(fn))
            ]
        elif env_name == 'CO2':
            candidates = [
                fn for fn in candidates
                if ('CO:CO2' in os.path.basename(fn)) or ('CO_CO2' in os.path.basename(fn))
            ]

        # Temperature filtering with float support (e.g., 10K and 10.0K)
        if temperature is not None and candidates:
            temp_target = float(temperature)
            with_temp = [(fn, _parse_temp_k(fn)) for fn in candidates]

            exact = [fn for fn, temp in with_temp if temp == temp_target]
            if exact:
                candidates = exact
            else:
                with_known_temp = [(fn, temp) for fn, temp in with_temp if temp is not None]
                if with_known_temp:
                    candidates = [
                        fn for fn, _ in sorted(
                            with_known_temp,
                            key=lambda item: abs(item[1] - temp_target)
                        )
                    ]

        # Deduplicate and sort
        return sorted(set(candidates))

    elif database == 'dream':
        comp = environment['dream_composition']
        pattern = f'{cache_dir}/dream_*{comp.replace(" ", "_")}*'
        return sorted(glob.glob(pattern))

    elif database == 'lida':
        candidates = []
        for term in environment['ocdb_search_terms']:
            pattern = f'{cache_dir}/*{term}*'
            candidates.extend(
                f for f in glob.glob(pattern)
                if 'ocdb' not in f and 'dream' not in f
            )
        return sorted(set(candidates))

    elif database == 'bergner':
        candidates = sorted(glob.glob(f'{cache_dir}/bergner_*'))

        if env_name == 'polar':
            candidates = [
                fn for fn in candidates
                if 'Polar-' in os.path.basename(fn)
            ]
        elif env_name == 'pure':
            candidates = [
                fn for fn in candidates
                if ('Apolar-' in os.path.basename(fn)) or ('_CO_' in os.path.basename(fn))
            ]
        elif env_name == 'CO2':
            candidates = [
                fn for fn in candidates
                if 'CO2' in os.path.basename(fn)
            ]

        if temperature is not None and candidates:
            temp_target = float(temperature)
            with_temp = [(fn, _parse_temp_k(fn)) for fn in candidates]

            exact = [fn for fn, temp in with_temp if temp == temp_target]
            if exact:
                candidates = exact
            else:
                with_known_temp = [(fn, temp) for fn, temp in with_temp if temp is not None]
                if with_known_temp:
                    candidates = [
                        fn for fn, _ in sorted(
                            with_known_temp,
                            key=lambda item: abs(item[1] - temp_target)
                        )
                    ]

        return sorted(set(candidates))

    else:
        raise ValueError(f"Unknown database '{database}'; "
                         "choose from 'ocdb', 'dream', 'lida', 'bergner'.")


def load_co_environment(environment, database='ocdb', temperature=10,
                        filename=None):
    """
    Load optical constants for CO in a specific ice environment.

    This is a convenience wrapper that searches the locally-cached database
    for the appropriate mixture and returns a table suitable for
    ``icemodels.absorbed_spectrum()``.

    If no laboratory data are found, a warning is issued and ``None`` is
    returned.  The caller can then fall back to
    ``pontoppidan_gaussian_tau()``.

    Parameters
    ----------
    environment : str
        One of 'pure', 'polar', 'CO2'.
    database : str
        'ocdb', 'dream', 'lida', or 'bergner'.
    temperature : int or None
        Target temperature in K (OCDB only).
    filename : str or None
        If given, load this specific file instead of searching.

    Returns
    -------
    astropy.table.Table or None
        Table with optical constants (columns include wavelength, n, k and
        density in metadata), or None if no matching file was found.
    """
    if filename is not None:
        files = [filename]
    else:
        files = find_co_mixture_files(environment, database=database,
                                      temperature=temperature)
    if not files:
        warnings.warn(
            f"No {database.upper()} files found for CO environment "
            f"'{environment}' at T={temperature} K.  "
            f"You may need to run icemodels.download_all_{database}() first, "
            f"or use pontoppidan_gaussian_tau() as a fallback.",
            stacklevel=2)
        return None

    # Pick the first match (user can pass filename= for finer control)
    fn = files[0]
    if database == 'ocdb':
        return core.read_ocdb_file(fn)
    elif database == 'dream':
        return core.read_dream_file(fn)
    elif database == 'lida':
        return core.read_lida_file(fn)
    elif database == 'bergner':
        return read_bergner_file(fn)


# ---------------------------------------------------------------------------
# Pontoppidan Gaussian fallback
# ---------------------------------------------------------------------------

def pontoppidan_gaussian_tau(environment, column, xarr,
                             bandstrength=1.1e-17 * u.cm):
    """
    Compute optical depth for a CO component using the Pontoppidan et al.
    (2003) Gaussian parameterisation.

    This is a *fallback* for when laboratory optical constants for a specific
    mixture are unavailable.  The Gaussian approximation captures the gross
    profile shape but not the detailed substructure that laboratory data
    provide.

    Parameters
    ----------
    environment : str
        One of 'pure', 'polar', 'CO2'.
    column : astropy.units.Quantity
        Column density of CO in this environment (cm⁻²).
    xarr : astropy.units.Quantity
        Wavelength array.
    bandstrength : astropy.units.Quantity
        Integrated band strength of the CO stretching mode.  Default
        1.1 × 10⁻¹⁷ cm/molecule (Jiang et al. 1975).

    Returns
    -------
    tau : numpy.ndarray
        Optical depth array on ``xarr``.
    """
    env = CO_ENVIRONMENTS[environment]
    center_um = env['center_um'] * u.um
    # Convert FWHM from cm⁻¹ to µm (local linearisation)
    fwhm_um = (env['fwhm_cm'] * u.cm**(-1)
               * (center_um.to(u.cm))**2).to(u.um)
    sigma_um = fwhm_um / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Gaussian in wavelength
    gauss = np.exp(-0.5 * ((xarr - center_um) / sigma_um)**2)
    # Normalise so that integral of tau over wavenumber = A * N
    # integral of Gaussian over wavelength = sigma * sqrt(2 pi)
    # convert to wavenumber integral: dν = dλ / λ²  (cgs)
    # Simplification: tau_peak = A * N / (sigma_nu * sqrt(2 pi))
    sigma_nu = (env['fwhm_cm'] / (2.0 * np.sqrt(2.0 * np.log(2.0)))) * u.cm**(-1)
    tau_peak = (bandstrength * column / (sigma_nu * np.sqrt(2 * np.pi))).decompose()

    return (tau_peak * gauss).value


# ---------------------------------------------------------------------------
# Composite CO profile
# ---------------------------------------------------------------------------

def co_composite_tau(columns, xarr, tables=None, database='ocdb',
                     temperature=10, use_gaussian_fallback=True,
                     bandstrength=1.1e-17 * u.cm):
    """
    Compute a composite CO optical depth profile from multiple ice
    environments.

    Parameters
    ----------
    columns : dict
        Dictionary mapping environment names ('pure', 'polar', 'CO2') to
        column densities (astropy Quantities in cm⁻²).
        Example: ``{'pure': 5e17*u.cm**-2, 'polar': 1e17*u.cm**-2}``
    xarr : astropy.units.Quantity
        Wavelength array.
    tables : dict or None
        Pre-loaded optical constant tables keyed by environment name.  If
        None, tables are loaded automatically via ``load_co_environment()``.
    database : str
        Database to use when auto-loading tables.
    temperature : int
        Temperature for OCDB lookup.
    use_gaussian_fallback : bool
        If True, use the Pontoppidan Gaussian when no lab data are found
        for a component.  If False, raise an error instead.
    bandstrength : astropy.units.Quantity
        Band strength for the Gaussian fallback.

    Returns
    -------
    tau_total : numpy.ndarray
        Total CO optical depth (sum of all components).
    tau_components : dict
        Individual optical depth arrays keyed by environment name.
    """
    if tables is None:
        tables = {}
        for env_name in columns:
            tables[env_name] = load_co_environment(
                env_name, database=database, temperature=temperature)

    tau_components = {}
    for env_name, col in columns.items():
        tbl = tables.get(env_name)
        if tbl is not None:
            if 'k' in tbl.colnames:
                env = CO_ENVIRONMENTS[env_name]
                tau = core.absorbed_spectrum(
                    ice_column=col,
                    ice_model_table=tbl,
                    molecular_weight=env['molecular_weight'],
                    xarr=xarr,
                    spectrum=np.ones(len(xarr)),   # unity spectrum -> tau only
                    return_tau=True,
                )
                tau_components[env_name] = tau
            elif 'absorbance' in tbl.colnames:
                tau_template = np.interp(
                    xarr.to(u.um).value,
                    tbl['Wavelength'].quantity.to(u.um).value,
                    np.array(tbl['absorbance']),
                    left=np.nanmedian(tbl['absorbance']),
                    right=np.nanmedian(tbl['absorbance']),
                )
                tau_template = tau_template - np.nanmedian(tau_template)
                tau_template = np.clip(tau_template, 0, np.inf)
                ref_col = float(tbl.meta.get('reference_column_cm2', 1e18))
                scale = col.to(u.cm**-2).value / ref_col
                tau_components[env_name] = tau_template * scale
            else:
                raise ValueError(
                    f"Table for environment '{env_name}' must contain either "
                    "'k' (optical constants) or 'absorbance' (profile) column."
                )
        elif use_gaussian_fallback:
            tau_components[env_name] = pontoppidan_gaussian_tau(
                env_name, col, xarr, bandstrength=bandstrength)
        else:
            raise RuntimeError(
                f"No lab data found for CO environment '{env_name}' and "
                f"use_gaussian_fallback=False.")

    tau_total = sum(tau_components.values())
    return tau_total, tau_components


def co_composite_absorbed_spectrum(columns, xarr, spectrum=None, **kwargs):
    """
    Apply a composite CO ice absorption to a background spectrum.

    Parameters
    ----------
    columns : dict
        As in ``co_composite_tau()``.
    xarr : astropy.units.Quantity
        Wavelength array.
    spectrum : array-like or None
        Background flux array.  If None, uses the default Phoenix 4000 K
        model from ``icemodels.core``.
    **kwargs
        Passed to ``co_composite_tau()``.

    Returns
    -------
    absorbed : numpy.ndarray
        Absorbed spectrum.
    tau_total : numpy.ndarray
        Total optical depth.
    tau_components : dict
        Component-wise optical depths.
    """
    if spectrum is None:
        default_spectrum = core.phx4000['fnu']
        default_wl = u.Quantity(core.phx4000['nu'], u.Hz).to(u.um, u.spectral())
        f = interp1d(default_wl.value, default_spectrum,
                     bounds_error=False, fill_value=1.0)
        spectrum = f(xarr.to(u.um).value)

    tau_total, tau_components = co_composite_tau(columns, xarr, **kwargs)
    absorbed = spectrum * np.exp(-tau_total)
    return absorbed, tau_total, tau_components


# ---------------------------------------------------------------------------
# Simple profile fitting
# ---------------------------------------------------------------------------

def fit_co_profile(observed_tau, xarr, tables=None, database='ocdb',
                   temperature=10, use_gaussian_fallback=True,
                   p0=None, bounds=None, bandstrength=1.1e-17 * u.cm):
    """
    Fit an observed CO optical depth profile as a linear combination of
    environment-dependent components.

    Parameters
    ----------
    observed_tau : numpy.ndarray
        Observed optical depth in the CO band region.
    xarr : astropy.units.Quantity
        Wavelength array corresponding to ``observed_tau``.
    tables : dict or None
        Pre-loaded optical constant tables.  Loaded automatically if None.
    database, temperature, use_gaussian_fallback, bandstrength
        As in ``co_composite_tau()``.
    p0 : array-like or None
        Initial guess for [N_pure, N_polar, N_CO2] in cm⁻².
        Default: [5e17, 1e17, 5e16].
    bounds : tuple or None
        Bounds for scipy.optimize.minimize in the form
        ((lo_pure, lo_polar, lo_CO2), (hi_pure, hi_polar, hi_CO2)).
        Default: all between 0 and 1e19.

    Returns
    -------
    result : dict
        'columns': dict of best-fit columns {env_name: Quantity},
        'tau_model': best-fit composite tau,
        'tau_components': dict of component taus,
        'residual': observed_tau - tau_model,
        'chi2': sum of squared residuals,
        'scipy_result': raw scipy OptimizeResult.
    """
    env_names = ['pure', 'polar', 'CO2']

    # Pre-load tables
    if tables is None:
        tables = {}
        for name in env_names:
            tables[name] = load_co_environment(
                name, database=database, temperature=temperature)

    # Precompute unit-column-density templates
    unit_col = 1e17 * u.cm**-2
    templates = {}
    for name in env_names:
        cols_unit = {name: unit_col}
        _, comps = co_composite_tau(
            cols_unit, xarr, tables={name: tables[name]},
            use_gaussian_fallback=use_gaussian_fallback,
            bandstrength=bandstrength)
        templates[name] = comps[name]

    # Optimise column densities
    if p0 is None:
        p0 = [5e17, 1e17, 5e16]

    if bounds is None:
        bounds = [(0, 1e19)] * 3

    def objective(params):
        model = np.zeros_like(observed_tau)
        for i, name in enumerate(env_names):
            scale = params[i] / unit_col.value
            model += scale * templates[name]
        return np.nansum((observed_tau - model)**2)

    res = minimize(objective, p0, bounds=bounds, method='L-BFGS-B')

    best_columns = {name: res.x[i] * u.cm**-2
                    for i, name in enumerate(env_names)}
    tau_model = np.zeros_like(observed_tau)
    tau_comps = {}
    for i, name in enumerate(env_names):
        scale = res.x[i] / unit_col.value
        tau_comps[name] = scale * templates[name]
        tau_model += tau_comps[name]

    return {
        'columns': best_columns,
        'tau_model': tau_model,
        'tau_components': tau_comps,
        'residual': observed_tau - tau_model,
        'chi2': res.fun,
        'scipy_result': res,
    }