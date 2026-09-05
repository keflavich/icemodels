"""
OCN-, HNCO, and supporting laboratory ice optical constants for the Gerakines,
Materese & Hudson 2025 cyanate-ion band-strength measurement.

Two paths to optical constants are provided:

1. ``make_ocn_table_synthetic``: construct k(nu) for OCN- as a Gaussian band
   anchored to the band strength A' = 1.51e-16 cm molecule^-1 measured by
   Gerakines+25 in H2O+HNCO+NH3 (~10:1:1) at 10 K (Table 1 of the paper),
   then derive n(nu) via Kramers-Kronig (Maclaurin's principal-value method).
   This needs no spectrum at all, only the published band strength and an
   assumed Gaussian profile.

2. ``make_ocn_table_from_mix_absorbance``: use the *measured* band profile.
   The Cosmic Ice Laboratory distributes the three 10 K absorbance spectra
   behind Gerakines+25 Figures 1-2 as ``Three_HNCO_ices.xlsx``, downloaded on
   first use by ``download_gerakines2025_ices``. The H2O+HNCO (10:1) ice is
   the NH3-free control for H2O+HNCO+NH3 (~10:1:1), so differencing the two
   removes the H2O and HNCO contributions experimentally
   (``ocn_absorbance_from_mix``) instead of subtracting published n,k for
   each component. The residual is converted to k by Beer's law and scaled
   so the band closes on the measured A'; n follows from the same K-K
   routine. Normalising on A' rather than on the film thickness -- which is
   not distributed with the spectra -- is equivalent to the thin-film recipe
   of Hudgins+1993 / Rocha & Pilling 2014 in the weak-absorber (tau << 1)
   limit, and inherits Gerakines+25's calibration exactly. The full Fresnel
   three-layer iteration is therefore not needed here and is not performed.

References
----------
Gerakines, Materese & Hudson 2025, MNRAS 537, 2918 (doi:10.1093/mnras/staf192)
Hudgins et al. 1993, ApJS 86, 713  -- iterative thin-film K-K
Rocha & Pilling 2014, Spectrochim. Acta A 123, 436  -- NKABS
Gerakines & Hudson 2020, ApJ 901, 52
Hudson et al. 2024 (ApJ, in press)  -- HNCO n,k
"""
import os

import numpy as np
import astropy.units as u
from astropy.table import Table

from icemodels.core import optical_constants_cache_dir


# ---- Gerakines+25 constants ----
OCN_BAND_NU0_CM = 2170.0           # cm^-1, band centre in H2O+HNCO+NH3 (10:1:1)
OCN_BAND_FWHM_CM = 30.0            # cm^-1, literature value for OCN- in H2O matrix
OCN_BAND_APRIME = 1.51e-16         # cm molecule^-1, integrated band strength
OCN_INT_LO_CM = 2100.0             # integration window low
OCN_INT_HI_CM = 2211.0             # integration window high
OCN_TEMPERATURE_K = 10.0
OCN_MOLWT = 42.017 * u.Da          # OCN- (CNO mass)
OCN_MATRIX_DENSITY = 0.93 * u.g / u.cm**3   # H2O-rich amorphous (Hudson+2020)
OCN_VIS_N0 = 1.32                  # n at 670 nm; H2O-rich amorphous
OCN_VIS_NU_CM = 1e7 / 670.0        # 670 nm in cm^-1 ~14925 cm^-1


def gaussian_k_from_aprime(nu_cm, *, nu0=OCN_BAND_NU0_CM,
                           fwhm=OCN_BAND_FWHM_CM,
                           A_prime=OCN_BAND_APRIME,
                           density=OCN_MATRIX_DENSITY,
                           molwt=OCN_MOLWT):
    """
    Build k(nu) as a Gaussian whose integrated A' matches the measured band
    strength.

    A' = (4 pi M_per_molecule / rho) * integral(nu * k(nu) d nu)

    M_per_molecule = molwt expressed as mass per molecule (i.e. molwt in
    Da, since astropy Da is already a per-particle mass unit). For a
    Gaussian centred on nu0 with FWHM ``fwhm`` (and nu0 >> sigma so nu
    varies negligibly across the band), the integral evaluates to
    nu0 * k_peak * sqrt(2 pi) * sigma. Solve for k_peak.
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))    # cm^-1
    coeff = (4.0 * np.pi * molwt / density).to(u.cm**3).value
    integral = A_prime / coeff                            # cm^-2
    k_peak = integral / (nu0 * np.sqrt(2.0 * np.pi) * sigma)
    return k_peak * np.exp(-(nu_cm - nu0)**2 / (2.0 * sigma**2))


def closure_aprime(nu_cm, k, *, density=OCN_MATRIX_DENSITY,
                   molwt=OCN_MOLWT, lo=OCN_INT_LO_CM, hi=OCN_INT_HI_CM):
    """
    Recover A' from k(nu) by integrating over [lo, hi] cm^-1.

    ``nu_cm`` may be in either order: tables in this package are stored
    wavelength-ascending, i.e. wavenumber-descending, which would otherwise
    integrate backwards and return a negative band strength.
    """
    nu_cm = np.asarray(nu_cm, dtype=float)
    k = np.asarray(k, dtype=float)
    order = np.argsort(nu_cm)
    nu_cm = nu_cm[order]
    k = k[order]
    sel = (nu_cm >= lo) & (nu_cm <= hi)
    coeff = (4.0 * np.pi * molwt / density).to(u.cm**3).value
    integral = np.trapezoid(nu_cm[sel] * k[sel], nu_cm[sel])
    return coeff * integral


def kk_maclaurin(nu_cm, k, n_anchor=OCN_VIS_N0, anchor_nu_cm=OCN_VIS_NU_CM):
    """
    Kramers-Kronig of k(nu) on a uniform wavenumber grid using Maclaurin's
    principal-value method (Ohta & Ishida 1988). Returns n(nu) anchored so
    that n(anchor_nu_cm) == n_anchor.

    n(nu) - 1 = (2/pi) P ∫_0^∞ nu' k(nu') / (nu'^2 - nu^2) d nu'

    Discretization: replace the integral by a sum over j with |j - i| odd,
    weighted by 2 * dnu (the trapezoidal-equivalent weight that skips the
    even-offset gridpoints to avoid the pole at j == i).
    """
    nu_cm = np.asarray(nu_cm, dtype=float)
    k = np.asarray(k, dtype=float)
    N = len(nu_cm)
    dnu = (nu_cm[-1] - nu_cm[0]) / (N - 1)
    if not np.allclose(np.diff(nu_cm), dnu, rtol=1e-6):
        raise ValueError("nu_cm must be uniformly spaced")

    nuk = nu_cm * k
    nu2 = nu_cm * nu_cm

    # Vectorize in chunks of rows to keep memory bounded.
    n_minus_1 = np.zeros(N, dtype=float)
    chunk = max(1, 5_000_000 // max(N, 1))
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        i = np.arange(start, end)[:, None]
        j = np.arange(N)[None, :]
        odd = ((j - i) & 1).astype(bool)
        denom = nu2[None, :] - nu2[i[:, 0], None]
        # Mask before dividing to avoid divide-by-zero on the diagonal,
        # which Maclaurin's method excludes anyway.
        safe_denom = np.where(odd, denom, 1.0)
        contrib = np.where(odd, nuk[None, :] / safe_denom, 0.0)
        n_minus_1[start:end] = contrib.sum(axis=1)
    n = 1.0 + (4.0 * dnu / np.pi) * n_minus_1

    # Anchor: shift so n(anchor_nu_cm) == n_anchor. K-K of a band-limited k
    # gives n -> n_inf at large nu; the anchor effectively sets n_inf.
    if anchor_nu_cm is not None:
        if anchor_nu_cm < nu_cm.min() or anchor_nu_cm > nu_cm.max():
            n_at_anchor = 1.0  # outside grid: K-K integral ~ 0 there
        else:
            n_at_anchor = float(np.interp(anchor_nu_cm, nu_cm, n))
        n += (n_anchor - n_at_anchor)
    return n


def make_ocn_table_synthetic(*, nu_lo=500.0, nu_hi=4000.0, dnu=0.5,
                             nu0=OCN_BAND_NU0_CM,
                             fwhm=OCN_BAND_FWHM_CM,
                             A_prime=OCN_BAND_APRIME,
                             n_anchor=OCN_VIS_N0,
                             save_path=None):
    """
    Build OCN- (n, k) table using the Gerakines+25 band strength and a
    Gaussian band shape. Returns an astropy Table with Wavelength (um), k,
    and n columns plus metadata matching the icemodels convention.

    Parameters
    ----------
    nu_lo, nu_hi, dnu : float
        Wavenumber grid in cm^-1.
    nu0, fwhm : float
        Band centre and FWHM in cm^-1.
    A_prime : float
        Integrated band strength (cm molecule^-1).
    n_anchor : float
        Visible refractive index used to anchor the K-K solution.
    save_path : str, optional
        If given, write the resulting Table as ECSV.
    """
    nu = np.arange(nu_lo, nu_hi + 0.5 * dnu, dnu)
    k = gaussian_k_from_aprime(nu, nu0=nu0, fwhm=fwhm, A_prime=A_prime)
    n = kk_maclaurin(nu, k, n_anchor=n_anchor)

    wl_um = (1.0 / nu) * 1e4  # cm^-1 -> um
    order = np.argsort(wl_um)
    tbl = Table({
        'Wavelength': wl_um[order] * u.um,
        'k': k[order],
        'n': n[order],
        'Wavenumber': nu[order] * u.cm**-1,
    })
    tbl.meta['composition'] = 'OCN- (1)'
    tbl.meta['molecule'] = 'OCN-'
    tbl.meta['author'] = 'Gerakines, Materese & Hudson 2025'
    tbl.meta['database'] = 'gerakines2025_synthetic'
    tbl.meta['temperature'] = OCN_TEMPERATURE_K * u.K
    tbl.meta['density'] = OCN_MATRIX_DENSITY
    tbl.meta['molwt'] = OCN_MOLWT
    tbl.meta['band_centre_cm-1'] = nu0
    tbl.meta['band_fwhm_cm-1'] = fwhm
    tbl.meta['A_prime_cm_per_molecule'] = A_prime
    tbl.meta['method'] = ('Synthetic Gaussian k profile constrained by the '
                          'Gerakines+25 OCN- band strength; n via Maclaurin K-K')

    if save_path:
        tbl.write(save_path, overwrite=True)
    return tbl


def read_hudson2024_hnco_nk(xlsx_path=None, save_path=None):
    """
    Read Hudson et al. 2024 amorphous-HNCO n,k from the Cosmic Ice Lab
    Excel file. Returns an astropy Table with Wavelength (um), k, n.

    The default path is the file in icemodels/data/ downloaded from
    https://science.gsfc.nasa.gov/691/cosmicice/constants/HNCO-H2CO-HCOOH/HNCO_H2CO_HCOOH_n_k_ApJ_2024.xlsx
    """
    if xlsx_path is None:
        xlsx_path = os.path.join(
            optical_constants_cache_dir,
            'HNCO_H2CO_HCOOH_Hudson2024',
            'HNCO_H2CO_HCOOH_n_k_ApJ_2024.xlsx')
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb['HNCO']
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    nu = []
    n = []
    k = []
    for r in rows:
        # Mid-IR triplet sits in columns D, E, F (indices 3,4,5).
        try:
            nu_val = float(r[3])
            n_val = float(r[4])
            k_val = float(r[5])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(nu_val) and np.isfinite(n_val) and np.isfinite(k_val)):
            continue
        nu.append(nu_val)
        n.append(n_val)
        k.append(k_val)
    nu = np.asarray(nu)
    n = np.asarray(n)
    k = np.asarray(k)

    wl_um = 1e4 / nu
    order = np.argsort(wl_um)
    tbl = Table({
        'Wavelength': wl_um[order] * u.um,
        'k': k[order],
        'n': n[order],
        'Wavenumber': nu[order] * u.cm**-1,
    })
    tbl.meta['composition'] = 'HNCO (1)'
    tbl.meta['molecule'] = 'HNCO'
    tbl.meta['author'] = 'Hudson et al. 2024'
    tbl.meta['database'] = 'cosmicicelab'
    tbl.meta['temperature'] = 10 * u.K
    tbl.meta['density'] = 1.102 * u.g / u.cm**3       # Gerakines+25 sec 2
    tbl.meta['molwt'] = 43.025 * u.Da

    if save_path:
        tbl.write(save_path, overwrite=True)
    return tbl


GERAKINES2025_ICES_URL = ('https://science.gsfc.nasa.gov/691/cosmicice/'
                          'spectra/HNCO_Ices/Three_HNCO_ices.xlsx')
GERAKINES2025_ICES_FILE = 'Three_HNCO_ices.xlsx'
# column pairs (wavenumber, absorbance) for each ice in the spreadsheet
GERAKINES2025_ICE_COLUMNS = {'HNCO': (3, 4),
                             'H2O_HNCO_10_1': (7, 8),
                             'H2O_HNCO_NH3_10_1_1': (11, 12)}


def download_gerakines2025_ices(target=None):
    """Fetch the Gerakines+2025 HNCO-ice spectra from the Cosmic Ice Lab."""
    import requests
    if target is None:
        target = os.path.join(optical_constants_cache_dir,
                              GERAKINES2025_ICES_FILE)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    resp = requests.get(GERAKINES2025_ICES_URL, timeout=120)
    resp.raise_for_status()
    with open(target, 'wb') as fh:
        fh.write(resp.content)
    return target


def load_gerakines2025_ices(path=None, download=True):
    """
    Measured 10 K absorbance spectra behind Gerakines+2025 Figures 1 & 2.

    Three amorphous ices, 450-5000 cm^-1: pure HNCO, H2O+HNCO (10:1), and
    H2O+HNCO+NH3 (~10:1:1).  The OCN- band at 2170 cm^-1 is present only in
    the NH3-bearing mixture, since OCN- forms by the acid-base reaction
    HNCO + NH3 -> NH4+ + OCN-.

    Downloads the spreadsheet on first use, mirroring the other
    ``download_*`` helpers in this package.

    Returns an astropy Table with columns Wavenumber, Wavelength, and
    absorbance_<ice>.
    """
    import pandas as pd
    if path is None:
        path = os.path.join(optical_constants_cache_dir,
                            GERAKINES2025_ICES_FILE)
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(path)
        download_gerakines2025_ices(path)

    raw = pd.read_excel(path, header=None)
    cols = {}
    for name, (cnu, cab) in GERAKINES2025_ICE_COLUMNS.items():
        nu = pd.to_numeric(raw[cnu], errors='coerce').values
        ab = pd.to_numeric(raw[cab], errors='coerce').values
        good = np.isfinite(nu) & np.isfinite(ab)
        order = np.argsort(nu[good])
        cols[name] = (nu[good][order], ab[good][order])

    ref = cols['HNCO'][0]
    tbl = Table()
    tbl['Wavenumber'] = ref / u.cm
    tbl['Wavelength'] = (1e4 / ref) * u.um
    for name, (nu, ab) in cols.items():
        tbl['absorbance_' + name] = np.interp(ref, nu, ab)
    tbl.meta['author'] = 'Gerakines, Materese & Hudson 2025'
    tbl.meta['reference'] = ('Gerakines, Materese & Hudson 2025, '
                             'MNRAS 537, 2918')
    tbl.meta['doi'] = '10.1093/mnras/staf192'
    tbl.meta['url'] = GERAKINES2025_ICES_URL
    tbl.meta['temperature'] = OCN_TEMPERATURE_K * u.K
    return tbl


def ocn_absorbance_from_mix(tbl=None, *, lo=OCN_INT_LO_CM, hi=OCN_INT_HI_CM,
                            base_lo=(2100.0, 2130.0), base_hi=(2205.0, 2230.0)):
    """
    Isolate the OCN- band absorbance from the measured mixture.

    The H2O+HNCO (10:1) ice is the NH3-free control for H2O+HNCO+NH3
    (~10:1:1): differencing removes the H2O and HNCO contributions
    experimentally rather than by subtracting literature optical constants,
    which is what step 2 of the full K-K recipe would otherwise require.
    A linear baseline fitted to the flanking windows removes the residual
    continuum offset between the two films.

    Returns (nu_cm, absorbance) over [lo, hi].
    """
    if tbl is None:
        tbl = load_gerakines2025_ices()
    nu = np.asarray(tbl['Wavenumber'], dtype=float)
    diff = (np.asarray(tbl['absorbance_H2O_HNCO_NH3_10_1_1'], dtype=float)
            - np.asarray(tbl['absorbance_H2O_HNCO_10_1'], dtype=float))
    wing = (((nu >= base_lo[0]) & (nu <= base_lo[1]))
            | ((nu >= base_hi[0]) & (nu <= base_hi[1])))
    coef = np.polyfit(nu[wing], diff[wing], 1)
    sel = (nu >= lo) & (nu <= hi)
    return nu[sel], diff[sel] - np.polyval(coef, nu[sel])


def make_ocn_table_from_mix_absorbance(*, nu_lo=500.0, nu_hi=4000.0, dnu=0.5,
                                       density=OCN_MATRIX_DENSITY,
                                       molwt=OCN_MOLWT,
                                       A_prime=OCN_BAND_APRIME,
                                       n_anchor=OCN_VIS_N0, save_path=None):
    """
    OCN- optical constants from the MEASURED Gerakines+2025 mixture spectrum.

    Unlike ``make_ocn_table_synthetic``, the band shape here is the laboratory
    profile, not a Gaussian.  The absolute scale is set by requiring closure
    against the measured band strength A' rather than by the film thickness,
    which is not distributed with the spectra; this is equivalent to steps 3-6
    of the thin-film recipe under the (excellent, tau << 1) weak-absorber
    approximation, and it inherits Gerakines+2025's calibration exactly.

    Returns a table with Wavelength, Wavenumber, k, n.
    """
    nu_band, a_band = ocn_absorbance_from_mix()
    grid = np.arange(nu_lo, nu_hi + dnu, dnu)
    a_grid = np.interp(grid, nu_band, a_band, left=0.0, right=0.0)
    a_grid[a_grid < 0] = 0.0

    # Beer's law up to the unknown thickness: k ∝ ln(10) * A / (4 pi nu)
    with np.errstate(divide='ignore', invalid='ignore'):
        k_shape = np.log(10.0) * a_grid / (4.0 * np.pi * grid)
    k_shape[~np.isfinite(k_shape)] = 0.0

    # scale so that the band integrates to the measured A'
    scale = A_prime / closure_aprime(grid, k_shape, density=density, molwt=molwt)
    k = k_shape * scale
    n = kk_maclaurin(grid, k, n_anchor=n_anchor)

    wl_um = 1e4 / grid
    order = np.argsort(wl_um)
    tbl = Table()
    tbl['Wavelength'] = wl_um[order] * u.um
    tbl['Wavenumber'] = grid[order] / u.cm
    tbl['k'] = k[order]
    tbl['n'] = n[order]
    tbl.meta['molecule'] = 'OCN-'
    tbl.meta['composition'] = 'OCN- (1)'
    tbl.meta['author'] = 'Gerakines, Materese & Hudson 2025'
    tbl.meta['database'] = 'gerakines2025_measured'
    tbl.meta['reference'] = 'Gerakines, Materese & Hudson 2025, MNRAS 537, 2918'
    tbl.meta['doi'] = '10.1093/mnras/staf192'
    tbl.meta['temperature'] = OCN_TEMPERATURE_K * u.K
    tbl.meta['density'] = density
    tbl.meta['molwt'] = molwt
    tbl.meta['band_strength'] = A_prime
    tbl.meta['derivation'] = ('measured H2O+HNCO+NH3 minus H2O+HNCO absorbance, '
                              'Beer law, scaled to A prime, K-K for n')
    if save_path:
        tbl.write(save_path, overwrite=True)
    return tbl
