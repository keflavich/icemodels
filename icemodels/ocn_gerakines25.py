"""
OCN-, HNCO, and supporting laboratory ice optical constants for the Gerakines,
Materese & Hudson 2025 cyanate-ion band-strength measurement.

Two paths to optical constants are provided:

1. ``make_ocn_table_synthetic``: construct k(nu) for OCN- as a Gaussian band
   anchored to the band strength A' = 1.51e-16 cm molecule^-1 measured by
   Gerakines+25 in H2O+HNCO+NH3 (~10:1:1) at 10 K (Table 1 of the paper),
   then derive n(nu) via Kramers-Kronig (Maclaurin's principal-value method).
   This is the recommended quick path: it directly uses the Gerakines+25
   result and a literature band shape, without needing the raw mixed-ice
   absorbance spectrum (which is not yet released as a digital file).

2. ``make_ocn_table_from_mix_absorbance``: full Hudgins-1993/Rocha-2014/
   Gerakines-Hudson-2020 thin-film K-K + Fresnel iteration. Requires the
   raw H2O+HNCO+NH3 absorbance A(nu) plus thickness d, density rho, n0(670),
   and the substrate index n_s. Subtracts H2O/HNCO/NH3/NH4+ contributions
   using published n,k (Mastrapa H2O, Hudson 2024 HNCO, LIDA NH3/NH4+),
   converts the OCN- residual via Beer's law, runs K-K, then iterates the
   Fresnel three-layer transmission against the measured T = 10^(-A) until
   k converges. **Currently a stub** -- raw absorbance file is not in the
   data tree; populate `gerakines_2025_data_path` and finish the loop when
   the data are available.

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
    """Recover A' from k(nu) by integrating over [lo, hi] cm^-1."""
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


def make_ocn_table_from_mix_absorbance(*args, **kwargs):
    """
    Full Hudgins-1993/Rocha-2014 thin-film K-K + Fresnel iteration on the
    measured H2O+HNCO+NH3 (~10:1:1) absorbance from Gerakines+25.

    Stub: implementation deferred until the raw absorbance spectrum is
    available as a digital data file (paper currently provides only a
    figure plus the integrated band strength). The pieces needed:

      1. Read the measured A(nu) and ice thickness d.
      2. Subtract H2O (Mastrapa+2009/Hudson+2025), HNCO (Hudson+2024),
         NH3, and NH4+ (LIDA) absorbance contributions using their
         literature optical constants and the ice composition ratios.
      3. Beer's law: k0(nu) = ln(10) * A_OCN(nu) / (4 pi nu d).
      4. K-K (kk_maclaurin) anchored at n0(670 nm) -> n(nu).
      5. Fresnel three-layer (vacuum / ice / CsI) forward-model T_calc;
         compare to T_meas = 10^(-A); update k by residual; redo K-K.
         Iterate to convergence.
      6. Closure: integrate (4 pi M / rho N_A) * nu * k over the band
         and require match to A' to within ~5 per cent.

    See module docstring for references.
    """
    raise NotImplementedError(
        "Raw H2O+HNCO+NH3 absorbance from Gerakines+25 not yet on disk. "
        "Use make_ocn_table_synthetic() until the spectrum is available.")
