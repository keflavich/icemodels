"""
Tests for the Gerakines+2025 OCN- optical constants.

The first two need ``Three_HNCO_ices.xlsx`` from the Cosmic Ice Laboratory, so
they are marked ``remote_data`` and only run under ``pytest --remote-data``;
the rest of the suite stays offline, as elsewhere in this package.
"""
import numpy as np
import pytest


@pytest.mark.remote_data
def test_measured_profile_closure():
    """The measured-absorbance OCN- table must reproduce the Gerakines+2025 A'."""
    from icemodels.ocn_gerakines25 import (make_ocn_table_from_mix_absorbance,
                                           closure_aprime, OCN_BAND_APRIME)
    t = make_ocn_table_from_mix_absorbance()
    nu = np.array(t['Wavenumber'])
    k = np.array(t['k'])
    assert np.isclose(closure_aprime(nu, k), OCN_BAND_APRIME, rtol=1e-3)
    # band peak at the published position
    assert abs(nu[np.argmax(k)] - 2170.0) < 3.0


@pytest.mark.remote_data
def test_ocn_band_only_in_nh3_mixture():
    """OCN- forms only when NH3 is present (acid-base product)."""
    from icemodels.ocn_gerakines25 import load_gerakines2025_ices
    t = load_gerakines2025_ices()
    nu = np.array(t['Wavenumber'])
    band = (nu > 2150) & (nu < 2190)
    wing = ((nu > 2100) & (nu < 2130)) | ((nu > 2205) & (nu < 2230))
    depth = {}
    for ice in ('HNCO', 'H2O_HNCO_10_1', 'H2O_HNCO_NH3_10_1_1'):
        a = np.array(t['absorbance_' + ice])
        depth[ice] = a[band].max() - np.median(a[wing])
    assert depth['H2O_HNCO_NH3_10_1_1'] > 5 * depth['H2O_HNCO_10_1']


def test_closure_is_row_order_independent():
    """
    closure_aprime must recover A' from a table however its rows are ordered.

    Tables here are stored wavelength-ascending, i.e. wavenumber-*descending*,
    so integrating in row order would otherwise return -A'. Needs no download.
    """
    from icemodels.ocn_gerakines25 import (make_ocn_table_synthetic,
                                           closure_aprime, OCN_BAND_APRIME)
    t = make_ocn_table_synthetic()
    nu = np.array(t['Wavenumber'])
    k = np.array(t['k'])
    assert nu[0] > nu[-1]        # wavelength-ascending, as stored
    assert np.isclose(closure_aprime(nu, k), OCN_BAND_APRIME, rtol=1e-3)
    assert np.isclose(closure_aprime(nu[::-1], k[::-1]), OCN_BAND_APRIME, rtol=1e-3)
